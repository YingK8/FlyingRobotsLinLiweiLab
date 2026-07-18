#include "PwmController.h"
#include <Arduino.h>
#include <math.h>

#ifndef APB_CLK_FREQ
#define APB_CLK_FREQ 80000000UL
#endif

namespace {
bool configureCarrierTimerForFreq(ledc_mode_t mode,
                                  ledc_timer_t timer,
                                  uint32_t freqHz,
                                  uint8_t &selectedBits) {
    // Estimate the highest useful resolution for the requested frequency.
    uint32_t ticksPerCycle = (freqHz > 0) ? (uint32_t)(APB_CLK_FREQ / freqHz) : 0;
    uint8_t startBits = 1;
    while ((startBits < 15) && ((1UL << startBits) <= ticksPerCycle)) {
        startBits++;
    }
    if (startBits > 1) startBits--; // Back off to last valid estimate.

    ledc_clk_cfg_t clkCfg = LEDC_AUTO_CLK;
#ifdef LEDC_USE_APB_CLK
    clkCfg = LEDC_USE_APB_CLK; // More deterministic than AUTO for carrier frequencies.
#endif

    for (int bits = startBits; bits >= 1; --bits) {
        ledc_timer_config_t ledc_timer = {
            .speed_mode = mode,
            .duty_resolution = (ledc_timer_bit_t)bits,
            .timer_num = timer,
            .freq_hz = freqHz,
            .clk_cfg = clkCfg
        };

        if (ledc_timer_config(&ledc_timer) == ESP_OK) {
            selectedBits = (uint8_t)bits;
            return true;
        }
    }

    return false;
}
} // namespace

PwmController* PwmController::_isrInstance = nullptr;

PwmController::PwmController(const gpio_num_t* pins, const float* phaseOffsetsDegrees, const float* dutyCycles, int numChannels) {
    _numChannels = numChannels;
    _periodicTimer = nullptr;
    _syncPin = GPIO_NUM_NC;  // Initialize to safe value to prevent garbage GPIO access
    
    // FIX 4: Initialize the RTOS spinlock properly so it doesn't crash on first lock
    _spinlock = portMUX_INITIALIZER_UNLOCKED;

    _pins = new gpio_num_t[_numChannels];
    _phaseOffsetsPct = new float[_numChannels];
    _dutyCycles = new float[_numChannels];
    _params = new PhaseParams[_numChannels];

    _lastSyncTimeUs = 0;
    _averagedPeriodUs = 20000;
    _lastIsrTimeUs = 0;
    _filterIdx = 0;
    _firstSyncReceived = false;

    // Carrier PWM initialization (multi-channel)
    _carrierPinsArray = nullptr;
    _carrierFreqHz = 10000.0;
    _carrierDutyCyclePct = nullptr;
    _carrierSpeedMode = LEDC_LOW_SPEED_MODE;
    _carrierTimer = LEDC_TIMER_0;
    _carrierDutyResolutionBits = 10;

    for(int i=0; i<FREQ_FILTER_SIZE; i++) _periodBuffer[i] = 20000;

    for (int i = 0; i < _numChannels; i++) {
        _pins[i] = pins[i];
        _phaseOffsetsPct[i] = constrain(phaseOffsetsDegrees[i], 0.0, 360.0) / 360.0;
        _dutyCycles[i] = constrain(dutyCycles[i], 0.0, 100.0);
    }
}

PwmController::~PwmController() {
    if (_periodicTimer) {
        esp_timer_stop(_periodicTimer);
        esp_timer_delete(_periodicTimer);
    }
    delete[] _pins; 
    delete[] _phaseOffsetsPct; 
    delete[] _dutyCycles; 
    delete[] _params; 
    if (_carrierPinsArray) delete[] _carrierPinsArray;
    if (_carrierDutyCyclePct) delete[] _carrierDutyCyclePct;
    if (_carrierLedcConfigured) delete[] _carrierLedcConfigured;
    if (_carrierLastDutyTicks) delete[] _carrierLastDutyTicks;
    if (_sense) delete _sense;
    if (_balance) delete _balance;
}

void PwmController::begin(float initialFreqHz) {
    for(int i = 0; i < _numChannels; i++) {
        // Validate pin before attempting to use it
        if (_pins[i] == GPIO_NUM_NC || _pins[i] > GPIO_NUM_39) {
            continue; // Skip invalid pins
        }
        gpio_reset_pin(_pins[i]);
        gpio_set_direction(_pins[i], GPIO_MODE_OUTPUT);
        
        // FIX 1: Set idle state to HIGH (1). 
        // Because of the logic inverter, setting this HIGH ensures the H-Bridge receives LOW, 
        // protecting it from phantom-powering via ESD diodes.
        gpio_set_level(_pins[i], 1); 
    }

    setGlobalFrequency(initialFreqHz);

    // COMPILE-TIME OPTIMIZED SETUP
    #if USE_SYNC
        #if SYNC_AS_SERVER
            // Master: Output Sync
            gpio_reset_pin(_syncPin);
            gpio_set_direction(_syncPin, GPIO_MODE_OUTPUT);
            gpio_set_level(_syncPin, 0);
            _lastSyncTimeUs = esp_timer_get_time();
        #else
            // Client: Input Sync
            gpio_reset_pin(_syncPin);
            gpio_set_direction(_syncPin, GPIO_MODE_INPUT);
            gpio_set_pull_mode(_syncPin, GPIO_PULLDOWN_ONLY);
            _isrInstance = this;
            attachInterrupt(digitalPinToInterrupt(_syncPin), _onSyncInterrupt, RISING);
        #endif
    #else
        _lastSyncTimeUs = esp_timer_get_time();
    #endif

    const esp_timer_create_args_t timer_args = {
        .callback = &_timerCallback,
        .arg = this,
        .dispatch_method = ESP_TIMER_TASK, // FIX 2: Moved to hardware ISR context to prevent RTOS starvation
        .name = "pwm_gen"
    };

    esp_timer_create(&timer_args, &_periodicTimer);
    esp_timer_start_periodic(_periodicTimer, 25); 
}

void IRAM_ATTR PwmController::_timerCallback(void* arg) {
    PwmController* self = (PwmController*)arg;
    int64_t now = esp_timer_get_time();

    int64_t lastSync;
    int64_t period;
    bool dc;

    // When dispatch_method=ESP_TIMER_TASK, callback runs in task context, not ISR.
    // Use portENTER_CRITICAL (task) not portENTER_CRITICAL_ISR
    portENTER_CRITICAL(&self->_spinlock);
    lastSync = self->_lastSyncTimeUs;
    period = self->_averagedPeriodUs;
    dc = self->_dcMode;
    portEXIT_CRITICAL(&self->_spinlock);

    // Client fallback: If waiting for first sync, use default period
    #if USE_SYNC && !SYNC_AS_SERVER
        if (!self->_firstSyncReceived) {
            // Defaults to 20000us if period is invalid
            if (period < 100) period = 20000; 
        }
    #endif

    // FIX 3: Safe 32-bit math for ISR. 
    // This prevents the ESP32 from crashing while trying to load 64-bit division 
    // library routines from flash memory during an interrupt.
    int64_t delta = now - lastSync;
    if (delta < 0) delta += period; // Handle case if sync pushed lastSync slightly into the future

    uint32_t delta32 = (uint32_t)delta;
    uint32_t period32 = (uint32_t)period;
    uint32_t timeInCycle = delta32 % period32;

    // DC mode: freeze the phase so the active-channel pattern is static (the
    // field doesn't rotate). Current is still gated by the carrier, so DC + 0%
    // carrier = fully off. Frozen at cycle-start (t=0) for a well-defined pattern.
    if (dc) timeInCycle = 0;

    // Master Sync Generation
    #if USE_SYNC && SYNC_AS_SERVER
        // Set sync high for first 50% of cycle
        gpio_set_level(self->_syncPin, (timeInCycle < (period32 / 2)) ? 1 : 0);
    #endif

    // Channel Generation
    #if USE_SYNC && SYNC_AS_SERVER
        const int channelLimit = 1;
    #else
        const int channelLimit = self->_numChannels;
    #endif

    for (int i = 0; i < channelLimit; i++) {
        // Skip invalid GPIO pins to prevent crashes
        if (self->_pins[i] == GPIO_NUM_NC || self->_pins[i] > GPIO_NUM_39) continue;
        
        // Local copies for speed
        uint32_t start = (uint32_t)self->_params[i].startUs;
        uint32_t end = (uint32_t)self->_params[i].endUs;
        
        bool active = self->_params[i].wraps ? 
                      (timeInCycle >= start || timeInCycle < end) : 
                      (timeInCycle >= start && timeInCycle < end);
        
        // ACTIVE LOW LOGIC: Due to the NC7SZ04P5X inverter, to turn the H-Bridge ON (HIGH), 
        // the ESP32 must output LOW (0). To turn it OFF, the ESP32 outputs HIGH (1).
        gpio_set_level(self->_pins[i], active ? 0 : 1);
    }
}

void IRAM_ATTR PwmController::_onSyncInterrupt() {
    #if USE_SYNC && !SYNC_AS_SERVER
        if (!_isrInstance) return;

        int64_t now = esp_timer_get_time();
        
        portENTER_CRITICAL_ISR(&_isrInstance->_spinlock);
        
        // 1. Measure Period with Glitch Filter
        if (_isrInstance->_lastIsrTimeUs > 0) {
            int64_t delta = now - _isrInstance->_lastIsrTimeUs;
            
            if (delta > 2000) {
                // Missed Pulse Compensation (if delta ~2x average)
                int64_t currentAvg = _isrInstance->_averagedPeriodUs;
                int64_t sample = delta;
                
                if (currentAvg > 0 && abs(delta - 2*currentAvg) < (currentAvg / 2)) {
                    sample = delta / 2;
                }

                _isrInstance->_periodBuffer[_isrInstance->_filterIdx] = sample;
                _isrInstance->_filterIdx = (_isrInstance->_filterIdx + 1) % FREQ_FILTER_SIZE;
                
                int64_t sum = 0;
                for(int i=0; i<FREQ_FILTER_SIZE; i++) sum += _isrInstance->_periodBuffer[i];
                _isrInstance->_averagedPeriodUs = sum / FREQ_FILTER_SIZE;
            }
        }
        
        _isrInstance->_lastIsrTimeUs = now;
        
        // 2. APPLY LATENCY COMPENSATION
        _isrInstance->_lastSyncTimeUs = now - SYNC_LATENCY_US;
        
        _isrInstance->_firstSyncReceived = true;
        
        portEXIT_CRITICAL_ISR(&_isrInstance->_spinlock);
    #endif
}

void PwmController::enableSync(gpio_num_t syncPin) {
    #if USE_SYNC
        _syncPin = syncPin;
    #endif
}

void PwmController::updatePhaseParams(int channel) {
    #if USE_SYNC && SYNC_AS_SERVER
        if (channel > 0) return;
    #endif
    
    float period = (float)_averagedPeriodUs; 
    float width = period * _dutyCycles[channel] / 100.0;
    
    #if USE_SYNC && SYNC_AS_SERVER
        float effectivePhasePct = 0.0f;
    #else
        float effectivePhasePct = _phaseOffsetsPct[channel];
    #endif

    _params[channel].startUs = (unsigned long)(period * effectivePhasePct);
    _params[channel].endUs = _params[channel].startUs + (unsigned long)width;

    if (_params[channel].endUs > period) {
        _params[channel].endUs -= (unsigned long)period;
        _params[channel].wraps = true;
    } else {
        _params[channel].wraps = false;
    }
}

void PwmController::setGlobalFrequency(float newHz) {
    #if USE_SYNC && !SYNC_AS_SERVER
        return; // Client ignores manual freq
    #endif

    // DC / stationary: a non-positive or sub-microhertz frequency (or NaN) means
    // "don't rotate the field." Hold the commutation pattern static instead of
    // dividing by zero (1e6/newHz would). The carrier still sets current, so DC
    // with 0% carrier is a safe fully-stopped idle; this is begin()'s default.
    if (!(newHz >= 1e-6f)) {           // !(>=) also catches NaN
        portENTER_CRITICAL(&_spinlock);
        _dcMode = true;
        _globalFreqHz = 0.0f;
        // _averagedPeriodUs keeps its last valid value (constructor seeds 20000us)
        // so the width/duty math and the ISR modulo stay well-defined; the ISR
        // freezes the phase while _dcMode is set, so no rotation occurs.
        for (int i = 0; i < _numChannels; i++) updatePhaseParams(i);
        portEXIT_CRITICAL(&_spinlock);
        return;
    }
    _dcMode = false;

    int64_t newPeriod = (int64_t)(1000000.0 / newHz);
    int64_t now = esp_timer_get_time();
    _globalFreqHz = newHz;

    portENTER_CRITICAL(&_spinlock);

    // === PHASE CONTINUITY CORRECTION ===
    if (_averagedPeriodUs > 0) {
        int64_t oldPos = (now - _lastSyncTimeUs) % _averagedPeriodUs;
        if (oldPos < 0) oldPos += _averagedPeriodUs;

        double phaseRatio = (double)oldPos / (double)_averagedPeriodUs;
        int64_t newPos = (int64_t)(phaseRatio * newPeriod);

        _lastSyncTimeUs = now - newPos;
    }

    _averagedPeriodUs = newPeriod;
    for(int i=0; i<FREQ_FILTER_SIZE; i++) _periodBuffer[i] = newPeriod;
    
    // Update params immediately inside lock to prevent tearing
    for(int i=0; i<_numChannels; i++) updatePhaseParams(i);
    
    portEXIT_CRITICAL(&_spinlock);
}

void PwmController::setDutyCycle(int channel, float dutyPercent) {
    if (channel < 0 || channel >= _numChannels) return;

    portENTER_CRITICAL(&_spinlock);
    _dutyCycles[channel] = constrain(dutyPercent, 0.0, 100.0);
    updatePhaseParams(channel);
    portEXIT_CRITICAL(&_spinlock);
}

void PwmController::setPhase(int channel, float degrees) {
    #if USE_SYNC && SYNC_AS_SERVER
        return; // Master ignores phase
    #endif

    float pct = degrees / 360.0;
    while(pct >= 1.0) pct -= 1.0;
    while(pct < 0.0) pct += 1.0;
    
    portENTER_CRITICAL(&_spinlock);
    _phaseOffsetsPct[channel] = pct;
    updatePhaseParams(channel);
    portEXIT_CRITICAL(&_spinlock);
}

float PwmController::getFrequency() const {
    if (_dcMode) return 0.0f;
    return 1000000.0 / _averagedPeriodUs;
}

float PwmController::getPhase(int channel) const { 
    #if USE_SYNC && SYNC_AS_SERVER
        return 0.0;
    #else
        return _phaseOffsetsPct[channel] * 360.0;
    #endif
}

float PwmController::getDutyCycle(int channel) const { return _dutyCycles[channel]; }

float PwmController::getCarrierDutyCycle(int channel) const {
    if (!_carrierDutyCyclePct || channel < 0 || channel >= _numChannels) return 0.0f;
    return _carrierDutyCyclePct[channel];
}

void PwmController::run() {
    static unsigned long lastUpdate = 0;
    if (esp_timer_get_time() - lastUpdate > 100000) {
        lastUpdate = esp_timer_get_time();

        portENTER_CRITICAL(&_spinlock);
        for(int i=0; i<_numChannels; i++) {
            updatePhaseParams(i);
        }
        portEXIT_CRITICAL(&_spinlock);
    }

    // Opted-in current sensing / overcurrent latch / PI balance. No-op if the
    // main never called enableCurrentSense(), so passthrough experiments are
    // untouched. Runs every call (NOT gated by the 100ms phase-drift block
    // above): the ADC is paced internally and the PI loop wants every iteration.
    _serviceCurrentLoop();
}

void PwmController::enableCurrentSense(const gpio_num_t *adcPins,
                                         const float *sensPerVolt,
                                         float overcurrentTripA) {
    if (!adcPins || !sensPerVolt) return;
    if (!_sense) _sense = new CurrentSense(adcPins, sensPerVolt);
    _sense->seed(); // coils must be OFF here (forceAllGatesLow + carriers at 0)
    _overcurrentTripA = overcurrentTripA;
    _tripped = false;
    _lastSenseUs = micros();
    _lastBalanceUs = _lastSenseUs;
}

void PwmController::enableCurrentBalance(const BalanceConfig &cfg,
                                           float startDuty) {
    // Balance needs the sensed currents; enableCurrentSense() must precede this.
    if (!_sense) return;
    if (!_balance) _balance = new CurrentBalanceController(cfg);
    _startDuty = startDuty;
    _balance->reset(startDuty);
    for (int i = 0; i < 4; i++) {
        _ceiling[i] = NAN;        // no carrier commanded yet -> parked off
        _balanceDuty[i] = startDuty;
    }
    _lastBalanceUs = micros();
}

void PwmController::setBalanceGains(float kp, float ki, float kd) {
    if (_balance) _balance->setGains(kp, ki, kd);
}

void PwmController::setBalanceRamp(float pctPerMs) {
    if (_balance) _balance->setRamp(pctPerMs);
}

const float *PwmController::measuredCurrents() const {
    return _sense ? _sense->i_meas : nullptr;
}

float PwmController::carrierCeiling(int channel) const {
    if (channel < 0 || channel >= _numChannels) return 0.0f;
    if (_balance && channel < 4) return _ceiling[channel];
    return getCarrierDutyCycle(channel);
}

void PwmController::_serviceCurrentLoop() {
    if (!_sense) return;

    unsigned long nowUs = micros();
    // ADC pacing: the ESP32 ADC needs real settling time between conversions,
    // separate from the control-loop rate below.
    float dtSenseMs = (float)(nowUs - _lastSenseUs) / 1000.0f;
    if (dtSenseMs >= 1.0f) {
        _sense->update(dtSenseMs);
        _lastSenseUs = nowUs;
    }

    // Hard overcurrent latch: once tripped, force every carrier to 0 and stay
    // there (only a reboot clears it), regardless of what the schedule commands.
    if (_overcurrentTripA > 0.0f && !_tripped) {
        for (int i = 0; i < _numChannels; i++) {
            if (i < 4 && _sense->i_meas[i] > _overcurrentTripA) {
                _tripped = true;
                break;
            }
        }
    }
    if (_tripped) {
        for (int i = 0; i < _numChannels; i++) _writeCarrier(i, 0.0f);
        return;
    }

    if (_balance) {
        float dtCtrlMs = (float)(nowUs - _lastBalanceUs) / 1000.0f;
        if (dtCtrlMs <= 0.0f) dtCtrlMs = 0.001f; // guard div-by-zero only
        _lastBalanceUs = nowUs;
        _balance->step(_sense->i_meas, dtCtrlMs, _ceiling, _balanceDuty);
        for (int i = 0; i < _numChannels && i < 4; i++)
            _writeCarrier(i, _balanceDuty[i]);
    }
}

void PwmController::initCarrierPWM(const gpio_num_t* pins, float freqHz, const float* dutyPercents) {
    if (!pins || !dutyPercents || freqHz <= 0.0f) return;

    if (!_carrierPinsArray) {
        _carrierPinsArray = new gpio_num_t[_numChannels];
        _carrierDutyCyclePct = new float[_numChannels];
        _carrierLedcConfigured = new bool[_numChannels];
        _carrierLastDutyTicks = new uint32_t[_numChannels];
    }

    _carrierFreqHz = freqHz;

    if (!configureCarrierTimerForFreq(_carrierSpeedMode,
                                      _carrierTimer,
                                      (uint32_t)freqHz,
                                      _carrierDutyResolutionBits)) {
        return;
    }

    uint32_t dutyMax = (1UL << _carrierDutyResolutionBits) - 1UL;

    for (int i = 0; i < _numChannels; i++) {
        gpio_num_t pin = pins[i];
        float duty = dutyPercents[i];
        _carrierPinsArray[i] = pin;
        _carrierDutyCyclePct[i] = duty;
        _carrierLedcConfigured[i] = false;
        _carrierLastDutyTicks[i] = 0;

        if (pin == GPIO_NUM_NC || pin > GPIO_NUM_39) continue;

        if (duty >= 100.0f) {
            _carrierDutyCyclePct[i] = 100.0f;
            gpio_reset_pin(pin);
            gpio_set_direction(pin, GPIO_MODE_OUTPUT);
            
            // 100% duty = carrier permanently ON. The carrier wires DIRECTLY into
            // the VNH5019 PWM pin (no inverter on this line), so full-on means
            // driving the pin HIGH.
            gpio_set_level(pin, 1);
            continue;
        }

        float clampedDuty = constrain(duty, 0.0f, 100.0f);

        // Active-high carrier: the pin drives the VNH5019 PWM input directly (no
        // inverter), so duty% = fraction of time the bridge is ON. output_invert=0
        // and standard duty are correct; do NOT invert here.
        _carrierDutyCyclePct[i] = clampedDuty;

        ledc_channel_config_t ledc_channel = {
            .gpio_num = pin,
            .speed_mode = _carrierSpeedMode,
            .channel = (ledc_channel_t)i,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = _carrierTimer,
            .duty = (uint32_t)((clampedDuty / 100.0f) * dutyMax),
            .hpoint = 0,
            .flags = {.output_invert = 0} // Can be set to 1 if you want LEDC to handle the hardware inversion automatically!
        };
        ledc_channel_config(&ledc_channel);
        _carrierLedcConfigured[i] = true;
        _carrierLastDutyTicks[i] = (uint32_t)((clampedDuty / 100.0f) * dutyMax);
    }
}

void PwmController::setCarrierDutyCycle(int channel, float dutyPercent) {
    if (channel < 0 || channel >= _numChannels) return;

    // Balance active: the schedule's commanded carrier is a CEILING, not a
    // direct drive. Stash it; run()/_serviceCurrentLoop() computes the actual
    // per-channel duty beneath it and writes the hardware. Passthrough
    // (balance off) writes straight through, exactly as before.
    if (_balance) {
        if (channel < 4) _ceiling[channel] = dutyPercent;
        return;
    }
    _writeCarrier(channel, dutyPercent);
}

void PwmController::_writeCarrier(int channel, float dutyPercent) {
    if (channel < 0 || channel >= _numChannels) return;
    if (!_carrierPinsArray || !_carrierDutyCyclePct) return;
    if (_carrierPinsArray[channel] == GPIO_NUM_NC) return;
    if (_carrierFreqHz <= 0.0f) return;

    if (dutyPercent >= 100.0f) {
        if (_carrierDutyCyclePct[channel] >= 100.0f && !_carrierLedcConfigured[channel]) {
            return; // already stopped, nothing to do
        }
        _carrierDutyCyclePct[channel] = 100.0f;
        // 100% duty = carrier permanently ON. No inverter on the carrier line, so
        // stop LEDC and park the pin HIGH (idle_level = 1) = full drive to the bridge.
        ledc_stop(_carrierSpeedMode, (ledc_channel_t)channel, 1);
        _carrierLedcConfigured[channel] = false; // pin must be re-attached on next PWM duty
        return;
    }

    float periodMs = 1000.0f / _carrierFreqHz;
    float minDutyCycle = (MIN_ON_OFF_MS / periodMs) * 100.0f;
    float maxDutyCycle = 100.0f - minDutyCycle;

    _carrierDutyCyclePct[channel] = constrain(dutyPercent, minDutyCycle, maxDutyCycle);

    uint32_t dutyMax = (1UL << _carrierDutyResolutionBits) - 1UL;
    uint32_t dutyValue = (uint32_t)((_carrierDutyCyclePct[channel] / 100.0f) * dutyMax);

    if (!_carrierLedcConfigured[channel]) {
        // (Re)attach the pin to LEDC. This is the only place the full channel
        // config runs; it resets hpoint and restarts the output, so doing it on
        // every duty update causes mid-cycle glitches on the carrier line.
        ledc_channel_config_t ledc_channel = {
            .gpio_num = _carrierPinsArray[channel],
            .speed_mode = _carrierSpeedMode,
            .channel = (ledc_channel_t)channel,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = _carrierTimer,
            .duty = dutyValue,
            .hpoint = 0,
            .flags = {.output_invert = 0}
        };
        esp_err_t err = ledc_channel_config(&ledc_channel);
        if (err != ESP_OK) {
            Serial.printf("[PwmController] ledc_channel_config ch%d pin%d failed: %d\n",
                          channel, (int)_carrierPinsArray[channel], (int)err);
            return; // stay unconfigured so the next call retries
        }
        // Explicit update: after ledc_stop() some IDF versions don't restart
        // the output from ledc_channel_config() alone.
        ledc_set_duty(_carrierSpeedMode, (ledc_channel_t)channel, dutyValue);
        ledc_update_duty(_carrierSpeedMode, (ledc_channel_t)channel);
        _carrierLedcConfigured[channel] = true;
        _carrierLastDutyTicks[channel] = dutyValue;
        return;
    }

    // Skip writes that don't change the duty at hardware resolution. This also
    // rate-limits ramps that call this function every loop() iteration.
    if (dutyValue == _carrierLastDutyTicks[channel]) return;

    // Glitch-free update: latched by hardware at the next PWM period boundary.
    ledc_set_duty(_carrierSpeedMode, (ledc_channel_t)channel, dutyValue);
    ledc_update_duty(_carrierSpeedMode, (ledc_channel_t)channel);
    _carrierLastDutyTicks[channel] = dutyValue;
}

void PwmController::shutdown(unsigned long rampMs) {
    // Snapshot the current carrier duty of every channel so each ramps from
    // wherever it is now (handles channels parked at 100%, which re-attach LEDC
    // automatically on the first sub-100% write).
    float startDuty[16];
    int n = _numChannels < 16 ? _numChannels : 16;
    for (int i = 0; i < n; i++)
        startDuty[i] = _carrierDutyCyclePct ? _carrierDutyCyclePct[i] : 0.0f;

    const int steps = 50;
    unsigned long stepMs = rampMs / steps;
    if (stepMs < 1) stepMs = 1;
    for (int s = 1; s <= steps; s++) {
        float frac = (float)s / (float)steps;          // 0 -> 1
        for (int i = 0; i < n; i++)
            setCarrierDutyCycle(i, startDuty[i] * (1.0f - frac));
        delay(stepMs);
    }

    // Force fully off (LEDC output held LOW = bridge disabled), then freeze the
    // phase GPIOs by stopping the periodic timer. Object/timer stay allocated.
    for (int i = 0; i < n; i++)
        setCarrierDutyCycle(i, 0.0f);
    if (_periodicTimer)
        esp_timer_stop(_periodicTimer);
}

bool PwmController::rampDownStep(float stepPct) {
    if (!_carrierDutyCyclePct) return true;
    if (stepPct <= 0.0f) stepPct = 0.1f;   // guard: must make progress
    bool allZero = true;
    for (int i = 0; i < _numChannels; i++) {
        float cur = _carrierDutyCyclePct[i];
        if (cur <= 0.0f) continue;         // already down
        float next = cur - stepPct;        // subtract from the LIVE value
        if (next < 0.0f) next = 0.0f;
        setCarrierDutyCycle(i, next);      // next < cur always -> monotonic
        if (next > 0.0f) allZero = false;
    }
    return allZero;
}