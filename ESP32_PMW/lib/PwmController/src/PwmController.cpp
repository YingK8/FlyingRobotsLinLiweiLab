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
    uint32_t ticksPerCycle = (freqHz > 0) ? (uint32_t)(APB_CLK_FREQ / freqHz) : 0;
    uint8_t startBits = 1;
    while ((startBits < 15) && ((1UL << startBits) <= ticksPerCycle)) {
        startBits++;
    }
    if (startBits > 1) startBits--;

    ledc_clk_cfg_t clkCfg = LEDC_AUTO_CLK;
#ifdef LEDC_USE_APB_CLK
    clkCfg = LEDC_USE_APB_CLK;
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
}

PwmController* PwmController::_isrInstance = nullptr;

PwmController::PwmController(const gpio_num_t* pins, const float* phaseOffsetsDegrees, const float* dutyCycles, int numChannels) {
    _numChannels = numChannels;
    _periodicTimer = nullptr;
    _syncPin = GPIO_NUM_NC;

    _spinlock = portMUX_INITIALIZER_UNLOCKED;

    _pins = new gpio_num_t[_numChannels];
    _phaseOffsetsPct = new float[_numChannels];
    _basePhaseDeg = new float[_numChannels];
    _dutyCycles = new float[_numChannels];
    _params = new PhaseParams[_numChannels];

    _lastSyncTimeUs = 0;
    _averagedPeriodUs = 20000;
    _lastIsrTimeUs = 0;
    _filterIdx = 0;
    _firstSyncReceived = false;

    _carrierPinsArray = nullptr;
    _carrierFreqHz = 10000.0;
    _carrierDutyCyclePct = nullptr;
    _carrierSpeedMode = LEDC_LOW_SPEED_MODE;
    _carrierTimer = LEDC_TIMER_0;
    _carrierDutyResolutionBits = 10;

    for(int i=0; i<FREQ_FILTER_SIZE; i++) _periodBuffer[i] = 20000;

    for (int i = 0; i < _numChannels; i++) {
        _pins[i] = pins[i];
        _basePhaseDeg[i] = constrain(phaseOffsetsDegrees[i], 0.0, 360.0);
        _phaseOffsetsPct[i] = _basePhaseDeg[i] / 360.0;
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
    delete[] _basePhaseDeg;
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
        if (_pins[i] == GPIO_NUM_NC || _pins[i] > GPIO_NUM_39) {
            continue;
        }
        gpio_reset_pin(_pins[i]);
        gpio_set_direction(_pins[i], GPIO_MODE_OUTPUT);

        // idle HIGH: the inverter keeps the bridge off.
        gpio_set_level(_pins[i], 1);
    }

    setGlobalFrequency(initialFreqHz);

    #if USE_SYNC
        #if SYNC_AS_SERVER
            gpio_reset_pin(_syncPin);
            gpio_set_direction(_syncPin, GPIO_MODE_OUTPUT);
            gpio_set_level(_syncPin, 0);
            _lastSyncTimeUs = esp_timer_get_time();
        #else
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
        .dispatch_method = ESP_TIMER_TASK,
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
    PhaseParams params[PWMC_MAX_CHANNELS];

    // task context, not ISR: portENTER_CRITICAL, never the ISR variant.
    portENTER_CRITICAL(&self->_spinlock);
    lastSync = self->_lastSyncTimeUs;
    period = self->_averagedPeriodUs;
    dc = self->_dcMode;
    // interrupt context: the ISR variant of the lk is mandatory here.
    portEXIT_CRITICAL(&self->_spinlock);

    // snapshot outside the lk on purpose: the lk covers the shared fields only.
    for (int i = 0; i < self->_numChannels && i < PWMC_MAX_CHANNELS; i++) {
        params[i] = self->_params[i];
    }

    #if USE_SYNC && !SYNC_AS_SERVER
        if (!self->_firstSyncReceived) {
            if (period < 100) period = 20000;
        }
    #endif

    // subtract from the live duty so the duty can only ever fall.
    int64_t delta = now - lastSync;
    if (delta < 0) delta += period;

    uint32_t delta32 = (uint32_t)delta;
    uint32_t period32 = (uint32_t)period;
    uint32_t timeInCycle = delta32 % period32;

    if (dc) timeInCycle = 0;

    #if USE_SYNC && SYNC_AS_SERVER
        gpio_set_level(self->_syncPin, (timeInCycle < (period32 / 2)) ? 1 : 0);
    #endif

    #if USE_SYNC && SYNC_AS_SERVER
        const int channelLimit = 1;
    #else
        const int channelLimit = (self->_numChannels < PWMC_MAX_CHANNELS)
                             ? self->_numChannels : PWMC_MAX_CHANNELS;
    #endif

    for (int i = 0; i < channelLimit; i++) {
        if (self->_pins[i] == GPIO_NUM_NC || self->_pins[i] > GPIO_NUM_39) continue;

        uint32_t start = (uint32_t)params[i].startUs;
        uint32_t end = (uint32_t)params[i].endUs;

        bool active = params[i].wraps ?
                      (timeInCycle >= start || timeInCycle < end) :
                      (timeInCycle >= start && timeInCycle < end);

        gpio_set_level(self->_pins[i], active ? 0 : 1);
    }
}

void IRAM_ATTR PwmController::_onSyncInterrupt() {
    #if USE_SYNC && !SYNC_AS_SERVER
        if (!_isrInstance) return;

        int64_t now = esp_timer_get_time();

        portENTER_CRITICAL_ISR(&_isrInstance->_spinlock);

        if (_isrInstance->_lastIsrTimeUs > 0) {
            int64_t delta = now - _isrInstance->_lastIsrTimeUs;

            if (delta > 2000) {
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

        _isrInstance->_lastSyncTimeUs = now - SYNC_LATENCY_US;

        _isrInstance->_firstSyncReceived = true;

        portEXIT_CRITICAL_ISR(&_isrInstance->_spinlock);
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
        return;
    #endif

    if (!(newHz >= 1e-6f)) {
        // held field: no rotation, so the trim is zeroed and the phase frozen.
        for (int i = 0; i < _numChannels && i < PWMC_MAX_CHANNELS; i++)
            _trimCacheDeg[i] = 0.0f;
        // one lock for all four: a callback between writes drives a field never asked for.
        portENTER_CRITICAL(&_spinlock);
        _dcMode = true;
        _globalFreqHz = 0.0f;
        for (int i = 0; i < _numChannels; i++) _applyPhase(i);
        portEXIT_CRITICAL(&_spinlock);
        return;
    }
    const bool leavingDc = _dcMode;
    _dcMode = false;

    int64_t newPeriod = (int64_t)(1000000.0 / newHz);
    int64_t now = esp_timer_get_time();
    _globalFreqHz = newHz;
    // recompute the trim outside the spinlock: four atanf calls with interrupts off is too long.
    _recomputeTrim();

    portENTER_CRITICAL(&_spinlock);

    if (leavingDc) {
        _lastSyncTimeUs = now;
    }
    else if (_averagedPeriodUs > 0) {
        int64_t oldPos = (now - _lastSyncTimeUs) % _averagedPeriodUs;
        if (oldPos < 0) oldPos += _averagedPeriodUs;

        double phaseRatio = (double)oldPos / (double)_averagedPeriodUs;
        int64_t newPos = (int64_t)(phaseRatio * newPeriod);

        _lastSyncTimeUs = now - newPos;
    }

    _averagedPeriodUs = newPeriod;
    for(int i=0; i<FREQ_FILTER_SIZE; i++) _periodBuffer[i] = newPeriod;

    for(int i=0; i<_numChannels; i++) _applyPhase(i);

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
        return;
    #endif

    if (channel < 0 || channel >= _numChannels) return;

    portENTER_CRITICAL(&_spinlock);
    _basePhaseDeg[channel] = degrees;
    _applyPhase(channel);
    portEXIT_CRITICAL(&_spinlock);
}

void PwmController::setPhases(const float *degrees) {
    #if USE_SYNC && SYNC_AS_SERVER
        return;
    #endif

    if (degrees == nullptr) return;

    portENTER_CRITICAL(&_spinlock);
    for (int i = 0; i < _numChannels; i++) {
        _basePhaseDeg[i] = degrees[i];
        _applyPhase(i);
    }
    portEXIT_CRITICAL(&_spinlock);
}

void PwmController::setPhaseTrim(const float *f0Hz, const float *q) {
    bool on = (f0Hz != nullptr && q != nullptr);
    if (on) {
        for (int i = 0; i < _numChannels; i++)
            if (!(f0Hz[i] > 0.0f) || !(q[i] > 0.0f)) { on = false; break; }
    }
    if (on)
        for (int i = 0; i < _numChannels && i < PWMC_MAX_CHANNELS; i++) {
            _trimF0Hz[i] = f0Hz[i];
            _trimQ[i] = q[i];
        }
    _trimOn = on;
    _recomputeTrim();

    portENTER_CRITICAL(&_spinlock);
    for (int i = 0; i < _numChannels; i++) _applyPhase(i);
    portEXIT_CRITICAL(&_spinlock);
}

float PwmController::phaseTrimDeg(int channel) const {
    if (!_trimOn || channel < 0 || channel >= _numChannels) return 0.0f;
    return _trimCacheDeg[channel];
}

void PwmController::_recomputeTrim() {
    const float f = _globalFreqHz;
    for (int i = 0; i < _numChannels && i < PWMC_MAX_CHANNELS; i++) {
        if (!_trimOn || _dcMode || !(f >= 1e-6f)) { _trimCacheDeg[i] = 0.0f; continue; }
        const float r = f / _trimF0Hz[i] - _trimF0Hz[i] / f;
        _trimCacheDeg[i] = atanf(_trimQ[i] * r) * 180.0f / (float)M_PI;
    }
}

void PwmController::_applyPhase(int channel) {
    // caller holds the spinlock; uses the cached trim only.
    _phaseOffsetsPct[channel] =
        _wrapPct(_basePhaseDeg[channel] - _trimCacheDeg[channel]);
    updatePhaseParams(channel);
}

float PwmController::_wrapPct(float degrees) {
    float pct = fmodf(degrees, 360.0f) / 360.0f;
    return (pct < 0.0f) ? pct + 1.0f : pct;
}

float PwmController::getFrequency() const {
    if (_dcMode) return 0.0f;
    return 1000000.0 / _averagedPeriodUs;
}

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

    _serviceCurrentLoop();
}

void PwmController::enableCurrentSense(const gpio_num_t *adcPins,
                                         const float *sensPerVolt,
                                         float overcurrentTripA) {
    if (!adcPins || !sensPerVolt) return;
    if (!_sense) _sense = new CurrentSense(adcPins, sensPerVolt);
    // coils must be off here: this captures the floating zero offset.
    _sense->seed();
    _overcurrentTripA = overcurrentTripA;
    _tripped = false;
    _lastSenseUs = micros();
    _lastBalanceUs = _lastSenseUs;
}

void PwmController::enableCurrentBalance(const BalanceConfig &cfg,
                                           float startDuty) {
    if (!_sense) return;
    if (!_balance) _balance = new CurrentBalanceController(cfg);
    _startDuty = startDuty;
    _balance->reset(startDuty);
    for (int i = 0; i < 4; i++) {
        _ceiling[i] = NAN;
        _balanceDuty[i] = startDuty;
    }
    _lastBalanceUs = micros();
}

const float *PwmController::measuredCurrents() const {
    return _sense ? _sense->i_meas : nullptr;
}

void PwmController::_serviceCurrentLoop() {
    if (!_sense) return;

    unsigned long nowUs = micros();
    // ADC pacing: the ESP32 ADC needs settling time between conversions.
    float dtSenseMs = (float)(nowUs - _lastSenseUs) / 1000.0f;
    if (dtSenseMs >= 1.0f) {
        _sense->update(dtSenseMs);
        _lastSenseUs = nowUs;
    }

    // latched until reboot: nothing but a reset clears this trip.
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
        if (dtCtrlMs <= 0.0f) dtCtrlMs = 0.001f;
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

            // no inverter on the carrier out: HIGH is full drive.
            gpio_set_level(pin, 1);
            continue;
        }

        float clampedDuty = constrain(duty, 0.0f, 100.0f);

        _carrierDutyCyclePct[i] = clampedDuty;

        ledc_channel_config_t ledc_channel = {
            .gpio_num = pin,
            .speed_mode = _carrierSpeedMode,
            .channel = (ledc_channel_t)i,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = _carrierTimer,
            .duty = (uint32_t)((clampedDuty / 100.0f) * dutyMax),
            .hpoint = 0,
            .flags = {.output_invert = 0}
        };
        ledc_channel_config(&ledc_channel);
        _carrierLedcConfigured[i] = true;
        _carrierLastDutyTicks[i] = (uint32_t)((clampedDuty / 100.0f) * dutyMax);
    }
}

void PwmController::setCarrierDutyCycle(int channel, float dutyPercent) {
    if (channel < 0 || channel >= _numChannels) return;

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
            return;
        }
        _carrierDutyCyclePct[channel] = 100.0f;
        // pin HIGH is full drive; the pin must be re-attached before the next PWM write.
        ledc_stop(_carrierSpeedMode, (ledc_channel_t)channel, 1);
        _carrierLedcConfigured[channel] = false;
        return;
    }

    float periodMs = 1000.0f / _carrierFreqHz;
    float minDutyCycle = (MIN_ON_OFF_MS / periodMs) * 100.0f;
    float maxDutyCycle = 100.0f - minDutyCycle;

    _carrierDutyCyclePct[channel] = constrain(dutyPercent, minDutyCycle, maxDutyCycle);

    uint32_t dutyMax = (1UL << _carrierDutyResolutionBits) - 1UL;
    uint32_t dutyValue = (uint32_t)((_carrierDutyCyclePct[channel] / 100.0f) * dutyMax);

    if (!_carrierLedcConfigured[channel]) {
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
            return;
        }
        // some IDF versions need this explicit step before LEDC restarts.
        ledc_set_duty(_carrierSpeedMode, (ledc_channel_t)channel, dutyValue);
        ledc_update_duty(_carrierSpeedMode, (ledc_channel_t)channel);
        _carrierLedcConfigured[channel] = true;
        _carrierLastDutyTicks[channel] = dutyValue;
        return;
    }

    if (dutyValue == _carrierLastDutyTicks[channel]) return;

    ledc_set_duty(_carrierSpeedMode, (ledc_channel_t)channel, dutyValue);
    ledc_update_duty(_carrierSpeedMode, (ledc_channel_t)channel);
    _carrierLastDutyTicks[channel] = dutyValue;
}

bool PwmController::rampDownStep(float stepPct) {
    if (!_carrierDutyCyclePct) return true;
    if (stepPct <= 0.0f) stepPct = 0.1f;
    bool allZero = true;
    for (int i = 0; i < _numChannels; i++) {
        float cur = _carrierDutyCyclePct[i];
        if (cur <= 0.0f) continue;
        float next = cur - stepPct;
        if (next < 0.0f) next = 0.0f;
        setCarrierDutyCycle(i, next);
        if (next > 0.0f) allZero = false;
    }
    return allZero;
}
