#include "PhaseController.h"

PhaseController* PhaseController::_isrInstance = nullptr;

PhaseController::PhaseController(const gpio_num_t* pins, const float* phaseOffsetsDegrees, int numChannels) {
    _numChannels = numChannels;
    _periodicTimer = nullptr;

    _pins = new gpio_num_t[_numChannels];
    _phaseOffsetsPct = new float[_numChannels];
    _dutyCycles = new float[_numChannels];
    _params = new PhaseParams[_numChannels];
    _freqsHz = new float[_numChannels];

    _lastSyncTimeUs = 0;
    _averagedPeriodUs = 20000;
    _lastIsrTimeUs = 0;
    _filterIdx = 0;
    _firstSyncReceived = false;

    for(int i=0; i<FREQ_FILTER_SIZE; i++) _periodBuffer[i] = 20000;

    for (int i = 0; i < _numChannels; i++) {
        _pins[i] = pins[i];
        _phaseOffsetsPct[i] = constrain(phaseOffsetsDegrees[i], 0.0, 360.0) / 360.0;
        _dutyCycles[i] = 50.0;
        _freqsHz[i] = 50.0;
    }
}

PhaseController::~PhaseController() {
    if (_periodicTimer) {
        esp_timer_stop(_periodicTimer);
        esp_timer_delete(_periodicTimer);
    }
    delete[] _pins; 
    delete[] _phaseOffsetsPct; 
    delete[] _dutyCycles; 
    delete[] _params; 
    delete[] _freqsHz;
}

void PhaseController::begin(float initialFreqHz) {
    for(int i = 0; i < _numChannels; i++) {
        gpio_reset_pin(_pins[i]);
        gpio_set_direction(_pins[i], GPIO_MODE_OUTPUT);
        gpio_set_level(_pins[i], 0);
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
        .dispatch_method = ESP_TIMER_TASK,
        .name = "pwm_gen"
    };

    esp_timer_create(&timer_args, &_periodicTimer);
    esp_timer_start_periodic(_periodicTimer, 25); 
}

void IRAM_ATTR PhaseController::_timerCallback(void* arg) {
    PhaseController* self = (PhaseController*)arg;
    int64_t now = esp_timer_get_time();

    int64_t lastSync;
    int64_t period;
    
    portENTER_CRITICAL_ISR(&self->_spinlock);
    lastSync = self->_lastSyncTimeUs;
    period = self->_averagedPeriodUs;
    portEXIT_CRITICAL_ISR(&self->_spinlock);

    // Client fallback: If waiting for first sync, use default period
    #if USE_SYNC && !SYNC_AS_SERVER
        if (!self->_firstSyncReceived) {
            // Defaults to 20000us if period is invalid
            if (period < 100) period = 20000; 
        }
    #endif

    // Calculate position
    int64_t timeInCycle = (now - lastSync) % period;
    if (timeInCycle < 0) timeInCycle += period;

    // Master Sync Generation
    #if USE_SYNC && SYNC_AS_SERVER
        // Set sync high for first 50% of cycle
        gpio_set_level(self->_syncPin, (timeInCycle < (period / 2)) ? 1 : 0);
    #endif

    // Channel Generation
    #if USE_SYNC && SYNC_AS_SERVER
        const int channelLimit = 1;
    #else
        const int channelLimit = self->_numChannels;
    #endif

    for (int i = 0; i < channelLimit; i++) {
        // Local copies for speed
        unsigned long start = self->_params[i].startUs;
        unsigned long end = self->_params[i].endUs;
        
        bool active = self->_params[i].wraps ? 
                      (timeInCycle >= start || timeInCycle < end) : 
                      (timeInCycle >= start && timeInCycle < end);
        
        gpio_set_level(self->_pins[i], active ? 1 : 0);
    }
}

void IRAM_ATTR PhaseController::_onSyncInterrupt() {
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

void PhaseController::enableSync(gpio_num_t syncPin) {
    #if USE_SYNC
        _syncPin = syncPin;
    #endif
}

void PhaseController::updatePhaseParams(int channel) {
    #if USE_SYNC && SYNC_AS_SERVER
        if (channel > 0) return;
    #endif

    // NOTE: This function should ideally be called within a critical section
    // or when the lock is held, as it reads and writes shared state.
    
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

void PhaseController::setGlobalFrequency(float newHz) {
    #if USE_SYNC && !SYNC_AS_SERVER
        return; // Client ignores manual freq
    #endif

    int64_t newPeriod = (int64_t)(1000000.0 / newHz);
    int64_t now = esp_timer_get_time();
    
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

void PhaseController::setFrequency(int channel, float newHz) {
    #if USE_SYNC && !SYNC_AS_SERVER
        return;
    #endif
    _freqsHz[channel] = newHz;
}

void PhaseController::setDutyCycle(int channel, float dutyPercent) {
    if (channel < 0 || channel >= _numChannels) return;

    // Fix: Critical section added to prevent race conditions during updates
    portENTER_CRITICAL(&_spinlock);
    _dutyCycles[channel] = constrain(dutyPercent, 0.0, 100.0);
    updatePhaseParams(channel);
    portEXIT_CRITICAL(&_spinlock);
}

void PhaseController::setPhase(int channel, float degrees) {
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

float PhaseController::getFrequency(int channel) const { 
    return 1000000.0 / _averagedPeriodUs; 
}
float PhaseController::getPhase(int channel) const { 
    #if USE_SYNC && SYNC_AS_SERVER
        return 0.0;
    #else
        return _phaseOffsetsPct[channel] * 360.0;
    #endif
}
float PhaseController::getDutyCycle(int channel) const { return _dutyCycles[channel]; }

void PhaseController::run() {
    // Ramps removed. This method now only handles sync drift compensation.
    // If the external sync frequency changes, we need to periodically recalculate 
    // the pulse widths (in microseconds) to maintain the correct duty cycle %.
    
    static unsigned long lastUpdate = 0;
    if (esp_timer_get_time() - lastUpdate > 100000) { 
        lastUpdate = esp_timer_get_time();
        
        portENTER_CRITICAL(&_spinlock);
        for(int i=0; i<_numChannels; i++) {
            updatePhaseParams(i);
        }
        portEXIT_CRITICAL(&_spinlock);
    }
}