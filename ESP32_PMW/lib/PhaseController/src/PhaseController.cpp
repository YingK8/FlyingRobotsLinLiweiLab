#include <Arduino.h>
#include "PhaseController.h"

// Initialise static member for timer sync functions!
PhaseController* PhaseController::_isrInstance = nullptr;

PhaseController::PhaseController(const int* pins, const float* phaseOffsetsDegrees, int numChannels) {
    _numChannels = numChannels;
    _syncEnabled = false; 

    _pins = new int[_numChannels];
    _phaseOffsetsPct = new float[_numChannels];
    _dutyCycles = new float[_numChannels];
    _params = new PhaseParams[_numChannels];
    
    _freqsHz = new float[_numChannels];
    _periodsUs = new float[_numChannels];
    _cyclePositions = new float[_numChannels];

    for (int i = 0; i < _numChannels; i++) {
        _pins[i] = pins[i];
        _phaseOffsetsPct[i] = constrain(phaseOffsetsDegrees[i], 0.0, 360.0) / 360.0;
        _dutyCycles[i] = 50.0;
        
        _freqsHz[i] = 10.0;
        _periodsUs[i] = 1000000.0 / _freqsHz[i];
        _cyclePositions[i] = 0;
    }

    _lastLoopMicros = 0;
    _hasSyncedOnce = false;
    _isrPulseReady = false;
}

PhaseController::~PhaseController() {
    delete[] _pins; delete[] _phaseOffsetsPct; delete[] _dutyCycles;
    delete[] _params; delete[] _freqsHz; delete[] _periodsUs; delete[] _cyclePositions;
}

void PhaseController::begin(float initialFreqHz) {
    // Pin Setup
    for(int i = 0; i < _numChannels; i++) {
        pinMode(_pins[i], OUTPUT);
        digitalWrite(_pins[i], LOW);
    }

    // Sync Pin Setup
    if (_syncEnabled) {
        if (_isServer) {
            pinMode(_syncPin, OUTPUT);
            // Startup Pulse: High for 3 seconds to reset clients
            digitalWrite(_syncPin, HIGH);
            delay(3000); 
            digitalWrite(_syncPin, LOW);
            _lastSyncTime = micros(); // Init timer
        } else {
            pinMode(_syncPin, INPUT_PULLDOWN); 
            // Client uses Interrupts
            _isrInstance = this;
            attachInterrupt(digitalPinToInterrupt(_syncPin), _onSyncInterrupt, CHANGE);
        }
    }

    setGlobalFrequency(initialFreqHz);
    _lastLoopMicros = micros();
}

void PhaseController::enableSync(bool isServer, int syncPin) {
    _isServer = isServer;
    _syncPin = syncPin;
    _syncEnabled = true;
}

// Static ISR for High-Speed Capture
void IRAM_ATTR PhaseController::_onSyncInterrupt() {
    if (_isrInstance) {
        bool state = digitalRead(_isrInstance->_syncPin);
        unsigned long now = micros();
        if (state) {
            // Rising Edge: Start Timer
            _isrInstance->_isrPulseStart = now;
        } else {
            // Falling Edge: Capture Width & Flag Ready
            _isrInstance->_isrPulseWidth = now - _isrInstance->_isrPulseStart;
            _isrInstance->_isrPulseReady = true;
        }
    }
}

void PhaseController::setFrequency(int channel, float newHz) {
    if (channel < 0 || channel >= _numChannels) return;
    if (newHz < 0.1 || newHz > 5000.0) return; 

    float oldPeriod = _periodsUs[channel];
    float newPeriod = 1000000.0 / newHz;

    // Elastic time scaling
    if (oldPeriod > 0) {
        _cyclePositions[channel] = (_cyclePositions[channel] / oldPeriod) * newPeriod;
    } else {
        _cyclePositions[channel] = 0;
    }

    _freqsHz[channel] = newHz;
    _periodsUs[channel] = newPeriod;
    updatePhaseParams(channel);
}

void PhaseController::setGlobalFrequency(float newHz) {
    for(int i = 0; i < _numChannels; i++) {
        setFrequency(i, newHz);
    }
}

void PhaseController::setDutyCycle(int channel, float dutyPercent) {
    if(channel < 0 || channel >= _numChannels) return;
    _dutyCycles[channel] = constrain(dutyPercent, 0.0, 100.0);
    updatePhaseParams(channel);
}

void PhaseController::setPhase(int channel, float degrees) {
    if(channel < 0 || channel >= _numChannels) return;
    float pct = degrees / 360.0;
    while(pct >= 1.0) pct -= 1.0;
    while(pct < 0.0) pct += 1.0;
    
    _phaseOffsetsPct[channel] = pct;
    updatePhaseParams(channel);
}

float PhaseController::getFrequency(int channel) const { return _freqsHz[channel]; }
float PhaseController::getPhase(int channel) const { return _phaseOffsetsPct[channel] * 360.0; }
float PhaseController::getDutyCycle(int channel) const { return _dutyCycles[channel]; }

void PhaseController::updatePhaseParams(int channel) {
    float width = _periodsUs[channel] * _dutyCycles[channel] / 100.0;
    _params[channel].start = _periodsUs[channel] * _phaseOffsetsPct[channel];
    _params[channel].end = _params[channel].start + width;

    if (_params[channel].end > _periodsUs[channel]) {
        _params[channel].end -= _periodsUs[channel];
        _params[channel].wraps = true;
    } else {
        _params[channel].wraps = false;
    }
}

void PhaseController::run() {
    unsigned long now = micros();
    unsigned long dt = now - _lastLoopMicros;
    _lastLoopMicros = now;

    if (dt > _periodsUs[0] * 2 && _periodsUs[0] > 0) dt = 0; 

    // --- HARDWARE SYNC LOGIC (GPIO 4) ---
    if (_syncEnabled) {
        if (_isServer) {
            // === SERVER LOGIC (Generates Pulse) ===
            // Generate pulse every 1 second
            if (now - _lastSyncTime >= SYNC_INTERVAL_US) {
                digitalWrite(_syncPin, HIGH);
                delayMicroseconds(1000); // 1ms pulse (Interrupts can catch short pulses easily)
                digitalWrite(_syncPin, LOW);
                _lastSyncTime = now;
            }
        } 
        else {
            // === CLIENT LOGIC (Process ISR Data) ===
            if (_isrPulseReady) {
                _isrPulseReady = false; // Reset flag
                
                unsigned long duration = _isrPulseWidth;
                unsigned long pulseTime = _isrPulseStart; // Captured in ISR

                // Case 1: Startup Pulse (> 2 seconds)
                if (duration > 2000000UL) {
                    for(int i=0; i<_numChannels; i++) _cyclePositions[i] = 0;
                    _lastSyncTime = pulseTime; 
                    _hasSyncedOnce = true;
                }
                // Case 2: Periodic Pulse (1ms)
                else if (_hasSyncedOnce) {
                    unsigned long actualInterval = pulseTime - _lastSyncTime;
                    long error = (long)(actualInterval - SYNC_INTERVAL_US);
                    
                    // Subtract drift
                    for(int i=0; i<_numChannels; i++) {
                        _cyclePositions[i] -= error;
                        
                        // Handle wrapping
                        while(_cyclePositions[i] < 0) _cyclePositions[i] += _periodsUs[i];
                        while(_cyclePositions[i] >= _periodsUs[i]) _cyclePositions[i] -= _periodsUs[i];
                    }
                    _lastSyncTime = pulseTime;
                }
                else {
                    _lastSyncTime = pulseTime;
                    _hasSyncedOnce = true;
                }
            }
        }
    }

    // Signal Generation Loop
    for(int i = 0; i < _numChannels; i++) {
        _cyclePositions[i] += dt;
        if (_cyclePositions[i] >= _periodsUs[i]) {
            _cyclePositions[i] -= _periodsUs[i];
        }
        bool active;
        if (_params[i].wraps) active = (_cyclePositions[i] >= _params[i].start || _cyclePositions[i] < _params[i].end);
        else active = (_cyclePositions[i] >= _params[i].start && _cyclePositions[i] < _params[i].end);
        digitalWrite(_pins[i], active); 
    }
}