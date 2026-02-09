#include "Arduino.h"
#include "PhaseSequencer.h"

PhaseSequencer::PhaseSequencer(const int* pins, const float* phaseOffsetsDegrees, int numChannels) 
    : PhaseController(pins, phaseOffsetsDegrees, numChannels) {
    
    _freqRamps = new RampState[numChannels];
    _phaseRamps = new RampState[numChannels];
    _dutyRamps = new RampState[numChannels];
    
    // Allocate Buffers
    _nextFreqs = new float[numChannels];
    _nextPhases = new float[numChannels];
    _nextDuties = new float[numChannels];
    
    _dirtyFreqs = new bool[numChannels];
    _dirtyPhases = new bool[numChannels];
    _dirtyDuties = new bool[numChannels];
    
    _lastRampUpdate = 0;

    // Initialize States
    for(int i=0; i<numChannels; i++) {
        _freqRamps[i].active = false;
        _phaseRamps[i].active = false;
        _dutyRamps[i].active = false;
        
        _dirtyFreqs[i] = false;
        _dirtyPhases[i] = false;
        _dirtyDuties[i] = false;
    }
}

PhaseSequencer::~PhaseSequencer() {
    delete[] _freqRamps; delete[] _phaseRamps; delete[] _dutyRamps;
    delete[] _nextFreqs; delete[] _nextPhases; delete[] _nextDuties;
    delete[] _dirtyFreqs; delete[] _dirtyPhases; delete[] _dirtyDuties;
}

void PhaseSequencer::rampFrequency(int channel, float targetHz, unsigned long durationMs, unsigned long delayMs) {
    if(channel < 0 || channel >= _numChannels) return;
    _freqRamps[channel] = {true, getFrequency(channel), targetHz, millis(), delayMs, durationMs};
}

void PhaseSequencer::rampPhase(int channel, float targetDeg, unsigned long durationMs, unsigned long delayMs) {
    if(channel < 0 || channel >= _numChannels) return;
    _phaseRamps[channel] = {true, getPhase(channel), targetDeg, millis(), delayMs, durationMs};
}

void PhaseSequencer::rampDuty(int channel, float targetPct, unsigned long durationMs, unsigned long delayMs) {
    if(channel < 0 || channel >= _numChannels) return;
    _dutyRamps[channel] = {true, getDutyCycle(channel), targetPct, millis(), delayMs, durationMs};
}

// --- Buffered Setters ---
void PhaseSequencer::setFrequencyNext(int channel, float hz) {
    if (channel < 0 || channel >= _numChannels || _freqRamps[channel].active) return;
    _nextFreqs[channel] = hz;
    _dirtyFreqs[channel] = true;
    _freqRamps[channel].active = false; // Override active ramp
}

void PhaseSequencer::setPhaseNext(int channel, float deg) {
    if(channel < 0 || channel >= _numChannels || _phaseRamps[channel].active) return;
    _nextPhases[channel] = deg;
    _dirtyPhases[channel] = true;
    _phaseRamps[channel].active = false;
}

void PhaseSequencer::setDutyNext(int channel, float pct) {
    if(channel < 0 || channel >= _numChannels || _dutyRamps[channel].active) return;
    _nextDuties[channel] = pct;
    _dirtyDuties[channel] = true;
    _dutyRamps[channel].active = false;
}

void PhaseSequencer::processRamp(RampState& ramp, int channel, int type) {
    if (!ramp.active) return;

    unsigned long now = millis();
    unsigned long elapsed = now - ramp.startTime;

    if (elapsed < ramp.delayMs) return; 

    unsigned long rampElapsed = elapsed - ramp.delayMs;
    float progress = 0.0;

    if (ramp.durationMs == 0) {
        progress = 1.0; 
    } else {
        progress = (float)rampElapsed / (float)ramp.durationMs;
        if (progress > 1.0) { progress = 1.0; }
    }

    float currentVal = interpolate(ramp.startVal, ramp.targetVal, progress);

    if (type == 0) PhaseController::setFrequency(channel, currentVal);
    else if (type == 1) PhaseController::setPhase(channel, currentVal);
    else if (type == 2) PhaseController::setDutyCycle(channel, currentVal);

    if (progress >= 1.0) { ramp.active = false; }
}

void PhaseSequencer::run() {
    // 0. Apply Buffered Changes (High Priority / Loop Updates)
    for(int i=0; i<_numChannels; i++) {
        if (_dirtyFreqs[i]) {
            PhaseController::setFrequency(i, _nextFreqs[i]);
            _dirtyFreqs[i] = false;
        }
        if (_dirtyPhases[i]) {
            PhaseController::setPhase(i, _nextPhases[i]);
            _dirtyPhases[i] = false;
        }
        if (_dirtyDuties[i]) {
            PhaseController::setDutyCycle(i, _nextDuties[i]);
            _dirtyDuties[i] = false;
        }
    }

    // 1. Base Run (Generate Signal)
    PhaseController::run();

    // 2. Process Ramps (Throttled background task)
    if (millis() - _lastRampUpdate >= RAMP_UPDATE_INTERVAL_MS) {
        _lastRampUpdate = millis();
        
        for(int i=0; i<_numChannels; i++) {
            processRamp(_freqRamps[i], i, 0); 
            processRamp(_phaseRamps[i], i, 1); 
            processRamp(_dutyRamps[i], i, 2); 
        }
    }
}