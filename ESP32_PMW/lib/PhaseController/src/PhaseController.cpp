#include <Arduino.h>
#include "PhaseController.h"

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
}

PhaseController::~PhaseController() {
    delete[] _pins; delete[] _phaseOffsetsPct; delete[] _dutyCycles;
    delete[] _params; delete[] _freqsHz; delete[] _periodsUs; delete[] _cyclePositions;
}

void PhaseController::begin(float initialFreqHz) {
    for(int i = 0; i < _numChannels; i++) {
        pinMode(_pins[i], OUTPUT);
        digitalWrite(_pins[i], LOW);
    }
    setGlobalFrequency(initialFreqHz);
    _lastLoopMicros = micros();
}

void PhaseController::enableSync(bool isServer, HardwareSerial* serial, int rxPin, int txPin, long baud) {
    _isServer = isServer;
    _syncSerial = serial;
    _syncEnabled = true;
    _syncSerial->begin(baud, SERIAL_8N1, rxPin, txPin);
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

    // Sync Logic
    if (_syncEnabled && !_isServer) {
        if (_syncSerial->available() >= 3) {
            if (_syncSerial->read() == 0xFF) { 
                uint8_t h = _syncSerial->read();
                uint8_t l = _syncSerial->read();
                uint16_t freqInt = (h << 8) | l;
                float syncedFreq = freqInt / 10.0;
                
                if (abs(syncedFreq - _freqsHz[0]) > 0.09) setGlobalFrequency(syncedFreq);
                for(int i=0; i<_numChannels; i++) _cyclePositions[i] = 0;
                while(_syncSerial->available()) _syncSerial->read();
            }
        }
    }

    // Signal Generation Loop
    for(int i = 0; i < _numChannels; i++) {
        _cyclePositions[i] += dt;
        if (_cyclePositions[i] >= _periodsUs[i]) {
            _cyclePositions[i] -= _periodsUs[i];
            if (i == 0 && _syncEnabled && _isServer) {
                uint16_t fScaled = (uint16_t)(_freqsHz[0] * 10);
                uint8_t packet[3] = {0xFF, (uint8_t)((fScaled >> 8) & 0xFF), (uint8_t)(fScaled & 0xFF)};
                _syncSerial->write(packet, 3);
            }
        }
        bool active;
        if (_params[i].wraps) active = (_cyclePositions[i] >= _params[i].start || _cyclePositions[i] < _params[i].end);
        else active = (_cyclePositions[i] >= _params[i].start && _cyclePositions[i] < _params[i].end);
        digitalWrite(_pins[i], active); 
    }
}