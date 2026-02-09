#ifndef PHASE_CONTROLLER_H
#define PHASE_CONTROLLER_H

#include "Arduino.h"

class PhaseController {
public:
    // Constructor: Define the 4 pins for 0, 90, 180, 270 degrees
    PhaseController(const int* pins, const float* phaseOffsetsDegrees, int numChannels);

    // Initialize pins and state
    void begin(float initialFreqHz = 10.0);

    // Main loop ticker - call this as fast as possible in void loop()
    void run();

    // Setters
    void setFrequency(float newHz);
    void setDutyCycle(int channel, float dutyPercent); // Channel 0-3
    void setGlobalDutyCycle(float dutyPercent);        // Set all at once
    
    // Getters
    float getFrequency() const;

private:
    // --- Hardware Config ---
    int _pins[4]; // Stores pin numbers for 0, 90, 180, 270

    // --- State Variables ---
    float _pwmFreqHz;
    float _periodUs;
    float _currentCyclePos;
    unsigned long _lastLoopMicros;

    // --- Duty Cycles (0.0 - 100.0) ---
    float _dutyCycles[4];

    // --- Optimization Structure ---
    struct PhaseParams {
        float start;
        float end;
        bool wraps;
    };

    // One param set for each channel
    PhaseParams _params[4]; 

    // --- Internal Logic ---
    void updatePhaseParams();
};

#endif
