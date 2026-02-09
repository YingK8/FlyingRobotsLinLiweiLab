#ifndef PHASE_SEQUENCER_H
#define PHASE_SEQUENCER_H

#include "Arduino.h"
#include "PhaseController.h"

class PhaseSequencer : public PhaseController {
  private:
    struct RampState {
        bool active;
        float startVal;
        float targetVal;
        unsigned long startTime;
        unsigned long delayMs;
        unsigned long durationMs;
    };

    // Arrays to store ramp state for each channel
    RampState* _freqRamps;
    RampState* _phaseRamps;
    RampState* _dutyRamps;

    // --- NEW: Buffered State for Loop Updates ---
    float* _nextFreqs;
    float* _nextPhases;
    float* _nextDuties;
    bool* _dirtyFreqs;
    bool* _dirtyPhases;
    bool* _dirtyDuties;

    unsigned long _lastRampUpdate;
    const unsigned long RAMP_UPDATE_INTERVAL_MS = 10; // Update control loop at 100Hz

    float interpolate(float start, float target, float progress) {
        return start + (target - start) * progress;
    }

    void processRamp(RampState& ramp, int channel, int type); // type: 0=Freq, 1=Phase, 2=Duty

  public:
    PhaseSequencer(const int* pins, const float* phaseOffsetsDegrees, int numChannels);
    ~PhaseSequencer();

    // --- High Level API ---

    // Smoothly change values over time
    void rampFrequency(int channel, float targetHz, unsigned long durationMs, unsigned long delayMs = 0);
    void rampPhase(int channel, float targetDeg, unsigned long durationMs, unsigned long delayMs = 0);
    void rampDuty(int channel, float targetPct, unsigned long durationMs, unsigned long delayMs = 0);

    // --- NEW: Buffered Setters ---
    // Use these in the main loop for dynamic control.
    // Changes are buffered and applied efficiently in the next run() call.
    void setFrequencyNext(int channel, float hz);
    void setPhaseNext(int channel, float deg);
    void setDutyNext(int channel, float pct);

    // Override run to handle buffers + ramps + signal generation
    void run(); 
};

#endif