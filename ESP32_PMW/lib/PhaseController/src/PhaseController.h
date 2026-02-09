#ifndef PHASE_CONTROLLER_H
#define PHASE_CONTROLLER_H

#include "Arduino.h"
#include "HardwareSerial.h"

class PhaseController {
  protected: // Changed to PROTECTED for inheritance
    int _numChannels;
    gpio_num_t* _pins;
    float* _phaseOffsetsPct;
    float* _dutyCycles;
    
    // Independent Time & State per Channel
    float* _freqsHz;        
    float* _periodsUs;      
    float* _cyclePositions; 
    
    unsigned long _lastLoopMicros;

    // Sync Variables
    bool _syncEnabled;
    bool _isServer;
    gpio_num_t _syncPin;
    
    // Hardware Sync State (ISR)
    static PhaseController* _isrInstance; // Static pointer for ISR
    volatile unsigned long _isrPulseStart;
    volatile unsigned long _isrPulseWidth;
    volatile bool _isrPulseReady;
    
    unsigned long _lastSyncTime;    
    bool _hasSyncedOnce;
    
    // Constants
    const unsigned long SYNC_INTERVAL_US = 1000000UL; // 1 Second (Faster Refresh)

    // ISR Function (Must be static and in IRAM)
    static void IRAM_ATTR _onSyncInterrupt();

    struct PhaseParams {
        float start;
        float end;
        bool wraps;
    };
    PhaseParams* _params;

    // Helper to update internal math
    void updatePhaseParams(int channel);

  public:
    PhaseController(const gpio_num_t* pins, const float* phaseOffsetsDegrees, int numChannels);
    virtual ~PhaseController(); // Virtual destructor
    
    void begin(float initialFreqHz);
    void enableSync(bool isServer, gpio_num_t syncPin);
    
    // Basic Setters
    void setFrequency(int channel, float newHz);
    void setGlobalFrequency(float newHz);
    void setDutyCycle(int channel, float dutyPercent);
    
    // Needed for the sequencer to control phase dynamically
    void setPhase(int channel, float degrees);

    // Getters
    float getFrequency(int channel) const;
    float getPhase(int channel) const;
    float getDutyCycle(int channel) const;

    // Core Loop
    void run();
};

#endif
