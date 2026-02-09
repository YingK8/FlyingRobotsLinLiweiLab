#ifndef PHASE_CONTROLLER_H
#define PHASE_CONTROLLER_H

#include "Arduino.h"
#include "HardwareSerial.h"

class PhaseController {
  protected: // Changed to PROTECTED for inheritance
    int _numChannels;
    int* _pins;
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
    HardwareSerial* _syncSerial;

    struct PhaseParams {
        float start;
        float end;
        bool wraps;
    };
    PhaseParams* _params;

    // Helper to update internal math
    void updatePhaseParams(int channel);

  public:
    PhaseController(const int* pins, const float* phaseOffsetsDegrees, int numChannels);
    virtual ~PhaseController(); // Virtual destructor
    
    void begin(float initialFreqHz);
    void enableSync(bool isServer, HardwareSerial* serial, int rxPin, int txPin, long baud);
    
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
