#pragma once

#include <Arduino.h>
#include "driver/gpio.h"
#include "esp_timer.h"

// Configuration Defines
// (Ensure these match your project settings or are defined here)
#ifndef USE_SYNC
#define USE_SYNC 1 
#endif

// Set to 1 for Master (Output Sync), 0 for Client (Input Sync)
#ifndef SYNC_AS_SERVER
#define SYNC_AS_SERVER 0
#endif

#define FREQ_FILTER_SIZE 5
#define SYNC_LATENCY_US 50 // Tuning value for latency compensation

struct PhaseParams {
    unsigned long startUs;
    unsigned long endUs;
    bool wraps;
};

class PhaseController {
public:
    PhaseController(const gpio_num_t* pins, const float* phaseOffsetsDegrees, int numChannels);
    ~PhaseController();

    void begin(float initialFreqHz);
    
    // Main loop task - Call this in loop()
    void run(); 

    // Configuration
    void setGlobalFrequency(float newHz);
    void setFrequency(int channel, float newHz);
    void setDutyCycle(int channel, float dutyPercent);
    void setPhase(int channel, float degrees);
    
    // Getters
    float getFrequency(int channel) const;
    float getPhase(int channel) const;
    float getDutyCycle(int channel) const;

    // Sync Configuration
    void enableSync(gpio_num_t syncPin);

private:
    // Internal methods
    static void IRAM_ATTR _timerCallback(void* arg);
    static void IRAM_ATTR _onSyncInterrupt();
    void updatePhaseParams(int channel);

    // Hardware
    int _numChannels;
    gpio_num_t* _pins;
    esp_timer_handle_t _periodicTimer;
    gpio_num_t _syncPin;
    
    // State Arrays
    float* _phaseOffsetsPct;
    float* _dutyCycles;
    PhaseParams* _params;
    float* _freqsHz; // Retained for API compatibility, though mainly unused logic-wise
    
    // Sync State
    int64_t _lastSyncTimeUs;
    int64_t _averagedPeriodUs;
    int64_t _periodBuffer[FREQ_FILTER_SIZE];
    int _filterIdx;
    
    // ISR variables
    static PhaseController* _isrInstance;
    int64_t _lastIsrTimeUs;
    bool _firstSyncReceived;
    
    // Concurrency
    portMUX_TYPE _spinlock = portMUX_INITIALIZER_UNLOCKED;
};