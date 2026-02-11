#ifndef PHASE_CONTROLLER_H
#define PHASE_CONTROLLER_H

#include <Arduino.h>
#include "driver/gpio.h"
#include "esp_timer.h"
#include "freertos/portmacro.h"

// --- COMPILE-TIME CONFIGURATION DEFAULTS ---

#ifndef USE_SYNC
#define USE_SYNC 0
#endif

#ifndef SYNC_AS_SERVER
#define SYNC_AS_SERVER 0 
#endif

// Latency Compensation in Microseconds
// A value of 150us compensates for roughly 3 degrees of lag at 50Hz.
// Increase this if phase angle is still too high (lagging).
// Decrease if phase angle becomes too low (leading).
#ifndef SYNC_LATENCY_US
#define SYNC_LATENCY_US 150 
#endif

#define FREQ_FILTER_SIZE 8

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
    void enableSync(gpio_num_t syncPin);

    // Configuration
    void setFrequency(int channel, float newHz);
    void setGlobalFrequency(float newHz);
    void setDutyCycle(int channel, float dutyPercent);
    void setPhase(int channel, float degrees);

    // Getters
    float getFrequency(int channel) const;
    float getPhase(int channel) const;
    float getDutyCycle(int channel) const;

    // Run loop
    void run();

private:
    void updatePhaseParams(int channel);
    
    // Callbacks
    static void IRAM_ATTR _onSyncInterrupt();
    static void IRAM_ATTR _timerCallback(void* arg);

    int _numChannels;
    gpio_num_t* _pins;
    
    float* _freqsHz;
    float* _dutyCycles;
    float* _phaseOffsetsPct;
    PhaseParams* _params;

    esp_timer_handle_t _periodicTimer;

    gpio_num_t _syncPin;
    static PhaseController* _isrInstance;
    
    portMUX_TYPE _spinlock = portMUX_INITIALIZER_UNLOCKED;
    volatile int64_t _lastSyncTimeUs;    
    volatile int64_t _averagedPeriodUs;  
    
    volatile int64_t _lastIsrTimeUs;
    volatile int64_t _periodBuffer[FREQ_FILTER_SIZE];
    volatile int _filterIdx;
    volatile bool _firstSyncReceived;
};

#endif