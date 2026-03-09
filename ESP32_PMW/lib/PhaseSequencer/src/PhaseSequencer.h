#pragma once
#include <Arduino.h>
#include <vector>
#include "PhaseController.h"
#include "esp_timer.h" 

// Define the types of commands our queue can execute
enum TaskType {
    TASK_SET_DUTY_CYCLES, 
    TASK_WAIT,            
    TASK_RAMP_LINEAR,     
    TASK_RAMP_EASE        
};

struct SequenceTask {
    TaskType type;
    int64_t durationUs; 
    float startFreq;
    float endFreq;
    float dutyCycles[4]; 
};

// --- NEW: Pre-computed Trajectory Point ---
// Stores the explicit state of all channels at a given microsecond in time
struct TrajectoryPoint {
    int64_t timeUs;
    float freq[4];
    float duty[4];
    float phase[4];
};

class PhaseSequencer {
public:
    PhaseSequencer(PhaseController* phaseCtrl);

    // Queue Builders
    void reserve(size_t size); 
    void addDutyCycleTask(float d0, float d1, float d2, float d3);
    void addWaitTask(uint32_t durationMs);
    void addLinearRampTask(float startHz, float endHz, uint32_t durationMs);
    void addEaseRampTask(float startHz, float endHz, uint32_t durationMs);
    
    // --- NEW: Compiler ---
    // Computes all float math upfront into a trajectory buffer
    void compile(uint32_t resolutionMs, float initialFreq, const float* initialDuty, const float* initialPhase);

    // Control
    void start();
    void run();
    bool isDone() const;

private:
    PhaseController* _phaseCtrl;
    std::vector<SequenceTask> _queue;
    std::vector<TrajectoryPoint> _trajectory; // The pre-computed LUT
    
    size_t _currentFrameIdx;
    int64_t _taskStartTimeUs; 

    float easeInOut(float t);
};