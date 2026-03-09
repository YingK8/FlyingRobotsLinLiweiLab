#include "PhaseSequencer.h"

PhaseSequencer::PhaseSequencer(PhaseController* phaseCtrl) {
    _phaseCtrl = phaseCtrl;
    _currentFrameIdx = 0;
    _taskStartTimeUs = 0;
}

void PhaseSequencer::reserve(size_t size) {
    _queue.reserve(size);
}

void PhaseSequencer::addDutyCycleTask(const float* dutyCycles, int numChannels) {
    SequenceTask task = {TASK_SET_DUTY_CYCLES, 0, 0.0, 0.0, {0,0,0,0}};
    for (int i = 0; i < numChannels; i++) {
        task.dutyCycles[i] = constrain(dutyCycles[i], 0.0, 100.0);
    }
    _queue.push_back(task);
}

void PhaseSequencer::addWaitTask(uint32_t durationMs) {
    SequenceTask task = {TASK_WAIT, (int64_t)durationMs * 1000LL, 0.0, 0.0, {0,0,0,0}};
    _queue.push_back(task);
}

void PhaseSequencer::addLinearRampTask(float startHz, float endHz, uint32_t durationMs) {
    if (durationMs == 0) durationMs = 1;
    SequenceTask task = {TASK_RAMP_LINEAR, (int64_t)durationMs * 1000LL, startHz, endHz, {0,0,0,0}};
    _queue.push_back(task);
}

void PhaseSequencer::addEaseRampTask(float startHz, float endHz, uint32_t durationMs) {
    if (durationMs == 0) durationMs = 1; 
    SequenceTask task = {TASK_RAMP_EASE, (int64_t)durationMs * 1000LL, startHz, endHz, {0,0,0,0}};
    _queue.push_back(task);
}

float PhaseSequencer::easeInOut(float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

// =========================================================
// TRAJECTORY COMPILER: Generates the buffer upfront
// =========================================================
void PhaseSequencer::compile(uint32_t resolutionMs, float initialFreq, const float* initialDuty, const float* initialPhase) {
    _trajectory.clear();
    
    int64_t currentTimeUs = 0;
    float curFreq[4] = {initialFreq, initialFreq, initialFreq, initialFreq};
    float curDuty[4] = {initialDuty[0], initialDuty[1], initialDuty[2], initialDuty[3]};
    float curPhase[4] = {initialPhase[0], initialPhase[1], initialPhase[2], initialPhase[3]};
    
    for (const auto& task : _queue) {
        if (task.type == TASK_SET_DUTY_CYCLES) {
            for(int i=0; i<4; i++) curDuty[i] = task.dutyCycles[i];
            
            TrajectoryPoint pt;
            pt.timeUs = currentTimeUs;
            for(int i=0; i<4; i++) { pt.freq[i] = curFreq[i]; pt.duty[i] = curDuty[i]; pt.phase[i] = curPhase[i]; }
            _trajectory.push_back(pt);
            
        } else if (task.type == TASK_WAIT) {
            currentTimeUs += task.durationUs;
            
            TrajectoryPoint pt;
            pt.timeUs = currentTimeUs;
            for(int i=0; i<4; i++) { pt.freq[i] = curFreq[i]; pt.duty[i] = curDuty[i]; pt.phase[i] = curPhase[i]; }
            _trajectory.push_back(pt);
            
        } else if (task.type == TASK_RAMP_LINEAR || task.type == TASK_RAMP_EASE) {
            int64_t stepUs = (int64_t)resolutionMs * 1000LL;
            int64_t elapsed = 0;
            
            // Generate intermediate points based on requested resolution
            while (elapsed <= task.durationUs) {
                float t = (float)elapsed / task.durationUs;
                if (task.type == TASK_RAMP_EASE) t = easeInOut(t);
                
                for(int i=0; i<4; i++) {
                    curFreq[i] = task.startFreq + t * (task.endFreq - task.startFreq);
                }
                
                TrajectoryPoint pt;
                pt.timeUs = currentTimeUs + elapsed;
                for(int i=0; i<4; i++) { pt.freq[i] = curFreq[i]; pt.duty[i] = curDuty[i]; pt.phase[i] = curPhase[i]; }
                _trajectory.push_back(pt);
                
                elapsed += stepUs;
            }
            
            // Ensure exact final value is snapped to at the end
            for(int i=0; i<4; i++) curFreq[i] = task.endFreq;
            currentTimeUs += task.durationUs;
            
            TrajectoryPoint pt;
            pt.timeUs = currentTimeUs;
            for(int i=0; i<4; i++) { pt.freq[i] = curFreq[i]; pt.duty[i] = curDuty[i]; pt.phase[i] = curPhase[i]; }
            _trajectory.push_back(pt);
        }
    }
}

void PhaseSequencer::start() {
    if (_trajectory.empty()) return; // Must compile first!
    _currentFrameIdx = 0;
    _taskStartTimeUs = esp_timer_get_time();
}

bool PhaseSequencer::isDone() const {
    return _currentFrameIdx >= _trajectory.size();
}

// =========================================================
// HIGH-SPEED HOT LOOP: No math, just pointer lookups
// =========================================================
void PhaseSequencer::run() {
    if (_trajectory.empty()) return;

    int64_t nowUs = esp_timer_get_time() - _taskStartTimeUs;

    // Fast-forward through the trajectory buffer based on elapsed time
    while (_currentFrameIdx < _trajectory.size() && nowUs >= _trajectory[_currentFrameIdx].timeUs) {
        
        TrajectoryPoint& pt = _trajectory[_currentFrameIdx];
        
        // Push pre-calculated states into the phase controller
        _phaseCtrl->setGlobalFrequency(pt.freq[0]); // Uses index 0 as global freq fallback
        
        for(int i = 0; i < 4; i++) {
            // Note: If PhaseController adds per-channel frequency later, uncomment this:
            // _phaseCtrl->setFrequency(i, pt.freq[i]); 
            _phaseCtrl->setDutyCycle(i, pt.duty[i]);
            _phaseCtrl->setPhase(i, pt.phase[i]);
        }
        
        _currentFrameIdx++;
    }
}