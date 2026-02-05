/**
 * ESP32 4-Channel Smooth Phase Controller (Debugged & Optimized)
 * * FIXES:
 * 1. Changed pwmFreqHz to FLOAT. (Integer truncation prevented 0.5Hz steps).
 * 2. Fixed Setup logic which jumped straight to 50Hz, skipping the sweep.
 * 3. Optimized Loop: Moved math out of the main loop to prevent jitter.
 */

#include "Arduino.h"

// --- 引脚定义 ---
const int PIN_PHASE_0   = 15; // A组
const int PIN_PHASE_90  = 33; // C组
const int PIN_PHASE_180 = 12; // B组
const int PIN_PHASE_270 = 27; // D组

// --- 动态运行变量 ---
float pwmFreqHz = 10.0; // Changed to FLOAT to support 0.5 increments
float periodUs = 1000000.0 / pwmFreqHz;

// --- 相位累加器 ---
float currentCyclePos = 0;       
unsigned long lastLoopMicros = 0;

// --- 频率控制 ---
unsigned long lastFreqUpdateTime = 0;
unsigned long loopStartTime = 0;
float targetFreqHz = 50.0;
float FreqStep = 0.5; 

// --- 独立 Duty Cycle (0.0 - 100.0) ---
float dutyCycle0   = 50.0; 
float dutyCycle90  = 50.0; 
float dutyCycle180 = 50.0; 
float dutyCycle270 = 50.0; 

// --- 预计算参数 (Optimization) ---
// Pre-calculate start and end times to keep the loop extremely fast
struct PhaseParams {
  float start;
  float end;
  bool wraps;
};
PhaseParams p0, p90, p180, p270;

// Helper to calculate parameters (Moved out of loop)
void updatePhaseParams() {
  auto calc = [](float offsetFactor, float duty) -> PhaseParams {
    PhaseParams p;
    float width = periodUs * duty / 100.0;
    p.start = periodUs * offsetFactor;
    p.end = p.start + width;
    
    // Handle wrapping (pulse crosses the cycle boundary)
    if (p.end > periodUs) {
      p.end -= periodUs;
      p.wraps = true;
    } else {
      p.wraps = false;
    }
    return p;
  };

  p0   = calc(0.00, dutyCycle0);
  p90  = calc(0.25, dutyCycle90);
  p180 = calc(0.50, dutyCycle180);
  p270 = calc(0.75, dutyCycle270);
}

/**
 * 设置频率并保持相位连续性
 */
void setFrequency(float newHz) {
  if (newHz < 1.0 || newHz > 1000.0) return;

  float oldPeriod = periodUs;
  float newPeriod = 1000000.0 / newHz;

  // 1. Scale current position to maintain phase continuity
  float ratio = currentCyclePos / oldPeriod;
  currentCyclePos = ratio * newPeriod;

  // 2. Update Globals
  pwmFreqHz = newHz;
  periodUs = newPeriod;

  // 3. Recalculate thresholds
  updatePhaseParams();
}

void setup() {
  Serial.begin(115200);
  Serial.println("--- ESP32 Debugged Controller ---");
  
  pinMode(PIN_PHASE_0, OUTPUT);
  pinMode(PIN_PHASE_90, OUTPUT);
  pinMode(PIN_PHASE_180, OUTPUT);
  pinMode(PIN_PHASE_270, OUTPUT);

  // Initialize at the START frequency (10Hz), not the target (50Hz)
  setFrequency(pwmFreqHz); 
  
  lastLoopMicros = micros();
  loopStartTime = millis();
}

void loop() {
  // 1. Time Stepping (Addition only - Fast)
  unsigned long now = micros();
  unsigned long dt = now - lastLoopMicros;
  lastLoopMicros = now;

  // 2. Frequency Sweep Logic
  // Wait 8 seconds, then sweep from 10Hz to 50Hz
  if (millis() - loopStartTime >= 8000) {
    if (now - lastFreqUpdateTime >= 100000) { // Every 0.1s
      lastFreqUpdateTime = now;
      
      // Floating point comparison handles the 0.5 steps correctly now
      if (pwmFreqHz < targetFreqHz) {
        setFrequency(pwmFreqHz + FreqStep);
      } else if (pwmFreqHz > targetFreqHz) {
        setFrequency(pwmFreqHz - FreqStep);
      }
    }
  }

  // 3. Advance Phase Accumulator
  currentCyclePos += dt;
  if (currentCyclePos >= periodUs) {
    currentCyclePos -= periodUs;
  }

  // 4. Pin Output (Comparison only - Very Fast)
  // Logic: "If wrapping, active if OUTSIDE the gap. If normal, active INSIDE the range."
  digitalWrite(PIN_PHASE_0,   p0.wraps   ? (currentCyclePos >= p0.start   || currentCyclePos < p0.end)   : (currentCyclePos >= p0.start   && currentCyclePos < p0.end));
  digitalWrite(PIN_PHASE_90,  p90.wraps  ? (currentCyclePos >= p90.start  || currentCyclePos < p90.end)  : (currentCyclePos >= p90.start  && currentCyclePos < p90.end));
  digitalWrite(PIN_PHASE_180, p180.wraps ? (currentCyclePos >= p180.start || currentCyclePos < p180.end) : (currentCyclePos >= p180.start && currentCyclePos < p180.end));
  digitalWrite(PIN_PHASE_270, p270.wraps ? (currentCyclePos >= p270.start || currentCyclePos < p270.end) : (currentCyclePos >= p270.start && currentCyclePos < p270.end));
}