#include "PhaseController.h"

// --- Pin Definitions ---
const int PIN_PHASE_0   = 15;
const int PIN_PHASE_90  = 33;
const int PIN_PHASE_180 = 12;
const int PIN_PHASE_270 = 27;

// --- Application Settings ---
const float START_FREQ = 10.0;
const float TARGET_FREQ = 50.0;
const float FREQ_STEP = 0.5;
const unsigned long SWEEP_DELAY_START = 8000; // Wait 8 seconds before sweeping

// --- Global Objects ---
PhaseController phaseCtrl(PIN_PHASE_0, PIN_PHASE_90, PIN_PHASE_180, PIN_PHASE_270);

// --- Timing Variables for Sweep ---
unsigned long appStartTime = 0;
unsigned long lastFreqUpdate = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("--- ESP32 Phase Controller (Library Version) ---");

  // Initialize controller at Start Frequency
  phaseCtrl.begin(START_FREQ);
  
  // Set individual duty cycles if needed (default is 50%)
  // phaseCtrl.setDutyCycle(0, 50.0);
  
  appStartTime = millis();
}

void loop() {
  // 1. HIGH PRIORITY: Update the signals
  // This must be called every loop and execution must be fast
  phaseCtrl.run();

  // 2. LOW PRIORITY: Application Logic (Frequency Sweep)
  unsigned long now = millis();

  // Check if 8 seconds have passed
  if (now - appStartTime >= SWEEP_DELAY_START) {
    
    // Update frequency every 100ms
    if (now - lastFreqUpdate >= 100) {
      lastFreqUpdate = now;
      
      float currentFreq = phaseCtrl.getFrequency();
      
      // Floating point comparison with small epsilon for safety
      if (currentFreq < TARGET_FREQ - 0.01) {
        phaseCtrl.setFrequency(currentFreq + FREQ_STEP);
        // Serial.printf("Freq: %.2f Hz\n", currentFreq + FREQ_STEP); // Debug (use sparingly to avoid blocking)
      } 
      else if (currentFreq > TARGET_FREQ + 0.01) {
        phaseCtrl.setFrequency(currentFreq - FREQ_STEP);
      }
    }
  }
}