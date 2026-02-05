// /**
//  * ESP32 4-Channel Software Phase-Shifted PWM
//  * * DYNAMIC DUTY CYCLE VERSION
//  * * METHOD: Non-blocking micros() polling (Time Slicing).
//  * * FREQUENCY: 10 Hz
//  * * NOTE: We use micros() for fine timing control (100ms period).
//  */

// #include "Arduino.h"

// // --- Configuration ---
// const int PIN_PHASE_0   = 15; //A
// const int PIN_PHASE_90  = 33; //C
// const int PIN_PHASE_180 = 12; //B
// const int PIN_PHASE_270 = 27; //D

// const int PWM_FREQ_HZ = 10;  // 10 Hz = 100ms period
// const unsigned long PERIOD_US = 1000000 / PWM_FREQ_HZ;

// // --- Dynamic Variables ---
// // Change this variable anywhere in your code to adjust brightness/width
// float dutyCyclePercent = 50.0; 

// // Calculated internally
// unsigned long dutyWidthUS; 
// unsigned long offset0, offset90, offset180, offset270;

// void setup() {
//   Serial.begin(115200);
//   Serial.println("Starting Dynamic PWM...");
//   Serial.print("Period: "); Serial.print(PERIOD_US); Serial.println(" us");

//   pinMode(PIN_PHASE_0, OUTPUT);
//   pinMode(PIN_PHASE_90, OUTPUT);
//   pinMode(PIN_PHASE_180, OUTPUT);
//   pinMode(PIN_PHASE_270, OUTPUT);

//   // Calculate fixed phase offsets (0%, 25%, 50%, 75% of period)
//   offset0   = 0;
//   offset90  = PERIOD_US / 4;
//   offset180 = PERIOD_US / 2;
//   offset270 = (PERIOD_US * 3) / 4;
// }

// // Helper to check if we are inside the "Active" window for a phase
// bool checkPhase(unsigned long currentPos, unsigned long shift, unsigned long width) {
//   unsigned long endPos = (shift + width) % PERIOD_US;
  
//   if (shift < endPos) {
//     // Normal active range (e.g., Start: 100, End: 200)
//     return (currentPos >= shift && currentPos < endPos);
//   } else {
//     // Wrapping active range (e.g., Start: 3000, End: 100)
//     return (currentPos >= shift || currentPos < endPos);
//   }
// }

// void loop() {
//   // 1. Time Capture & Sync
//   // We capture time ONCE. All logic uses this snapshot.
//   // This provides the "compensation" for calculation delays, ensuring
//   // all pins switch based on the exact same moment in time.
//   unsigned long now = micros();
//   unsigned long cyclePos = now % PERIOD_US;

//   // 2. Update Pulse Width (allows dynamic adjustment)
//   dutyWidthUS = (unsigned long)((PERIOD_US * dutyCyclePercent) / 100.0);

//   // 3. Calculate States
//   // We calculate all states BEFORE writing to minimize time gap between pins
//   bool s0   = checkPhase(cyclePos, offset0,   dutyWidthUS);
//   bool s90  = checkPhase(cyclePos, offset90,  dutyWidthUS);
//   bool s180 = checkPhase(cyclePos, offset180, dutyWidthUS);
//   bool s270 = checkPhase(cyclePos, offset270, dutyWidthUS);

//   // 4. Write to Pins
//   digitalWrite(PIN_PHASE_0,   s0);
//   digitalWrite(PIN_PHASE_90,  s90);
//   digitalWrite(PIN_PHASE_180, s180);
//   digitalWrite(PIN_PHASE_270, s270);
// }