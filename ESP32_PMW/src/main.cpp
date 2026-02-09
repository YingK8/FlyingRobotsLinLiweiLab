#include "Arduino.h"
#include "PhaseSequencer.h"

const bool IS_SERVER = true; 
const int PINS[] = {15, 33, 12, 27}; 
const float PHASES[] = {0.0, 90.0, 180.0, 270.0};
const int NUM_CHANNELS = 4;

const int SYNC_PIN = 4;

PhaseSequencer* sequencer;

void setup() {
  Serial.begin(115200);
  Serial.print("--- ESP32 Phase Sequencer: ");
  Serial.println(IS_SERVER ? "SERVER" : "CLIENT");

  sequencer = new PhaseSequencer(PINS, PHASES, NUM_CHANNELS);
  
  // Update: Use GPIO 4 Sync
  sequencer->enableSync(IS_SERVER, SYNC_PIN);
  
  sequencer->begin(10.0); // initialFreqHz = 10Hz

  if (IS_SERVER) {
      Serial.println("Starting Realtime Control...");
  }
}

void loop() {
  sequencer->run();
  
  // Example of using buffered setters for dynamic control:
  // if (analogRead(A0) > 2000) {
  //    sequencer->setFrequencyNext(0, 60.0); // Applies in next run()
  // }

  for (int channel = 0; channel < NUM_CHANNELS; channel++) {
    sequencer->setFrequencyNext(channel, 10.0);
    sequencer->rampFrequency(channel, 100.0,
                             10000); // Ramp channel to 100Hz over 10 seconds
  }
}