#include "Arduino.h"
#include "PhaseSequencer.h"

const bool IS_SERVER = true; 
const int PINS[] = {15, 33, 12, 27}; 
const float PHASES[] = {0.0, 90.0, 180.0, 270.0};
const int NUM_CHANNELS = 4;

const int PIN_RX = 16;
const int PIN_TX = 17;

PhaseSequencer* sequencer;

void setup() {
  Serial.begin(115200);
  Serial.print("--- ESP32 Phase Sequencer: ");
  Serial.println(IS_SERVER ? "SERVER" : "CLIENT");

  sequencer = new PhaseSequencer(PINS, PHASES, NUM_CHANNELS);
  sequencer->enableSync(IS_SERVER, &Serial2, PIN_RX, PIN_TX, 921600);
  sequencer->begin(10.0);
}

void loop() {
  sequencer->run();
  
  // Example of using buffered setters for dynamic control:
  // if (analogRead(A0) > 2000) {
  //    sequencer->setFrequencyNext(0, 60.0); // Applies in next run()
  // }
}