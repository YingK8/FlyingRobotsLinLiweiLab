#include <Arduino.h>

#include "SerialComm.h"

SerialComm comm;

void setup() { Serial.begin(115200); }

void loop() {
  String line = comm.handleSerialComm();
  if (line.length()) {
    comm.handleSerialComm("echo: " + line);
  }
}
