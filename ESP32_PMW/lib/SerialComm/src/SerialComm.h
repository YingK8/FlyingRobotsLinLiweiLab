#pragma once

#include <Arduino.h>

// Non-blocking, newline-framed serial link
// Framing: ASCII lines terminated by \n, \r, or \r\n

class SerialComm {
public:
  explicit SerialComm(Stream &port = Serial) : _port(port) {}

  // Call once per loop() iteration. non-blocking.
  String step(const String &outgoing = String());

private:
  static const size_t MAX_LINE_LEN = 128; // overflow guard against garbage

  Stream &_port;
  String _rxBuf;
};
