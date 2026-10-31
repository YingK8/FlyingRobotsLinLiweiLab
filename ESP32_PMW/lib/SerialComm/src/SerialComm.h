#pragma once

#include <Arduino.h>

// no handshake: a baud mismatch is silent
class SerialComm {
public:
  explicit SerialComm(Stream &port = Serial) : _port(port) {}

  // returns at most one line per call; the rest wait for the next call
  String handleSerialComm(const String &outgoing = String());

private:
  static const size_t MAX_LINE_LEN = 128; // an over-long out is discarded whole, never truncated

  Stream &_port;
  String _rxBuf;
};
