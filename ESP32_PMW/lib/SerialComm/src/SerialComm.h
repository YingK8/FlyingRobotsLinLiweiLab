#pragma once

#include <Arduino.h>

class SerialComm {
public:
  explicit SerialComm(Stream &port = Serial) : _port(port) {}

  // returns at most one line per call; the rest wait for the next call
  String handleSerialComm(const String &outgoing = String());

  static uint8_t checksum(const char *text, size_t length);

private:
  static const size_t MAX_LINE_LEN = 128; // an over-long out is discarded whole, never truncated
  bool checkAndStripChecksum(String &line);

  Stream &_port;
  String _rxBuf;
};
