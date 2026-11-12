#pragma once

#include <Arduino.h>

// Non-blocking, newline-framed serial link. Framing: ASCII lines ending in \n,
// \r, or \r\n.
class SerialComm {
public:
  explicit SerialComm(Stream &port = Serial) : _port(port) {}

  // Call once per loop(); never blocks. Writes `outgoing` + '\n' if non-empty.
  // Returns the first complete line drained this call (buffer cleared), else "".
  // Extra buffered lines stay in Serial for the next call. Lines longer than
  // MAX_LINE_LEN are discarded whole rather than truncated.
  String handleSerialComm(const String &outgoing = String());

  static uint8_t checksum(const char *text, size_t length);

private:
  static const size_t MAX_LINE_LEN = 128; // overflow guard against garbage
  bool checkAndStripChecksum(String &line);

  Stream &_port;
  String _rxBuf;
};
