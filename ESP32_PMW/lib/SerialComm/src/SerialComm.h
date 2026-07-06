#pragma once

#include <Arduino.h>

// Non-blocking, newline-framed serial link. Generalizes the ad-hoc reader in
// main_tilt.cpp so any sketch can send/receive without re-deriving the
// buffering. Framing: ASCII lines terminated by \n, \r, or \r\n. No checksum,
// no binary framing -- matches the existing PC-side tools (run_experiment.py,
// trigger_reset_log.py) and their python mirror (tools/serial_comm.py).
class SerialComm {
public:
  explicit SerialComm(Stream &port = Serial) : _port(port) {}

  // Call once per loop() iteration. Never blocks.
  // If `outgoing` is non-empty, writes it followed by '\n'.
  // Drains whatever bytes are currently available; if a full line completes
  // during this call, returns it (buffer cleared). Otherwise returns "".
  // If more than one line's worth of bytes is waiting, only the first
  // completed line is returned/cleared this call -- nothing is dropped, the
  // rest stays in Serial's own buffer for the next call.
  String handleSerialComm(const String &outgoing = String());

private:
  static const size_t MAX_LINE_LEN = 128; // overflow guard against garbage

  Stream &_port;
  String _rxBuf;
};
