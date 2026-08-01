#include "SerialComm.h"

String SerialComm::handleSerialComm(const String &outgoing) {
  if (outgoing.length()) {
    _port.print(outgoing);
    _port.print('\n');
  }

  while (_port.available()) {
    char c = (char)_port.read();
    if (c == '\n' || c == '\r') {
      String line = _rxBuf;
      _rxBuf = "";
      if (line.length() && line.length() <= MAX_LINE_LEN) return line;
    } else if (_rxBuf.length() <= MAX_LINE_LEN) {
      _rxBuf += c; // stop appending once over the cap; the line is already doomed
    }
  }
  return String();
}
