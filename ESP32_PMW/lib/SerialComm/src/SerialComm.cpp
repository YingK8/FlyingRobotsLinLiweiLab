#include "SerialComm.h"

String SerialComm::handleSerialComm(const String &outgoing) {
  if (outgoing.length()) {
    _port.print(outgoing);
    _port.print('\n');
  }

  while (_port.available()) {
    char c = (char)_port.read();
    if (c == '\n' || c == '\r') {
      if (_rxBuf.length()) {
        String line = _rxBuf;
        _rxBuf = "";
        return line;
      }
      // bare CR/LF with nothing buffered (e.g. second half of CRLF) -- keep draining
    } else {
      _rxBuf += c;
      if (_rxBuf.length() > MAX_LINE_LEN) _rxBuf = ""; // overflow guard
    }
  }
  return String();
}
