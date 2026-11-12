#include "SerialComm.h"

uint8_t SerialComm::checksum(const char *text, size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; i++) {
    crc ^= (uint8_t)text[i];
    for (int bit = 0; bit < 8; bit++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  }
  return crc;
}

bool SerialComm::checkAndStripChecksum(String &line) {
  const int length = line.length();
  if (length < 3 || line[length - 3] != '*') return true;
  const bool isHex = isxdigit(line[length - 2]) && isxdigit(line[length - 1]);
  const long expected = strtol(line.c_str() + length - 2, nullptr, 16);
  if (!isHex || checksum(line.c_str(), length - 3) != expected) {
    _port.print("!crc ");
    _port.println(line);
    return false;
  }
  line.remove(length - 3);
  return true;
}

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
      if (line.length() && line.length() <= MAX_LINE_LEN && checkAndStripChecksum(line)) return line;
    } else if (_rxBuf.length() <= MAX_LINE_LEN) {
      _rxBuf += c; // stop appending past the cap; an over-long line is already doomed
    }
  }
  return String();
}
