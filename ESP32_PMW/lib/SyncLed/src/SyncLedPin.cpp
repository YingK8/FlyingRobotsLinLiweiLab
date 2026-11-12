#include "SyncLedPin.h"

namespace sync_led {

void PinLed::attachTo(uint8_t pin) {
  pin_ = pin;
  pinMode(pin_, OUTPUT);
  machine_.holdOff();
  write();
}

void PinLed::flip() {
  if (machine_.isOn()) holdOff();
  else holdOn();
}

// call every loop or the second edge never happens
void PinLed::advance() {
  if (machine_.advanceTo(millis())) write();
}

void PinLed::write() {
  if (pin_ == unattached) return;
  digitalWrite(pin_, machine_.isOn() ? HIGH : LOW);
}

}  // namespace sync_led
