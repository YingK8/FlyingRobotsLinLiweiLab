#pragma once

#include <Arduino.h>

#include "SyncLed.h"

namespace sync_led {

class PinLed {
 public:
  void attachTo(uint8_t pin) {
    pin_ = pin;
    pinMode(pin_, OUTPUT);
    machine_.holdOff();
    write();
  }

  void sendStartSignal() { machine_.sendStartSignal(millis()); write(); }
  void sendStopSignal() { machine_.sendStopSignal(millis()); write(); }
  void sendBlink() { machine_.sendBlink(millis()); write(); }
  void holdOn() { machine_.holdOn(); write(); }
  void holdOff() { machine_.holdOff(); write(); }

  void flip() {
    if (machine_.isOn()) holdOff();
    else holdOn();
  }

  void advance() {
    if (machine_.advanceTo(millis())) write();
  }

  bool isOn() const { return machine_.isOn(); }
  bool isSendingSignal() const { return machine_.isSendingSignal(); }
  const PulseTiming &timing() const { return machine_.timing(); }

 private:
  void write() { digitalWrite(pin_, machine_.isOn() ? HIGH : LOW); }

  uint8_t pin_ = 255;
  SyncLed machine_;
};

inline PinLed &led() {
  static PinLed instance;
  return instance;
}

}  // namespace sync_led
