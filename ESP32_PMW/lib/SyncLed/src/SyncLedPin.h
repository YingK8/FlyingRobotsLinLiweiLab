#pragma once

#include <Arduino.h>

#include "SyncLed.h"

namespace sync_led {

// the pin is handed over once by driveBoot()
class PinLed {
 public:
  void attachTo(uint8_t pin);

  void sendStartSignal() { machine_.sendStartSignal(millis()); write(); }
  void sendStopSignal() { machine_.sendStopSignal(millis()); write(); }
  void sendBlink() { machine_.sendBlink(millis()); write(); }
  void sendCode(uint8_t word7) { machine_.sendCode(word7, millis()); write(); }
  void holdOn() { machine_.holdOn(); write(); }
  void holdOff() { machine_.holdOff(); write(); }

  void flip();

  void advance();

  bool isOn() const { return machine_.isOn(); }
  bool isSendingSignal() const { return machine_.isSendingSignal(); }
  const PulseTiming &timing() const { return machine_.timing(); }

 private:
  void write();

  static constexpr uint8_t unattached = 255;
  uint8_t pin_ = unattached;
  SyncLed machine_;
};

inline PinLed &led() {
static PinLed instance;
  return instance;
}

}  // namespace sync_led
