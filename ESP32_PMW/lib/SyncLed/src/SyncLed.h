#pragma once

#include <stdint.h>

namespace sync_led {

struct PulseTiming {
  uint16_t on_ms = 30;
  uint16_t gap_ms = 30;
  uint16_t blink_ms = 30;
  uint16_t code_ms = 80; // controller/pose/scene.py CODE_BIT_S
};

enum class Signal : uint8_t { None, Start, Stop, Blink, Code };

class SyncLed {
 public:
  explicit SyncLed(const PulseTiming &timing = PulseTiming()) : timing_(timing) {}

  const PulseTiming &timing() const { return timing_; }
  Signal signal() const { return signal_; }
  bool isSendingSignal() const { return signal_ != Signal::None; }
  bool isOn() const { return isOn_; }

  bool sendStartSignal(uint32_t now_ms) { return beginSignal(Signal::Start, true, now_ms); }
  bool sendStopSignal(uint32_t now_ms) { return beginSignal(Signal::Stop, false, now_ms); }
  bool sendBlink(uint32_t now_ms) { return beginSignal(Signal::Blink, true, now_ms); }
  bool sendCode(uint8_t word7, uint32_t now_ms) { word_ = word7; return beginSignal(Signal::Code, true, now_ms); }

  bool holdOn() { return endSignal(true); }
  bool holdOff() { return endSignal(false); }

  bool advanceTo(uint32_t now_ms);

 private:
  bool beginSignal(Signal signal, bool on, uint32_t now_ms);

  bool endSignal(bool on);

  bool setOn(bool on);

  PulseTiming timing_;
  Signal signal_ = Signal::None;
  uint8_t edgesWritten_ = 0;
  uint32_t lastEdgeMs_ = 0;
  bool isOn_ = false; // starts dark so the first rising edge is unambiguous
  uint8_t word_ = 0;
};

}  // namespace sync_led
