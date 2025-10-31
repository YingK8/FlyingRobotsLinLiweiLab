#ifndef PMW_CONFIG_H
#define PMW_CONFIG_H

#include <Arduino.h>

// Frequency Configuration
#define TIMER_FREQ_10kHz

#if defined TIMER_FREQ_10kHz
constexpr int ICAP_NDiv = 800;
constexpr int DUTY_25_PerCent_VALUE = 200;
constexpr int DEAD_TIME = 4;
constexpr int PHASE_OFFSET = 65535 - 400;
constexpr int PRESCALE = 1;
constexpr int ADJ_CONST = 3;
constexpr int WAIT_CONST = 33;
constexpr int Duty_Numeric_Boundary = 5;
const String OP_Frequency = "10 kHz";
#endif

// Pin Definitions
constexpr uint8_t btn_DutyIncrease_PinA15 = A15;
constexpr uint8_t btn_DutyDecrease_PinA13 = A13;
constexpr uint8_t btn_START_PinA11 = A11;
constexpr uint8_t btn_STOP_PinA9 = A9;
constexpr uint8_t swt_SEQUENCE_DIRECTION_PinA7 = A7;

#endif // PMW_CONFIG_H