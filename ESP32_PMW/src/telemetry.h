#pragma once

#include "constants.h"
#include <Arduino.h>

// Shared "I[A]: A=.. B=.. C=.. D=.. | duty[%]: A=.. B=.. C=.. D=.." fragment;
// callers wrap it with their own prefix/suffix and own the trailing newline.
inline void printCurrentAndDuty(const float iMeas[NUM_CHANNELS],
                                const float dutyPct[NUM_CHANNELS]) {
  Serial.printf("I[A]: A=%.2f B=%.2f C=%.2f D=%.2f | "
                "duty[%%]: A=%.1f B=%.1f C=%.1f D=%.1f",
                iMeas[0], iMeas[1], iMeas[2], iMeas[3], dutyPct[0],
                dutyPct[1], dutyPct[2], dutyPct[3]);
}
