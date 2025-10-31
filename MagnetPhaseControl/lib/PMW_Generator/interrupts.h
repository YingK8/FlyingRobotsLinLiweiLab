#ifndef INTERRUPTS_H
#define INTERRUPTS_H

#include "pwm_controller.h"

void setupInterrupts();
void ISR_START();
void ISR_STOP();

#endif // INTERRUPTS_H