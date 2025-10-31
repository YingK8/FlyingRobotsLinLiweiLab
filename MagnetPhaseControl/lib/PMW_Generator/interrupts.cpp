#include "interrupts.h"
#include <EnableInterrupt.h>
#include "pwm_controller.h"
#include "pmw_config.h"

extern PWMController pwmController;

void setupInterrupts() {
    enableInterrupt(btn_START_PinA11, ISR_START, FALLING);
    enableInterrupt(btn_STOP_PinA9, ISR_STOP, FALLING);
}

void ISR_START() {
    if (pwmController.opState != OperationState::STOPPED) return;

    switch (pwmController.opDirection) {
        case OperationDirection::FIXED_FORWARD_DIR:
            pwmController.currentRequest = RequestState::REQ_START_FWD;
            break;
        case OperationDirection::FIXED_REVERSE_DIR:
            pwmController.currentRequest = RequestState::REQ_START_REV;
            break;
        default:
            if (digitalRead(swt_SEQUENCE_DIRECTION_PinA7) == LOW) {
                pwmController.opDirection = OperationDirection::SWT_REVERSE_DIR;
                pwmController.currentRequest = RequestState::REQ_START_REV;
            } else {
                pwmController.opDirection = OperationDirection::SWT_FORWARD_DIR;
                pwmController.currentRequest = RequestState::REQ_START_FWD;
            }
    }
}

void ISR_STOP() {
    if (pwmController.opState == OperationState::ACTIVE) {
        pwmController.currentRequest = RequestState::REQ_STOP;
    }
}