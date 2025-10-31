#include <Arduino.h>
#include "pwm_controller.h"
#include "pmw_config.h"

PWMController pwmController;

void PWMController::initialize() {
    // Set startup duty value
    switch (startUpDuty) {
        case DutyStartType::DUTY_25_PERCENT:
            startUpDutyCycleValue = (dutyCycleRange == DutyRangeType::MAX_25_MINUS_DEADTIME) 
                ? (DUTY_25_PerCent_VALUE - DEAD_TIME) 
                : DUTY_25_PerCent_VALUE;
            break;
        case DutyStartType::DUTY_10_PERCENT:
            startUpDutyCycleValue = static_cast<int>(ICAP_NDiv * 0.100);
            break;
        case DutyStartType::DUTY_02_PERCENT:
            startUpDutyCycleValue = static_cast<int>(ICAP_NDiv * 0.020);
            break;
    }

    // Configure pins
    pinMode(btn_DutyIncrease_PinA15, INPUT_PULLUP);
    pinMode(btn_DutyDecrease_PinA13, INPUT_PULLUP);
    pinMode(btn_START_PinA11, INPUT_PULLUP);
    pinMode(btn_STOP_PinA9, INPUT_PULLUP);
    pinMode(swt_SEQUENCE_DIRECTION_PinA7, INPUT_PULLUP);
    pinMode(14, OUTPUT); // Logic analyzer trigger

#ifdef DEBUG
    Serial.println(F("Phase Shift PWM Initialized"));
    Serial.print(F("Frequency: "));
    Serial.println(OP_Frequency);
#endif
}

void PWMController::handleRequests() {
    if (currentRequest == RequestState::NO_REQUEST_PENDING) return;

    switch (currentRequest) {
        case RequestState::REQ_START_FWD:
            currentRequest = RequestState::NO_REQUEST_PENDING;
            startPWMForward();
            break;
            
        case RequestState::REQ_START_REV:
            currentRequest = RequestState::NO_REQUEST_PENDING;
            startPWMReverse();
            break;
            
        case RequestState::REQ_STOP:
            currentRequest = RequestState::NO_REQUEST_PENDING;
            stopPWM();
            break;
            
        default:
            break;
    }
}

void PWMController::handleDutyAdjustment() {
    if (opState != OperationState::ACTIVE) return;

    if (digitalRead(btn_DutyIncrease_PinA15) == LOW) {
        dutyAdjust(+ADJ_CONST);
        delay(WAIT_CONST);
    }

    if (digitalRead(btn_DutyDecrease_PinA13) == LOW) {
        dutyAdjust(-ADJ_CONST);
        delay(WAIT_CONST);
    }
}

void PWMController::startPWMForward() {
    if (opState != OperationState::STOPPED) return;

    GTCCR |= 0x81; // Stop timers
    
    DDRB &= 0x9F; DDRE &= 0xE7; // Disable outputs

    // Timer 1 setup
    TCCR1B = 0x18; TCCR1A = 0x50;
    OCR1B = startUpDutyCycleValue;
    OCR1A = (ICAP_NDiv - startUpDutyCycleValue);
    TCCR1B = 0x10; TCCR1A = 0xE2;
    ICR1 = ICAP_NDiv;

    // Timer 3 setup
    TCCR3B = 0x18; TCCR3A = 0x50;
    OCR3B = startUpDutyCycleValue;
    OCR3A = (ICAP_NDiv - startUpDutyCycleValue);
    TCCR3B = 0x10; TCCR3A = 0xE2;
    ICR3 = ICAP_NDiv;

    TCCR1B |= PRESCALE; TCCR3B |= PRESCALE;
    TCNT1 = 65535; TCNT3 = PHASE_OFFSET;

    // Trigger and enable
    noInterrupts();
    digitalWrite(14, HIGH); digitalWrite(14, LOW);
    interrupts();
    
    DDRB |= 0x60; DDRE |= 0x18;
    GTCCR = 0x0;
    opState = OperationState::ACTIVE;

#ifdef DEBUG
    Serial.println(F("STARTED: Forward"));
    Serial.print(F("Duty: ")); Serial.print(getDutyValue()); Serial.println(F(" %"));
#endif
}

void PWMController::startPWMReverse() {
    if (opState != OperationState::STOPPED) return;

    GTCCR |= 0x81;
    DDRB &= 0x9F; DDRE &= 0xE7;

    // Timer 1 setup (reversed)
    TCCR1B = 0x18; TCCR1A = 0x50;
    OCR1A = startUpDutyCycleValue;
    OCR1B = (ICAP_NDiv - startUpDutyCycleValue);
    TCCR1B = 0x10; TCCR1A = 0xB2;
    ICR1 = ICAP_NDiv;

    // Timer 3 setup (reversed)
    TCCR3B = 0x18; TCCR3A = 0x50;
    OCR3A = startUpDutyCycleValue;
    OCR3B = (ICAP_NDiv - startUpDutyCycleValue);
    TCCR3B = 0x10; TCCR3A = 0xB2;
    ICR3 = ICAP_NDiv;

    TCCR1B |= PRESCALE; TCCR3B |= PRESCALE;
    TCNT1 = PHASE_OFFSET; TCNT3 = 65535;

    DDRB |= 0x60; DDRE |= 0x18;
    noInterrupts();
    digitalWrite(14, HIGH); digitalWrite(14, LOW);
    interrupts();
    GTCCR = 0x0;
    opState = OperationState::ACTIVE;

#ifdef DEBUG
    Serial.println(F("STARTED: Reverse"));
    Serial.print(F("Duty: ")); Serial.print(getDutyValue()); Serial.println(F(" %"));
#endif
}

void PWMController::stopPWM() {
    if (opState != OperationState::ACTIVE) return;

    if (dutyDownRamp == DutyRampType::RAMP_ENABLED) {
        dutyRampDown();
    }

    noInterrupts();
    digitalWrite(14, HIGH); digitalWrite(14, LOW);
    interrupts();

    DDRB &= 0x9F; DDRE &= 0xE7;
    TCCR1A = 0x02; TCCR3A = 0x02;
    TCCR1B = 0x10; TCCR3B = 0x10;

    // Clean up timers
    TCCR1B = 0x18; TCCR1A = 0x50; TCCR1A = 0xA0; TCCR1C = 0xC0;
    TCCR3B = 0x18; TCCR3A = 0x50; TCCR3A = 0xA0; TCCR3C = 0xC0;

    DDRB |= 0x60; DDRE |= 0x18;
    opState = OperationState::STOPPED;

#ifdef DEBUG
    Serial.print(F("STOPPED at Duty: ")); Serial.print(getDutyValue()); Serial.println(F(" %"));
#endif
}

void PWMController::dutyAdjust(int up_down) {
    bool isForward = (opDirection == OperationDirection::FIXED_FORWARD_DIR || 
                     opDirection == OperationDirection::SWT_FORWARD_DIR);

    if (isForward) up_down = -up_down;

    ocrResult = OCR1A + up_down;
    ocrResult = constrain(ocrResult, Duty_Numeric_Boundary, ICR1 - Duty_Numeric_Boundary);

    // Apply duty limits
    if (dutyCycleRange != DutyRangeType::NO_LIMIT) {
        int maxDuty = (dutyCycleRange == DutyRangeType::MAX_25) ? 
            DUTY_25_PerCent_VALUE : (DUTY_25_PerCent_VALUE - DEAD_TIME);
        
        if (isForward) {
            ocrResult = max(ocrResult, ICR1 - maxDuty);
        } else {
            ocrResult = min(ocrResult, maxDuty);
        }
    }

    OCR1A = OCR3A = ocrResult;
    OCR1B = OCR3B = ICR1 - ocrResult;
}

float PWMController::getDutyValue() {
    bool isForward = (opDirection == OperationDirection::FIXED_FORWARD_DIR || 
                     opDirection == OperationDirection::SWT_FORWARD_DIR);
    
    return isForward ? 
        ((float(OCR1B) / float(ICAP_NDiv)) * 100.00) :
        ((float(OCR1A) / float(ICAP_NDiv)) * 100.00);
}

void PWMController::dutyRampDown() {
#ifdef DEBUG
    Serial.print(F("Ramping down from ")); 
    Serial.print(getDutyValue()); Serial.println(F(" %"));
#endif

    while (getDutyValue() > 1.0) {
        dutyAdjust(-ADJ_CONST);
        delay(WAIT_CONST);
    }
}