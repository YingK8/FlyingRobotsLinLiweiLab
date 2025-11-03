#include <Arduino.h>
#include "PMW.h"

const int Duty_Numeric_Boundary = 20;

#if defined TIMER_FREQ_50Hz
const int ICAP_NDiv = 20000;
const int DUTY_25_PerCent_VALUE = 5000;
const int DEAD_TIME = 25;
const int PHASE_OFFSET = 65535 - 10000;
const int PRESCALE = 2;
const int ADJ_CONST = 91;
const int WAIT_CONST = 16;
const String OP_Frequency = "50 Hz";
#elif defined TIMER_FREQ_100Hz
const int ICAP_NDiv = 10000;
const int DUTY_25_PerCent_VALUE = 2500;
const int DEAD_TIME = 13;
const int PHASE_OFFSET = 65535 - 5000;
const int PRESCALE = 2;
const int ADJ_CONST = 51;
const int WAIT_CONST = 17;
const String OP_Frequency = "100 Hz";
#elif defined TIMER_FREQ_200Hz
const int ICAP_NDiv = 5000;
const int DUTY_25_PerCent_VALUE = 1250;
const int DEAD_TIME = 6;
const int PHASE_OFFSET = 65535 - 2500;
const int PRESCALE = 2;
const int ADJ_CONST = 23;
const int WAIT_CONST = 16;
const String OP_Frequency = "200 Hz";
#elif defined TIMER_FREQ_400Hz
const int ICAP_NDiv = 20000;
const int DUTY_25_PerCent_VALUE = 5000;
const int DEAD_TIME = 25;
const int PHASE_OFFSET = 65535 - 10000;
const int PRESCALE = 1;
const int ADJ_CONST = 81;
const int WAIT_CONST = 16;
const String OP_Frequency = "400 Hz";
#elif defined TIMER_FREQ_500Hz
const int ICAP_NDiv = 16000;
const int DUTY_25_PerCent_VALUE = 4000;
const int DEAD_TIME = 20;
const int PHASE_OFFSET = 65535 - 8000;
const int PRESCALE = 1;
const int ADJ_CONST = 67;
const int WAIT_CONST = 16;
const String OP_Frequency = "500 Hz";
#endif

// --- Pin/Const Definitions ---
const int ZERO_PHASE_REFERENCE = 65535;
const int btn_DutyIncrease_PinA15 = A15;
const int btn_DutyDecrease_PinA13 = A13;
const int btn_START_PinA11 = A11;
const int btn_STOP_PinA9 = A9;
const int swt_SEQUENCE_DIRECTION_PinA7 = A7;

// Helper function to check direction
bool isForwardDirection() {
    return (OP_DIRECTION == FIXED_FORWARD_DIR) || (OP_DIRECTION == SWT_FORWARD_DIR);
}

float Get_DutyValue()
{
    // obtain duty cycle value as percent 
    if (isForwardDirection()) {
        return ((float(OCR1B) / float(ICAP_NDiv)) * 100.0);
    } else { // reverse dir
        return ((float(OCR1A) / float(ICAP_NDiv)) * 100.0);
    }
}

void DUTY_Ramp_Down() // Ramp Duty down to minimum before Stopping 
{
    if (OP_STATE != ACTIVE) return;

    Serial.println(F("Ramping Down..."));

    // Ramp Duty down
    while (Get_DutyValue() > 1.0)
    {
        DUTY_Adjust(-ADJ_CONST); // adjust-n-wait vals defined in Freq Code Sections
        delay(WAIT_CONST);
    }
}

void ISR_START()
{
    // test if PSPWM stopped
    if (OP_STATE != STOPPED) return;

    if (OP_DIRECTION == FIXED_FORWARD_DIR) {
        CURRENT_REQ_STATE = REQ_START_FWD;
    } else if (OP_DIRECTION == FIXED_REVERSE_DIR) {
        CURRENT_REQ_STATE = REQ_START_REV;
    } else {
        // Use hardware switch
        if (digitalRead(swt_SEQUENCE_DIRECTION_PinA7) == LOW) {
            OP_DIRECTION = SWT_REVERSE_DIR;
            CURRENT_REQ_STATE = REQ_START_REV;
        } else {
            OP_DIRECTION = SWT_FORWARD_DIR;
            CURRENT_REQ_STATE = REQ_START_FWD;
        }
    }
}

void ISR_STOP()
{
    // test if PSPWM running
    if (OP_STATE == ACTIVE) {
        CURRENT_REQ_STATE = REQ_STOP;
    }
}

void START_PSPWM(bool isForward)
{
    // test if PSPWM stopped
    if (OP_STATE != STOPPED) return;

    // STOP TIMERS 0, 1, 3, 4, 5 and Clear preScalers
    GTCCR |= 0x81;

    // disable 4 (PULLED-DOWN) output pins 
    DDRB &= 0x9F; // -> 1001 1111
    DDRE &= 0xE7; // -> 1110 0111 

    // Set values based on direction
    uint8_t tccr_a_val = isForward ? 0xE2 : 0xB2;
    int ocr_a_val = isForward ? (ICAP_NDiv - StartUp_Duty_Cycle_Value) : StartUp_Duty_Cycle_Value;
    int ocr_b_val = isForward ? StartUp_Duty_Cycle_Value : (ICAP_NDiv - StartUp_Duty_Cycle_Value);

    // Set up Timer-1
    TCCR1B = 0x18; // Mode-12 CTC 
    TCCR1A = 0x50; // IMMEDIATE UPDATE TO OCR
    OCR1A = ocr_a_val;
    OCR1B = ocr_b_val;
    TCCR1B = 0x10; // Mode 10 Phase
    TCCR1A = tccr_a_val; // Assert chan-B normal, chan-A comp (or reverse)
    ICR1 = ICAP_NDiv; // Frequency control  

    // Set up Timer-3
    TCCR3B = 0x18;  // Mode-12 CTC
    TCCR3A = 0x50; // IMMEDIATE UPDATE TO OCRs
    OCR3A = ocr_a_val;
    OCR3B = ocr_b_val;
    TCCR3B = 0x10;   // Mode 10
    TCCR3A = tccr_a_val;   // Assert chan-B normal, chan-A comp (or reverse)
    ICR3 = ICAP_NDiv; // Frequency control 

    // set Prescale-Clock selectors
    TCCR1B |= PRESCALE;
    TCCR3B |= PRESCALE;

    // Set 90 degree phase shift between Timers
    if (isForward) {
        TCNT1 = ZERO_PHASE_REFERENCE;   // T1 LEADS T3
        TCNT3 = PHASE_OFFSET;
    } else {
        TCNT1 = PHASE_OFFSET; // T1 Output LAGS  T3
        TCNT3 = ZERO_PHASE_REFERENCE;
    }

    // fire Logic Analyzer trigger pulse (~4 uSec width)
    noInterrupts();
    digitalWrite(14, HIGH);
    digitalWrite(14, LOW);
    interrupts();

    // Enable output pins
    DDRB |= 0x60;
    DDRE |= 0x18;

    // start TIMERS 0, 1, 3, 4, 5 and associated Prescalers
    GTCCR = 0x0;

    // Update Status
    OP_STATE = ACTIVE;
    
    if (isForward) {
        Serial.println(F("Started Forward"));
    } else {
        Serial.println(F("Started Reverse"));
    }
}

void STOP_PSPWM()
{
    // test if PSPWM running
    if (OP_STATE != ACTIVE) return;

    if (DUTY_DOWN_RAMP == RAMP_ENABLED)
    {
        // Ramp DUTY-CYCLE Down to low-level before STOP
        DUTY_Ramp_Down();
    }

    // fire Logic Analyzer trigger pulse (~4 uSec width)
    noInterrupts();
    digitalWrite(14, HIGH);
    digitalWrite(14, LOW);
    interrupts();

    // begin STOP
    DDRB &= 0x9F;  // disable (PULLED-DOWN) output pins
    DDRE &= 0xE7;

    TCCR1A = 0x02;  // change Timer output to Normal Port-Pin 
    TCCR3A = 0x02;

    TCCR1B = 0x10;  // STOP Timer-1 (no clock)
    TCCR3B = 0x10;  // STOP Timer-3 (no clock)

    // Clean Up T1 WaveGens for Next Start
    TCCR1B = 0x18;  // partial mode 12 CTC
    TCCR1A = 0x50;  // assert Mode 12 CTC
    TCCR1A = 0xA0;  // FOC setup
    TCCR1C = 0xC0;  // FOC strobe Clears them

    // Clean Up T3 WaveGens for Next Start
    TCCR3B = 0x18;  // partial mode 12 CTC
    TCCR3A = 0x50;  // assert Mode 12 CTC
    TCCR3A = 0xA0;  // FOC setup
    TCCR3C = 0xC0;  // FOC strobe Clears them

    // Enable (PULLED-DOWN) output pins
    DDRB |= 0x60;
    DDRE |= 0x18;

    Serial.println(F("Stopped"));

    // Update Status
    OP_STATE = STOPPED;
}

void DUTY_Adjust(int up_down) // adjust PWM-duty on Timers T1,  T3 
{
    bool IS_FORWARD_DIR = isForwardDirection();

    // keeps btn Increase and Decrease operation Consistent! 
    if (IS_FORWARD_DIR) {
        up_down = -up_down; // Flip polarity if Dir == Fwd
    }

    OCR_Result = OCR1A + up_down;

    // constrain, OCRs should never be < boundary or > ICR - boundary
    if (OCR_Result < Duty_Numeric_Boundary) OCR_Result = Duty_Numeric_Boundary;
    if (OCR_Result > ICR1 - Duty_Numeric_Boundary) OCR_Result = (ICR1 - Duty_Numeric_Boundary);

    // constrain to MAX allowed duty (25% or slightly less) if Limit in force
    if (DUTY_CYCLE_RANGE != NO_LIMIT)
    {
        int MAX_DUTY_TO_USE = (DUTY_CYCLE_RANGE == MAX_25) ? DUTY_25_PerCent_VALUE : (DUTY_25_PerCent_VALUE - DEAD_TIME);

        if (IS_FORWARD_DIR) {
            if (OCR_Result < (ICR1 - MAX_DUTY_TO_USE)) {
                OCR_Result = (ICR1 - MAX_DUTY_TO_USE);
            }
        } else { // Reverse
            if (OCR_Result > MAX_DUTY_TO_USE) {
                OCR_Result = MAX_DUTY_TO_USE;
            }
        }
    }

    // Assert regular Duty Cycle to A-channels 
    OCR1A = OCR_Result;
    OCR3A = OCR_Result;
    // Assert mirrored Duty Cycle to B-channels 
    OCR1B = (ICR1 - OCR_Result);
    OCR3B = (ICR1 - OCR_Result);
}
