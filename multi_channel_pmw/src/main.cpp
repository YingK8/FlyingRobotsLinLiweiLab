#include <Arduino.h>
#include <EnableInterrupt.h> // Pin-Change Interrupts
#include "PMW.h"

enum OPERATION_STATES_TYPE OP_STATE;
enum OPERATION_DIRECTION_TYPE OP_DIRECTION;
enum REQUEST_STATES_TYPE CURRENT_REQ_STATE;
enum DUTY_START_TYPE START_UP_DUTY;
enum DUTY_RANGE_TYPE DUTY_CYCLE_RANGE;
enum DUTY_RAMPDOWN_TYPE DUTY_DOWN_RAMP;

int OCR_Result;
int StartUp_Duty_Cycle_Value;

void setup()
{
    Serial.begin(115200);

    // User Configuration
    //---------------------------------
    // SET -to- NO_LIMIT -or- MAX_25 -or- MAX_25_MINUS_DEADTIME
    DUTY_CYCLE_RANGE = NO_LIMIT;

    // SET -to- DUTY_02_PERCENT -or- DUTY_10_PERCENT -or- DUTY_25_PERCENT
    START_UP_DUTY = DUTY_25_PERCENT;

    // SET -to- RAMP_ENABLED -or- RAMP_DISABLED
    DUTY_DOWN_RAMP = RAMP_DISABLED;

    //  SET -to- FIXED -or- SWITCHED DIRECTION
    // Ignores Switch ==  FIXED_FORWARD_DIR,   FIXED_REVERSE_DIR
    // Checks Switch ==  SWT_FORWARD_DIR,    SWT_REVERSE_DIR
    OP_DIRECTION = SWT_FORWARD_DIR;
    //---------------------------------
    

    // Determine which Duty Value to use starting up....
    switch (START_UP_DUTY)
    {
        // 25% used to test for 4-equal length pulses ACTIVE 90 degrees 
        case DUTY_25_PERCENT:
            if (DUTY_CYCLE_RANGE == MAX_25_MINUS_DEADTIME) {
                StartUp_Duty_Cycle_Value = (DUTY_25_PerCent_VALUE - DEAD_TIME); // < 25%
            } else {
                StartUp_Duty_Cycle_Value = DUTY_25_PerCent_VALUE; // 25% near exact 
            }
            break;

        case DUTY_10_PERCENT:
            StartUp_Duty_Cycle_Value = (int)(ICAP_NDiv * 0.100);  // ~10% duty 
            break;

        case DUTY_02_PERCENT:
            StartUp_Duty_Cycle_Value = (int)(ICAP_NDiv * 0.020);  // ~2% duty 
            break;
    }

    // PUSH BUTTONS require Pulled-Up Inputs
    // pinMode(btn_DutyIncrease_PinA15, INPUT_PULLUP);
    // pinMode(btn_DutyDecrease_PinA13, INPUT_PULLUP);
    // pinMode(btn_START_PinA11, INPUT_PULLUP);
    // pinMode(btn_STOP_PinA9, INPUT_PULLUP);

    // SLIDE SWITCH requires Pulled-Up Input
    // pinMode(swt_SEQUENCE_DIRECTION_PinA7, INPUT_PULLUP); 

    // Assign Interrupts for START and STOP buttons 
    // enableInterrupt(A11, ISR_START, FALLING);
    // enableInterrupt(A9, ISR_STOP, FALLING);

    // Marker Pulse pin-D14, Logic Analyzer Trigger
    pinMode(14, OUTPUT);

    // init states
    CURRENT_REQ_STATE = NO_REQUEST_PENDING;
    OP_STATE = STOPPED;

    Serial.println("PSPWM " + OP_Frequency + " Initialized");
} // end setup


void loop()
{
    // Check for ISR-generated SYSTEM REQUESTS
    if (CURRENT_REQ_STATE != NO_REQUEST_PENDING)
    {
        switch (CURRENT_REQ_STATE)
        {
            case (REQ_START_FWD): // Forward Direction Start
                CURRENT_REQ_STATE = NO_REQUEST_PENDING; // Clear Request
                START_PSPWM(true); // CALL SUB
                break;

            case (REQ_START_REV): // Reverse Direction Start 
                CURRENT_REQ_STATE = NO_REQUEST_PENDING; // Clear Request
                START_PSPWM(false); // CALL SUB
                break;

            case (REQ_STOP): // Halt System
                CURRENT_REQ_STATE = NO_REQUEST_PENDING; // Clear Request
                STOP_PSPWM(); // CALL SUB
                break;
            
            case (NO_REQUEST_PENDING):
                // Do nothing
                break;
        }
    }

    // Check for manual Duty-Cycle Adjust 
    if (OP_STATE == ACTIVE)
    {
		DUTY_Adjust(+ADJ_CONST); // Increase duty
        delay(WAIT_CONST);
		DUTY_Adjust(+ADJ_CONST); // Increase duty
        delay(WAIT_CONST);
		DUTY_Adjust(+ADJ_CONST); // Increase duty
        delay(WAIT_CONST);

		DUTY_Adjust(-ADJ_CONST); // Decrease duty
        delay(WAIT_CONST);
		DUTY_Adjust(-ADJ_CONST); // Decrease duty
        delay(WAIT_CONST);
		DUTY_Adjust(-ADJ_CONST); // Decrease duty
        delay(WAIT_CONST);
    }
}// end loop()