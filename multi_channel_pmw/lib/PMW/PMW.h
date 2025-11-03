#ifndef PMW_H
#define PMW_H

#include <Arduino.h>

// --- Select ONE frequency (This is the only place you should change it) ---
//#define TIMER_FREQ_50Hz 
//#define TIMER_FREQ_100Hz 
#define TIMER_FREQ_200Hz 
//#define TIMER_FREQ_400Hz 
//#define TIMER_FREQ_500Hz 

// --- CONSTANT DECLARATIONS (using 'extern const') ---
extern const int Duty_Numeric_Boundary;

#if defined TIMER_FREQ_50Hz
extern const int ICAP_NDiv;
extern const int DUTY_25_PerCent_VALUE;
extern const int DEAD_TIME;
extern const int PHASE_OFFSET;
extern const int PRESCALE;
extern const int ADJ_CONST;
extern const int WAIT_CONST;
extern const String OP_Frequency;
#elif defined TIMER_FREQ_100Hz
extern const int ICAP_NDiv;
extern const int DUTY_25_PerCent_VALUE;
extern const int DEAD_TIME;
extern const int PHASE_OFFSET;
extern const int PRESCALE;
extern const int ADJ_CONST;
extern const int WAIT_CONST;
extern const String OP_Frequency;
#elif defined TIMER_FREQ_200Hz
extern const int ICAP_NDiv;
extern const int DUTY_25_PerCent_VALUE;
extern const int DEAD_TIME;
extern const int PHASE_OFFSET;
extern const int PRESCALE;
extern const int ADJ_CONST;
extern const int WAIT_CONST;
extern const String OP_Frequency;
#elif defined TIMER_FREQ_400Hz
extern const int ICAP_NDiv;
extern const int DUTY_25_PerCent_VALUE;
extern const int DEAD_TIME;
extern const int PHASE_OFFSET;
extern const int PRESCALE;
extern const int ADJ_CONST;
extern const int WAIT_CONST;
extern const String OP_Frequency;
#elif defined TIMER_FREQ_500Hz
extern const int ICAP_NDiv;
extern const int DUTY_25_PerCent_VALUE;
extern const int DEAD_TIME;
extern const int PHASE_OFFSET;
extern const int PRESCALE;
extern const int ADJ_CONST;
extern const int WAIT_CONST;
extern const String OP_Frequency;
#endif

// --- ENUMERATION TYPE DEFINITIONS ---
// This defines the *type*, not the variable
enum OPERATION_STATES_TYPE { ACTIVE, STOPPED };
enum OPERATION_DIRECTION_TYPE { SWT_FORWARD_DIR, SWT_REVERSE_DIR, FIXED_FORWARD_DIR, FIXED_REVERSE_DIR };
enum REQUEST_STATES_TYPE { REQ_START_FWD, REQ_START_REV, REQ_STOP, NO_REQUEST_PENDING };
enum DUTY_START_TYPE { DUTY_25_PERCENT, DUTY_10_PERCENT, DUTY_02_PERCENT };
enum DUTY_RANGE_TYPE { NO_LIMIT, MAX_25, MAX_25_MINUS_DEADTIME };
enum DUTY_RAMPDOWN_TYPE { RAMP_ENABLED, RAMP_DISABLED };

// --- CONSTS and PINS (using 'extern const') ---
extern const int ZERO_PHASE_REFERENCE;
extern const int btn_DutyIncrease_PinA15;
extern const int btn_DutyDecrease_PinA13;
extern const int btn_START_PinA11;
extern const int btn_STOP_PinA9;
extern const int swt_SEQUENCE_DIRECTION_PinA7;

// --- GLOBAL VARIABLE DECLARATIONS (using 'extern') ---
// This *declares* the variables. They are *defined* in main.cpp
extern enum OPERATION_STATES_TYPE OP_STATE;
extern enum OPERATION_DIRECTION_TYPE OP_DIRECTION;
extern enum REQUEST_STATES_TYPE CURRENT_REQ_STATE;
extern enum DUTY_START_TYPE START_UP_DUTY;
extern enum DUTY_RANGE_TYPE DUTY_CYCLE_RANGE;
extern enum DUTY_RAMPDOWN_TYPE DUTY_DOWN_RAMP;

extern int OCR_Result;
extern int StartUp_Duty_Cycle_Value;

// --- FUNCTION PROTOTYPES ---
bool isForwardDirection();
void ISR_START();
void ISR_STOP();
void START_PSPWM(bool isForward);
void STOP_PSPWM();
void DUTY_Adjust(int up_down);
float Get_DutyValue();
void DUTY_Ramp_Down();

#endif // PMW_H