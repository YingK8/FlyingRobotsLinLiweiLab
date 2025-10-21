#include <Arduino.h>
#include "PMW_setup.h"
/////////////////////////////////////////////////////////////////////////////
// Handy  Variables

int btn_DutyIncrease_Pin = A13; // Using ADC pins for push-buttons
int btn_DutyDecrease_Pin = A15;

int OCR_Result; // a calculation variable
int const Duty_Numeric_Boundary = 20; // Numeric LIMIT for adjustable duty range
/////////////////////////////////////////////////////////////////////////////

ISR(TIMER5_COMPA_vect)
{
TCCR3B |= PRESCALE; // DELAYED START Timer 3, 60 degrees
}

ISR(TIMER5_COMPB_vect)
{
TCCR4B |= PRESCALE; // DELAYED START Timer 4, 120 degrees
TIMSK5=0x0; // STOP Interrupts from Timer5
}

ISR(TIMER4_COMPB_vect)
// ENABLE WAVEFORM PINS ONCE TIMERS STABILIZE
{
pinMode(11, OUTPUT); // OC1a, 0 degrees phase shift
pinMode(12, OUTPUT); // OC1b, 180

pinMode(5, OUTPUT);  // OC3a, 60 degrees from Timer-3
pinMode(2, OUTPUT);  // OC3b, 240 degrees from Timer-3

pinMode(6, OUTPUT);  // OC4a, 120 degrees from Timer-4
pinMode(7, OUTPUT);  // OC4b, 300 degrees from Timer-4

TIMSK4=0x0; // STOP Interrupts from Timer4
}

void setup() // ********
{
// PUSH BUTTON PINS need Pulled-Up Inputs
pinMode(btn_DutyIncrease_Pin, INPUT_PULLUP); // Push-button on ADC pin
pinMode(btn_DutyDecrease_Pin, INPUT_PULLUP); // Push-button on ADC pin

pinMode(14, OUTPUT); // Marker Pulse pin

// Set Timer-5 freq SAME as other Timers!
// Using mode-14 FAST-PWM, single slope, up counter

TCCR5B = 0x18; // 0001 1000, Disable Timer
TCCR5A = 0x02; // 0101 0000

ICR5 = T5_ICAP;
TCNT5=0x0;
OCR5A = (int) (ICR5 * 0.166666); // 60 degrees at Freq
OCR5B = (int) (ICR5 * 0.333333); // 120 degrees at Freq
TCNT5=0x0;

TIMSK5 = 0x06; // Enable CompA and CompB Interrupts   <<**********************

/////////////////////////////////////////////////////////////////////////////

// Timer-1 16-bit, Mode-10 Phase, Top=ICR
// Clock is 16 MHz

TCCR1B = 0x10; // 0001 1000, Disable Timer
TCCR1A = 0xB2; // 1000 0010

ICR1 = ICAP_NDiv;
OCR1A = (int) (ICR1 * 0.166666);
OCR1B = (int) (ICR1 * 0.833333);
TCNT1=0x0;

/////////////////////////////////////////////////////////////////////////////

// Timer-3 16-bit, Mode-10 Phase, Top=ICR
// Clock is 16 MHz

TCCR3B = 0x10; // 0001 1000, Disable Timer
TCCR3A = 0xB2; // 1010 0010

ICR3 = ICAP_NDiv;
OCR3A = (int) (ICR3 * 0.166666);
OCR3B = (int) (ICR3 * 0.833333);
TCNT3=0x0;

/////////////////////////////////////////////////////////////////////////////

// Timer-4 16-bit, Mode-10 Phase, Top=ICR
// Clock is 16 MHz

TCCR4B = 0x10; // 0001 1000, Disable Timer
TCCR4A = 0xB2; // 1010 0010

ICR4 = ICAP_NDiv;
OCR4A = (int) (ICR4 * 0.166666);
OCR4B = (int) (ICR4 * 0.833333);
TCNT4=0x0;

TIMSK4 = 0x04; // Interrupt to enable Output Pins

digitalWrite(14, HIGH); // MARKER PULSE  <<-----------------------------
digitalWrite(14, LOW);

//  UPDATE 9-21-2022 by stockvu
//  Synchronize Start of Timers 1 and 5

GTCCR=0x81; // Timer Sync Control -- STOP ALL TIMERS, Clear preScalers

TCCR5B |= PRESCALE; // Prescale=X, ENABLE Timer 5 clock
TCCR1B |= PRESCALE; // Prescale=X, ENABLE Timer 1 clock

GTCCR=0x0; // START ALL TIMERS

} // end setup

void PWM_adjust(int up_down) // Adjust PWM-duty on Timers Together T1,  T3 and T4
{
// calculate
OCR_Result = OCR1A;
OCR_Result += up_down;

// constrain (old-school style), OCRs can never be < 1 or >= ICR ! 
if (OCR_Result < Duty_Numeric_Boundary) OCR_Result = Duty_Numeric_Boundary;
if (OCR_Result > ICR1-Duty_Numeric_Boundary) OCR_Result = ICR1-Duty_Numeric_Boundary;

// Assert regular Duty Cycle to A-channels 
OCR1A = (int) OCR_Result;
OCR3A = (int) OCR_Result; 
OCR4A = (int) OCR_Result;

// Assert mirrored Duty Cycle to B-channels
OCR1B = (int) ICR1-OCR_Result;
OCR3B = (int) ICR1-OCR_Result; 
OCR4B = (int) ICR1-OCR_Result;
}

void loop() // ********
{
int state_DutyIncrease;
int state_DutyDecrease;

  // read state of pushbutton(s):
  state_DutyIncrease = digitalRead(btn_DutyIncrease_Pin);
  state_DutyDecrease = digitalRead(btn_DutyDecrease_Pin);

  // check if pushbutton pressed.
  if (state_DutyIncrease == LOW)
  {
  PWM_adjust(+ADJ_CONST);
  delay(WAIT_CONST); // PACE loop speed a bit
  }
 
  // check if pushbutton pressed.
  if (state_DutyDecrease == LOW)
  {
  PWM_adjust(-ADJ_CONST);
  delay(WAIT_CONST); // PACE loop speed a bit
  }
}// end loop()