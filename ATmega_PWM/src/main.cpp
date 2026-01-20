#include <Arduino.h>

/*
 * 6 Independent PWM Signals on Arduino Mega 2560
 * * Target Board: Arduino Mega 2560
 * * GLOBAL FREQUENCY CONFIGURATION:
 * Change the #define below to set the frequency for ALL 6 channels.
 * * WARNING: Timer 0 (Pin 13) controls millis() and delay(). 
 * Changing the global prescaler to anything other than 64 (Standard) 
 * will break time-keeping functions.
 */

// =========================================================================
// GLOBAL FREQUENCY SETTING
// Uncomment EXACTLY ONE of the following lines to set the global speed.
// =========================================================================
// #define PRESCALER_1      // ~62.5 kHz
// #define PRESCALER_8      // ~7.8 kHz
#define PRESCALER_64     // ~976 Hz   (Standard Arduino PWM / Safe for millis)
// #define PRESCALER_256    // ~244 Hz
// #define PRESCALER_1024   // ~61 Hz
// =========================================================================

// Automatic Bit Mapping based on selection above
#if defined(PRESCALER_1)
  #define PWM_CS_BITS     0x01  // No Prescaling
  #define PWM_CS_BITS_T2  0x01  // No Prescaling
#elif defined(PRESCALER_8)
  #define PWM_CS_BITS     0x02  // /8
  #define PWM_CS_BITS_T2  0x02  // /8
#elif defined(PRESCALER_64)
  #define PWM_CS_BITS     0x03  // /64
  #define PWM_CS_BITS_T2  0x04  // /64 (Note: Timer 2 uses different bit map)
#elif defined(PRESCALER_256)
  #define PWM_CS_BITS     0x04  // /256
  #define PWM_CS_BITS_T2  0x06  // /256
#elif defined(PRESCALER_1024)
  #define PWM_CS_BITS     0x05  // /1024
  #define PWM_CS_BITS_T2  0x07  // /1024
#else
  #error "Please uncomment exactly one PRESCALER option at the top of the file."
#endif

void setup() {
  // Set all used PWM pins to Output
  pinMode(13, OUTPUT); // Timer 0
  pinMode(11, OUTPUT); // Timer 1
  pinMode(10, OUTPUT); // Timer 2
  pinMode(5,  OUTPUT); // Timer 3
  pinMode(6,  OUTPUT); // Timer 4
  pinMode(46, OUTPUT); // Timer 5

  // ==========================================================
  // TIMER 0 (8-bit) - Pin 13
  // ==========================================================
  TCCR0A = 0; TCCR0B = 0;
  TCCR0A |= (1 << WGM00) | (1 << WGM01); // Fast PWM
  TCCR0B |= (1 << WGM02);
  TCCR0A |= (1 << COM0A1);               // Non-inverting
  TCCR0B |= PWM_CS_BITS;                 // Apply Global Frequency
  OCR0A = 128;

  // ==========================================================
  // TIMER 1 (16-bit) - Pin 11
  // ==========================================================
  TCCR1A = 0; TCCR1B = 0;
  TCCR1A |= (1 << WGM10);                // Fast PWM 8-bit
  TCCR1B |= (1 << WGM12);
  TCCR1A |= (1 << COM1A1);               // Non-inverting
  TCCR1B |= PWM_CS_BITS;                 // Apply Global Frequency
  OCR1A = 128;

  // ==========================================================
  // TIMER 2 (8-bit) - Pin 10
  // ==========================================================
  TCCR2A = 0; TCCR2B = 0;
  TCCR2A |= (1 << WGM20) | (1 << WGM21); // Fast PWM
  TCCR2A |= (1 << COM2A1);               // Non-inverting
  TCCR2B |= PWM_CS_BITS_T2;              // Apply Global Frequency (T2 mapping)
  OCR2A = 128;

  // ==========================================================
  // TIMER 3 (16-bit) - Pin 5
  // ==========================================================
  TCCR3A = 0; TCCR3B = 0;
  TCCR3A |= (1 << WGM30);                // Fast PWM 8-bit
  TCCR3B |= (1 << WGM32);
  TCCR3A |= (1 << COM3A1);               // Non-inverting
  TCCR3B |= PWM_CS_BITS;                 // Apply Global Frequency
  OCR3A = 128;

  // ==========================================================
  // TIMER 4 (16-bit) - Pin 6
  // ==========================================================
  TCCR4A = 0; TCCR4B = 0;
  TCCR4A |= (1 << WGM40);                // Fast PWM 8-bit
  TCCR4B |= (1 << WGM42);
  TCCR4A |= (1 << COM4A1);               // Non-inverting
  TCCR4B |= PWM_CS_BITS;                 // Apply Global Frequency
  OCR4A = 128;

  // ==========================================================
  // TIMER 5 (16-bit) - Pin 46
  // ==========================================================
  TCCR5A = 0; TCCR5B = 0;
  TCCR5A |= (1 << WGM50);                // Fast PWM 8-bit
  TCCR5B |= (1 << WGM52);
  TCCR5A |= (1 << COM5A1);               // Non-inverting
  TCCR5B |= PWM_CS_BITS;                 // Apply Global Frequency
  OCR5A = 128;
}

void loop() {
  // DEMONSTRATION:
  // Independently fading all 6 signals at different speeds to prove independence.
  // The Frequency is GLOBAL, but Duty Cycle (brightness) is INDEPENDENT.
  
  OCR0A = (millis() / 2) % 255;  // Timer 0
  OCR1A = (millis() / 4) % 255;  // Timer 1
  OCR2A = (millis() / 8) % 255;  // Timer 2
  OCR3A = (millis() / 16) % 255; // Timer 3
  OCR4A = (millis() / 32) % 255; // Timer 4
  OCR5A = (millis() / 64) % 255; // Timer 5

  delay(10);
}