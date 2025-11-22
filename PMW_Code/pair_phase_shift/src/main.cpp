#include <Arduino.h>

// This code demonstrates how to generate two output signals
// with variable phase shift between them using an AVR Timer 

// The output shows up on Arduino pin 9, 10

// More AVR Timer Tricks at http://josh.com

void setup() {
  pinMode( 9 , OUTPUT );    // Arduino Pin  9 = OCR1A
  pinMode( 10 , OUTPUT );   // Arduino Pin 10 = OCR1B

  // Both outputs in toggle mode  
  TCCR1A = _BV( COM1A0 ) |_BV( COM1B0 );


  // CTC Waveform Generation Mode
  // TOP=ICR1  
  // Note clock is left off for now

  TCCR1B = _BV( WGM13) | _BV( WGM12);

  OCR1A = 0;    // First output is the base, it always toggles at 0


}

// prescaler of 1 will get us 8MHz - 488Hz
// User a higher prescaler for lower freqncies

#define PRESCALER 8
#define PRESCALER_BITS 0x01

#define CLK 16000000UL    // Default clock speed is 16MHz on Arduino Uno

// Output phase shifted wave forms on Arduino Pins 9 & 10
// freq = freqnecy in Hertz (  122 < freq <8000000 )
// shift = phase shift in degrees ( 0 <= shift < 180 )

// Do do shifts 180-360 degrees, you could invert the OCR1B by doing an extra toggle using FOC

/// Note phase shifts will be rounded down to the next neared possible value so the higher the frequency, the less phase shift resolution you get. At 8Mhz, you can only have 0 or 180 degrees because there are only 2 clock ticks per cycle.  

int setWaveforms( unsigned long freq , int shift ) {

  // This assumes prescaler = 1. For lower freqnecies, use a larger prescaler.

  unsigned long clocks_per_toggle = (CLK / freq) / 2;    // /2 becuase it takes 2 toggles to make a full wave

  ICR1 = clocks_per_toggle;

  unsigned long offset_clocks = (clocks_per_toggle * shift) / 180UL; // Do mult first to save precision

  OCR1B= offset_clocks;

  switch(PRESCALER) {
      case 1:    TCCR1B |= _BV(CS10); break;
      case 8:    TCCR1B |= _BV(CS11); break;
      case 64:   TCCR1B |= _BV(CS11) | _BV(CS10); break;
      case 256:  TCCR1B |= _BV(CS12); break; 
      case 1024: TCCR1B |= _BV(CS12) | _BV(CS10); break;
      default:   TCCR1B |= _BV(CS10); break; // fallback to no prescale
  }
}

// Demo by cycling through some phase shifts at 50Khz  
int angle = 0;
void loop() {
  angle = (angle + 1) % 360;


  // setWaveforms( 50000 , 0 );

  // delay(1000); 

  // setWaveforms( 50000 , 90 );

  // delay(1000); 

  setWaveforms(300, angle);
  Serial.println(angle);

  delay(1000); 


}