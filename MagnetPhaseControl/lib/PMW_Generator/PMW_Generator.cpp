#include <Arduino.h>
#include "PMW_Generator.h"

class SimplePWM {
private:
    struct PWMChannel {
        uint8_t pin;
        uint8_t timer;
        volatile uint16_t *OCR;
        volatile uint16_t *ICR;
        uint16_t phase_offset;
        float duty_cycle;
    };

    PWMChannel channels[4];  // Fixed array instead of dynamic allocation
    uint8_t channel_count;
    float frequency;
    uint16_t timer_period;

    // Timer configuration helper
    struct TimerConfig {
        volatile uint16_t *ICR;
        volatile uint16_t *TCNT;
        volatile uint8_t *TCCRA;
        volatile uint8_t *TCCRB;
    };

    const TimerConfig timer1 = {&ICR1, &TCNT1, &TCCR1A, &TCCR1B};
    const TimerConfig timer3 = {&ICR3, &TCNT3, &TCCR3A, &TCCR3B};
    const TimerConfig timer4 = {&ICR4, &TCNT4, &TCCR4A, &TCCR4B};
    const TimerConfig timer5 = {&ICR5, &TCNT5, &TCCR5A, &TCCR5B};

    const TimerConfig* get_timer_config(uint8_t timer_num) {
        switch(timer_num) {
            case 1: return &timer1;
            case 3: return &timer3;
            case 4: return &timer4;
            case 5: return &timer5;
            default: return nullptr;
        }
    }

    uint8_t get_timer_from_pin(uint8_t pin) {
        // Common PWM pins on ATmega2560
        switch(pin) {
            case 11: case 12: return 1;  // Timer1
            case 5: case 2: case 3: return 3;  // Timer3  
            case 6: case 7: case 8: return 4;  // Timer4
            case 44: case 45: case 46: return 5; // Timer5
            default: return 0;
        }
    }

    volatile uint16_t* get_OCR_register(uint8_t timer, uint8_t channel_idx) {
        switch (timer) {
            case 1: return (channel_idx == 0) ? &OCR1A : &OCR1B;
            case 3: return (channel_idx == 0) ? &OCR3A : &OCR3B;
            case 4: return (channel_idx == 0) ? &OCR4A : &OCR4B;
            case 5: return (channel_idx == 0) ? &OCR5A : &OCR5B;
            default: return nullptr;
        }
    }

    void setup_timer(uint8_t channel_idx) {
        PWMChannel& ch = channels[channel_idx];
        const TimerConfig* timer = get_timer_config(ch.timer);
        if (!timer) return;

        // Stop timer and reset
        *timer->TCCRA = 0;
        *timer->TCCRB = 0;
        
        // Set frequency and duty cycle
        *timer->ICR = timer_period;
        *ch.OCR = (uint16_t)(timer_period * ch.duty_cycle / 100.0f);
        
        // Configure Fast PWM mode
        *timer->TCCRA = _BV(COM1A1) | _BV(WGM11);
        *timer->TCCRB = _BV(WGM13) | _BV(WGM12);
        
        // Set phase offset
        *timer->TCNT = ch.phase_offset;
        
        pinMode(ch.pin, OUTPUT);
    }

    void start_timer(uint8_t timer_num) {
        const TimerConfig* timer = get_timer_config(timer_num);
        if (!timer) return;
        
        // Calculate and set prescaler
        const uint32_t prescalers[] = {1, 8, 64, 256, 1024};
        const uint8_t prescaler_bits[] = {_BV(CS10), _BV(CS11), _BV(CS11)|_BV(CS10), _BV(CS12), _BV(CS12)|_BV(CS10)};
        
        for (uint8_t i = 0; i < 5; i++) {
            uint32_t test_period = (F_CPU / (prescalers[i] * frequency)) - 1;
            if (test_period <= 65535) {
                timer_period = test_period;
                *timer->TCCRB |= prescaler_bits[i];
                return;
            }
        }
        
        // Fallback to prescaler 1024
        timer_period = (F_CPU / (1024 * frequency)) - 1;
        *timer->TCCRB |= _BV(CS12) | _BV(CS10);
    }
