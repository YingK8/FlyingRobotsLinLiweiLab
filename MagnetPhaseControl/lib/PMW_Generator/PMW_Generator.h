#include <Arduino.h>

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

    uint8_t get_timer_from_pin(uint8_t pin);

    volatile uint16_t* get_OCR_register(uint8_t timer, uint8_t channel_idx);

    void setup_timer(uint8_t channel_idx);

    void start_timer(uint8_t timer_num);
    }
