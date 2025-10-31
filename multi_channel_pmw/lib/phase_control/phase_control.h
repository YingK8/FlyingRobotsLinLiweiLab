#ifndef PHASE_CONTROL_H
#define PHASE_CONTROL_H

#include <Arduino.h>

class PMWChannel {
private:
    int _out_pin;
    int _fault_pin;
    int _current_pin;

    float _frequency;
    float _duty;
    float _phase; // Phase in degrees (0-360)
    unsigned long _period_us;
    unsigned long _pulse_width_us; // The "on" time
    unsigned long _phase_offset_us; // The delay
    bool _is_high;
    bool _enabled;

    void _recalculate_timing();

public:
    PMWChannel(int out_pin, int fault_pin, int current_pin, float frequency = 300.0, float duty = 0.5, float phase = 0.0);
    bool stop();
    bool start();
    void change_phase(float phase);
    void change_frequency(float frequency);
    void change_duty(float duty);
    void shift_phase(float d_phase);
    void increase_frequency(float d_frequency);
    void increase_duty(float d_duty);
    void update(unsigned long current_time_us);
};

#endif // PHASE_CONTROL_H