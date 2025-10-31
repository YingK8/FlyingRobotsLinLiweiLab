#include <Arduino.h>
#include "phase_control.h"

PMWChannel::PMWChannel(int out_pin, int fault_pin, int current_pin, float frequency, float duty, float phase)
    : _out_pin(out_pin), 
      _fault_pin(fault_pin), 
      _current_pin(current_pin),
      _frequency(frequency), 
      _duty(duty), 
      _phase(phase),
      _is_high(false),
      _enabled(false) 
{
    // Configure pins
    pinMode(_out_pin, OUTPUT);
    digitalWrite(_out_pin, LOW); // Start in a known LOW state
    
    // Inputs (example setup)
    pinMode(_fault_pin, INPUT_PULLUP); // Assuming active-low fault
    pinMode(_current_pin, INPUT);     // Analog input
    
    // Calculate initial timing values
    _recalculate_timing();
}

void PMWChannel::_recalculate_timing() {
    // 1. Calculate Period in microseconds
    _period_us = (unsigned long)(1000000.0 / _frequency);
    
    // 2. Calculate Pulse Width (ON time) in microseconds
    _pulse_width_us = (unsigned long)(_period_us * fmod(fmod(_duty, 1.0) + 1.0, 1.0));
    
    // 3. Calculate Phase Offset (delay) in microseconds
    _phase_offset_us = (unsigned long)(_period_us * fmod(fmod(_phase, 360.0) + 360.0, 360.0));
}

bool PMWChannel::start() {
    _enabled = true;
    return true;
}

bool PMWChannel::stop() {
    _enabled = false;
    digitalWrite(_out_pin, LOW);
    _is_high = false;
    return true;
}

void PMWChannel::change_phase(float phase) {
    _phase = phase;
    _recalculate_timing();
}

void PMWChannel::change_duty(float duty) {
    _duty = duty;
    _recalculate_timing();
}

void PMWChannel::change_frequency(float frequency) {
    _frequency = frequency;
    _recalculate_timing();
}

void PMWChannel::shift_phase(float d_phase) {
    _phase += d_phase;
    _recalculate_timing();
}

void PMWChannel::increase_frequency(float d_frequency) {
    _frequency += d_frequency;
    _recalculate_timing();
}

void PMWChannel::increase_duty(float duty) {
    _duty += duty;
    _recalculate_timing();
}

void PMWChannel::update(unsigned long current_time_us) {
    if (!_enabled) {
        return; // Do nothing if stopped
    }

    // --- This is the Modular Arithmetic ---
    //
    // We calculate the time *within* this channel's specific cycle,
    // which is shifted by its phase offset.
    //
    // (current_time + period - offset) % period
    //
    // We add _period_us before subtracting the offset to ensure
    // the value is positive, preventing underflow with unsigned longs.
    
    unsigned long time_in_cycle = (current_time_us + _period_us - _phase_offset_us) % _period_us;

    // Check if we should be in the "HIGH" part of the cycle
    bool should_be_high = (time_in_cycle < _pulse_width_us);

    // Only write to the pin if a change is needed
    if (should_be_high && !_is_high) {
        digitalWrite(_out_pin, HIGH);
        _is_high = true;
    } else if (!should_be_high && _is_high) {
        digitalWrite(_out_pin, LOW);
        _is_high = false;
    }
}