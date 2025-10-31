#include <Arduino.h>
#include <phase_control.h>

// Create two channels
// PMWChannel(out_pin, fault_pin, current_pin, freq, duty, phase)
PMWChannel channel_A(9, A0, A1, 300.0, 0.5, 0.0);   // Phase 0 (Reference)
PMWChannel channel_B(10, A2, A3, 300.0, 0.5, 90.0); // Phase 90
PMWChannel channel_C(10, A2, A3, 300.0, 0.5, 0.0);

void setup() {
    Serial.begin(115200);
    
    // Start the channels
    channel_A.start();
    channel_B.start();
    channel_C.start();
}

void loop() {
    // 1. Get the current time ONCE
    unsigned long now = micros();

    // 2. Update all channels with the same timestamp
    channel_A.update(now);
    channel_B.update(now);
    channel_C.update(now);

    // 3. (Optional) You can change parameters at any time
    // Example: Sweep the phase of channel C
    channel_C.shift_phase(0.036); // add 1 deg per sec
    channel_C.increase_duty(0.001); // 1 sec per sec
}