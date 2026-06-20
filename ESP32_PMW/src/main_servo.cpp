#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"

// ============================================================================
//  main_servo.cpp  --  serial-commanded actuator for closed-loop height hold.
//
//  The ESP32 is a THIN ACTUATOR here: a host PC runs the camera + control loop
//  (host/visual_servo/) and streams a target rotation FREQUENCY over USB serial.
//  Vertical lift scales with the rotating-field spin rate, so frequency is the
//  throttle; per-coil carrier amplitudes are held fixed in this 1-D height mode.
//
//  Protocol (ASCII, '\n'-terminated, non-blocking):
//    F<hz>          set target frequency       e.g.  F185.0
//    A<ch>,<duty>   set one carrier duty %     e.g.  A0,80      (reserved: attitude)
//    S              safe descent (ramp to F_LAND)
//    P              ping -> emit one telemetry line immediately
//  Telemetry out (CSV, ~50 Hz):
//    T,<millis>,<freqApplied>,<dutyA>,<dutyB>,<dutyC>,<dutyD>\n
//
//  Safety: commanded frequency is clamped to [F_MIN, F_MAX] and slew-limited.
//  If no F/S/P command arrives for CMD_TIMEOUT_MS, the controller ramps the
//  frequency down to F_LAND (controlled descent) -- it never hard-cuts, which
//  would drop the craft.
// ============================================================================

// CONFIGURATION
const int NUM_CHANNELS = 4;

// --- GOLDEN, UNTOUCHABLE pin map (see canonical-gpio-pin-map). Do NOT do
//     arithmetic on these constants; duty lives in the duty arrays. ---
const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_23;
const gpio_num_t C_PWM_PIN = GPIO_NUM_27;
const gpio_num_t D_PWM_PIN = GPIO_NUM_25;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_13;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_26;

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,     C_PWM_PIN,     D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN, C_CARRIER_PIN, D_CARRIER_PIN};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

const int PWM_FREQ = 15000;
const float carrier_duty = 100.0;  // 100 = constant HIGH = full on (direct carriers)
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {
    carrier_duty, carrier_duty, carrier_duty, carrier_duty};

// --- SAFETY / CONTROL LIMITS ----------------------------------------------
const float F_MIN  = 60.0f;    // min spin to stay aloft / catch synchronously
const float F_MAX  = 320.0f;   // hard ceiling on commanded frequency
const float F_LAND = 60.0f;    // descent target on loss-of-comms / 'S'
const float MAX_SLEW_HZ_S = 400.0f;  // max change rate of applied frequency

const unsigned long SPINUP_DURATION_MS = 6000;   // eased 0 -> F_MIN spin-up
const unsigned long CMD_TIMEOUT_MS     = 250;    // watchdog: no cmd -> descend
const unsigned long TELEM_PERIOD_MS    = 20;     // ~50 Hz telemetry

// --- INDICATOR LED ---
const int LED_PIN = 2;
bool led_state = false;

// globals
PhaseController* controller;
PhaseSequencer*  seq;

enum ServoState { SPINUP, COMMAND };
ServoState servo_state = SPINUP;

float target_freq  = F_MIN;   // where we want to be (set by F / S / watchdog)
float applied_freq = F_MIN;   // slew-limited value actually written to hardware

unsigned long last_cmd_ms   = 0;
unsigned long last_telem_ms = 0;
unsigned long last_loop_ms  = 0;

// serial line buffer
char line_buf[64];
uint8_t line_len = 0;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void sendTelemetry() {
  Serial.printf("T,%lu,%.2f,%.1f,%.1f,%.1f,%.1f\n",
                millis(), applied_freq,
                controller->getMeasuredDuty(0), controller->getMeasuredDuty(1),
                controller->getMeasuredDuty(2), controller->getMeasuredDuty(3));
  controller->resetMeasuredCounts();
}

// Parse one complete command line (already stripped of the terminator).
void handleCommand(const char* s) {
  if (s[0] == '\0') return;

  switch (s[0]) {
    case 'F': case 'f': {
      float hz = atof(s + 1);
      target_freq = clampf(hz, F_MIN, F_MAX);
      last_cmd_ms = millis();
      break;
    }
    case 'A': case 'a': {
      // A<ch>,<duty>  -- reserved for future attitude control
      const char* comma = strchr(s, ',');
      if (comma) {
        int ch = atoi(s + 1);
        float duty = atof(comma + 1);
        if (ch >= 0 && ch < NUM_CHANNELS) {
          controller->setCarrierDutyCycle(ch, clampf(duty, 0.0f, 100.0f));
        }
      }
      last_cmd_ms = millis();
      break;
    }
    case 'S': case 's': {
      target_freq = F_LAND;            // controlled descent
      last_cmd_ms = millis();
      break;
    }
    case 'P': case 'p': {
      sendTelemetry();                 // ping -> immediate telemetry
      last_cmd_ms = millis();
      break;
    }
    default:
      break;
  }
}

// Non-blocking: drain the UART, assemble lines, dispatch on '\n'.
void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (line_len > 0) {
        line_buf[line_len] = '\0';
        handleCommand(line_buf);
        line_len = 0;
      }
    } else if (line_len < sizeof(line_buf) - 1) {
      line_buf[line_len++] = c;
    } else {
      line_len = 0;  // overflow -> drop the garbled line
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new PhaseSequencer(controller);

  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

  // One-shot eased spin-up so the rotor catches the field synchronously.
  seq->addEaseRampTask(1.0f, F_MIN, SPINUP_DURATION_MS);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);
  seq->start();

  last_cmd_ms = last_telem_ms = last_loop_ms = millis();
}

void loop() {
  controller->run();
  unsigned long now = millis();

  if (servo_state == SPINUP) {
    seq->run();  // sequencer drives frequency during spin-up ONLY
    if (seq->isDone()) {
      // Hand off to command mode at the spin-up endpoint. From here we never
      // call seq->run() again (it would overwrite F commands every frame).
      target_freq = applied_freq = F_MIN;
      controller->setGlobalFrequency(applied_freq);
      last_cmd_ms = now;
      servo_state = COMMAND;
    }
    return;
  }

  // ----- COMMAND MODE -----
  pollSerial();

  // Watchdog: stale comms -> controlled descent (never a hard cut).
  if (now - last_cmd_ms > CMD_TIMEOUT_MS) {
    target_freq = F_LAND;
  }

  // Slew-limit the applied frequency toward the target.
  float dt = (now - last_loop_ms) * 0.001f;
  last_loop_ms = now;
  if (dt > 0.0f) {
    float max_step = MAX_SLEW_HZ_S * dt;
    float err = target_freq - applied_freq;
    if (err >  max_step) err =  max_step;
    if (err < -max_step) err = -max_step;
    applied_freq = clampf(applied_freq + err, F_MIN, F_MAX);
    controller->setGlobalFrequency(applied_freq);
  }

  // Telemetry heartbeat (also lets the host log measured duty).
  if (now - last_telem_ms >= TELEM_PERIOD_MS) {
    last_telem_ms = now;
    sendTelemetry();
    led_state = !led_state;
    digitalWrite(LED_PIN, led_state ? HIGH : LOW);
  }
}
