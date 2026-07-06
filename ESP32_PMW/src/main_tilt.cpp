#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include <FS.h>
#include <SPIFFS.h>

// #include <gpio_viewer.h> // Must me the first include in your project
// GPIOViewer gpio_viewer;

// CONFIGURATION
#include "constants.h"

// TRIMS ARE DIRECTION-SPECIFIC: coupled power transfer ~ sin(dphi) reverses
// with rotation, so each phase map needs its own calibration. Keep each
// INITIAL_PHASES line paired with its CARRIER_TRIM line (both from env:comp_test
// scope-in-the-loop calibration, 2026-07-04, normalized so the strongest
// channel caps at 100%).
// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
// const float CARRIER_TRIM[NUM_CHANNELS] = {63.0, 100.0, 73.8, 63.7}; // % CCW (comp_ccw_iter2: 1.97->1.05)
// rotation is clockwise: D -> B -> C -> A  (ACTIVE for tilt-direction test)
// const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float CARRIER_TRIM[NUM_CHANNELS] = {100.0, 64.8, 59.4, 75.0}; // % CW (comp_ff_iter2: 1.88->1.07)
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_8;



const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {
    CARRIER_TRIM[0], CARRIER_TRIM[1], CARRIER_TRIM[2], CARRIER_TRIM[3]};

const float start_freq = 1.0f;
const float end_freq = 210.0f;
const unsigned long ramp_duration_ms = 30000;

// FREQUENCY-SCHEDULED TRIM: the coupling the trim cancels grows ~linearly with
// drive frequency (coupled EMF = w*M*I, zero at DC), so a static trim is only
// correct at the 210 Hz calibration point and IS the field imbalance everywhere
// below it (measured: per-coil ramp currents follow raw duty at low f, AC/BD
// supply ratio 0.92 at 50-100 Hz). Blend the correction in proportionally:
// equal duties at f=0 morphing into the calibrated pattern at end_freq, at
// constant mean drive. TRIM_MEAN is CARRIER_TRIM re-normalized to mean 1.0;
// TRIM_BASE_PCT is the equal-duty starting level, sized so the strongest
// channel lands exactly on 100% when the blend completes.
const float TRIM_BASE_PCT = (CARRIER_TRIM[0] + CARRIER_TRIM[1] +
                             CARRIER_TRIM[2] + CARRIER_TRIM[3]) / 4.0f;

// Scheduled carrier duty (%) for channel i at drive frequency f: all channels
// equal at TRIM_BASE_PCT when f~0, morphing linearly into CARRIER_TRIM[] at
// end_freq (the strongest channel reaches 100% there since the trim set is
// max-normalized). Mean duty stays TRIM_BASE_PCT at every frequency.
float scheduledDutyPct(int i, float f) {
  float blend = f / end_freq;
  if (blend < 0.0f) blend = 0.0f;
  if (blend > 1.0f) blend = 1.0f;
  return TRIM_BASE_PCT + (CARRIER_TRIM[i] - TRIM_BASE_PCT) * blend;
}

// --- INDICATOR LED CONFIGURATION ---
bool led_state = false; 

// declare as global POINTERS (do not initialize yet)
PhaseController* controller;
PhaseSequencer* seq;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);
  SPIFFS.begin(true); 

  Serial.println("running main tilt");

  // gpio_viewer.connectToWifi("Your SSID network", "Your WiFi Password");
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 
  
  // instantiate the objects dynamically NOW that the OS has booted
  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new PhaseSequencer(controller);

  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  // Arm de-energized: hold all carriers at 0 until the PC sends 'start', so the
  // scope can be armed before any current flows.
  for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, 0.0f);

  seq->addRampTask(start_freq, end_freq, ramp_duration_ms, TaskType::PWM_FREQ, TaskMode::EASE);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);
  // Do NOT seq->start() here — the serial 'start' command launches it (see
  // handleSerialCommands) so the recording can be coordinated from the PC.
  Serial.println("armed: b=begin  s=stop(e-stop)  e=end(ramp down)");
}

// --- STATE MACHINE / COMMAND MODEL ---
// IDLE     : armed, coils de-energized, waiting for 'begin' (b) so the scope can
//            be armed before any current flows.
// RAMP_UP/HOLD/TRANSITION : the running sweep.
// DONE     : sweep finished naturally (B & C at 0, A & D still at trimmed carrier).
// ENDING   : graceful ramp-down of all coils to 0 (from 'end' e).
// STOPPED  : latched off — all coils at 0, held down until the next 'begin'.
//            Reached by 'stop' (s, immediate e-stop) or after ENDING completes.
enum SweepState { IDLE, RAMP_UP, HOLD, TRANSITION, DONE, ENDING, STOPPED };
SweepState sweep_state = IDLE;

float duty_start = 100.0f;   // duty at the beginning of the current transition
float duty_target = 100.0f;  // duty at the end of the current transition
const float d_duty_step = 10.0f;

const unsigned long hold_time_ms = 2500;        // hold at each duty level
const unsigned long transition_time_ms = 500;   // cubic ramp between levels (3s total per step)
unsigned long state_start_time = 0;

// Monotonic ramp-down (ENDING): the library subtracts ramp_step_pct from each
// coil's live duty every ramp_tick_ms, so output can only decrease. stop=fast,
// end=graceful; steps sized so 100%->0 takes ~stop_ms / ~end_ms respectively.
const unsigned long ramp_tick_ms = 20;
const float stop_step_pct = 8.0f;   // ~250 ms from full
const float end_step_pct = 2.0f;    // ~1 s from full
float ramp_step_pct = end_step_pct;
unsigned long last_ramp_ms = 0;

// FIELD TRIM: live per-coil factor on top of the (current-balance) coupling
// trims, adjusted over serial (a+/a-/b+/b-/... in 2% steps) while watching the
// disk. Rationale: CS-based trims equalize supply CURRENT, but field-per-amp
// differs per coil (mounting height/tilt/geometry) — the disk is the only
// sensor that sees that, and it shows B/C effectively stronger (tilt toward
// B/C, fixed in lab frame under rotation reversal). Null the tilt by lowering
// b/c (or raising a/d where headroom allows), then bake the values in here.
float FIELD_TRIM[NUM_CHANNELS] = {1.0f, 1.0f, 1.0f, 1.0f};
const float FIELD_TRIM_STEP = 0.02f;
const float FIELD_TRIM_MIN = 0.50f, FIELD_TRIM_MAX = 1.50f;
float last_sweep_pct = 100.0f;   // current sweep level, for live re-apply

static inline float dutyClamp(float d) {
  return d < 0.0f ? 0.0f : (d > 100.0f ? 100.0f : d);
}

// only channels 1(B) and 2(C) sweep down; 0(A) and 3(D) stay at their trim.
// `pct` is the sweep level (0-100%); it scales each channel's trimmed base so the
// per-channel current balance is preserved throughout the sweep.
void setSweepCarrierDuty(float pct) {
  last_sweep_pct = pct;
  controller->setCarrierDutyCycle(
      1, dutyClamp(pct * CARRIER_TRIM[1] / 100.0f * FIELD_TRIM[1]));
  controller->setCarrierDutyCycle(
      2, dutyClamp(pct * CARRIER_TRIM[2] / 100.0f * FIELD_TRIM[2]));
}

// begin (b): launch/relaunch the compiled sweep. Valid from IDLE, DONE, or
// STOPPED — re-running just re-runs the same compiled trajectory.
void beginExperiment() {
  duty_start = duty_target = 100.0f;
  seq->start();  // start() sets carriers to NAN (no-change), so energize below.
  for (int i = 0; i < NUM_CHANNELS; i++)   // equal duties at the ramp's start
    controller->setCarrierDutyCycle(
        i, dutyClamp(scheduledDutyPct(i, start_freq) * FIELD_TRIM[i]));
  state_start_time = millis();
  sweep_state = RAMP_UP;
  Serial.println("begin: started");
}

// stop (s) and end (e) both enter ENDING — a monotonic library ramp-down that
// can only decrease (rampDownStep subtracts from the live duty). stop just uses
// a bigger step so it falls faster. Safe from any state; from an already-down
// state the ramp completes on the first tick.
void beginRampDown(float stepPct, const char *msg) {
  ramp_step_pct = stepPct;
  last_ramp_ms = millis();
  sweep_state = ENDING;
  Serial.println(msg);
}

void printFieldTrim() {
  Serial.printf("field trim: a=%.2f b=%.2f c=%.2f d=%.2f\n", FIELD_TRIM[0],
                FIELD_TRIM[1], FIELD_TRIM[2], FIELD_TRIM[3]);
}

// Adjust one coil's live field trim and re-apply duties in the current state
// (RAMP_UP refreshes on its own 50 ms tick; the static channels 0/3 and the
// sweep pair need an explicit push).
void nudgeFieldTrim(int i, float delta) {
  FIELD_TRIM[i] += delta;
  if (FIELD_TRIM[i] < FIELD_TRIM_MIN) FIELD_TRIM[i] = FIELD_TRIM_MIN;
  if (FIELD_TRIM[i] > FIELD_TRIM_MAX) FIELD_TRIM[i] = FIELD_TRIM_MAX;
  if (sweep_state == HOLD || sweep_state == TRANSITION || sweep_state == DONE) {
    controller->setCarrierDutyCycle(
        0, dutyClamp(INITIAL_CARRIER_DUTY_CYCLES[0] * FIELD_TRIM[0]));
    controller->setCarrierDutyCycle(
        3, dutyClamp(INITIAL_CARRIER_DUTY_CYCLES[3] * FIELD_TRIM[3]));
    setSweepCarrierDuty(last_sweep_pct);
  }
  printFieldTrim();
}

void dispatchCommand(const String &cmd) {
  if (cmd == "b" || cmd == "begin" || cmd == "start" || cmd == "g") {
    if (sweep_state == IDLE || sweep_state == DONE || sweep_state == STOPPED)
      beginExperiment();
    else
      Serial.println("ignored: already running (stop first)");
  } else if (cmd == "s" || cmd == "stop") {
    beginRampDown(stop_step_pct, "stop: ramping down (fast)");
  } else if (cmd == "e" || cmd == "end" || cmd == "down") {
    beginRampDown(end_step_pct, "end: ramping down");
  } else if (cmd.length() == 2 && cmd[0] >= 'a' && cmd[0] <= 'd' &&
             (cmd[1] == '+' || cmd[1] == '-')) {
    // per-coil field-trim nudge: a+/a-/b+/b-/c+/c-/d+/d- (2% per step)
    nudgeFieldTrim(cmd[0] - 'a', cmd[1] == '+' ? FIELD_TRIM_STEP
                                               : -FIELD_TRIM_STEP);
  } else if (cmd == "-" || cmd == "_") {
    // A and D together: weaken both 2% (BC-decrease didn't move the tilt)
    nudgeFieldTrim(0, -FIELD_TRIM_STEP);
    nudgeFieldTrim(3, -FIELD_TRIM_STEP);
  } else if (cmd == "+" || cmd == "=") {
    nudgeFieldTrim(0, FIELD_TRIM_STEP);
    nudgeFieldTrim(3, FIELD_TRIM_STEP);
  } else if (cmd == "t") {
    printFieldTrim();
  } else {
    Serial.printf("unknown cmd '%s' (b=begin s=stop e=end -/+=AD field trim "
                  "a+/a-..d+/d-=per-coil t=show)\n", cmd.c_str());
  }
}

// Non-blocking serial reader. Accumulates bytes into a static buffer ACROSS
// loop() iterations and dispatches on end-of-line, so it does not depend on a
// whole command arriving inside one Serial.setTimeout window — robust to USB
// packet fragmentation and char-by-char typing (CR, LF, or CRLF all end a line).
void handleSerialCommands() {
  static String buf;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      buf.trim();
      buf.toLowerCase();
      if (buf.length()) dispatchCommand(buf);
      buf = "";
    } else {
      buf += c;
      if (buf.length() > 32) buf = "";   // overflow guard against garbage
    }
  }
}

void loop() {
  handleSerialCommands();
  controller->run();
  // Only drive the compiled trajectory during the active sweep. In IDLE/ENDING/
  // STOPPED the sequencer must not touch the coils, so a stop stays down.
  if (sweep_state == RAMP_UP || sweep_state == HOLD || sweep_state == TRANSITION)
    seq->run();

  unsigned long current_millis = millis();

  // NOTE: the indicator LED is intentionally NOT blinked here. It toggles once
  // per carrier-duty step (see the HOLD case), so its state acts as a visual
  // marker aligned with each duty change on the scope trace.

  // TEMP DEBUG: 1Hz status print. ledc readback = duty the hardware actually
  // has latched (0..4095 at 12-bit); 0 means LEDC is not driving that pin.
  static unsigned long last_dbg_ms = 0;
  if (current_millis - last_dbg_ms >= 1000) {
    last_dbg_ms = current_millis;
    Serial.printf("state=%d freq=%.1f target=%.1f ledc0=%u ledc3=%u\n",
                  (int)sweep_state, controller->getFrequency(), duty_target,
                  (unsigned)ledc_get_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0),
                  (unsigned)ledc_get_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_3));
  }

  switch (sweep_state) {
    case IDLE:
    case STOPPED:
      break;   // de-energized, latched down, waiting for 'begin'

    case RAMP_UP: {
      // Blend the coupling trim in with frequency (see scheduledDutyPct).
      static unsigned long last_trim_ms = 0;
      if (current_millis - last_trim_ms >= 50) {
        last_trim_ms = current_millis;
        float f = controller->getFrequency();
        for (int i = 0; i < NUM_CHANNELS; i++)
          controller->setCarrierDutyCycle(
              i, dutyClamp(scheduledDutyPct(i, f) * FIELD_TRIM[i]));
      }
      if (controller->getFrequency() >= end_freq - 0.5f) {
        // Land exactly on the calibrated full-trim pattern before the sweep.
        for (int i = 0; i < NUM_CHANNELS; i++)
          controller->setCarrierDutyCycle(
              i, dutyClamp(INITIAL_CARRIER_DUTY_CYCLES[i] * FIELD_TRIM[i]));
        sweep_state = HOLD;
        state_start_time = current_millis;
      }
      break;
    }

    case HOLD:
      if (current_millis - state_start_time >= hold_time_ms) {
        if (duty_target <= 0.0f) {
          sweep_state = DONE;
          break;
        }
        duty_start = duty_target;
        duty_target = duty_target - d_duty_step;
        if (duty_target < 0.0f) {
          duty_target = 0.0f;
        }

        led_state = !led_state;
        digitalWrite(LED_PIN, led_state ? HIGH : LOW);

        sweep_state = TRANSITION;
        state_start_time = current_millis;
      }
      break;

    case TRANSITION: {
      float t = (float)(current_millis - state_start_time) / (float)transition_time_ms;
      if (t >= 1.0f) {
        setSweepCarrierDuty(duty_target);
        sweep_state = HOLD;
        state_start_time = current_millis;
      } else {
        // cubic smoothstep: ease in/out between duty levels
        float ease = t * t * (3.0f - 2.0f * t);
        setSweepCarrierDuty(duty_start + (duty_target - duty_start) * ease);
      }
      break;
    }

    case ENDING:
      // Monotonic ramp: one library step per tick until every coil hits 0.
      if (current_millis - last_ramp_ms >= ramp_tick_ms) {
        last_ramp_ms = current_millis;
        if (controller->rampDownStep(ramp_step_pct)) {
          digitalWrite(LED_PIN, LOW);
          sweep_state = STOPPED;
          Serial.println("coils off");
        }
      }
      break;

    case DONE:
      break;   // sweep finished; A & D still energized — send e/s to de-energize
  }
}