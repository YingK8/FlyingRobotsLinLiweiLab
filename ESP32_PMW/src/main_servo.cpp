// Dumb serial -> PwmController shim for the host visual-servo loop
// (controller/visual_servo). No state machine, no ramp, no watchdog: the host
// owns all of that and this just executes what it is told, one line at a time.
// Commands (newline, 115200):
//   F<hz>         setGlobalFrequency
//   A<ch>,<pct>   setCarrierDutyCycle (ch 0..3 -> A,B,C,D)
//   S             all carriers to 0 (coils off)
// Telemetry is driveTelemetry()'s shared 2 Hz line, same as every other main.
#include "drive_common.h"
#include "SerialComm.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static SerialComm comm;

static void dispatch(String cmd) {
  cmd.trim();
  if (cmd.startsWith("F")) {
    float hz = cmd.substring(1).toFloat();
    ctl.setGlobalFrequency(hz);
  } else if (cmd.startsWith("A")) {
    int comma = cmd.indexOf(',');
    int ch = cmd.substring(1, comma).toInt();
    ctl.setCarrierDutyCycle(ch, constrain(cmd.substring(comma + 1).toFloat(), 0.0f, 100.0f));
  } else if (cmd == "S") {
    for (int i = 0; i < NUM_CHANNELS; i++) ctl.setCarrierDutyCycle(i, 0.0f);
  }
}

void setup() {
  driveBoot();
  ctl.begin(); // DC until the host sends the first F
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f);
  // ctl.enableCurrentBalance();  // OFF: CS is blind to the disk (tilt_ccw_nodisk), the
  // imbalance is static magnetic coupling that feedforward trims already fix (1.97 ->
  // 1.046), and the loop confounds the az sweep. Re-enable only with evidence.
}

void loop() {
  String line = comm.handleSerialComm();
  if (line.length()) dispatch(line);

  ctl.run(); // sense + balance + overcurrent trip
  driveTelemetry(ctl);
}
