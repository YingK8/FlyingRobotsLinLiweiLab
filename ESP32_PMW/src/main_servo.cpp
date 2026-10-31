#include "drive_common.h"
#include "SerialComm.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static SerialComm comm;

// no watchdog: a dead host leaves the coils at their last command
static void dispatch(String cmd) {
  cmd.trim();
  if (cmd.startsWith("F")) {
    float hz = cmd.substring(1).toFloat();
    ctl.setGlobalFrequency(hz);
  } else if (cmd.startsWith("A")) {
    int comma = cmd.indexOf(',');
    int ch = cmd.substring(1, comma).toInt();
    ctl.setCarrierDutyCycle(ch, constrain(cmd.substring(comma + 1).toFloat(), 0.0f, 100.0f));
  } else if (cmd.startsWith("P")) {
    // P<ch>,<deg>: at F0 the phase picks which leg is held high (cs_cal.py)
    int comma = cmd.indexOf(',');
    ctl.setPhase(cmd.substring(1, comma).toInt(), cmd.substring(comma + 1).toFloat());
  } else if (cmd == "S" || cmd == "stop") {  // "stop" so safe_off.py reaches this env
    for (int i = 0; i < NUM_CHANNELS; i++) ctl.setCarrierDutyCycle(i, 0.0f);
  // balance stays off: the current sense is blind to the disk
  }
}

void setup() {
  driveBoot();
  ctl.begin();
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS,   10.0f);
}

void loop() {
  String line = comm.handleSerialComm();
  if (line.length()) dispatch(line);

  ctl.run();
  driveTelemetry(ctl);
}
