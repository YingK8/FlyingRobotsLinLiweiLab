"""Closed-loop height-hold main loop.

    frame -> height (vision) -> commanded frequency (controller) -> F<hz> (serial)

Logs every step to CSV for tuning. Press 'q' (in the video window) or Ctrl-C to
stop; either way the serial link sends 'S' (controlled descent) on exit.

Usage:
    python run_servo.py --z-ref 80 --port /dev/tty.usbserial-0001
    python run_servo.py --z-ref 80 --kp 1.0 --kd 0.2 --f-hover 160 --log run1.csv
"""
import argparse
import csv
import time

import cv2

from config import DEFAULT_GAINS, CMD_PERIOD_S, SERIAL_PORT
from vision import open_camera, HeightTracker
from controller import HeightController
from serial_link import SerialLink


def parse_args():
    p = argparse.ArgumentParser(description="Visual-servo height hold")
    p.add_argument("--z-ref", type=float, required=True, help="target height (mm)")
    p.add_argument("--port", default=SERIAL_PORT)
    p.add_argument("--camera", type=int, default=None, help="camera index override")
    p.add_argument("--f-hover", type=float, default=DEFAULT_GAINS.f_hover)
    p.add_argument("--kp", type=float, default=DEFAULT_GAINS.kp)
    p.add_argument("--kd", type=float, default=DEFAULT_GAINS.kd)
    p.add_argument("--ki", type=float, default=DEFAULT_GAINS.ki)
    p.add_argument("--log", default="servo_log.csv")
    p.add_argument("--latency", type=float, default=0.0,
                   help="seconds to predict height ahead (camera/pipeline delay)")
    p.add_argument("--no-window", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    gains = DEFAULT_GAINS
    gains.f_hover, gains.kp, gains.kd, gains.ki = (
        args.f_hover, args.kp, args.kd, args.ki)

    cap = open_camera(args.camera) if args.camera is not None else open_camera()
    tracker = HeightTracker(cap, draw=not args.no_window)
    controller = HeightController(gains)

    log_f = open(args.log, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["t", "z_mm", "z_ref", "zdot", "u_cmd_hz",
                     "freq_meas_hz", "duty_a", "duty_b", "duty_c", "duty_d"])

    print(f"Closed-loop hold: z_ref={args.z_ref} mm  f_hover={gains.f_hover}  "
          f"Kp={gains.kp} Kd={gains.kd} Ki={gains.ki}")
    print("  'q' in the window or Ctrl-C to stop (controlled descent on exit).")

    t0 = time.monotonic()
    last_cmd_t = t0
    last_loop_t = t0

    try:
        with SerialLink(args.port) as link:
            while True:
                z, zdot, det, frame = tracker.step()
                now = time.monotonic()
                dt = now - last_loop_t
                last_loop_t = now

                # latency compensation via the Kalman predictor
                z_used = z
                if z is not None and args.latency > 0.0:
                    z_used = tracker.kf.predict_ahead(args.latency)

                u = controller.update(args.z_ref, z_used, zdot, dt)

                # Command + heartbeat at >= 50 Hz to keep the firmware watchdog fed.
                if now - last_cmd_t >= CMD_PERIOD_S:
                    link.set_frequency(u)
                    last_cmd_t = now

                tel = link.latest()
                writer.writerow([
                    f"{now - t0:.4f}",
                    "" if z is None else f"{z:.2f}", f"{args.z_ref:.2f}",
                    "" if zdot is None else f"{zdot:.2f}", f"{u:.2f}",
                    f"{tel.freq:.2f}", *[f"{d:.1f}" for d in tel.duty]])

                if frame is not None and not args.no_window:
                    cv2.line(frame, (0, int(tracker.kf.x[0, 0])),  # debug only
                             (0, int(tracker.kf.x[0, 0])), (255, 0, 0), 1)
                    cv2.imshow("visual_servo", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        print("\nInterrupted -> controlled descent.")
    finally:
        log_f.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
