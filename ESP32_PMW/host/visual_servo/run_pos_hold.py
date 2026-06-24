"""Closed-loop height + lateral position hold (v2).

Two cameras:
  - FRONT camera  -> height  -> commanded frequency   (F<hz>)        [v1 loop]
  - DOWNWARD cam  -> (x, y)  -> tilt -> carrier duties (A<ch>,<duty>) [v2 loop]

The two loops are near-decoupled (frequency sets thrust magnitude -> vertical;
carrier asymmetry tilts the thrust -> lateral). Lateral starts DISARMED: get a
stable hover first, then press SPACE to arm the centring loop. Disarming (or
exiting) levels the carriers; the firmware watchdog is the backstop.

Usage:
    python run_pos_hold.py --z-ref 80 --f-hover 160
    python run_pos_hold.py --z-ref 80 --cam-front 0 --cam-down 1 --arm-lateral
Keys (in a video window): SPACE arm/disarm lateral, 'q' quit (controlled descent).
"""
import argparse
import csv
import time

import cv2

from config import (
    DEFAULT_GAINS, DEFAULT_LATERAL_GAINS, CMD_PERIOD_S, SERIAL_PORT,
    CARRIER_NOMINAL,
)
from vision import open_camera, HeightTracker
from controller import HeightController
from lateral import open_down_camera, LateralTracker, LateralController
from mixer import tilt_to_duties
from serial_link import SerialLink

N_COILS = 4  # A, B, C, D (mixer order)


def parse_args():
    p = argparse.ArgumentParser(description="Height + lateral position hold (v2)")
    p.add_argument("--z-ref", type=float, required=True, help="target height (mm)")
    p.add_argument("--x-ref", type=float, default=None,
                   help="target x (px); default = down-frame centre")
    p.add_argument("--y-ref", type=float, default=None,
                   help="target y (px); default = down-frame centre")
    p.add_argument("--port", default=SERIAL_PORT)
    p.add_argument("--cam-front", type=int, default=None, help="front camera index")
    p.add_argument("--cam-down", type=int, default=None, help="downward camera index")
    # height gains
    p.add_argument("--f-hover", type=float, default=DEFAULT_GAINS.f_hover)
    p.add_argument("--kp", type=float, default=DEFAULT_GAINS.kp)
    p.add_argument("--kd", type=float, default=DEFAULT_GAINS.kd)
    p.add_argument("--ki", type=float, default=DEFAULT_GAINS.ki)
    # lateral gains
    p.add_argument("--lkp", type=float, default=DEFAULT_LATERAL_GAINS.kp)
    p.add_argument("--lkd", type=float, default=DEFAULT_LATERAL_GAINS.kd)
    p.add_argument("--lki", type=float, default=DEFAULT_LATERAL_GAINS.ki)
    p.add_argument("--arm-lateral", action="store_true",
                   help="arm the lateral loop at startup (default: disarmed)")
    p.add_argument("--latency", type=float, default=0.0,
                   help="seconds to predict height ahead (camera/pipeline delay)")
    p.add_argument("--log", default="pos_hold_log.csv")
    p.add_argument("--no-window", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    gains = DEFAULT_GAINS
    gains.f_hover, gains.kp, gains.kd, gains.ki = (
        args.f_hover, args.kp, args.kd, args.ki)
    lat_gains = DEFAULT_LATERAL_GAINS
    lat_gains.kp, lat_gains.kd, lat_gains.ki = (args.lkp, args.lkd, args.lki)

    cap_front = open_camera(args.cam_front) if args.cam_front is not None else open_camera()
    cap_down = open_down_camera(args.cam_down) if args.cam_down is not None else open_down_camera()

    htracker = HeightTracker(cap_front, draw=not args.no_window)
    hcontroller = HeightController(gains)
    ltracker = LateralTracker(cap_down, draw=not args.no_window)
    lcontroller = LateralController(lat_gains)

    armed = args.arm_lateral
    x_ref, y_ref = args.x_ref, args.y_ref   # filled from frame centre on first frame

    log_f = open(args.log, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["t", "z_mm", "z_ref", "zdot", "u_hz", "freq_meas",
                     "x_px", "y_px", "x_ref", "y_ref", "tilt_x", "tilt_y",
                     "armed", "duty_a", "duty_b", "duty_c", "duty_d"])

    print(f"Height+lateral hold: z_ref={args.z_ref}mm f_hover={gains.f_hover} "
          f"Kp={gains.kp} Kd={gains.kd} | lateral Kp={lat_gains.kp} Kd={lat_gains.kd}")
    print(f"  lateral {'ARMED' if armed else 'DISARMED'} -- SPACE to toggle, 'q' to quit.")

    t0 = time.monotonic()
    last_cmd_t = t0
    last_loop_t = t0
    duties = [CARRIER_NOMINAL] * N_COILS

    try:
        with SerialLink(args.port) as link:
            while True:
                z, zdot, _zdet, f_front = htracker.step()
                x, xdot, y, ydot, _ldet, f_down = ltracker.step()
                now = time.monotonic()
                dt = now - last_loop_t
                last_loop_t = now

                # default lateral target = downward frame centre
                if (x_ref is None or y_ref is None) and f_down is not None:
                    h, w = f_down.shape[:2]
                    x_ref = w / 2.0 if x_ref is None else x_ref
                    y_ref = h / 2.0 if y_ref is None else y_ref

                # --- height loop -> frequency ---
                z_used = z
                if z is not None and args.latency > 0.0:
                    z_used = htracker.kf.predict_ahead(args.latency)
                u = hcontroller.update(args.z_ref, z_used, zdot, dt)

                # --- lateral loop -> tilt -> carrier duties ---
                tilt_x = tilt_y = 0.0
                if armed and x_ref is not None:
                    tilt_x, tilt_y = lcontroller.update(
                        x_ref, y_ref, x, y, xdot, ydot, dt)
                    duties = tilt_to_duties(tilt_x, tilt_y)
                else:
                    duties = [CARRIER_NOMINAL] * N_COILS

                # --- command + heartbeat at >= 50 Hz (feeds the watchdog) ---
                if now - last_cmd_t >= CMD_PERIOD_S:
                    link.set_frequency(u)
                    if armed:
                        for ch, d in enumerate(duties):
                            link.set_carrier(ch, d)
                    last_cmd_t = now

                tel = link.latest()
                writer.writerow([
                    f"{now - t0:.4f}",
                    "" if z is None else f"{z:.2f}", f"{args.z_ref:.2f}",
                    "" if zdot is None else f"{zdot:.2f}", f"{u:.2f}",
                    f"{tel.freq:.2f}",
                    "" if x is None else f"{x:.1f}", "" if y is None else f"{y:.1f}",
                    "" if x_ref is None else f"{x_ref:.1f}",
                    "" if y_ref is None else f"{y_ref:.1f}",
                    f"{tilt_x:.3f}", f"{tilt_y:.3f}", int(armed),
                    *[f"{d:.1f}" for d in duties]])

                if not args.no_window:
                    if f_front is not None:
                        cv2.imshow("height (front)", f_front)
                    if f_down is not None:
                        if x_ref is not None:
                            cv2.drawMarker(f_down, (int(x_ref), int(y_ref)),
                                           (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(f_down,
                                    f"{'ARMED' if armed else 'DISARMED'} "
                                    f"tilt=({tilt_x:+.2f},{tilt_y:+.2f})",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 255), 2)
                        cv2.imshow("lateral (down)", f_down)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord(" "):
                        armed = not armed
                        if not armed:
                            lcontroller.reset()
                            for ch in range(N_COILS):
                                link.set_carrier(ch, CARRIER_NOMINAL)
                        print(f"  lateral {'ARMED' if armed else 'DISARMED'}")
    except KeyboardInterrupt:
        print("\nInterrupted -> controlled descent.")
    finally:
        log_f.close()
        cap_front.release()
        cap_down.release()
        cv2.destroyAllWindows()
        print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
