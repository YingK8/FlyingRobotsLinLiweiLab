"""Calibration & characterization, to run BEFORE closed-loop flight.

Subcommands:
  scale        Click two points a known mm apart -> prints PX_PER_MM and Z_REF_PX.
  hover        Open-loop: slowly raise frequency until the craft floats; the
               operator marks the equilibrium -> prints F_HOVER.
  sweep        Open-loop lift-vs-frequency sweep. Confirms control authority is
               MONOTONIC over [F_MIN, F_MAX] (rising-side-of-resonance check) and
               writes a CSV of (freq, marker height).
  threshold    Live HSV trackbars to dial in the marker mask -> prints HSV bounds
               (legacy; MOG2 background subtraction needs no threshold).
  lateral      (v2) With the craft hovering, poke +x/+y tilt and watch the drift on
               the downward camera -> prints the 2x2 LATERAL_CAL (axis/sign) matrix.

Examples:
  python calibrate.py scale --mm 100
  python calibrate.py hover --port /dev/tty.usbserial-0001
  python calibrate.py sweep --f-start 80 --f-end 300 --step 5 --dwell 1.5 --out sweep.csv
  python calibrate.py threshold
  python calibrate.py lateral --f-hover 160 --z-ref 80
"""
import argparse
import csv
import time

import cv2
import numpy as np

from config import SERIAL_PORT, F_MIN, F_MAX, CARRIER_NOMINAL, DEFAULT_GAINS
from vision import open_camera, HeightTracker, find_marker, pixel_v_to_mm
from controller import HeightController
from lateral import open_down_camera, LateralTracker
from mixer import tilt_to_duties
from serial_link import SerialLink


# --- scale: px -> mm -------------------------------------------------------
def cmd_scale(args):
    cap = open_camera()
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow("scale")
    cv2.setMouseCallback("scale", on_mouse)
    print(f"Click two points exactly {args.mm} mm apart (vertical preferred). 'r' resets, 'q' quits.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        for p in pts:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)
        if len(pts) == 2:
            cv2.line(frame, pts[0], pts[1], (0, 255, 0), 2)
            dpx = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
            px_per_mm = dpx / args.mm
            z_ref_px = (pts[0][1] + pts[1][1]) / 2.0
            cv2.putText(frame, f"PX_PER_MM={px_per_mm:.3f}  Z_REF_PX={z_ref_px:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("scale", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(pts) == 2:
        print(f"\nPut these in config.py:\n  PX_PER_MM = {px_per_mm:.3f}\n  Z_REF_PX = {z_ref_px:.1f}")


# --- hover: find F_HOVER ---------------------------------------------------
def cmd_hover(args):
    cap = open_camera()
    tracker = HeightTracker(cap, draw=True)
    print("Raising frequency slowly. Press SPACE when the craft holds a steady hover, 'q' to abort.")
    with SerialLink(args.port) as link:
        f = F_MIN
        last = time.monotonic()
        while True:
            z, zdot, det, frame = tracker.step()
            now = time.monotonic()
            if now - last >= 0.05:
                f = min(f + args.rate * (now - last), F_MAX)
                link.set_frequency(f)
                last = now
            if frame is not None:
                cv2.putText(frame, f"F={f:.1f} Hz", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("hover", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                print(f"\nF_HOVER = {f:.1f}   (put in config.py: ControlGains.f_hover)")
                break
            if k == ord("q") or f >= F_MAX:
                print(f"\nAborted at F={f:.1f}")
                break
    cap.release()
    cv2.destroyAllWindows()


# --- sweep: lift vs frequency ---------------------------------------------
def cmd_sweep(args):
    cap = open_camera()
    tracker = HeightTracker(cap, draw=True)
    rows = []
    print(f"Sweep {args.f_start}->{args.f_end} Hz step {args.step}, dwell {args.dwell}s each.")
    with SerialLink(args.port) as link:
        f = args.f_start
        while f <= args.f_end + 1e-6:
            link.set_frequency(f)
            t_end = time.monotonic() + args.dwell
            samples = []
            while time.monotonic() < t_end:
                z, zdot, det, frame = tracker.step()
                if z is not None:
                    samples.append(z)
                if frame is not None:
                    cv2.putText(frame, f"F={f:.1f} Hz  n={len(samples)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("sweep", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        f = args.f_end + 1.0
                        break
            z_mean = float(np.mean(samples)) if samples else float("nan")
            rows.append((f, z_mean, len(samples)))
            print(f"  F={f:6.1f} Hz -> z={z_mean:7.2f} mm  ({len(samples)} samples)")
            f += args.step
        link.safe_descent()

    cap.release()
    cv2.destroyAllWindows()
    with open(args.out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["freq_hz", "z_mm", "n_samples"])
        w.writerows(rows)
    print(f"\nWrote {args.out}. CHECK: z should rise MONOTONICALLY with freq over your")
    print("intended hover band. If it peaks then falls, you are past resonance -> the")
    print("control sign inverts; keep the operating band on the rising side.")


# --- lateral: axis/sign (LATERAL_CAL) calibration --------------------------
def _segment(link, htracker, hctl, ltracker, z_ref, tilt, duration, title):
    """Hold height while applying a constant tilt command (or None=level) for
    `duration` s. Returns net filtered (dx, dy) drift on the down camera, or None.
    """
    t_end = time.monotonic() + duration
    last = time.monotonic()
    start_xy = None
    last_xy = (None, None)
    while time.monotonic() < t_end:
        z, zdot, _zd, ff = htracker.step()
        x, xdot, y, ydot, _ld, fd = ltracker.step()
        now = time.monotonic(); dt = now - last; last = now

        link.set_frequency(hctl.update(z_ref, z, zdot, dt))  # keep it aloft
        duties = ([CARRIER_NOMINAL] * 4 if tilt is None
                  else tilt_to_duties(tilt[0], tilt[1]))
        for ch, d in enumerate(duties):
            link.set_carrier(ch, d)

        if x is not None:
            if start_xy is None:
                start_xy = (x, y)
            last_xy = (x, y)
        if ff is not None:
            cv2.imshow("cal front", ff)
        if fd is not None:
            cv2.putText(fd, title, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.imshow("cal down", fd)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None
    if start_xy is None or last_xy[0] is None:
        return None
    return (last_xy[0] - start_xy[0], last_xy[1] - start_xy[1])


def cmd_lateral(args):
    cap_front = (open_camera(args.cam_front) if args.cam_front is not None
                 else open_camera())
    cap_down = (open_down_camera(args.cam_down) if args.cam_down is not None
                else open_down_camera())
    htracker = HeightTracker(cap_front, draw=True)
    ltracker = LateralTracker(cap_down, draw=True)
    gains = DEFAULT_GAINS
    gains.f_hover = args.f_hover
    hctl = HeightController(gains)

    print("LATERAL CAL: get a STABLE, CENTRED hover first (height loop is running).")
    print(f"  It will poke tilt={args.tilt} for {args.dwell}s on +x then +y,")
    print("  levelling for --settle between. Keep a safety net under the craft.")
    print("  Press SPACE to begin, 'q' to abort.")

    with SerialLink(args.port) as link:
        # Idle: hold height, level carriers, wait for SPACE.
        last = time.monotonic()
        while True:
            z, zdot, _z, ff = htracker.step()
            _x, _xd, _y, _yd, _l, fd = ltracker.step()
            now = time.monotonic(); dt = now - last; last = now
            link.set_frequency(hctl.update(args.z_ref, z, zdot, dt))
            for ch in range(4):
                link.set_carrier(ch, CARRIER_NOMINAL)
            if ff is not None:
                cv2.imshow("cal front", ff)
            if fd is not None:
                cv2.imshow("cal down", fd)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                break
            if k == ord("q"):
                link.safe_descent()
                cap_front.release(); cap_down.release(); cv2.destroyAllWindows()
                return

        m = args.tilt
        _segment(link, htracker, hctl, ltracker, args.z_ref, None, args.settle, "settle")
        d_x = _segment(link, htracker, hctl, ltracker, args.z_ref, (m, 0.0),
                       args.dwell, "poke +x")
        _segment(link, htracker, hctl, ltracker, args.z_ref, None, args.settle, "settle")
        d_y = _segment(link, htracker, hctl, ltracker, args.z_ref, (0.0, m),
                       args.dwell, "poke +y")
        _segment(link, htracker, hctl, ltracker, args.z_ref, None, args.settle, "settle")
        link.safe_descent()

    cap_front.release(); cap_down.release(); cv2.destroyAllWindows()

    if d_x is None or d_y is None:
        print("\nABORTED or no detection during a poke -- no matrix produced.")
        return

    # Columns = drift produced by a unit +x and +y tilt command. cal = J^-1 makes
    # the closed-loop response decoupled and correctly signed; normalise so the
    # largest entry is ~1 (fold absolute gain into Kp afterwards).
    J = np.array([[d_x[0], d_y[0]],
                  [d_x[1], d_y[1]]], dtype=float) / m
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if abs(det) < 1e-9:
        print(f"\nDegenerate response (det~0): J={J.tolist()}. Increase --tilt/--dwell.")
        return
    cal = np.linalg.inv(J)
    cal = cal / np.max(np.abs(cal))
    print("\nMeasured drift (px):  +x ->", [f"{v:.1f}" for v in d_x],
          "  +y ->", [f"{v:.1f}" for v in d_y])
    print("Put this in config.py:")
    print(f"  LATERAL_CAL = (({cal[0,0]:.4f}, {cal[0,1]:.4f}),")
    print(f"                 ({cal[1,0]:.4f}, {cal[1,1]:.4f}))")
    print("Then re-tune lateral Kp/Kd from a step (gain changed by the normalisation).")


# --- threshold tuner -------------------------------------------------------
def cmd_threshold(args):
    cap = open_camera()
    win = "threshold (set H/S/V, read values, copy to config.py)"
    cv2.namedWindow(win)
    for name, init, maxv in [("H_lo", 0, 179), ("S_lo", 0, 255), ("V_lo", 220, 255),
                             ("H_hi", 179, 179), ("S_hi", 80, 255), ("V_hi", 255, 255)]:
        cv2.createTrackbar(name, win, init, maxv, lambda v: None)
    print("Tune until only the marker is white. 'q' to print bounds and quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        g = lambda n: cv2.getTrackbarPos(n, win)
        lo = (g("H_lo"), g("S_lo"), g("V_lo"))
        hi = (g("H_hi"), g("S_hi"), g("V_hi"))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        det = find_marker(frame, lo, hi)
        if det:
            cv2.circle(frame, (int(det[0]), int(det[1])), 6, (0, 255, 0), 2)
        cv2.imshow(win, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"\nHSV_LOWER = {lo}\nHSV_UPPER = {hi}")
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="visual_servo calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scale"); s.add_argument("--mm", type=float, required=True)
    s.set_defaults(func=cmd_scale)

    h = sub.add_parser("hover")
    h.add_argument("--port", default=SERIAL_PORT)
    h.add_argument("--rate", type=float, default=5.0, help="Hz per second ramp")
    h.set_defaults(func=cmd_hover)

    w = sub.add_parser("sweep")
    w.add_argument("--port", default=SERIAL_PORT)
    w.add_argument("--f-start", type=float, default=F_MIN)
    w.add_argument("--f-end", type=float, default=min(F_MAX, 300.0))
    w.add_argument("--step", type=float, default=5.0)
    w.add_argument("--dwell", type=float, default=1.5)
    w.add_argument("--out", default="sweep.csv")
    w.set_defaults(func=cmd_sweep)

    t = sub.add_parser("threshold"); t.set_defaults(func=cmd_threshold)

    L = sub.add_parser("lateral")
    L.add_argument("--port", default=SERIAL_PORT)
    L.add_argument("--cam-front", type=int, default=None)
    L.add_argument("--cam-down", type=int, default=None)
    L.add_argument("--z-ref", type=float, default=80.0, help="hold height (mm)")
    L.add_argument("--f-hover", type=float, default=DEFAULT_GAINS.f_hover)
    L.add_argument("--tilt", type=float, default=0.4, help="poke tilt magnitude")
    L.add_argument("--dwell", type=float, default=1.5, help="poke duration (s)")
    L.add_argument("--settle", type=float, default=2.0, help="level/settle time (s)")
    L.set_defaults(func=cmd_lateral)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
