"""Calibration & characterization, to run BEFORE closed-loop flight.

Subcommands:
  scale        Click two points a known mm apart -> prints PX_PER_MM and Z_REF_PX.
  hover        Open-loop: slowly raise frequency until the craft floats; the
               operator marks the equilibrium -> prints F_HOVER.
  sweep        Open-loop lift-vs-frequency sweep. Confirms control authority is
               MONOTONIC over [F_MIN, F_MAX] (rising-side-of-resonance check) and
               writes a CSV of (freq, marker height).
  threshold    Live HSV trackbars to dial in the marker mask -> prints HSV bounds.

Examples:
  python calibrate.py scale --mm 100
  python calibrate.py hover --port /dev/tty.usbserial-0001
  python calibrate.py sweep --f-start 80 --f-end 300 --step 5 --dwell 1.5 --out sweep.csv
  python calibrate.py threshold
"""
import argparse
import csv
import time

import cv2
import numpy as np

from config import SERIAL_PORT, F_MIN, F_MAX
from vision import open_camera, HeightTracker, find_marker, pixel_v_to_mm
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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
