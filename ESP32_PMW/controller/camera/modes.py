"""What frame rate each sensor mode actually delivers, and why some do not.

A planning sweep measured this camera at 121.4 fps on its native 1280x800 and
209.9 at 640x480 -- matching the datasheet -- but 98.8 at 800x600 against 120
asked, 271.3 at 640x400 against 210, and 285.4 at 160x120 against 640. Three
anomalies, and the sweep that found them could not say why: 100 reads after a
10-frame warmup, which at 640 fps is 16 ms and settles nothing, with a `cvtColor`
running inside the timed section.

This exists to separate the competing explanations rather than to re-print the
numbers:

**Consumer-bound, not camera-bound.** `sources.CameraSource` grabs on its own
thread, so the camera's rate and the rate a Python loop can *consume* are two
different numbers, and only the first is a property of the camera. Both are
reported. `n_grabbed / elapsed` is the camera; frames the consumer never saw are
counted in `n_dropped`. Reading `n_dropped` as camera loss is exactly how the
160x120 anomaly would get blamed on hardware.

**Colour conversion inside the measurement.** `grayscale=True` converts on the
grabber thread. The sensor is mono, so MJPG returns a replicated grey and the
conversion is pure overhead -- it costs frames without adding anything. The sweep
defaults to `--no-grayscale`, and `--grayscale` runs the A/B.

**A mode the sensor does not have.** A driver will synthesise a resolution by
scaling or decimating a neighbouring one, at a cost that looks like a mysterious
rate limit. `--probe-formats` asks ffmpeg for the modes AVFoundation reports as
real, which shares no code with the sweep and so cannot fail the same way.

**Too short a sample.** Warmup is by *time*, not frame count, the default sample
is 600 frames, and `--repeat` gives the run-to-run spread. `--frames 100
--warmup-s 0` reproduces the original conditions; reproducing an anomaly is the
regression test for the measurement itself.

Usage:
    uv run python controller/elp/modes.py --index 0
    uv run python controller/elp/modes.py --index 0 --modes 1280x800@120,640x400@210
    uv run python controller/elp/modes.py --probe-formats
    uv run python controller/elp/modes.py --frames 100 --warmup-s 0   # reproduce the old run
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. camera is stage 1 of 4.
sys.path[:0] = [str(HERE)]

import elp as cammod  # noqa: E402

RESULTS = HERE.parents[1] / "results" / "elp"


def sweep_mode(index, mode, frames=600, warmup_s=1.0, grayscale=False, timeout=2.0):
    """Time real reads in one mode. Returns a row dict, or one with ``error`` set."""
    mode = cammod.parse_mode(mode)
    try:
        cam = cammod.open_elp(index=index, mode=mode, grayscale=grayscale, strict=False)
    except OSError as e:
        return {"asked_w": mode.width, "asked_h": mode.height, "asked_fps": mode.fps,
                "error": str(e)}

    try:
        actual = cam.actual
        # Warm up by time. Ten frames is a sixteenth of a second at 640 fps --
        # exposure, gain and the USB pipeline have all not settled by then, which
        # is the leading suspect for the original 160x120 number.
        t_end = time.monotonic() + warmup_s
        while time.monotonic() < t_end:
            cam.read(timeout)
        g0, d0 = cam.n_grabbed, cam.n_dropped

        stamps = []
        t0 = time.monotonic()
        for _ in range(frames):
            item = cam.read(timeout)
            if item is None:
                break
            stamps.append(item[0])
        elapsed = time.monotonic() - t0
        grabbed = cam.n_grabbed - g0
        dropped = cam.n_dropped - d0
    finally:
        cam.close()

    return _stats(stamps, grabbed, dropped, elapsed, mode, actual, grayscale)


def _stats(stamps, grabbed, dropped, elapsed, mode, actual, grayscale):
    n = len(stamps)
    d = np.diff(np.asarray(stamps)) * 1e3 if n > 1 else np.array([np.nan])
    return {
        "asked_w": mode.width, "asked_h": mode.height, "asked_fps": mode.fps,
        "got_w": actual["width"], "got_h": actual["height"], "prop_fps": actual["fps"],
        # The camera's own rate: what the grabber thread pulled off the device,
        # whether or not this process kept up with it.
        "fps_grabbed": grabbed / elapsed if elapsed > 0 else 0.0,
        # What a single-threaded consumer sustained. The one that bounds a
        # control loop, and the smaller of the two whenever compute is binding.
        "fps_consumed": (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0.0,
        "median_ms": float(np.median(d)), "p95_ms": float(np.percentile(d, 95)),
        "p99_ms": float(np.percentile(d, 99)), "max_ms": float(np.max(d)),
        "n_frames": n, "n_grabbed": grabbed,
        # Consumer misses, NOT camera loss -- see the module docstring.
        "n_dropped": dropped,
        "grayscale": int(bool(grayscale)),
    }


def fit_frame_time(rows):
    """Least squares on ``t_frame = a + b * W * H``, over modes that are not rate-capped.

    ``a`` is fixed per-frame overhead in ms -- USB transaction, MJPEG decode, the
    Python round trip -- and ``b`` is ms per megapixel. Only modes running below
    their requested rate carry information: one that hits its asked rate is
    telling you about the *request*, not about the cost.

    Its value is as much in what it fails to absorb. A mode that undershoots while
    having *fewer* pixels than a faster one cannot be explained by any model of
    this shape, and a large residual there is what promotes "the sensor does not
    have this mode" from a guess to the leading hypothesis.
    """
    pts = [(r["asked_w"] * r["asked_h"], 1e3 / r["fps_grabbed"])
           for r in rows if r.get("fps_grabbed", 0) > 0 and r.get("asked_fps")
           and r["fps_grabbed"] < 0.95 * r["asked_fps"]]
    if len(pts) < 2:
        return {}
    px = np.array([p[0] for p in pts], float)
    ms = np.array([p[1] for p in pts], float)
    A = np.column_stack([np.ones_like(px), px])
    (a, b), *_ = np.linalg.lstsq(A, ms, rcond=None)
    resid = ms - (a + b * px)
    return {"a_ms": float(a), "b_ms_per_mpx": float(b * 1e6),
            "n_points": len(pts), "resid_max_ms": float(np.max(np.abs(resid)))}


def classify(row, native=None):
    """A one-word verdict per mode, so the table can be read without arithmetic."""
    if row.get("error"):
        return "error"
    asked, grabbed, consumed = row.get("asked_fps"), row["fps_grabbed"], row["fps_consumed"]
    if (row["got_w"], row["got_h"]) != (row["asked_w"], row["asked_h"]):
        return "substituted"
    if native is not None and not any(
            m["width"] == row["asked_w"] and m["height"] == row["asked_h"] for m in native):
        return "not-native"
    if grabbed > 1.5 * consumed:
        return "consumer-bound"
    if asked and grabbed >= 0.95 * asked:
        return "native"
    return "overhead-limited"


def _write_metadata(fh, meta):
    """The repo's ``# key, value`` provenance block, read back with
    ``pandas.read_csv(path, comment="#")``.

    Eight lines rather than an import of `pose/recorder.py`, which has the same
    helper: this is stage 1 and that is stage 3, so importing it would make the
    camera depend on the estimator for a header format.
    """
    fh.write(f"# generated, {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    for k, v in meta.items():
        fh.write(f"# {k}, {v}\n")


def _write_csv(path, rows, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["asked_w", "asked_h", "asked_fps", "got_w", "got_h", "prop_fps",
            "fps_grabbed", "fps_consumed", "median_ms", "p95_ms", "p99_ms",
            "max_ms", "n_frames", "n_grabbed", "n_dropped", "grayscale", "verdict"]
    with open(path, "w") as fh:
        _write_metadata(fh, meta)
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--modes", default="all",
                    help="'all' for the profile's list, or WxH@FPS,WxH@FPS")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--grayscale", action="store_true",
                    help="convert on the grabber thread; the A/B against the default")
    ap.add_argument("--repeat", type=int, default=1, help="run-to-run spread")
    ap.add_argument("--probe-formats", action="store_true",
                    help="ask ffmpeg which modes the device really has")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-config", action="store_true",
                    help="write measured_fps and verdicts back into elp_camera.json")
    args = ap.parse_args(argv)

    native = cammod.native_modes_ffmpeg(args.index) if args.probe_formats else None
    if args.probe_formats:
        if native is None:
            print("ffmpeg not found or gave nothing -- skipping the native-mode check")
        else:
            print(f"AVFoundation reports {len(native)} native modes:")
            for m in native:
                print(f"    {m['width']}x{m['height']}  {m['fps_min']:g}-{m['fps_max']:g} fps")
            print()

    wanted = (cammod.modes() if args.modes == "all"
              else [cammod.parse_mode(s) for s in args.modes.split(",")])
    if not wanted:
        print("no modes to sweep (is elp_camera.json present?)")
        return 1

    print(f"{'asked':>15} {'got':>11} {'camera':>8} {'consumed':>9} "
          f"{'median':>7} {'p99':>7} {'drop':>6}  verdict")
    rows = []
    for m in wanted:
        reps = [sweep_mode(args.index, m, args.frames, args.warmup_s, args.grayscale)
                for _ in range(args.repeat)]
        r = min(reps, key=lambda x: x.get("fps_grabbed", 0)) if args.repeat > 1 else reps[0]
        if args.repeat > 1:
            r["fps_grabbed_spread"] = (max(x.get("fps_grabbed", 0) for x in reps)
                                       - min(x.get("fps_grabbed", 0) for x in reps))
        if r.get("error"):
            print(f"{m.width}x{m.height}@{m.fps or '-':<4}  ERROR {r['error']}")
            rows.append(r)
            continue
        r["verdict"] = classify(r, native)
        rows.append(r)
        print(f"{m.width}x{m.height}@{str(m.fps or '-'):<4} {r['got_w']:>5}x{r['got_h']:<5} "
              f"{r['fps_grabbed']:>8.1f} {r['fps_consumed']:>9.1f} "
              f"{r['median_ms']:>7.2f} {r['p99_ms']:>7.2f} {r['n_dropped']:>6}  {r['verdict']}")

    fit = fit_frame_time(rows)
    if fit:
        print(f"\nframe time ~ {fit['a_ms']:.2f} ms + {fit['b_ms_per_mpx']:.2f} ms/Mpx "
              f"over {fit['n_points']} rate-limited modes "
              f"(worst residual {fit['resid_max_ms']:.2f} ms)")
        if fit["resid_max_ms"] > 1.0:
            print("  a residual this large is not pixel cost -- suspect a non-native mode")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": " ".join(["modes.py"] + (argv or sys.argv[1:])),
            "index": args.index, "frames": args.frames, "warmup_s": args.warmup_s,
            "grayscale": args.grayscale, "repeat": args.repeat,
            "cv2": cv2.__version__, "platform": platform.platform(),
            "backend": "AVFOUNDATION" if sys.platform == "darwin" else "V4L2",
            "n_samples": args.frames}
    csv_path = Path(args.out) if args.out else RESULTS / f"modes_{stamp}.csv"
    _write_csv(csv_path, rows, meta)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({"meta": meta, "rows": rows, "fit": fit,
                                     "native": native}, indent=2))
    print(f"\nwrote {csv_path}\n      {json_path}")

    if args.write_config:
        prof = cammod.load_profile()
        by_size = {(r["asked_w"], r["asked_h"]): r for r in rows if not r.get("error")}
        for m in prof.get("modes", []):
            r = by_size.get((m["width"], m["height"]))
            if r:
                m["measured_fps"] = round(r["fps_grabbed"], 1)
                m["verdict"] = r["verdict"]
        prof["created"] = meta["created"]
        prof["source"] = meta["source"]
        prof["n_samples"] = args.frames
        prof.pop("caveat", None)
        cammod.DEFAULT_PATH.write_text(json.dumps(prof, indent=2) + "\n")
        print(f"      {cammod.DEFAULT_PATH} (measured_fps updated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
