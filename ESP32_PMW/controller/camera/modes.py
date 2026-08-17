"""
Time what frame rate each sensor mode actually delivers.

Two rates, and only the first is a property of the camera: `MonoCamera` grabs on
its own thread, so ``n_grabbed / elapsed`` is the sensor while frames the consumer
never saw are `n_dropped`. Reading dropped frames as camera loss is the mistake
this reports both numbers to prevent.

`grayscale=True` converts on the grabber thread. The sensor is mono, so MJPG
returns a replicated grey and the conversion is pure overhead; it defaults off, and
passing it runs the A/B.

Warmup is by *time*, not frame count. `probe_formats=True` asks ffmpeg which modes
AVFoundation reports as real, which shares no code with the sweep and so cannot
fail the same way -- a driver will synthesise a missing resolution by scaling a
neighbour, at a cost that looks like a mysterious rate limit.

    modes.sweep()                                        # the profile's list
    modes.sweep(modes=["1280x800@120", "640x400@210"])   # a subset
    modes.sweep(probe_formats=True)                      # what is native
    modes.sweep(write_config=True)                       # fold into elp_camera.json
"""

from __future__ import annotations

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


def sweep_mode(index, mode, frames=600, warmup_s=1.0, grayscale=False):
    """
    Time real reads in one mode. Returns a row dict, or one with ``error`` set.
    """

    mode = cammod.parse_mode(mode)
    try:
        cam = cammod.open_elp(index=index, mode=mode, grayscale=grayscale, strict=False)
    except OSError as e:
        return {
            "asked_w": mode.width,
            "asked_h": mode.height,
            "asked_fps": mode.fps,
            "error": str(e),
        }

    try:
        actual = cam.actual
        # Warm up by time. Ten frames is a sixteenth of a second at 640 fps --
        # exposure, gain and the USB pipeline have all not settled by then, which
        # is the leading suspect for the original 160x120 number.
        t_end = time.monotonic() + warmup_s
        while time.monotonic() < t_end:
            cam.read()
        g0, d0 = cam.n_grabbed, cam.n_dropped

        stamps = []
        t0 = time.monotonic()
        for _ in range(frames):
            item = cam.read()
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
        "asked_w": mode.width,
        "asked_h": mode.height,
        "asked_fps": mode.fps,
        "got_w": actual["width"],
        "got_h": actual["height"],
        "prop_fps": actual["fps"],
        # The camera's own rate: what the grabber thread pulled off the device,
        # whether or not this process kept up with it.
        "fps_grabbed": grabbed / elapsed if elapsed > 0 else 0.0,
        # What a single-threaded consumer sustained. The one that bounds a
        # control loop, and the smaller of the two whenever compute is binding.
        "fps_consumed": (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0.0,
        "median_ms": float(np.median(d)),
        "p95_ms": float(np.percentile(d, 95)),
        "p99_ms": float(np.percentile(d, 99)),
        "max_ms": float(np.max(d)),
        "n_frames": n,
        "n_grabbed": grabbed,
        # Consumer misses, NOT camera loss -- see the module docstring.
        "n_dropped": dropped,
        "grayscale": int(bool(grayscale)),
    }


def fit_frame_time(rows):
    """
    Least squares on ``t_frame = a + b * W * H``, over modes that are not rate-capped.

    ``a`` is fixed per-frame overhead in ms -- USB transaction, MJPEG decode, the
    Python round trip -- and ``b`` is ms per megapixel. Only modes running below
    their requested rate carry information: one that hits its asked rate is
    telling you about the *request*, not about the cost.

    Its value is as much in what it fails to absorb. A mode that undershoots while
    having *fewer* pixels than a faster one cannot be explained by any model of
    this shape, and a large residual there is what promotes "the sensor does not
    have this mode" from a guess to the leading hypothesis.
    """

    pts = [
        (r["asked_w"] * r["asked_h"], 1e3 / r["fps_grabbed"])
        for r in rows
        if r.get("fps_grabbed", 0) > 0
        and r.get("asked_fps")
        and r["fps_grabbed"] < 0.95 * r["asked_fps"]
    ]
    if len(pts) < 2:
        return {}
    px = np.array([p[0] for p in pts], float)
    ms = np.array([p[1] for p in pts], float)
    A = np.column_stack([np.ones_like(px), px])
    (a, b), *_ = np.linalg.lstsq(A, ms, rcond=None)
    resid = ms - (a + b * px)
    return {
        "a_ms": float(a),
        "b_ms_per_mpx": float(b * 1e6),
        "n_points": len(pts),
        "resid_max_ms": float(np.max(np.abs(resid))),
    }


def classify(row, native=None):
    """
    A one-word verdict per mode, so the table can be read without arithmetic.
    """

    if row.get("error"):
        return "error"
    asked, grabbed, consumed = (
        row.get("asked_fps"),
        row["fps_grabbed"],
        row["fps_consumed"],
    )
    if (row["got_w"], row["got_h"]) != (row["asked_w"], row["asked_h"]):
        return "substituted"
    if native is not None and not any(
        m["width"] == row["asked_w"] and m["height"] == row["asked_h"] for m in native
    ):
        return "not-native"
    if grabbed > 1.5 * consumed:
        return "consumer-bound"
    if asked and grabbed >= 0.95 * asked:
        return "native"
    return "overhead-limited"


def _write_metadata(fh, meta):
    """
    The repo's ``# key, value`` provenance block, read back with
    ``pandas.read_csv(path, comment="#")``.

    Eight lines rather than an import of `pose/recorder.py`, which has the same
    helper: this is stage 1 and that is stage 3, so importing it would make the
    camera depend on the estimator for a header format.
    """

    fh.write(
        f"# generated, {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    )
    for k, v in meta.items():
        fh.write(f"# {k}, {v}\n")


def _write_csv(path, rows, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "asked_w",
        "asked_h",
        "asked_fps",
        "got_w",
        "got_h",
        "prop_fps",
        "fps_grabbed",
        "fps_consumed",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "n_frames",
        "n_grabbed",
        "n_dropped",
        "grayscale",
        "verdict",
    ]
    with open(path, "w") as fh:
        _write_metadata(fh, meta)
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(
                ",".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols)
                + "\n"
            )
    return path


def sweep(
    index=0,
    modes=None,
    frames=600,
    warmup_s=1.0,
    grayscale=False,
    repeat=1,
    probe_formats=False,
    out=None,
    write_config=False,
):
    """
    Time every mode and write the result. Returns ``(rows, fit, paths)``.

        ``modes`` defaults to the profile's list; pass `Mode` tuples or ``"WxH@FPS"``
        strings to sweep a subset. ``repeat`` > 1 keeps the *worst* run, since a mode
        that only sometimes holds its rate has not held it. ``write_config`` folds
        ``measured_fps`` and the verdicts back into `elp_camera.json`, which is the
        only way that file should ever change.
    """

    native = cammod.native_modes_ffmpeg(index) if probe_formats else None
    if probe_formats:
        if native is None:
            print("ffmpeg not found or gave nothing -- skipping the native-mode check")
        else:
            print(f"AVFoundation reports {len(native)} native modes:")
            for m in native:
                print(
                    f"    {m['width']}x{m['height']}  "
                    f"{m['fps_min']:g}-{m['fps_max']:g} fps"
                )

    wanted = cammod.modes() if modes is None else [cammod.parse_mode(m) for m in modes]
    if not wanted:
        raise ValueError("no modes to sweep -- is elp_camera.json present?")

    print(
        f"\n{'asked':>15} {'got':>11} {'camera':>8} {'consumed':>9} "
        f"{'median':>7} {'p99':>7} {'drop':>6}  verdict"
    )
    rows = []
    for m in wanted:
        reps = [
            sweep_mode(index, m, frames, warmup_s, grayscale) for _ in range(repeat)
        ]
        r = min(reps, key=lambda x: x.get("fps_grabbed", 0))
        rows.append(r)
        # One field, padded once: built piecewise it comes out 12-13 wide against
        # a 15-wide header and every column downstream shifts row to row. ``:g``
        # because `parse_mode` returns a float, and "@120.0" is noise.
        asked = f"{m.width}x{m.height}@{f'{m.fps:g}' if m.fps else '-'}"
        if r.get("error"):
            print(f"{asked:>15}  ERROR {r['error']}")
            continue
        r["verdict"] = classify(r, native)
        print(
            f"{asked:>15} {r['got_w']:>5}x{r['got_h']:<5} "
            f"{r['fps_grabbed']:>8.1f} {r['fps_consumed']:>9.1f} "
            f"{r['median_ms']:>7.2f} {r['p99_ms']:>7.2f} {r['n_dropped']:>6}  {r['verdict']}"
        )

    fit = fit_frame_time(rows)
    if fit:
        print(
            f"\nframe time ~ {fit['a_ms']:.2f} ms + {fit['b_ms_per_mpx']:.2f} ms/Mpx "
            f"over {fit['n_points']} rate-limited modes "
            f"(worst residual {fit['resid_max_ms']:.2f} ms)"
        )
        if fit["resid_max_ms"] > 1.0:
            print(
                "  a residual this large is not pixel cost -- suspect a non-native mode"
            )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": (
            f"modes.sweep(index={index}, frames={frames}, "
            f"warmup_s={warmup_s}, grayscale={grayscale}, repeat={repeat})"
        ),
        "index": index,
        "warmup_s": warmup_s,
        "grayscale": grayscale,
        "repeat": repeat,
        "cv2": cv2.__version__,
        "platform": platform.platform(),
        "backend": "AVFOUNDATION" if sys.platform == "darwin" else "V4L2",
        "n_samples": frames,
    }
    csv_path = Path(out) if out else RESULTS / f"modes_{stamp}.csv"
    _write_csv(csv_path, rows, meta)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"meta": meta, "rows": rows, "fit": fit, "native": native}, indent=2)
    )
    print(f"\nwrote {csv_path}\n      {json_path}")

    if write_config:
        prof = cammod.load_profile()
        by_size = {(r["asked_w"], r["asked_h"]): r for r in rows if not r.get("error")}
        for m in prof.get("modes", []):
            r = by_size.get((m["width"], m["height"]))
            if r:
                m["measured_fps"] = round(r["fps_grabbed"], 1)
                m["verdict"] = r["verdict"]
        prof["created"] = meta["created"]
        prof["source"] = meta["source"]
        prof["n_samples"] = frames
        cammod.DEFAULT_PATH.write_text(json.dumps(prof, indent=2) + "\n")
        print(f"      {cammod.DEFAULT_PATH} (measured_fps updated)")

    return rows, fit, {"csv": csv_path, "json": json_path}
