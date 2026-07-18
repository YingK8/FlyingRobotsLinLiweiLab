#!/usr/bin/env python3
"""Long-duration STREAMING recorder for the PicoScope 5443D (ps5000a).

The companion `picoscope_capture.py` is block-mode (tens of milliseconds, for
waveform/THD analysis). This tool instead does continuous **streaming** capture
over tens of seconds — the whole tilt / carrier-ramp experiment — and writes a CSV
in the exact format the analysis pipeline expects:

    Time,Channel A,Channel B,Channel C,Channel D
    (s),(V),(V),(V),(V)
    <blank>
    0.000000,0.001,...

so `pico/calibrate_multipoint.py` and the tilt analysis read it directly
(`pd.read_csv(path, skiprows=[1])`).

Recording strategy (default): sample fast (raw interval) and let the driver
**aggregate-downsample** to a slower output rate, writing the per-bin PEAK (max).
For a current-sense signal that IS the rectified current envelope — exactly what
the downstream rolling-max wants — and it captures the 190 Hz crest that a naive
slow sample would alias past. Use --raw to disable aggregation and store the
plain decimated samples instead.

Reuses the device open / USB-power / range handling from picoscope_capture.py.

HARDWARE-FREE: `--self-test` validates the CSV-format contract (round-trips
through the same pandas call the analysis uses) without a scope or the runtime.

GROUNDING SAFETY: see picoscope_capture.py / ai/README.md. Scope BNC grounds
are common and earth-referenced — use a differential probe on any floating node.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Reuse the validated scope/range plumbing from the block-mode tool.
from picoscope_capture import (
    ChannelSpec,
    RANGE_FULLSCALE_V,
    RANGE_SHORTHAND,
    open_scope,
    configure_channel,
    close_scope,
    _HAVE_PS5000A,
    _require_ps5000a,
)

if _HAVE_PS5000A:
    import ctypes
    from picosdk.ps5000a import ps5000a as ps
    from picosdk.functions import adc2mV
else:
    import ctypes  # stdlib


# --------------------------------------------------------------------------- #
# Streaming capture
# --------------------------------------------------------------------------- #

def _enum(d, key, fallback):
    """Look up a picosdk enum value by key, tolerating naming differences."""
    try:
        return d[key]
    except Exception:
        return fallback


def run_streaming(chandle, specs, max_adc, raw_interval_us, duration_s,
                  downsample, aggregate):
    """Stream `specs` channels for `duration_s`, return (time_s, {label: volts*scale}).

    raw_interval_us : requested raw sample interval (driver returns the actual one).
    downsample      : driver downsample ratio (>=1). Output rate = raw / downsample.
    aggregate       : if True, AGGREGATE mode and store per-bin max (peak envelope);
                      else DECIMATE mode (every Nth raw sample).
    """
    _require_ps5000a()

    downsample = max(1, int(downsample))
    if downsample > 1:
        ratio_mode = (_enum(ps.PS5000A_RATIO_MODE, "PS5000A_RATIO_MODE_AGGREGATE", 1)
                      if aggregate
                      else _enum(ps.PS5000A_RATIO_MODE, "PS5000A_RATIO_MODE_DECIMATE", 2))
    else:
        ratio_mode = _enum(ps.PS5000A_RATIO_MODE, "PS5000A_RATIO_MODE_NONE", 0)

    # Overview buffer per channel (chunk size the driver hands back per callback).
    buf_len = 65536
    drv_max, drv_min = {}, {}
    for spec in specs:
        ch = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{spec.channel}"]
        bmax = np.zeros(buf_len, dtype=np.int16)
        bmin = np.zeros(buf_len, dtype=np.int16)
        drv_max[spec.label] = (bmax, spec)
        drv_min[spec.label] = bmin
        # SetDataBuffers(handle, channel, bufferMax, bufferMin, bufferLth, segment, mode)
        from picosdk.functions import assert_pico_ok
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            chandle, ch,
            bmax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            bmin.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            buf_len, 0, ratio_mode))

    # Per-channel accumulators (downsampled output samples).
    acc = {spec.label: [] for spec in specs}

    # Requested raw interval -> driver fills in the actual value it used.
    sample_interval = ctypes.c_int32(int(raw_interval_us))
    time_units = _enum(ps.PS5000A_TIME_UNITS, "PS5000A_US", 3)
    raw_total = int(duration_s * 1e6 / max(1, raw_interval_us))  # raw samples to autostop

    from picosdk.functions import assert_pico_ok
    assert_pico_ok(ps.ps5000aRunStreaming(
        chandle, ctypes.byref(sample_interval), time_units,
        0,                      # maxPreTriggerSamples
        raw_total,              # maxPostTriggerSamples (autostop count, raw)
        1,                      # autoStop
        downsample, ratio_mode, buf_len))

    actual_raw_us = sample_interval.value
    out_interval_s = actual_raw_us * 1e-6 * downsample

    state = {"stop": False}

    def _callback(handle, n, start, overflow, trigger_at, triggered, auto_stop, param):
        if n > 0:
            for label, (bmax, _spec) in drv_max.items():
                acc[label].append(bmax[start:start + n].copy())
        if auto_stop:
            state["stop"] = True

    try:
        StreamingReadyType = ps.StreamingReadyType
    except AttributeError:
        StreamingReadyType = ctypes.CFUNCTYPE(
            None, ctypes.c_int16, ctypes.c_int32, ctypes.c_uint32, ctypes.c_int16,
            ctypes.c_uint32, ctypes.c_int16, ctypes.c_int16, ctypes.c_void_p)
    cb = StreamingReadyType(_callback)

    t_start = time.time()
    while not state["stop"]:
        ps.ps5000aGetStreamingLatestValues(chandle, cb, None)
        if time.time() - t_start > duration_s + 10.0:
            print("WARNING: streaming wall-clock timeout; stopping early.")
            break
        time.sleep(0.005)

    # ADC -> volts*scale per channel.
    data = {}
    n_out = min(len(np.concatenate(acc[s.label])) if acc[s.label] else 0
                for s in specs) if specs else 0
    for spec in specs:
        bmax, _ = drv_max[spec.label]
        chunks = acc[spec.label]
        raw = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        raw = raw[:n_out]
        rng = ps.PS5000A_RANGE[spec.range_name]
        mv = np.array(adc2mV(raw.tolist(), rng, max_adc), dtype=float)
        data[spec.label] = (mv / 1000.0) * spec.scale
    time_s = np.arange(n_out) * out_interval_s
    print(f"streamed {n_out} samples/ch @ {1.0/out_interval_s:.1f} S/s "
          f"(raw {1e6/actual_raw_us:.0f} S/s / {downsample}x"
          f"{' aggregate-peak' if ratio_mode else ' decimate'}), "
          f"{time_s[-1] if n_out else 0:.1f} s")
    return time_s, data


# --------------------------------------------------------------------------- #
# CSV export — pipeline format (Time, Channel A..D, units row, blank line)
# --------------------------------------------------------------------------- #

def save_overview_plot(time_s, data, csv_path, win_s=0.5):
    """Best-effort rolling-RMS overview PNG saved next to the CSV, one per run.

    The current sense is AC, so its RMS is the physical current magnitude; a
    running RMS (sqrt of a moving-average of x^2) turns each channel into a clean
    envelope over the whole capture. Failures never abort a recording.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # matplotlib optional
        print(f"(plot skipped: {e})")
        return None
    time_s = np.asarray(time_s, float)
    if time_s.size < 2:
        return None
    dt = float(np.median(np.diff(time_s))) or 1.0
    win = max(1, int(round(win_s / dt)))
    kern = np.ones(win) / win
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, v in data.items():
        v = np.asarray(v, float); vv = v - v.mean()
        rms = np.sqrt(np.convolve(vv * vv, kern, mode="same")) * 1000.0
        ax.plot(time_s, rms, lw=1.0, label=label)
    ax.set_title(f"Rolling-RMS envelope ({win_s*1000:.0f} ms) — {os.path.basename(csv_path)}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("RMS current sense (mV)")
    ax.set_xlim(0, time_s[-1]); ax.grid(alpha=0.3)
    ax.legend(ncol=max(1, len(data)), fontsize=8)
    out = csv_path.rsplit(".", 1)[0] + "_rms.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def export_pipeline_csv(time_s, data, path):
    labels = list(data.keys())
    with open(path, "w", newline="") as f:
        f.write("Time," + ",".join(labels) + "\n")
        f.write("(s)," + ",".join("(V)" for _ in labels) + "\n")
        f.write("\n")
        for i in range(len(time_s)):
            row = [f"{time_s[i]:.8f}"] + [f"{data[l][i]:.8f}" for l in labels]
            f.write(",".join(row) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def default_channels(range_name):
    """A,B,C,D, DC, scale 1, labelled to match the analysis pipeline."""
    return [ChannelSpec(c, range_name, "DC", 1.0, "V", f"Channel {c}")
            for c in ("A", "B", "C", "D")]


def parse_channel_arg(s):
    """CHANNEL:RANGE[:COUPLING:SCALE:UNIT:LABEL]; label defaults to 'Channel X'."""
    parts = s.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--ch '{s}' must be CHANNEL:RANGE[:COUPLING:SCALE:UNIT:LABEL]")
    channel = parts[0].upper()
    range_name = RANGE_SHORTHAND.get(parts[1].lower(), parts[1])
    if range_name not in RANGE_FULLSCALE_V:
        raise argparse.ArgumentTypeError(f"unknown range '{parts[1]}' in --ch '{s}'")
    coupling = parts[2].upper() if len(parts) > 2 and parts[2] else "DC"
    scale = float(parts[3]) if len(parts) > 3 and parts[3] else 1.0
    unit = parts[4] if len(parts) > 4 and parts[4] else "V"
    label = parts[5] if len(parts) > 5 and parts[5] else f"Channel {channel}"
    return ChannelSpec(channel, range_name, coupling, scale, unit, label)


def self_test():
    """Round-trip the CSV format through the pipeline's own read call."""
    import pandas as pd
    import tempfile, os
    n = 50
    t = np.arange(n) * 0.01
    data = {f"Channel {c}": np.sin(2 * np.pi * 5 * t + i)
            for i, c in enumerate("ABCD")}
    path = os.path.join(tempfile.mkdtemp(), "stream_selftest.csv")
    export_pipeline_csv(t, data, path)
    df = pd.read_csv(path, skiprows=[1])          # same call the analysis uses
    ok = (list(df.columns) == ["Time", "Channel A", "Channel B", "Channel C",
                               "Channel D"]
          and len(df) == n
          and np.allclose(df["Channel A"].values, data["Channel A"], atol=1e-6))
    print(f"CSV columns: {list(df.columns)}")
    print(f"rows: {len(df)} (expected {n})")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(
        description="PicoScope 5443D long streaming recorder -> pipeline CSV.")
    p.add_argument("--seconds", type=float, default=80.0,
                   help="capture duration (s)")
    p.add_argument("--interval-us", type=float, default=50.0,
                   help="raw sample interval in microseconds (default 50us = 20kS/s)")
    p.add_argument("--downsample", type=int, default=20,
                   help="driver downsample ratio; output rate = raw/this "
                        "(default 20 -> 1kS/s out)")
    p.add_argument("--raw", action="store_true",
                   help="store plain decimated samples instead of aggregate peak")
    p.add_argument("--resolution", default="12", choices=["8", "12", "14", "15", "16"])
    p.add_argument("--range", default="1v",
                   help="default range for A-D when no --ch given (e.g. 500mv, 1v, 2v)")
    p.add_argument("--ch", dest="channels", action="append", type=parse_channel_arg,
                   metavar="A:RANGE[:COUPLING:SCALE:UNIT:LABEL]",
                   help="repeatable; default = A-D at --range, DC, labelled 'Channel X'")
    p.add_argument("--out", default=None, help="output CSV path")
    p.add_argument("--self-test", action="store_true",
                   help="validate CSV format only, no scope")
    p.add_argument("--dry-run", action="store_true",
                   help="print config and exit (no acquisition)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the auto rolling-RMS overview PNG saved per run")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    range_name = RANGE_SHORTHAND.get(args.range.lower(), args.range)
    if range_name not in RANGE_FULLSCALE_V:
        raise SystemExit(f"unknown --range '{args.range}'")
    specs = args.channels or default_channels(range_name)
    out = args.out or f"picoscope_stream_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    if args.dry_run:
        print("DRY RUN — streaming config:")
        print(f"  duration {args.seconds}s, raw interval {args.interval_us}us, "
              f"downsample {args.downsample}x "
              f"({'decimate' if args.raw else 'aggregate-peak'})")
        for c in specs:
            print(f"  {c.label}: ch{c.channel} {c.range_name} {c.coupling} "
                  f"scale={c.scale}")
        print(f"  -> {out}")
        return 0

    chandle, max_adc = open_scope(args.resolution)
    try:
        for c in specs:
            configure_channel(chandle, c)
        time_s, data = run_streaming(
            chandle, specs, max_adc, args.interval_us, args.seconds,
            args.downsample, aggregate=not args.raw)
    finally:
        close_scope(chandle)

    print("CSV ->", export_pipeline_csv(time_s, data, out))
    if not args.no_plot:
        png = save_overview_plot(time_s, data, out)
        if png:
            print("plot ->", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
