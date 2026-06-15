#!/usr/bin/env python3
"""PicoScope 5443D (5000D / ps5000a) capture + analysis tool.

Debugs the ESP32 -> VNH5019 H-bridge -> series-RLC coil chain by capturing one or
more analog channels, classifying each waveform (sine / square / triangle), and
measuring the fundamental (~190 Hz), THD and (optionally) the voltage-vs-current
phase that tells you whether a coil branch is at resonance.

KEY PHYSICS (read this before concluding anything):
  The H-bridge OUTPUT (OUTA/OUTB) is ALWAYS a ~190 Hz bipolar SQUARE wave. At 100%
  carrier there is no 15 kHz ripple; below 100% the carrier rides on top. The
  SINUSOID is the COIL CURRENT (or the voltage across L or C alone), NOT the bridge
  output voltage. Seeing "square" at the bridge output is EXPECTED, not a fault.
  To see the sine, measure current (series sense resistor read with a differential
  probe, or a current probe).

DRIVER NOTE: the 5443D uses the `ps5000a` driver via the official `picosdk` Python
wrapper -- NOT `pypicosdk` (that only supports 6000E/3000E series).

Hardware-free modes (`--self-test`, `--dry-run`) work without the scope or the
native PicoSDK runtime, so the analysis can be validated on any machine.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy import signal as sp_signal
    from scipy.fft import rfft, rfftfreq
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy missing
    _HAVE_SCIPY = False
    rfft = None
    rfftfreq = None
    sp_signal = None

# ps5000a (and its native runtime) are only needed for real captures. Guard the
# import so --self-test / --dry-run run anywhere.
_PS_IMPORT_ERROR = None
try:
    import ctypes
    from picosdk.ps5000a import ps5000a as ps
    from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
    _HAVE_PS5000A = True
except Exception as exc:  # native lib or package missing
    _HAVE_PS5000A = False
    _PS_IMPORT_ERROR = exc
    import ctypes  # ctypes is stdlib, always available


# --------------------------------------------------------------------------- #
# Constants / lookup tables
# --------------------------------------------------------------------------- #

# Acquisition presets: (target sample rate Hz, sample count).
# envelope: see the 190 Hz waveform shape over ~19 cycles, 10 Hz FFT bins.
# carrier : resolve the 15 kHz carrier / PWM duty over ~1 drive cycle.
PRESETS = {
    "envelope": (200_000.0, 20_000),
    "carrier": (5_000_000.0, 25_000),
}

# Plausible band for the drive (direction-reversal) fundamental.
DRIVE_BAND_HZ = (100.0, 400.0)

# Resolution name -> PS5000A_DEVICE_RESOLUTION key suffix.
RESOLUTION_KEYS = {
    "8": "PS5000A_DR_8BIT",
    "12": "PS5000A_DR_12BIT",
    "14": "PS5000A_DR_14BIT",
    "15": "PS5000A_DR_15BIT",
    "16": "PS5000A_DR_16BIT",
}

# Friendly range shorthand -> PS5000A_RANGE key.
RANGE_SHORTHAND = {
    "10mv": "PS5000A_10MV", "20mv": "PS5000A_20MV", "50mv": "PS5000A_50MV",
    "100mv": "PS5000A_100MV", "200mv": "PS5000A_200MV", "500mv": "PS5000A_500MV",
    "1v": "PS5000A_1V", "2v": "PS5000A_2V", "5v": "PS5000A_5V",
    "10v": "PS5000A_10V", "20v": "PS5000A_20V", "50v": "PS5000A_50V",
}

# Nominal full-scale volts per PS5000A_RANGE key (for synthetic/dry-run sizing).
RANGE_FULLSCALE_V = {
    "PS5000A_10MV": 0.01, "PS5000A_20MV": 0.02, "PS5000A_50MV": 0.05,
    "PS5000A_100MV": 0.1, "PS5000A_200MV": 0.2, "PS5000A_500MV": 0.5,
    "PS5000A_1V": 1.0, "PS5000A_2V": 2.0, "PS5000A_5V": 5.0,
    "PS5000A_10V": 10.0, "PS5000A_20V": 20.0, "PS5000A_50V": 50.0,
}

# Channel letter -> overflow bit / buffer index.
CHANNEL_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

# Threshold direction codes (PS5000A_THRESHOLD_DIRECTION).
DIRECTION_CODE = {"rising": 2, "falling": 3}

DECISION_TREE = """\
DEBUG DECISION TREE  (you are probing the BRIDGE OUTPUT, OUTA/OUTB)
------------------------------------------------------------------
1. Bridge output = clean ~190 Hz SQUARE (verdict SQUARE, f0~190, symmetric, no DC)
   -> EXPECTED & HEALTHY. This is NOT the fault. The bridge always outputs square;
      the sinusoid is the CURRENT. Add the sense-resistor current channel and re-run:
        --ch B:PS5000A_500MV:DC:<1/Rsense>:A:coil_current --vi bridge_out,coil_current
      The CURRENT should read SINE at resonance.

2. Bridge output is NOT a clean 190 Hz square
   (wrong freq / DC offset / asymmetric / missing half-cycles / distorted)
   -> FIRMWARE / DRIVE issue: carrier duty, phase/direction signal, or channel
      mapping. Capture the ESP32 carrier + phase GPIOs (3.3 V logic, safe on analog
      channels) and confirm carrier=15 kHz, phase toggles at the drive frequency.

3. Current channel added -- what does the CURRENT look like?
   * SINE, in phase with bridge output (|phase|<10 deg), ~190 Hz -> healthy tank. Done.
   * TRIANGLE  -> series cap J3/J4 MISSING or SHORTED (current = integral of square).
   * SINE but leading/lagging -> OFF-RESONANCE; retune per the printed phase direction.
"""


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class ChannelSpec:
    channel: str                 # "A".."D"
    range_name: str              # PS5000A_RANGE key
    coupling: str = "DC"         # "DC" / "AC"
    scale: float = 1.0           # multiply volts -> physical units (probe atten, 1/Rsense)
    unit: str = "V"
    label: str = ""
    enabled: bool = True

    def __post_init__(self):
        self.channel = self.channel.upper()
        if not self.label:
            self.label = f"ch{self.channel}"


@dataclass
class TriggerSpec:
    enabled: bool = True
    source_channel: str = "A"
    threshold_v: float = 0.0     # in the source channel's PHYSICAL units
    direction: str = "rising"
    auto_trigger_ms: int = 1000


@dataclass
class CaptureConfig:
    resolution_name: str = "12"
    channels: list = field(default_factory=list)
    target_sample_rate_hz: float = 200_000.0
    n_samples: int = 20_000
    pretrig_frac: float = 0.1
    trigger: TriggerSpec = field(default_factory=TriggerSpec)
    preset: str = "envelope"


@dataclass
class CaptureResult:
    time_s: np.ndarray
    data: dict                   # label -> np.ndarray in physical units
    units: dict                  # label -> unit string
    fs_hz: float = 0.0
    actual_interval_ns: float = 0.0
    overflow: dict = field(default_factory=dict)  # label -> bool (clipped)
    auto_triggered: bool = False
    config: CaptureConfig = None


@dataclass
class ChannelAnalysis:
    label: str
    unit: str
    f0_hz: float
    f0_amp: float
    rms: float
    pp: float
    thd: float
    harmonic_ratios: dict        # k -> A_k/A_1
    verdict: str
    confidence: float
    clipped: bool = False


# --------------------------------------------------------------------------- #
# Analysis layer (pure numpy/scipy -- no scope needed, unit-testable)
# --------------------------------------------------------------------------- #

def compute_fft(samples, fs, window="hann"):
    """Return (freqs, amplitude_spectrum, complex_spectrum).

    Amplitude is corrected for window coherent gain so a pure sinusoid of
    amplitude A appears as a peak of height ~A.
    """
    x = np.asarray(samples, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if window == "hann":
        w = np.hanning(n)
    else:
        w = np.ones(n)
    cg = np.sum(w) / n               # coherent gain (0.5 for Hann)
    xf = rfft(x * w)
    freqs = rfftfreq(n, d=1.0 / fs)
    mag = (2.0 / (n * cg)) * np.abs(xf)
    return freqs, mag, xf


def _peak_near(freqs, mag, f_target, tol_bins=2):
    """Peak magnitude (and its frequency) within +/- tol_bins of f_target."""
    if f_target <= 0 or f_target > freqs[-1]:
        return 0.0, f_target
    bin0 = int(np.argmin(np.abs(freqs - f_target)))
    lo = max(0, bin0 - tol_bins)
    hi = min(len(mag), bin0 + tol_bins + 1)
    local = mag[lo:hi]
    k = int(np.argmax(local))
    return float(local[k]), float(freqs[lo + k])


def measure_fundamental(freqs, mag, band=DRIVE_BAND_HZ):
    """Fundamental frequency + amplitude, with parabolic (sub-bin) refinement."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0, 0.0
    idx_band = np.where(mask)[0]
    k_rel = int(np.argmax(mag[mask]))
    k = idx_band[k_rel]
    df = freqs[1] - freqs[0]
    # 3-point parabolic interpolation on log-magnitude.
    if 0 < k < len(mag) - 1:
        a, b, c = (np.log(mag[k - 1] + 1e-30),
                   np.log(mag[k] + 1e-30),
                   np.log(mag[k + 1] + 1e-30))
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    f0 = (k + delta) * df
    return float(f0), float(mag[k])


def compute_thd(freqs, mag, f0, n_harmonics=10):
    """Return (thd_ratio, {k: A_k}) for k=1..n_harmonics."""
    amps = {}
    for k in range(1, n_harmonics + 1):
        a, _ = _peak_near(freqs, mag, k * f0, tol_bins=2)
        amps[k] = a
    a1 = amps[1] if amps[1] > 0 else 1e-30
    harm_sq = sum(amps[k] ** 2 for k in range(2, n_harmonics + 1))
    thd = float(np.sqrt(harm_sq) / a1)
    return thd, amps


def classify_waveform(thd, harmonic_amps):
    """Classify as SINE / SQUARE / TRIANGLE / DISTORTED with a confidence [0,1]."""
    a1 = harmonic_amps.get(1, 0.0)
    if a1 <= 0:
        return "NO-SIGNAL", 0.0
    r = {k: harmonic_amps[k] / a1 for k in harmonic_amps}
    even_energy = np.sqrt(r.get(2, 0) ** 2 + r.get(4, 0) ** 2 + r.get(6, 0) ** 2)

    # Odd-harmonic roll-off slope: square ~ 1/k (slope -1), triangle ~ 1/k^2 (slope -2).
    odd_k = [k for k in (3, 5, 7) if r.get(k, 0) > 1e-4]
    slope = 0.0
    if len(odd_k) >= 2:
        logk = np.log([k for k in odd_k])
        logr = np.log([r[k] for k in odd_k])
        slope = float(np.polyfit(logk, logr, 1)[0])

    if thd < 0.05:
        # Confidence grows as THD shrinks toward 0.
        return "SINE", float(np.clip(1.0 - thd / 0.05, 0.0, 1.0))

    if (0.35 < thd < 0.60 and 0.25 < r.get(3, 0) < 0.42
            and 0.13 < r.get(5, 0) < 0.27 and even_energy < 0.10):
        conf = 1.0 - (abs(r.get(3, 0) - 1 / 3) / (1 / 3)
                      + abs(r.get(5, 0) - 1 / 5) / (1 / 5)) / 2.0
        return "SQUARE", float(np.clip(conf, 0.0, 1.0))

    if (0.07 < thd < 0.18 and 0.07 < r.get(3, 0) < 0.16
            and even_energy < 0.10 and slope < -1.5):
        conf = 1.0 - abs(r.get(3, 0) - 1 / 9) / (1 / 9)
        return "TRIANGLE", float(np.clip(conf, 0.0, 1.0))

    return "DISTORTED/OTHER", 0.0


def vi_phase(v_samples, i_samples, fs, f0):
    """Phase of current relative to voltage at f0, in degrees, + regime string.

    Positive => current LEADS (capacitive, below resonance).
    """
    freqs, _, xv = compute_fft(v_samples, fs)
    _, _, xi = compute_fft(i_samples, fs)
    bin0 = int(np.argmin(np.abs(freqs - f0)))
    cross = xi[bin0] * np.conj(xv[bin0])
    phase = float(np.degrees(np.angle(cross)))

    if abs(phase) < 10.0:
        regime = "AT RESONANCE (|phase|<10 deg) -- drive frequency matches the tank."
    elif phase > 0:
        regime = ("current LEADS -> BELOW resonance (capacitive): raise drive "
                  "frequency or decrease series C.")
    else:
        regime = ("current LAGS -> ABOVE resonance (inductive): lower drive "
                  "frequency or increase series C (or the cap is missing).")
    return phase, regime


def analyze_channel(label, unit, samples, fs, clipped=False):
    """Full single-channel analysis."""
    freqs, mag, _ = compute_fft(samples, fs)
    f0, _ = measure_fundamental(freqs, mag)
    thd, amps = compute_thd(freqs, mag, f0)
    a1 = amps.get(1, 0.0)
    ratios = {k: (amps[k] / a1 if a1 > 0 else 0.0) for k in amps}
    if clipped:
        verdict, conf = "CLIPPED", 0.0
    else:
        verdict, conf = classify_waveform(thd, amps)
    x = np.asarray(samples, dtype=float)
    return ChannelAnalysis(
        label=label, unit=unit, f0_hz=f0, f0_amp=a1,
        rms=float(np.sqrt(np.mean((x - np.mean(x)) ** 2))),
        pp=float(np.ptp(x)), thd=thd, harmonic_ratios=ratios,
        verdict=verdict, confidence=conf, clipped=clipped,
    )


# --------------------------------------------------------------------------- #
# Scope-control layer (requires picosdk + native runtime)
# --------------------------------------------------------------------------- #

def _require_ps5000a():
    if not _HAVE_PS5000A:
        raise SystemExit(
            "ERROR: could not import the ps5000a driver.\n"
            f"  {type(_PS_IMPORT_ERROR).__name__}: {_PS_IMPORT_ERROR}\n"
            "Fix: `pip install picosdk` AND install the native PicoSDK runtime from\n"
            "https://www.picotech.com/downloads (the wrapper dlopen()s libps5000a).\n"
            "Do NOT install pypicosdk -- it does not support the 5000D/5443D.\n"
            "You can still run --self-test and --dry-run without the scope."
        )


def open_scope(resolution_name):
    """Open the unit, handle USB power, return (chandle, max_adc)."""
    _require_ps5000a()
    chandle = ctypes.c_int16()
    status = {}
    resolution = ps.PS5000A_DEVICE_RESOLUTION[RESOLUTION_KEYS[resolution_name]]
    status["open"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    try:
        assert_pico_ok(status["open"])
    except Exception:
        power = status["open"]
        if power in (282, 286):  # power-supply-not-connected / non-USB3 port
            status["power"] = ps.ps5000aChangePowerSource(chandle, power)
            assert_pico_ok(status["power"])
        else:
            raise
    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))
    return chandle, max_adc


def configure_channel(chandle, spec):
    ch = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{spec.channel}"]
    coupling = ps.PS5000A_COUPLING[f"PS5000A_{spec.coupling}"]
    rng = ps.PS5000A_RANGE[spec.range_name]
    assert_pico_ok(ps.ps5000aSetChannel(
        chandle, ch, int(spec.enabled), coupling, rng, 0.0))


def pick_timebase(chandle, target_fs, n_samples):
    """Scan GetTimebase2 for the timebase whose actual rate is >= target (closest).

    Returns (timebase, actual_interval_ns).
    """
    target_interval_ns = 1e9 / target_fs
    best = None          # (interval, timebase) with interval <= target (fastest enough)
    fastest = None       # smallest valid interval seen (fallback if target too fast)
    interval = ctypes.c_float()
    returned_max = ctypes.c_int32()
    for tb in range(0, 4000):
        rc = ps.ps5000aGetTimebase2(
            chandle, tb, n_samples, ctypes.byref(interval),
            ctypes.byref(returned_max), 0)
        if rc != 0:
            continue                                  # invalid at this res/depth
        if returned_max.value < n_samples:
            continue                                  # not enough capture memory
        iv = float(interval.value)
        if fastest is None or iv < fastest[0]:
            fastest = (iv, tb)
        if iv <= target_interval_ns:
            if best is None or iv > best[0]:          # largest interval still <= target
                best = (iv, tb)
        elif best is not None:
            break                                     # intervals only grow -> done
    chosen = best or fastest
    if chosen is None:
        raise SystemExit("Could not find a valid timebase for the requested "
                         "sample count -- reduce --samples or change resolution.")
    return chosen[1], chosen[0]


def set_trigger(chandle, spec, source_spec, max_adc):
    if not spec.enabled:
        return
    src = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{spec.source_channel}"]
    rng = ps.PS5000A_RANGE[source_spec.range_name]
    # threshold is in physical units -> volts -> mV -> adc.
    threshold_volts = spec.threshold_v / (source_spec.scale or 1.0)
    threshold_adc = int(mV2adc(threshold_volts * 1000.0, rng, max_adc))
    assert_pico_ok(ps.ps5000aSetSimpleTrigger(
        chandle, 1, src, threshold_adc, DIRECTION_CODE[spec.direction], 0,
        spec.auto_trigger_ms))


def run_block_capture(chandle, config, max_adc):
    """Run one block and return a CaptureResult in physical units."""
    enabled = [c for c in config.channels if c.enabled]
    timebase, interval_ns = pick_timebase(
        chandle, config.target_sample_rate_hz, config.n_samples)
    fs = 1e9 / interval_ns

    n = config.n_samples
    pre = int(n * config.pretrig_frac)
    post = n - pre

    buffers = {}
    for spec in enabled:
        bmax = (ctypes.c_int16 * n)()
        bmin = (ctypes.c_int16 * n)()
        src = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{spec.channel}"]
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            chandle, src, ctypes.byref(bmax), ctypes.byref(bmin), n, 0, 0))
        buffers[spec.label] = (bmax, spec)

    assert_pico_ok(ps.ps5000aRunBlock(
        chandle, pre, post, timebase, None, 0, None, None))

    ready = ctypes.c_int16(0)
    t0 = time.time()
    while ready.value == 0:
        ps.ps5000aIsReady(chandle, ctypes.byref(ready))
        if time.time() - t0 > max(5.0, config.trigger.auto_trigger_ms / 1000.0 + 5.0):
            raise SystemExit("Capture timed out waiting for IsReady.")
        time.sleep(0.001)

    c_samples = ctypes.c_int32(n)
    overflow = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aGetValues(
        chandle, 0, ctypes.byref(c_samples), 0, 0, 0, ctypes.byref(overflow)))
    got = c_samples.value

    data, units, ovf = {}, {}, {}
    for label, (bmax, spec) in buffers.items():
        rng = ps.PS5000A_RANGE[spec.range_name]
        mv = np.array(adc2mV(bmax, rng, max_adc)[:got], dtype=float)
        data[label] = (mv / 1000.0) * spec.scale          # volts -> physical units
        units[label] = spec.unit
        ovf[label] = bool(overflow.value & (1 << CHANNEL_INDEX[spec.channel]))

    time_s = np.arange(got) * interval_ns * 1e-9
    return CaptureResult(
        time_s=time_s, data=data, units=units, fs_hz=fs,
        actual_interval_ns=interval_ns, overflow=ovf, config=config)


def close_scope(chandle):
    try:
        ps.ps5000aStop(chandle)
    finally:
        ps.ps5000aCloseUnit(chandle)


# --------------------------------------------------------------------------- #
# Output layer
# --------------------------------------------------------------------------- #

def export_csv(result, path):
    labels = list(result.data.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# preset", result.config.preset])
        w.writerow(["# resolution_bits", result.config.resolution_name])
        w.writerow(["# sample_rate_hz", f"{result.fs_hz:.3f}"])
        w.writerow(["# actual_interval_ns", f"{result.actual_interval_ns:.3f}"])
        w.writerow(["# channels", ", ".join(f"{l}({result.units[l]})" for l in labels)])
        w.writerow(["time_s"] + labels)
        cols = [result.time_s] + [result.data[l] for l in labels]
        for row in zip(*cols):
            w.writerow([f"{row[0]:.9f}"] + [f"{v:.6g}" for v in row[1:]])
    return path


def make_figure(result, analyses, path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(result.data.keys())
    n = len(labels)
    fig, axes = plt.subplots(n + 1, 1, figsize=(10, 2.6 * (n + 1)))
    if n + 1 == 1:
        axes = [axes]
    if title:
        fig.suptitle(title)

    for ax, label in zip(axes[:n], labels):
        ax.plot(result.time_s * 1e3, result.data[label], lw=0.8)
        a = analyses[label]
        clip = "  [CLIPPED]" if a.clipped else ""
        ax.set_title(f"{label}: {a.verdict}  f0={a.f0_hz:.1f} Hz  "
                     f"THD={a.thd*100:.1f}%{clip}", fontsize=9)
        ax.set_ylabel(result.units[label])
        ax.grid(True, alpha=0.3)
    axes[n - 1].set_xlabel("time (ms)")

    axfft = axes[n]
    for label in labels:
        freqs, mag, _ = compute_fft(result.data[label], result.fs_hz)
        axfft.semilogy(freqs, mag + 1e-9, lw=0.8, label=label)
    axfft.set_xlim(0, min(5000, result.fs_hz / 2))
    axfft.set_xlabel("frequency (Hz)")
    axfft.set_ylabel("amplitude")
    axfft.set_title("Single-sided spectrum", fontsize=9)
    axfft.grid(True, alpha=0.3)
    axfft.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary(result, analyses, vi_pairs):
    print("\n" + "=" * 70)
    print(f"Capture: {result.config.preset} preset, "
          f"{result.fs_hz/1e3:.1f} kS/s, {len(result.time_s)} samples "
          f"({result.time_s[-1]*1e3:.1f} ms)")
    if result.auto_triggered:
        print("  NOTE: auto-triggered (no edge found) -- phase alignment not guaranteed.")
    print("-" * 70)
    for label, a in analyses.items():
        flag = "  <-- CLIPPED, increase range" if a.clipped else ""
        print(f"  {label:>14}: {a.verdict:<16} f0={a.f0_hz:7.1f} Hz  "
              f"THD={a.thd*100:5.1f}%  RMS={a.rms:.4g}{a.unit}  conf={a.confidence:.2f}{flag}")
        ratios = "  ".join(f"A{k}/A1={a.harmonic_ratios.get(k,0):.3f}"
                           for k in (2, 3, 5, 7) if k in a.harmonic_ratios)
        print(f"                  {ratios}")
    for v_label, i_label in vi_pairs:
        if v_label in result.data and i_label in result.data:
            f0 = analyses[i_label].f0_hz or analyses[v_label].f0_hz
            phase, regime = vi_phase(result.data[v_label], result.data[i_label],
                                     result.fs_hz, f0)
            print("-" * 70)
            print(f"  V-I phase ({i_label} vs {v_label}) @ {f0:.1f} Hz: "
                  f"{phase:+.1f} deg")
            print(f"    -> {regime}")
    print("=" * 70)
    print(DECISION_TREE)


# --------------------------------------------------------------------------- #
# Self-test (no hardware): validate the analysis math
# --------------------------------------------------------------------------- #

def self_test():
    if not _HAVE_SCIPY:
        raise SystemExit("self-test needs scipy (pip install scipy).")
    fs, f0, n = 200_000.0, 190.0, 20_000
    t = np.arange(n) / fs
    cases = {
        "sine": np.sin(2 * np.pi * f0 * t),
        "square": sp_signal.square(2 * np.pi * f0 * t),
        "triangle": sp_signal.sawtooth(2 * np.pi * f0 * t, width=0.5),
    }
    expect = {"sine": "SINE", "square": "SQUARE", "triangle": "TRIANGLE"}
    ok = True
    print("Self-test (synthetic 190 Hz signals):")
    for name, x in cases.items():
        a = analyze_channel(name, "V", x, fs)
        passed = a.verdict == expect[name]
        ok &= passed
        print(f"  {name:>9}: verdict={a.verdict:<16} f0={a.f0_hz:6.1f} "
              f"THD={a.thd*100:5.1f}%  {'OK' if passed else 'FAIL expected '+expect[name]}")
    # V-I phase: current lags voltage by 30 deg -> expect ~ -30.
    v = np.sin(2 * np.pi * f0 * t)
    i = np.sin(2 * np.pi * f0 * t - np.deg2rad(30))
    phase, _ = vi_phase(v, i, fs, f0)
    phase_ok = abs(phase - (-30.0)) < 2.0
    ok &= phase_ok
    print(f"  V-I phase: {phase:+.1f} deg (expected -30) "
          f"{'OK' if phase_ok else 'FAIL'}")
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #

def parse_channel_arg(s):
    """Parse 'A:RANGE:COUPLING:SCALE:UNIT:LABEL' (RANGE shorthand allowed)."""
    parts = s.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--ch '{s}' must be CHANNEL:RANGE[:COUPLING:SCALE:UNIT:LABEL]")
    channel = parts[0].upper()
    rng_in = parts[1]
    range_name = RANGE_SHORTHAND.get(rng_in.lower(), rng_in)
    if range_name not in RANGE_FULLSCALE_V:
        raise argparse.ArgumentTypeError(f"unknown range '{rng_in}' in --ch '{s}'")
    coupling = parts[2].upper() if len(parts) > 2 and parts[2] else "DC"
    scale = float(parts[3]) if len(parts) > 3 and parts[3] else 1.0
    unit = parts[4] if len(parts) > 4 and parts[4] else "V"
    label = parts[5] if len(parts) > 5 and parts[5] else f"ch{channel}"
    return ChannelSpec(channel=channel, range_name=range_name, coupling=coupling,
                       scale=scale, unit=unit, label=label)


def build_config(args):
    if args.channels:
        channels = args.channels
    else:  # default: bridge output on channel A
        channels = [ChannelSpec("A", "PS5000A_5V", "DC", 1.0, "V", "bridge_out")]

    if args.preset in PRESETS:
        fs, n = PRESETS[args.preset]
    else:
        fs, n = 200_000.0, 20_000
    if args.sample_rate:
        fs = args.sample_rate
    if args.samples:
        n = args.samples

    trig_src = args.trigger_channel or channels[0].channel
    trigger = TriggerSpec(
        enabled=(trig_src.upper() != "NONE"),
        source_channel=trig_src.upper(),
        threshold_v=args.trigger_level,
        direction=args.trigger_edge,
        auto_trigger_ms=args.auto_trigger_ms,
    )
    return CaptureConfig(
        resolution_name=args.resolution, channels=channels,
        target_sample_rate_hz=fs, n_samples=n, pretrig_frac=args.pretrig,
        trigger=trigger, preset=args.preset)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="PicoScope 5443D capture + sine/square/triangle analysis.")
    p.add_argument("--resolution", choices=list(RESOLUTION_KEYS), default="12")
    p.add_argument("--preset", choices=["envelope", "carrier", "custom"],
                   default="envelope")
    p.add_argument("--sample-rate", type=float, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--pretrig", type=float, default=0.1)
    p.add_argument("--ch", dest="channels", action="append", type=parse_channel_arg,
                   metavar="A:RANGE:COUPLING:SCALE:UNIT:LABEL",
                   help="repeatable; default = A:PS5000A_5V:DC:1:V:bridge_out")
    p.add_argument("--trigger-channel", default=None)
    p.add_argument("--trigger-level", type=float, default=0.0)
    p.add_argument("--trigger-edge", choices=["rising", "falling"], default="rising")
    p.add_argument("--auto-trigger-ms", type=int, default=1000)
    p.add_argument("--vi", dest="vi", action="append", default=[],
                   metavar="V_LABEL,I_LABEL",
                   help="compute V-I phase between two channel labels (repeatable)")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument("--self-test", action="store_true",
                   help="validate analysis on synthetic signals (no scope)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate config + chosen timebase, no acquisition")
    p.add_argument("--list-units", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    config = build_config(args)
    vi_pairs = [tuple(s.split(",")) for s in args.vi]

    if args.list_units:
        chandle, max_adc = open_scope(config.resolution_name)
        print(f"Opened 5000D unit OK. maxADC={max_adc.value}, "
              f"resolution={config.resolution_name}-bit.")
        close_scope(chandle)
        return 0

    if args.dry_run:
        print("DRY RUN -- configuration:")
        print(f"  resolution: {config.resolution_name}-bit  preset: {config.preset}")
        print(f"  target fs : {config.target_sample_rate_hz:.0f} Hz  "
              f"samples: {config.n_samples}")
        for c in config.channels:
            print(f"  channel {c.channel}: {c.range_name} {c.coupling} "
                  f"scale={c.scale} unit={c.unit} label={c.label}")
        print(f"  trigger   : {config.trigger}")
        if _HAVE_PS5000A:
            chandle, max_adc = open_scope(config.resolution_name)
            for c in config.channels:
                configure_channel(chandle, c)
            tb, iv = pick_timebase(chandle, config.target_sample_rate_hz,
                                   config.n_samples)
            print(f"  -> timebase {tb}, actual interval {iv:.1f} ns "
                  f"({1e9/iv/1e3:.1f} kS/s)")
            close_scope(chandle)
        else:
            print("  (ps5000a not available -- skipping device timebase query)")
        return 0

    # Real capture.
    chandle, max_adc = open_scope(config.resolution_name)
    try:
        for c in config.channels:
            configure_channel(chandle, c)
        set_trigger(chandle, config.trigger, config.channels[0], max_adc)
        result = run_block_capture(chandle, config, max_adc)
    finally:
        close_scope(chandle)

    analyses = {l: analyze_channel(l, result.units[l], result.data[l],
                                   result.fs_hz, clipped=result.overflow.get(l, False))
                for l in result.data}

    prefix = args.out_prefix or f"picoscope_{config.preset}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not args.no_csv:
        print("CSV  ->", export_csv(result, prefix + "_results.csv"))
    if not args.no_plot:
        try:
            print("PLOT ->", make_figure(result, analyses, prefix + ".png", args.title))
        except Exception as exc:
            print(f"(plot skipped: {exc})")

    print_summary(result, analyses, vi_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
