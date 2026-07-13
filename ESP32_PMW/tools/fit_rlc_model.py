#!/usr/bin/env python3
"""Joint fit of per-channel R_i, L_i, C_i (series coil resistance,
inductance, and the PCB's series coil capacitor -- see
docs/PCB_Design_Documentation.md sec.6) from a solo-channel frequency sweep
(tools/gen_solo_sweep_experiment.py + a main_experiment.cpp telemetry log).

Physics: at fixed duty, |Z_i(omega)| = V_fund(omega) / I_meas(omega) =
sqrt(R_i^2 + (omega*L_i - 1/(omega*C_i))^2), where V_fund = (4/pi) *
(duty/100) * V_supply is the square-wave fundamental of the chopped drive.
R, L, C are fit JOINTLY, not measured as separate low/high-frequency limits:
because of the series cap, near-DC is capacitance-dominated (the cap blocks
DC), not resistance-dominated, so R can't be isolated from a low-frequency
point alone -- it's only cleanly exposed right at resonance where the
reactances cancel (|Z|=R). See the plan's "Pre-flight corrections" section.

Usage:
  uv run python tools/fit_rlc_model.py serial.log --v-supply 11.9 --duty 30 \
      --channels A B C D --out tools/rlc_fit.json --plot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import RESONANCE_BAND_HZ, parse_experiment_log

_LABEL_RE = re.compile(r"SWEEP_([ABCD])_F([\d.]+)")

# CurrentSense.h's EMA time constant (default tauFilterMs=50.0) -- the solo
# sweep's commutation frequency (1-210Hz) is in-band with this filter's own
# ~3.2Hz cutoff, so i_meas at sweep frequencies above a few Hz is the TRUE
# coil current attenuated by this filter, not the coil current alone. Fitting
# z_model against raw i_meas without this term silently absorbs the filter's
# rolloff into R/L/C, producing self-consistent but physically wrong values
# (confirmed against hardware: uncorrected fit gave L/C ~20x/~12x off known
# nominal component values; correcting for this filter brought both within
# ~2x). This is NOT a fitted parameter -- it's a known firmware constant.
CS_FILTER_TAU_S = 0.050


def extract_sweep_points(data: dict, channel: str, settle_frac: float = 0.5):
    """For labels 'SWEEP_{channel}_F{freq}', keep the LAST settle_frac of each
    contiguous label run (post-transient, closer to steady state) and return
    (freqs_hz, i_mean, duty_mean) arrays sorted by frequency."""
    labels = data["label"]
    i_col = data[f"i_{channel.lower()}"]
    d_col = data[f"d_{channel.lower()}"]
    freqs, i_means, d_means = [], [], []
    idx = 0
    n = len(labels)
    while idx < n:
        m = _LABEL_RE.match(labels[idx])
        if not (m and m.group(1) == channel):
            idx += 1
            continue
        f_val = float(m.group(2))
        j = idx
        while j < n and labels[j] == labels[idx]:
            j += 1
        k0 = idx + int((j - idx) * settle_frac)
        if k0 < j:
            freqs.append(f_val)
            i_means.append(float(np.mean(i_col[k0:j])))
            d_means.append(float(np.mean(d_col[k0:j])))
        idx = j
    order = np.argsort(freqs)
    return np.array(freqs)[order], np.array(i_means)[order], np.array(d_means)[order]


def z_model(omega, R, L, C):
    return np.sqrt(R ** 2 + (omega * L - 1.0 / (omega * C)) ** 2)


def _cs_filter_gain(omega):
    """Magnitude response of CurrentSense's single-pole EMA at CS_FILTER_TAU_S,
    the sensor-side attenuation that i_meas has already been through."""
    return 1.0 / np.sqrt(1.0 + (omega * CS_FILTER_TAU_S) ** 2)


def _i_model(xdata, R, L, C):
    omega, v_fund = xdata
    i_true = v_fund / z_model(omega, R, L, C)
    return i_true * _cs_filter_gain(omega)


def fit_channel(freqs_hz, i_meas_a, duty_pct, v_supply,
                 p0=(1.5, 0.02, 30e-6)):
    """Fit R, L, C directly against MEASURED CURRENT (not impedance
    Z=V/I). Fitting Z blows up the low-frequency/high-impedance corner
    (tiny, ADC-noise-dominated currents there get divided into huge Z
    values) and an unweighted least-squares fit ends up dominated by
    exactly the noisiest, least-relevant points -- while the near-resonance
    region (largest, best-measured currents, and the thing this model
    actually needs to get right) gets swamped. Fitting current directly
    keeps the residual scale tied to what's actually measured, so the
    largest-magnitude (best-SNR, near-resonance) points naturally dominate
    the unweighted fit instead."""
    omega = 2 * np.pi * np.asarray(freqs_hz)
    v_fund = (4.0 / np.pi) * (np.asarray(duty_pct) / 100.0) * v_supply
    i_meas_a = np.asarray(i_meas_a)
    popt, _pcov = curve_fit(_i_model, (omega, v_fund), i_meas_a, p0=p0,
                             bounds=(0, np.inf), maxfev=20000)
    R, L, C = popt
    f0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
    resid_rms = float(np.sqrt(np.mean((_i_model((omega, v_fund), *popt) - i_meas_a) ** 2)))
    return {"R": float(R), "L": float(L), "C": float(C), "f0_hz": float(f0),
            "resid_rms_a": resid_rms, "n_points": int(len(freqs_hz))}


def _plot_channel(ch, freqs, i_meas, duty, v_supply, fit, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    omega = 2 * np.pi * freqs
    v_fund = (4.0 / np.pi) * (duty / 100.0) * v_supply
    # i_meas has already been through the CS EMA filter -- divide its gain
    # back out so this plotted "measured |Z|" is comparable to the pure-coil
    # z_model curve (see CS_FILTER_TAU_S note above _i_model).
    z_meas = v_fund / np.maximum(i_meas, 1e-3) * _cs_filter_gain(omega)
    omega_fine = 2 * np.pi * np.linspace(freqs.min(), freqs.max(), 400)
    z_fit = z_model(omega_fine, fit["R"], fit["L"], fit["C"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(freqs, z_meas, "o", label="measured |Z|")
    ax.plot(omega_fine / (2 * np.pi), z_fit, "-", label="fit")
    ax.axvline(fit["f0_hz"], color="r", linestyle="--", label=f"f0={fit['f0_hz']:.1f}Hz")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [ohm]")
    ax.set_title(f"channel {ch}: R={fit['R']:.3f} L={fit['L']*1e3:.2f}mH "
                 f"C={fit['C']*1e6:.2f}uF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--v-supply", type=float, required=True,
                     help="bench supply voltage AT TEST TIME -- log it, don't "
                          "assume the 12V nominal design spec")
    ap.add_argument("--channels", nargs="+", default=list("ABCD"), choices=list("ABCD"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "rlc_fit.json"))
    ap.add_argument("--plot", action="store_true",
                     help="write <log>_rlc_<channel>.png per channel -- a bad "
                          "fit should be visually obvious before it's trusted")
    args = ap.parse_args()

    data = parse_experiment_log(args.log)
    results = {}
    for ch in args.channels:
        freqs, i_meas, duty = extract_sweep_points(data, ch)
        if len(freqs) < 4:
            print(f"channel {ch}: not enough sweep points ({len(freqs)}), skipping")
            continue
        fit = fit_channel(freqs, i_meas, duty, args.v_supply)
        results[ch] = fit
        lo, hi = RESONANCE_BAND_HZ
        in_band = lo <= fit["f0_hz"] <= hi
        print(f"{ch}: R={fit['R']:.3f} ohm  L={fit['L']*1e3:.2f} mH  "
              f"C={fit['C']*1e6:.2f} uF  f0={fit['f0_hz']:.1f} Hz "
              f"({'IN' if in_band else 'NOT in'} the {lo:.0f}-{hi:.0f} Hz "
              f"resonance band)  resid={fit['resid_rms_a']*1e3:.1f} mA "
              f"(n={fit['n_points']})")
        if args.plot:
            out_png = args.log.rsplit(".", 1)[0] + f"_rlc_{ch}.png"
            _plot_channel(ch, freqs, i_meas, duty, args.v_supply, fit, out_png)
            print(f"  wrote {out_png}")

    if not results:
        raise SystemExit("no channel had enough sweep points to fit -- check "
                          "the log and --channels")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    print("note: fitted R includes VNH5019 bridge drop + dead-time effects, "
          "not pure coil copper resistance")


if __name__ == "__main__":
    main()
