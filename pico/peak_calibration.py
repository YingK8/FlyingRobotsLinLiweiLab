"""PEAK current-sense calibration for magnetic-field use.

We want PEAK coil current (B-field amplitude ~ N * I_peak), so every metric here is a
PEAK, not an average or RMS:

    I_peak(V_CS) = g_peak * (V_CS_peak - offset)

  offset       : zero-current CS voltage (ramp baseline, drive off)
  V_CS_peak    : peak of the CS envelope (15 kHz carrier removed, 190 Hz crest)
  g_peak [A/V] : calibrated against PEAK current-probe current, NOT supply current
  K_eff        : R_shunt * g_peak

------------------------------------------------------------------------------------
TWO WAYS TO RUN
------------------------------------------------------------------------------------
MODE A  (quick, two-point): measure PEAK coil current with a series current probe at
        100% duty for each channel, type it into PROBE_PEAK_100 below. Gain comes from
        that point + the ramp offset. Run: `python3 peak_calibration.py`

MODE B  (full multipoint): capture ONE synchronized CSV per channel while ramping duty,
        with columns: Time, CS (V), Probe (A).  Point PROBE_SWEEP at it. The script
        low-passes CS to kill the 15 kHz carrier, takes the PEAK envelope of both CS and
        probe per 190 Hz cycle, and least-squares fits g_peak + offset over the sweep.
        Run: `python3 peak_calibration.py sweep`
------------------------------------------------------------------------------------
"""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt

CH     = ["Channel A", "Channel B", "Channel C", "Channel D"]
COLORS = {"Channel A":"C0","Channel B":"C1","Channel C":"C2","Channel D":"C3"}
R      = {"Channel A":2532.,"Channel B":2530.,"Channel C":2543.,"Channel D":2540.}

# ---- MODE A inputs: PEAK coil current (A) from a series probe at 100% duty ----
# Fill these in once you measure them. Leave as None to skip a channel.
# NOTE: provisional — using single-point SUPPLY currents (averages) as a stand-in for the
# probe PEAK current. Replace with true series-probe peak readings for a valid field cal.
PROBE_PEAK_100 = {"Channel A": 5.466, "Channel B": 3.618, "Channel C": 5.234, "Channel D": 5.247}

# ---- MODE B input: synchronized sweep CSV (Time, CS V, Probe A) ----
PROBE_SWEEP = "cs_probe_sweep.csv"   # create this from a PicoScope capture

RAMP = "seperate_ground_10sec_linear_carrier_ramp_15kHz_ABCD.csv"
W = 41
def peak_env(y):  # upper (peak) envelope
    return pd.Series(y).rolling(W,center=True,min_periods=1).max() \
             .rolling(W,center=True,min_periods=1).mean().values


def mode_A():
    """Two-point peak calibration: ramp gives offset + V_CS peak plateau; probe gives I_peak."""
    df = pd.read_csv(RAMP, skiprows=[1]); t = df["Time"].values
    print("MODE A — two-point PEAK calibration (probe @100% duty + ramp offset)\n")
    print(f"{'ch':3}{'offset mV':>10}{'Vcs_pk V':>10}{'I_pk(A)':>9}{'g_peak A/V':>12}{'K_eff':>8}")
    any_done = False
    for ch in CH:
        offset = float(np.median(df[ch].values[t < 1.0]))
        vpk    = float(np.median(peak_env(df[ch].values)[t > t.max()-2.0]))  # plateau peak
        ipk    = PROBE_PEAK_100[ch]
        if ipk is None:
            print(f"{ch[-1]:3}{offset*1000:10.1f}{vpk:10.4f}{'--':>9}{'(fill probe)':>12}{'':>8}")
            continue
        g = ipk / (vpk - offset)
        print(f"{ch[-1]:3}{offset*1000:10.1f}{vpk:10.4f}{ipk:9.2f}{g:12.2f}{R[ch]*g:8.0f}")
        any_done = True
    if not any_done:
        print("\n>> No probe peak currents entered yet. Measure PEAK coil current with a")
        print(">> series current probe at 100% duty for each channel and fill PROBE_PEAK_100.")


def mode_B():
    """Multipoint peak calibration from a synchronized CS+probe sweep CSV."""
    try:
        raw = pd.read_csv(PROBE_SWEEP)
    except FileNotFoundError:
        print(f"MODE B — sweep file '{PROBE_SWEEP}' not found.\n")
        print("Capture one CSV per channel (or a combined one) with columns:")
        print("    Time (s),  CS (V),  Probe (A)")
        print("while ramping duty 0->100%. Then re-run: python3 peak_calibration.py sweep")
        return
    # expect columns: Time, CS, Probe (case-insensitive, first matching)
    def col(key): return next(c for c in raw.columns if key in c.lower())
    cs = raw[col("cs")].values; ip = raw[col("probe")].values
    cs_pk = peak_env(cs); ip_pk = peak_env(ip)
    m = ip_pk > 0.05*np.nanmax(ip_pk)               # use the meaningful current range
    # fit V_CS_peak = ip/g + offset   ->  cs_pk = a*ip_pk + b ; g = 1/a
    a, b = np.polyfit(ip_pk[m], cs_pk[m], 1)
    g = 1.0/a; offset = b
    r2 = 1 - np.var(cs_pk[m]-(a*ip_pk[m]+b))/np.var(cs_pk[m])
    print("MODE B — multipoint PEAK calibration\n")
    print(f"  g_peak = {g:.3f} A/V   offset = {offset*1000:.1f} mV   R^2 = {r2:.4f}")
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(ip_pk[m], cs_pk[m], ".", ms=3, alpha=0.5)
    xx = np.array([0, np.nanmax(ip_pk)])
    ax.plot(xx, a*xx+b, "k-", lw=1.5, label=f"Vcs = {offset*1000:.0f}mV + I/{g:.2f}")
    ax.set_xlabel("Probe PEAK current (A)"); ax.set_ylabel("CS PEAK envelope (V)")
    ax.set_title(f"Peak CS calibration (R^2={r2:.3f})"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("peak_calibration_fit.png", dpi=140)
    print("saved peak_calibration_fit.png")


if __name__ == "__main__":
    (mode_B if (len(sys.argv) > 1 and sys.argv[1] == "sweep") else mode_A)()
