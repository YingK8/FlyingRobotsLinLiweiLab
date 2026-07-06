#!/usr/bin/env python3
"""Extract the mutual-coupling matrix from a pairwise coupling sweep.

Firmware sequence (main_coupling_test.cpp): 4 solos, 6 pairs, ALL, one loop.
Each coil's current SHIFT in a pair vs its solo current is caused by the
partner's mutual EMF: same-sign shift on both coils = reactive coupling
(M*cos dphi), opposite-sign = real-power transfer (M*sin dphi, the effect that
imbalances the supplies). Magnitude of the shift ~ the coupling coefficient.

Handles the scope<->firmware channel swap (B<->D) found empirically.
Usage: uv run python tools/coupling_matrix.py CSV [--t0 SEC]
"""
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--t0", type=float, default=12.0, help="center of solo-A burst (s)")
ap.add_argument("--slot", type=float, default=5.0, help="segment slot length (s)")
ap.add_argument("--half", type=float, default=0.8, help="half-window to average (s)")
args = ap.parse_args()

df = pd.read_csv(args.csv, skiprows=[1]); t = df["Time"].values
scope = ["Channel A","Channel B","Channel C","Channel D"]
dt = np.median(np.diff(t)); fs = 1/dt
w = int(round(0.3*fs))
def rms(x): x=x-np.mean(x); return np.sqrt(pd.Series(x*x).rolling(w,1,center=True).mean().values)*1000
R = {c: rms(df[c].values) for c in scope}

# firmware coil -> scope channel: AUTO-DETECTED from the 4 solo segments (robust
# to any coil<->channel rewiring). Each solo drives one firmware coil; whichever
# scope channel dominates that window is where that coil's current sense is wired.
FW2SCOPE = {}
_used = set()
for _k, _coil in enumerate("ABCD"):          # first 4 firmware segments = solos
    _c = args.t0 + args.slot * _k
    _m = (t >= _c - args.half) & (t <= _c + args.half)
    _means = {c: np.nanmean(R[c][_m]) for c in scope}
    _best = max((c for c in scope if c not in _used), key=lambda c: _means[c])
    FW2SCOPE[_coil] = _best; _used.add(_best)
print("detected firmware->scope map:", {k: v[-1] for k, v in FW2SCOPE.items()})
# firmware drive phases (INITIAL_PHASES {A,B,C,D} indices) for interpretation
PHASE = {"A":270.0,"B":90.0,"C":180.0,"D":0.0}
# segment order in firmware
SEG = [["A"],["B"],["C"],["D"],
       ["A","B"],["A","C"],["A","D"],["B","C"],["B","D"],["C","D"],
       ["A","B","C","D"]]

def cur(coil, center):
    m = (t >= center-args.half) & (t <= center+args.half)
    return float(np.nanmean(R[FW2SCOPE[coil]][m]))

# measure current of each driven coil in each segment
meas = []
for k, seg in enumerate(SEG):
    c = args.t0 + args.slot*k
    meas.append({coil: cur(coil, c) for coil in seg})

solo = {SEG[i][0]: meas[i][SEG[i][0]] for i in range(4)}
print(f"{args.csv}")
print("solo currents (mV):", {k: round(v) for k,v in solo.items()})

print(f"\n{'pair':>6}{'dphi':>6} | per-coil shift vs solo        | type")
coup = {}   # symmetric coupling strength per pair
for k in range(4, 10):
    i, j = SEG[k]
    ci, cj = meas[k][i], meas[k][j]
    si, sj = ci/solo[i]-1, cj/solo[j]-1
    dphi = abs((PHASE[i]-PHASE[j]+180) % 360 - 180)
    typ = "reactive (same-sign)" if si*sj > 0 else "REAL-POWER transfer (opp-sign)"
    print(f"{i+'+'+j:>6}{dphi:>5.0f}° | {i}:{si*100:+5.1f}%  {j}:{sj*100:+5.1f}%   | {typ}")
    coup[(i,j)] = (abs(si)+abs(sj))/2*100

# ALL cross-check
allc = {c: cur(c, args.t0+args.slot*10) for c in "ABCD"}
print("\nALL segment (mV):", {k: round(v) for k,v in allc.items()},
      " -> A change vs solo:", f"{allc['A']/solo['A']-1:+.0%}")

# coupling matrix (symmetric, % mean |shift|)
coils = ["A","B","C","D"]
M = np.full((4,4), np.nan)
for (i,j),v in coup.items():
    a,b = coils.index(i), coils.index(j)
    M[a,b]=M[b,a]=v
print("\ncoupling strength matrix (mean |current shift| %, higher = stronger):")
print("      "+"".join(f"{c:>7}" for c in coils))
for a,c in enumerate(coils):
    print(f"  {c}  "+"".join(f"{M[a,b]:7.1f}" if not np.isnan(M[a,b]) else f"{'-':>7}" for b in range(4)))

# plot: matrix heatmap + shift bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
im = ax1.imshow(np.nan_to_num(M), cmap="magma", vmin=0)
ax1.set_xticks(range(4)); ax1.set_xticklabels(coils)
ax1.set_yticks(range(4)); ax1.set_yticklabels(coils)
for a in range(4):
    for b in range(4):
        if not np.isnan(M[a,b]): ax1.text(b,a,f"{M[a,b]:.0f}%",ha="center",va="center",color="w")
ax1.set_title("Coupling strength (mean |current shift| %)")
fig.colorbar(im, ax=ax1, label="%")

pairs=[f"{i}+{j}" for (i,j) in coup]; vals=[coup[k] for k in coup]
dphis=[abs((PHASE[i]-PHASE[j]+180)%360-180) for (i,j) in coup]
colors=["#4e79a7" if abs(d-90)<1 else "#e15759" for d in dphis]
ax2.bar(pairs, vals, color=colors)
ax2.set_ylabel("mean |shift| %"); ax2.set_title("blue = 90° phase (power transfer)   red = 180° (reactive)")
ax2.grid(alpha=0.3, axis="y")
fig.tight_layout()
out = args.csv.rsplit(".",1)[0]+"_matrix.png"; fig.savefig(out, dpi=110)
print("\nwrote", out)
