#!/usr/bin/env python3
"""Analyze a coupling-sweep capture (solo A/B/C/D + ALL segments).

Detects each drive segment by which scope channel dominates, reports the driven
vs undriven RMS current, and plots the labeled RMS envelope. The current on an
undriven coil (vs the all-off noise floor) is the mutually-coupled current;
the solo->ALL current redistribution is the coupling seen on driven coils.
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

csv = sys.argv[1] if len(sys.argv) > 1 else "coupling_sweep_20260704_142357.csv"
df = pd.read_csv(csv, skiprows=[1]); t = df["Time"].values
chs = ["Channel A","Channel B","Channel C","Channel D"]
col = {"Channel A":"#e15759","Channel B":"#4e79a7","Channel C":"#59a14f","Channel D":"#f28e2b"}
dt = np.median(np.diff(t)); fs = 1/dt
w = int(round(0.2*fs))
def rms(x): x=x-np.mean(x); return np.sqrt(pd.Series(x*x).rolling(w,1,center=True).mean().values)*1000
R = {c: rms(df[c].values) for c in chs}

# noise floor from the all-off tail (last 4 s)
tail = t >= t[-1]-4
floor = {c: np.nanmedian(R[c][tail]) for c in chs}
print("all-off noise floor (mV):", {c[-1]:round(floor[c],1) for c in chs})

# find solo segments: windows where exactly one channel is >3x floor
THRESH = {c: 3*floor[c] for c in chs}
active = np.array([[R[c][i] > THRESH[c] for c in chs] for i in range(len(t))])
nact = active.sum(1)
# label contiguous runs
segs=[]; i=0
while i < len(t):
    if nact[i]==0: i+=1; continue
    j=i
    while j<len(t) and nact[j]>=1 and (active[j]==active[i]).all(): j+=1
    if (j-i) > 0.5*fs:   # >0.5s
        mask=active[i]; segs.append((t[i],t[j-1],mask))
    i=j

print(f"\ndetected {len(segs)} drive segments:")
print(f"  {'t0':>5}{'t1':>6}  driven      | undriven-coil RMS (mV) vs floor")
solo_rows=[]
for t0,t1,mask in segs:
    m=(t>=t0)&(t<=t1)
    vals={c:np.nanmean(R[c][m]) for c in chs}
    driven=[chs[k] for k in range(4) if mask[k]]
    dn=",".join(d[-1] for d in driven)
    others=" ".join(f"{c[-1]}={vals[c]:5.0f}(+{vals[c]-floor[c]:4.0f})"
                     for c in chs if c not in driven)
    print(f"  {t0:5.0f}{t1:6.0f}  {dn:11s} | {others}")
    if len(driven)==1: solo_rows.append((driven[0],vals))

# solo vs ALL redistribution
allsegs=[s for s in segs if s[2].sum()==4]
if allsegs:
    t0,t1,_=allsegs[0]; m=(t>=t0)&(t<=t1)
    allv={c:np.nanmean(R[c][m]) for c in chs}
    print("\nsolo vs ALL (driven-current redistribution = coupling on driven coils):")
    print(f"  {'coil':>6}{'solo':>8}{'ALL':>8}{'change':>9}")
    solo_by={d:v for d,v in solo_rows}
    for c in chs:
        if c in solo_by:
            sv=solo_by[c][c]; av=allv[c]
            print(f"  {c[-1]:>6}{sv:8.0f}{av:8.0f}{(av/sv-1)*100:+8.0f}%")

# plot
fig,ax=plt.subplots(figsize=(13,6))
for c in chs: ax.plot(t,R[c],lw=1.0,color=col[c],label=c)
for t0,t1,mask in segs:
    dn="".join(chs[k][-1] for k in range(4) if mask[k])
    ax.axvspan(t0,t1,color="gray",alpha=0.08)
    ax.text((t0+t1)/2, ax.get_ylim()[1]*0.95, dn, ha="center", va="top", fontsize=8)
ax.set_title(f"Coupling sweep — {csv}\nsolo bursts (one coil driven) + ALL; flat undriven = no coupled current (open coils)")
ax.set_xlabel("time (s)"); ax.set_ylabel("RMS current sense (mV)")
ax.set_xlim(0,t[-1]); ax.grid(alpha=0.3); ax.legend(ncol=4,fontsize=9)
out=csv.rsplit(".",1)[0]+"_analysis.png"; fig.tight_layout(); fig.savefig(out,dpi=110)
print("\nwrote",out)
