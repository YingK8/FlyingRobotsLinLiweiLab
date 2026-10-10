import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("20260617.csv", skiprows=[1])
f = df["Frequency"].values
def sm(y,w=201): return pd.Series(y).rolling(w,center=True,min_periods=1).median().values
S={c:sm(df[c].values) for c in ["Channel A","Channel B","Channel C","Channel D"]}

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5),sharey=True)
a1.set_title("Board 1: A & C (should match)")
a1.plot(f,S["Channel A"],label="A",color="C0")
a1.plot(f,S["Channel C"],label="C",color="C2")
a2.set_title("Board 2: B & D (should match)")
a2.plot(f,S["Channel B"],label="B",color="C1")
a2.plot(f,S["Channel D"],label="D",color="C3")
for a in (a1,a2):
    a.axvspan(200,250,color="k",alpha=0.08,label="200-250 Hz")
    a.set_xlim(0,1000); a.set_xlabel("Frequency (Hz)"); a.grid(alpha=0.3); a.legend()
a1.set_ylabel("Current (dBu)")
fig.tight_layout(); fig.savefig("20260617_boards.png",dpi=140)
print("saved 20260617_boards.png")
