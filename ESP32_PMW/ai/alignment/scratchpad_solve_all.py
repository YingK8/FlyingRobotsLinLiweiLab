#!/usr/bin/env python3
"""Solve every good take of the campaign that has no `axis_minor.csv` yet.

Separate from `alignment_rate --sweep` so it can run alongside the sweep and be restarted
without redoing work: it skips anything already solved.
"""

import csv
import sys
from pathlib import Path

from controller.pose import disc_axis

ROOT = Path("results/alignment_rate/20260909_205843")
STRIDE = 4

rows = [r for f in sorted(ROOT.glob("*/index.csv")) for r in csv.DictReader(open(f))]
todo = []
for r in rows:
    if r["outcome"] != "ok":
        continue
    take = Path(r["flight"])
    if (take / "axis_minor.csv").exists():
        continue
    if not (take / "meta.json").exists():
        continue
    todo.append((r["freq_hz"], take))

print(f"{len(todo)} takes to solve (stride {STRIDE})", flush=True)
for i, (f, take) in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {f} Hz {take.name}", flush=True)
    try:
        disc_axis.solve(take, progress=False, stride=STRIDE)
    except Exception as e:                      # a corrupt take must not stop the batch
        print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
print("done", flush=True)
