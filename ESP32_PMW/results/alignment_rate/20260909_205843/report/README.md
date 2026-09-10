# Alignment-rate campaign report — 2026-09-09/10

How fast the robot re-orients after coils A and C are cut, against drive frequency.
Every file here is regenerated from the takes; nothing in this directory is a capture.

All commands run from the repo root, against `results/alignment_rate/20260909_205843`.

| file | made by |
|---|---|
| `campaign.csv` | `alignment_rate.py --campaign <root>` — the authoritative per-frequency table |
| `rate_vs_frequency.png` | same command |
| `transients.png` | same command |
| `rate_scatter.png` | `alignment_rate.py --scatter <root>` — swing and rate, every repeat |
| `trial_fits.png` | `alignment_rate.py --trials <root>` — one median-rate trial per frequency |
| `trials_0NNhz.png` | `alignment_rate.py --trials <root> --only-hz NN` — EVERY repeat at that frequency |
| `rate_fits.png` | `alignment_rate.py --fits <root>` — a DIAGNOSTIC, not a result: what the same rule reads off the POOLED curve, against the per-repeat median |
| `coupling.png` | `coupling.py <root>` — radial/azimuth shared lines and what deprojecting them does |

## Stale — do not read

`summary.csv` and `transients.csv` are dated 2026-09-09 21:18 and were written by
`alignment_rate.py <take>` (the single-take path) before the campaign existed. They predate
the 2026-09-10 rewrite of the rate metric and disagree with `campaign.csv` on every number.
Kept only because nothing in `results/` is deleted. **`campaign.csv` is the current table.**

## What the rate means, as of 2026-09-10

The metric is the gradient of the response's rising edge. Find the first swing that reaches
`PEAK_FRAC` (30%) of the record's height in the second after the cut; the edge runs from the
last visit to the bottom 10% band up to the FIRST entry into the top 10% band, so a response
that reaches its level and then sits there contributes its rise and not its plateau. Trim
`BUFFER_FRAC` off each end, fit the middle, and take the dead time as where that fitted line,
run backwards, crosses the level the trace held before it.

See `alignment_rate.relu_window` for why each clause is what it is — every one of them is a
failure that was found in these takes and fixed, and each names the take that found it.
`PEAK_FRAC` in particular is not a judgement call: it was scanned over all 101 kill windows
and 0.30 sits mid-plateau (fits landing later than 200 ms after the cut go 5, 4, 3, 2, 1, 1,
1, 1 as it falls 0.70 -> 0.20, while the number of windows fitted stays at 89-92).

**Pooled and per-repeat rates are both available, and they are not the same number.**
`rate_fits.png` runs the identical estimator on the pooled curve; `campaign.csv` and
`rate_scatter.png` report per-repeat. Pooling reads **19% of the per-repeat rate** (range 11%
to 35%), because repeats do not share a dead time — 33-102 ms across this campaign — so
averaging smears an edge that is itself only ~50 ms wide, and the pooled edge begins a median
of 40 ms before the cut.

The consequence that matters for reading the sweep: **pooling also flattens the frequency
dependence.** Per-repeat the rate runs 353-2012 deg/s with a clear rise to 80 Hz; pooled it
sits at 139-256 deg/s across 20-110 Hz with almost no structure. Quote one or the other
consistently, and say which.

The negative-lag refusal (`LAG_TOL_S`) is a PER-REPEAT precondition — it asks whether that
run's rise was already underway at the cut — so it is disabled for pooled curves via
`transient(pooled=True)`. On an average a negative lag is the smearing, not evidence that any
run responded early.

The angle plotted is the **separation between the robot's lean direction now and its
pre-cut lean direction**, an `arccos` of two unit vectors. It is in [0, 180] and never
negative; there is no unwrap and no branch anywhere in it, so it cannot flip sign.

## Trials that are refused, and why

A repeat reports no rate when the record cannot support one. These are flight conditions,
not analysis failures, and they are worth reading before deciding what to re-run:

* the fitted ramp's onset extrapolates to before the cut — the rise was already underway
  when the coils dropped, so it is not the response to them (e.g. 50 Hz `2026-09-09_234425`)
* the trace falls below its pre-cut level within 0.4 s of the cut — the pre-cut window was
  not a settled reference (e.g. 10 Hz `2026-09-09_210122`)
* the pre-cut lean is below 5 deg, so there is no lean DIRECTION to measure a change of
  (60 Hz `2026-09-10_000732`: median pre-cut lean 2.9 deg, 3 of 206 samples usable)
* the robot was already swinging hard at the cut (60 Hz `2026-09-09_212930`: 12.7 deg of
  scatter on the *smoothed* pre-cut trace)

## Known caveats

* **40 Hz mixes two hold times.** Reps 11-14 (`073235`-`073606`) flew the 0.5 s frequency
  hold; reps 15-19 (`073823`-`074326`) flew the original. Not like-for-like.
* **A few repeats respond LATE.** 60 Hz `2026-09-09_213341` and `2026-09-10_000608` sit
  still for 145-190 ms and then climb; 30 Hz `2026-09-10_010923` takes 146 ms. Seven of the
  101 windows have a dead time over 120 ms against a median of 59. They are fitted, not
  discarded -- but whether they are a second mode or a bad cut is not yet established.
* **70 Hz has 2 usable repeats.** Any number quoted from it is a pair, not a statistic.
* Takes solved below twice their drive frequency are refused outright and listed by the
  `--campaign` run; the once-per-rev wobble aliases below that.
