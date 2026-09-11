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
| `settling.csv` | `alignment_rate.py --settle <root>` — per repeat: resting axes, settling time, rise time, coning |
| `settling_by_freq.csv` | same command — the per-frequency medians |
| `axis_vs_time_0NNhz.png` | same command — settling, coning envelope, and the cone's shape, every repeat at that frequency |
| `settle_vs_frequency.png` | same command — settling time, swing and coning against drive |
| `settle_report.html` | `settle_report.py <root>` — the standalone report, every figure inlined |
| `settle_step_response.png` | same command — one repeat marked: t_c, initial average, final average, ±10% band |
| `settle_coning_overlay.png` | same command — the disc drawn coning about its average axis, swing removed |
| `settle_coning_time.png` | same command — the coning's own settling time and its fitted decay |
| `settle_spiral.png` | same command — the residual coloured by time; the coning spiralling in |
| `settle_envelope_heatmap.png` | same command — coning half-angle against time and drive |
| `settle_metrics.png` | same command — rate, rise time and settling time side by side |

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

## Regenerated 2026-09-10: five takes recovered

`campaign.csv`, its figures and `coupling.png` were regenerated after a fix to
`alignment_rate.timeline()`. Some takes lose their opening `FREQ_<f>HZ` label to boot-time
serial garbage while keeping `KILL_` and `DOWN_`; the parser required `FREQ_` to open a point,
so those kills were discarded. Five takes come back — two at 20 Hz, one each at 30, 60 and 70.

Medians move under 1.3%, which is the check that they belong to the same population. The
scatter tightens: 70 Hz goes from 4 repeats to 5 and its rotation MAD from 28.17 to 19.97 deg.
Full account in `control/theory.md` 24.11. **Numbers quoted from the pre-2026-09-10 table will
differ in the fourth significant figure at 20/30/60 Hz and the second at 70.**

## The coning has its own settling time (2026-09-10)

`settling.csv` carries two settling times, and they answer different questions.

* `settle_10pct_s` — how long the AXIS takes to arrive at its new attitude.
* `cone_settle_s` — how long the CONING takes to stop ringing about it. The coning is a
  pulse rather than a step, so the band is ±10% of the excursion (peak − final), not a swing.

**The coning settles later than the axis at every frequency where both exist** — 2.41–3.25 s
against 0.66–2.41 s. The robot arrives well before it stops wobbling.

`cone_tau_s` is a fitted exponential decay constant, and it is reported because the band often
refuses where the fit does not: `DROP_MS` is 5 s and the envelope is still falling at the end
of it, so **97 of 107 repeats give a τ against 50 that give a settling time**. τ runs
0.64–1.38 s and is flat at ~1.2 s over 30–90 Hz.

**Renamed 2026-09-10: coning, not precession.** What is measured sits at 1.00× the drive
frequency at every frequency from 20 to 100 Hz, which is a body-fixed asymmetry carried round
by the rotor — synchronous coning. `control/theory.md` 11.3's free precession is a *different*
mode at a roughly constant ~1.2 Hz, and 24.10's 2–4 Hz line is a third, mechanical, tracking
neither. Three frequencies, three modes; one word for all of them hid that. Columns are now
`cone_*`. Argument in `control/theory.md` 25.10, and 25.11 on why the decomposition is only
half an Euler transform.

**Correction.** An earlier note here said the cut makes the coning worse. It excites it — the
half-angle rises on 101 of 107 repeats, median 2.62° → 6.79° peak — but it then damps to a
settled 1.81°, at or below where it started. True of the peak, false of the steady state.
Full account in `control/theory.md` 25.9.

## Settling is a different question from the rate, and a different number

`--settle` (added 2026-09-10) does not measure the rate. It measures where the axis ENDS UP
and how long until it stays there: the initial and final resting axes, the time to enter and
remain in a ±10% band about the final one, the classical 10–90% rise time, and the coning
cone about the running average axis. Full argument in `control/theory.md` 25.

Three things to know before quoting it.

* **Settling time and rise time are not the same quantity, and neither is the rate.** 25.1.
  The rate is a gradient in deg/s; the rise time is 10→90% of the final value; the settling
  time is the last exit from the band. Say which.
* **The swing here is not `rot_median_deg`.** That column is the change in lean DIRECTION;
  `swing_deg` is the angle the axis actually turned through, which is the great-circle arc
  `2·asin(sinθ·sin(Δφ/2))` and is much smaller — 12 deg at 40 Hz against a 47 deg azimuth
  swing. The relation holds across the campaign to better than 0.5 deg (25.2), which is the
  check that the two agree; they are not interchangeable numbers.
* **The settling time RISES with drive (0.73 s at 20 Hz to 2.41 s at 110) while the rate rises
  too.** Both are true and they measure opposite ends of the same transient. A report that
  quotes only the rate is telling half of it (25.8).

67 of 102 repeats settle. The rest are refused, mostly because the trace still wanders wider
than the band where it is supposed to have arrived — every row carries its own
`tail_over_band` so that judgement can be re-made from the CSV rather than taken on trust
(25.6).

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
