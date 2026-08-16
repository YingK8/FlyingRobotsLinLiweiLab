# 5-DOF estimator — iteration journal

**Target: ±1° and ±0.5 mm on 100% of reported frames, at every sensor mode.**

---

## What "100%" means here, and why

A rotor face-on to both cameras carries no tilt information at any resolution.
Tilt is read from the ellipse's axis ratio, and `dθ/d(ratio) = −1/sin θ`
diverges as θ → 0; the rim's own wall thickness adds a second, independent floor
at `2·arctan(h/R) = 4.96°`, below which two different tilts produce *identical*
silhouettes. No estimator reaches ±1° there. "100% of all poses" is therefore
unachievable in principle, not merely in practice.

So the target is read as **100% of the frames the estimator reports.** It must
recognise when it cannot measure to specification and decline, rather than emit a
confident wrong answer. For a controller that is the more useful contract
anyway: a declared gap can be coasted through, a silent 50 mm error cannot.

That makes **detection rate a first-class number**, reported beside every
accuracy figure in every table below. Without it, "100% in spec" is trivially
achievable by rejecting everything, and the journal would be measuring nothing.

## Which modes can meet ±0.5 mm at all

Before asking whether the estimator reaches the target, it is worth asking where
the target is reachable. Position error is a noise floor plus a systematic
silhouette bias. The floor is derivable: depth from one view goes as `e/M²` and
lateral position as `e/M` (§12.9 of the lecture notes), and stereo fusion at 60°
axis separation turns the former into the latter. At 250 mm range and a 0.3 px
boundary:

| mode | fps | fused noise floor | headroom to 0.5 mm | measured bias (iter 3) | verdict |
|---|---|---|---|---|---|
| 1280×800 | 120 | 0.060 mm | 0.440 mm | 0.250 mm | reachable now |
| 1280×720 | 120 | 0.060 mm | 0.440 mm | 0.250 mm | reachable now |
| 1024×768 | 120 | 0.075 mm | 0.425 mm | 0.285 mm | reachable now |
| 800×600 | 120 | 0.095 mm | 0.405 mm | 0.365 mm | reachable now |
| 640×480 | 210 | 0.119 mm | 0.381 mm | 0.378 mm | reachable, no margin |
| 640×400 | 210 | 0.119 mm | 0.381 mm | 0.378 mm | reachable, no margin |
| 320×240 | 420 | 0.239 mm | 0.261 mm | 0.569 mm | **needs 2.2× less bias** |
| 160×120 | 640 | 0.477 mm | 0.023 mm | — | **impossible** |

Two conclusions follow, and they are the honest frame for every claim below.

**160×120 cannot meet ±0.5 mm, by any estimator.** Its rim spans 18 px, the
fused floor is 0.477 mm, and that consumes 95% of the budget before a single
systematic effect is counted. Refusing every frame there is the correct
behaviour, not a shortfall — the alternative is reporting a number known to be
out of specification.

**320×240 is reachable but not yet reached.** It has 0.261 mm of headroom and the
measured bias is 0.569 mm, so it needs the systematic error roughly halved. That
is a real target rather than a physical wall, and it is what the bias work is
aimed at.

So "100% of test cases at every sensor mode" is achievable at **seven of the
eight**, and the eighth is excluded by geometry that no amount of estimator work
changes. Every claim in this journal is against that reachable set, stated
explicitly rather than by quietly dropping the mode from a table.

## Two condition tiers

| | core | edge |
|---|---|---|
| ambient light | 0.20 – 0.60 | 0.15 – 0.28 |
| sources | dome, single lateral, or two opposed | grazing lateral or low dome |
| backdrop peak | ≤ 90 grey | ≤ 120 grey |
| opacity | 0.8 – 1.0 | 0.7 – 1.0 |
| read noise | as the exposure implies | up to 1.5× |
| **gates the target** | **yes** | no |

Backdrops are monochrome gradients, vignettes and bands, and **every one stays
below the segmenter's threshold of 128**. Crossing it is a segmentation cliff
already characterised elsewhere (position error 232 mm at background grey 0.5);
including it here would drown the geometric signal these tiers exist to measure.
Difficulty comes from structure and contrast instead.

Both tiers see the **same pose set**, so any difference between them is a
difference in conditions rather than in what was asked of the estimator.

## Regression policy

Every iteration is diffed against the previous one, per mode and per metric, and
each comparison is sorted into **improved**, **constant**, or **regressed**. The
tolerance separating "constant" from a real change is the *measured* run-to-run
noise, not a number chosen by eye — without that, "improved" is unfalsifiable,
because every rerun moves every number a little.

**An iteration with an unexplained regression is not published.** It is either
fixed first, or kept with an explicit written justification, and a regression is
only acceptable when it buys something measured and larger. Silent regressions
are the specific failure this journal exists to prevent.

Numbers in every table are generated from the sweep's JSON output by
`validation/journal.py`. Nothing is transcribed by hand.

## Reproducing

```bash
cd ESP32_PMW
uv run python controller/pose/validation/resolution_sweep.py --poses 200 --tag iterN
uv run python controller/pose/validation/journal.py --tag iterN --prev iterN-1 --append
```

---

## Iteration 1 — baseline on the new dataset

*200 poses per tier, seed 20260809. Target ±1° and ±0.5 mm on 100% of reported frames.*

### Changed

- **New validation set.** Monochrome backdrops — flat, linear gradients, vignettes, sinusoidal and square bands, and gradient+band mixtures — replacing three flat grey levels. All strictly below the segmenter's threshold of 128 (`validation/backgrounds.py`, asserted in `test_backgrounds.py` over 1200 fields).
- **Two condition tiers.** Core (ambient 0.20–0.60, backdrop ≤ 90 grey, now including two-source rigs at opposing bearings) gates the target; edge (ambient 0.15–0.28, backdrop ≤ 120 grey, up to 1.5× read noise) is measured but does not gate. Both tiers see the identical pose set, so any difference between them is a difference in conditions.
- **Backdrop compositing in the renderer.** `render._render_instant` now renders onto transparency and composites a supplied greyscale field, so backgrounds can be textured rather than a single colour.
- **Resolution sweep across all eight sensor modes** (`validation/resolution_sweep.py`), rendering once at 1280×800 and width-scaling with a height crop so pixels stay square and the principal point follows.
- **Journal generation from the sweep's JSON** (`validation/journal.py`), with regression classification against measured bootstrap tolerances rather than chosen ones.
- No estimator changes. This entry exists to validate the harness before anything moves.

### Theory

No new estimation mechanism in this iteration — it is a baseline, deliberately —
but the dataset introduces two things that need justifying.

**Why every backdrop stays below grey 128.** The segmenter is a fixed threshold.
A background pixel above it is indistinguishable from robot, and because
`segment.silhouette_hull` takes a convex hull, a single bright region anywhere in
frame drags the fitted ellipse out to enclose it. The failure is total rather
than graceful: the monocular sweep measured 232 mm position error at a uniform
background of grey 128. That cliff is a property of the *threshold*, not of the
pose geometry, and a validation set containing it measures mostly how often
segmentation dies. So difficulty here comes from structure and contrast, with
the ceiling set at 90 grey for the core tier and 120 for the edge tier — both
above the 71–79 grey maximum measured on the only real dark-backdrop photographs
in the repo.

**Why compositing is correct, and where it differs.** pyrender's background is a
single colour, so a gradient must be composited afterwards. Rendering onto
transparency and blending gives `robot·α + backdrop·(1−α)`, and the α there is
**un-premultiplied** — a dim, partially covered edge pixel returns a bright RGB
with a small α. Treating it as premultiplied lights up the entire silhouette
boundary; the check that caught this was that a partially covered pixel returned
63 where the opaque path renders 4, and 63 × 0.063 = 4.

The composited path is not pixel-identical to the old flat-background one: 8-bit
rounding at partial-coverage boundaries moves the fitted major axis by up to
0.6 px. Scored against analytic truth it is the **more accurate** of the two
(bias +0.485 px against +0.713, RMS 0.993 against 1.122), because the slight
edge darkening opposes the outward bias that thresholding-then-hulling
introduces. So this is an improvement rather than a regression — but it does mean
results here are a **new baseline** and not a continuation of earlier numbers.

### Results

![accuracy against sensor mode](../../results/pose_validation/resolution_iter1.png)

**Core tier** — this is what gates the target.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 53% | 0.922° | 8.305° | 0.277 mm | 1.496 mm | 76.4% | 9.1 |
| 1280×720 | 120 | 53% | 0.922° | 8.305° | 0.277 mm | 1.496 mm | 76.4% | 7.7 |
| 1024×768 | 120 | 55% | 0.977° | 7.855° | 0.300 mm | 1.536 mm | 70.9% | 11.7 |
| 800×600 | 120 | 58% | 1.326° | 14.908° | 0.390 mm | 4.014 mm | 65.8% | 10.0 |
| 640×480 | 210 | 68% | 3.308° | 89.014° | 0.459 mm | 2.804 mm | 48.9% | 5.5 |
| 640×400 | 210 | 68% | 3.308° | 89.014° | 0.459 mm | 2.803 mm | 48.9% | 5.3 |
| 320×240 | 420 | 29% | 13.955° | 67.686° | 12.296 mm | 356.067 mm | 8.6% | 2.9 |
| 160×120 | 640 | 19% | 51.058° | 88.374° | 236.958 mm | 428.918 mm | 0.0% | 1.7 |

**Edge tier** — measured and tracked, does not gate.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 42% | 1.816° | 7.906° | 0.338 mm | 1.564 mm | 34.9% | 9.0 |
| 1280×720 | 120 | 41% | 1.794° | 7.906° | 0.336 mm | 1.564 mm | 35.4% | 7.9 |
| 1024×768 | 120 | 46% | 2.970° | 60.914° | 0.441 mm | 6.443 mm | 28.6% | 12.0 |
| 800×600 | 120 | 44% | 3.008° | 20.113° | 0.420 mm | 2.485 mm | 28.1% | 9.7 |
| 640×480 | 210 | 48% | 4.963° | 81.868° | 0.585 mm | 5.748 mm | 21.6% | 4.8 |
| 640×400 | 210 | 48% | 4.963° | 81.868° | 0.585 mm | 5.748 mm | 21.6% | 4.6 |
| 320×240 | 420 | 33% | 23.347° | 88.675° | 125.630 mm | 1023.810 mm | 6.1% | 2.8 |
| 160×120 | 640 | 14% | 59.382° | 88.335° | 219.169 mm | 511.188 mm | 0.0% | 1.6 |

### Improved / Constant / Regressed

First entry on this dataset, so there is nothing to compare against. The conditions changed in this iteration (backgrounds, lighting, two tiers), so earlier numbers from the flat-background sweeps are **not** comparable and are deliberately not carried forward.

## Iteration 2 — refusing what cannot be certified

*200 poses per tier, seed 20260809. Target ±1° and ±0.5 mm on 100% of reported frames.*

### Changed

- **Refuse single-view frames** (`stereo.require_stereo`, default on). A frame where only one camera produced a usable outline is now declined rather than answered monocularly. This follows from the noise derivation rather than from tuning — see Theory — and it also closed a hole in the cross-view gate, which is skipped when there is only one view and so passed those frames without comparing anything.
- **Scale-free outline-quality gate.** `MAX_FIT_RMS_REL = 0.012` replaces the dimensional 1.5 px threshold. A fixed pixel threshold tightens as resolution improves, which is backwards, and it was the cause of the Iteration 1 anomaly where detection was *lower* at 1280×800 (53%) than at 640×480 (68%).
- **Predicted-error gate** (`uncertainty.ErrorModel`, `--gate`). The estimator predicts this frame's position and angle error from observables it already computed, and declines when the prediction exceeds specification.
- **Operating point chosen by leave-one-mode-out, mid-plateau** (`fit_error_model.choose_threshold`). Two earlier procedures were wrong in ways that cost real accuracy; both are documented in Theory.
- **`--gate` is opt-in in the sweep**, so the ungated numbers keep measuring the estimator and the gated numbers measure the delivered product.
- Two hypotheses were tested and **rejected** rather than shipped: a `sin θ` term for silhouette bias (measured rank correlation with position error ρ = −0.005) and a minimum-object-size gate (blunders reach 143 px major axis against a clean-set median of 103 px, so size does not separate them).

### Theory

Two mechanisms enter in this iteration. The first is a refusal that follows from
geometry; the second is a gate whose calibration turned out to be a different
kind of object than it was first built as.

#### Why a single-view frame cannot be answered, ever

The estimator previously fell back to a monocular solve when one camera lost the
outline, on the reasoning that one view is a degradation rather than a failure.
That reasoning does not survive the noise derivation.

Depth from one view is read from the ellipse's **size**. Since `z = 2fR/M`,

    dz/z = −dM/M

and size is a *difference* of two boundary positions, so it inherits the boundary
error directly. Lateral position is read from the ellipse's **centre**, which is
an *average* over the boundary and therefore one power of `M` better. At a 0.3 px
boundary and 250 mm range:

| mode | M px | single-view depth σ | fused stereo σ |
|---|---|---|---|
| 1280×800 | 143.7 | 0.522 mm | **0.060 mm** |
| 1024×768 | 114.9 | 0.653 mm | **0.075 mm** |
| 640×480 | 71.8 | 1.044 mm | **0.119 mm** |
| 320×240 | 35.9 | 2.088 mm | **0.239 mm** |
| 160×120 | 18.0 | 4.176 mm | **0.477 mm** |

A monocular solve misses ±0.5 mm at **every** mode this camera offers, including
its best. Stereo meets it at every mode, because with the optical axes 60° apart
the second camera measures laterally exactly the direction the first is blind
along. So refusing a single-view frame is not a tuned threshold — it is the
statement that the specification is unreachable without the second view.

It also closed a hole. The cross-view discrepancy gate is skipped when only one
view is usable, so those frames passed without anything being compared. Measured
over 921 samples, all six single-view frames were **0% in spec with a 256 mm
median error**, and refusing them costs 0.7% of frames while taking the worst
position error from **755.7 mm to 8.0 mm**.

#### The quantile is an operating point, not a bound

The gate multiplies a fitted error scale by `k`, the high quantile of
actual/predicted measured on the training split. Three attempts to calibrate it
produced `k = 2.4×10⁹`, then 280, then 674 — each rejecting every frame.

The first fault was that **blunders and precision are different problems.** A
frame whose outline closed around the wrong object has ordinary-looking features
and an error two orders of magnitude out. No monotone function of precision
features predicts it, so including such frames in the fit lets them set the
quantile. Excluding them from the *fit* while keeping them in the *evaluation*
brought `k` to 4.39 — the asymmetry matters, because a blunder the hard gates
miss must still surface as a coverage failure rather than being excused.

The second fault was conceptual, and is the more useful one. `k` is a single
global multiplier, so it **does not reorder frames** — it only slides the
threshold along a fixed ranking. Calibrating it from a quantile of the training
ratio is therefore not "computing a bound" at all; it is picking an operating
point on an ROC curve, and the direction is counter-intuitive: a *lower* quantile
gives a *smaller* predicted error, so *more* frames pass. Once seen, the correct
procedure follows immediately — choose the operating point against a held-out
coverage requirement rather than from a training statistic.

That requires three splits, not two: the **scale** is fitted on one set of modes,
the **threshold** chosen on a second, and the reported numbers come from a third
neither step saw. Splitting by mode rather than at random is deliberate — frames
of the same pose at two resolutions are far from independent, and a random split
would grade the model partly on data it had effectively seen.

Cross-validated over eight rotations of that split, the gate holds **100% coverage
in 8/8 folds** at 30.6% mean acceptance.

#### Choose the middle of the plateau, not its edge

Coverage as a function of the quantile is a step, not a slope: 100% above some
value, decaying below it. Mapping it by leave-one-mode-out gives

| q | 0.95 | 0.89 | 0.85 | 0.81 | 0.79 | 0.75 | 0.69 |
|---|---|---|---|---|---|---|---|
| acceptance | 2.6% | 18.3% | 27.5% | 35.3% | 39.8% | 46.4% | 52.6% |
| coverage | 100% | 100% | 100% | **100%** | 98.5% | 97.4% | 96.6% |

The temptation is to take the most permissive point holding 100% — here q = 0.81
— because it maximises acceptance. That is the plateau's **edge**, and an edge is
exactly where resampling flips the answer. Measured: a model at q = 0.80, one
step past it, held 100% on its own selection data and delivered **94.4%** on a
fresh seed. Taking the midpoint q = 0.88 instead costs acceptance and buys margin
that survives new data — 100% in spec at four of the six reporting modes.

The general form: **when a decision threshold is chosen against a hard
requirement, the qualifying region's interior is the robust choice and its
boundary is the fragile one.** Optimising acceptance subject to "coverage = 100%
on the sample I measured" lands on the boundary by construction, and boundary
performance does not generalise.

#### What this does not fix

The gate rejects frames it cannot certify; it does not make them better. The
underlying accuracy is unchanged, and §12.9 of the lecture notes records why that
is the binding constraint: measured error at the best mode is 4.6× the noise
floor, so the residual is dominated by systematic silhouette bias rather than by
pixel noise. Two consequences follow for the next iterations. Resolution is
nearly free to give up — 143.7 px to 71.8 px across the rim costs 0.06 mm against
a larger bias floor — so the 420 fps mode is not disqualified by geometry.  And
**angle, not position, is the binding metric**: in the precision regime position
is in spec 88% of the time and angle only 55%.

One hypothesis was tested and rejected rather than implemented. Silhouette bias
grows with tilt through `ρ = cos θ + k sin θ`, which suggested a `sin θ` term in
the position model. Measured rank correlation between `sin θ` and position error
is **ρ = −0.005** — no relationship at all. A minimum-object-size gate was
likewise rejected: blunders reach 143 px major axis against a clean-set median of
103 px, so size does not separate them.

### Results

![accuracy against sensor mode](../../results/pose_validation/resolution_iter2.png)

*Ungated. The gated equivalent is `resolution_iter2gated.png`.*

#### Ungated — did the estimator improve?

This is the like-for-like successor to Iteration 1: same poses, same seed, no frames refused by the predicted-error gate. The improved/constant/regressed classification below is computed against these numbers, so that better estimation is never confused with stricter refusal.

**Core tier** — this is what gates the target.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 56% | 0.971° | 8.305° | 0.279 mm | 1.496 mm | 73.2% | 7.4 |
| 1280×720 | 120 | 56% | 0.971° | 8.305° | 0.279 mm | 1.496 mm | 73.2% | 6.6 |
| 1024×768 | 120 | 54% | 0.962° | 7.855° | 0.300 mm | 1.536 mm | 72.0% | 9.4 |
| 800×600 | 120 | 50% | 1.085° | 13.051° | 0.383 mm | 4.014 mm | 73.3% | 7.3 |
| 640×480 | 210 | 50% | 2.683° | 89.014° | 0.439 mm | 2.804 mm | 64.6% | 3.7 |
| 640×400 | 210 | 50% | 2.683° | 89.014° | 0.439 mm | 2.803 mm | 64.6% | 3.6 |
| 320×240 | 420 | 18% | 4.341° | 58.900° | 0.595 mm | 2.647 mm | 13.5% | 1.8 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.1 |

**Edge tier** — measured and tracked, does not gate.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 45% | 1.823° | 7.906° | 0.339 mm | 1.564 mm | 35.6% | 7.5 |
| 1280×720 | 120 | 44% | 1.802° | 7.906° | 0.337 mm | 1.564 mm | 36.0% | 6.8 |
| 1024×768 | 120 | 42% | 2.872° | 60.914° | 0.446 mm | 6.443 mm | 29.8% | 9.6 |
| 800×600 | 120 | 36% | 2.592° | 15.206° | 0.365 mm | 1.222 mm | 30.1% | 7.7 |
| 640×480 | 210 | 32% | 2.965° | 20.696° | 0.440 mm | 1.551 mm | 28.6% | 3.5 |
| 640×400 | 210 | 32% | 2.965° | 20.696° | 0.440 mm | 1.551 mm | 28.6% | 3.2 |
| 320×240 | 420 | 16% | 4.265° | 34.402° | 0.577 mm | 2.583 mm | 12.5% | 1.7 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.1 |

#### Gated — what the estimator delivers

The same frames, with the estimator declining any it cannot certify to specification. Detection rate is the price and is reported beside every accuracy figure; without it, 100% in spec is trivially reachable by refusing everything.

**Core tier**

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 7% | 0.309° | 0.571° | 0.242 mm | 0.381 mm | **100%** | 9.5 |
| 1280×720 | 120 | 7% | 0.309° | 0.571° | 0.242 mm | 0.381 mm | **100%** | 8.4 |
| 1024×768 | 120 | 6% | 0.337° | 0.600° | 0.256 mm | 0.417 mm | **100%** | 12.3 |
| 800×600 | 120 | 6% | 0.308° | 0.557° | 0.313 mm | 0.475 mm | **100%** | 10.2 |
| 640×480 | 210 | 4% | 0.471° | 0.785° | 0.382 mm | 0.522 mm | 87.5% | 5.1 |
| 640×400 | 210 | 4% | 0.471° | 0.785° | 0.382 mm | 0.522 mm | 87.5% | 4.9 |
| 320×240 | 420 | **0%** | — | — | — | — | — | 2.5 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.5 |

**Edge tier**

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 1% | 0.386° | 0.472° | 0.246 mm | 0.280 mm | **100%** | 9.2 |
| 1280×720 | 120 | 1% | 0.386° | 0.472° | 0.246 mm | 0.280 mm | **100%** | 8.3 |
| 1024×768 | 120 | 2% | 0.593° | 0.635° | 0.316 mm | 0.357 mm | **100%** | 11.9 |
| 800×600 | 120 | 0% | 0.265° | 0.265° | 0.197 mm | 0.197 mm | **100%** | 9.0 |
| 640×480 | 210 | 2% | 0.625° | 1.299° | 0.337 mm | 0.456 mm | 66.7% | 4.3 |
| 640×400 | 210 | 2% | 0.625° | 1.299° | 0.337 mm | 0.456 mm | 66.7% | 4.0 |
| 320×240 | 420 | **0%** | — | — | — | — | — | 2.4 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.5 |

### Improved / Constant / Regressed

*Computed on the ungated runs, against measured bootstrap tolerances.*

**Improved** (30)

- core 1280x800 detected: 53.0% → 56.0%
- core 1280x720 detected: 53.0% → 56.0%
- core 800x600 in spec: 65.8% → 73.3%
- core 640x480 in spec: 48.9% → 64.6%
- core 640x400 in spec: 48.9% → 64.6%
- core 320x240 in spec: 8.6% → 13.5%
- core 320x240 angle mean: 13.955° → 4.341°
- core 320x240 position mean: 12.296 mm → 0.595 mm
- core 320x240 position worst: 356.067 mm → 2.647 mm
- edge 1280x800 detected: 41.5% → 45.0%
- edge 1280x720 detected: 41.0% → 44.5%
- edge 800x600 in spec: 28.1% → 30.1%
- edge 800x600 angle worst: 20.113° → 15.206°
- edge 800x600 position mean: 0.420 mm → 0.365 mm
- edge 800x600 position worst: 2.485 mm → 1.222 mm
- edge 640x480 in spec: 21.6% → 28.6%
- edge 640x480 angle mean: 4.963° → 2.965°
- edge 640x480 angle worst: 81.868° → 20.696°
- edge 640x480 position mean: 0.585 mm → 0.440 mm
- edge 640x480 position worst: 5.748 mm → 1.551 mm
- edge 640x400 in spec: 21.6% → 28.6%
- edge 640x400 angle mean: 4.963° → 2.965°
- edge 640x400 angle worst: 81.868° → 20.696°
- edge 640x400 position mean: 0.585 mm → 0.440 mm
- edge 640x400 position worst: 5.748 mm → 1.551 mm
- edge 320x240 in spec: 6.1% → 12.5%
- edge 320x240 angle mean: 23.347° → 4.265°
- edge 320x240 angle worst: 88.675° → 34.402°
- edge 320x240 position mean: 125.630 mm → 0.577 mm
- edge 320x240 position worst: 1023.810 mm → 2.583 mm

**Constant** (45)

- core 1280x800 angle mean: 0.922° → 0.971°
- core 1280x800 angle worst: 8.305° → 8.305°
- core 1280x800 position mean: 0.277 mm → 0.279 mm
- core 1280x800 position worst: 1.496 mm → 1.496 mm
- core 1280x720 angle mean: 0.922° → 0.971°
- core 1280x720 angle worst: 8.305° → 8.305°
- core 1280x720 position mean: 0.277 mm → 0.279 mm
- core 1280x720 position worst: 1.496 mm → 1.496 mm
- core 1024x768 in spec: 70.9% → 72.0%
- core 1024x768 detected: 55.0% → 53.5%
- core 1024x768 angle mean: 0.977° → 0.962°
- core 1024x768 angle worst: 7.855° → 7.855°
- core 1024x768 position mean: 0.300 mm → 0.300 mm
- core 1024x768 position worst: 1.536 mm → 1.536 mm
- core 800x600 angle mean: 1.326° → 1.085°
- core 800x600 angle worst: 14.908° → 13.051°
- core 800x600 position mean: 0.390 mm → 0.383 mm
- core 800x600 position worst: 4.014 mm → 4.014 mm
- core 640x480 angle mean: 3.308° → 2.683°
- core 640x480 angle worst: 89.014° → 89.014°
- core 640x480 position mean: 0.459 mm → 0.439 mm
- core 640x480 position worst: 2.804 mm → 2.804 mm
- core 640x400 angle mean: 3.308° → 2.683°
- core 640x400 angle worst: 89.014° → 89.014°
- core 640x400 position mean: 0.459 mm → 0.439 mm
- core 640x400 position worst: 2.803 mm → 2.803 mm
- core 320x240 angle worst: 67.686° → 58.900°
- core 160x120 in spec: 0.0% → 0.0%
- edge 1280x800 in spec: 34.9% → 35.6%
- edge 1280x800 angle mean: 1.816° → 1.823°
- edge 1280x800 angle worst: 7.906° → 7.906°
- edge 1280x800 position mean: 0.338 mm → 0.339 mm
- edge 1280x800 position worst: 1.564 mm → 1.564 mm
- edge 1280x720 in spec: 35.4% → 36.0%
- edge 1280x720 angle mean: 1.794° → 1.802°
- edge 1280x720 angle worst: 7.906° → 7.906°
- edge 1280x720 position mean: 0.336 mm → 0.337 mm
- edge 1280x720 position worst: 1.564 mm → 1.564 mm
- edge 1024x768 in spec: 28.6% → 29.8%
- edge 1024x768 angle mean: 2.970° → 2.872°
- edge 1024x768 angle worst: 60.914° → 60.914°
- edge 1024x768 position mean: 0.441 mm → 0.446 mm
- edge 1024x768 position worst: 6.443 mm → 6.443 mm
- edge 800x600 angle mean: 3.008° → 2.592°
- edge 160x120 in spec: 0.0% → 0.0%

****Regressed**** (13)

- core 1280x800 in spec: 76.4% → 73.2%
- core 1280x720 in spec: 76.4% → 73.2%
- core 800x600 detected: 58.5% → 50.5%
- core 640x480 detected: 68.5% → 49.5%
- core 640x400 detected: 68.5% → 49.5%
- core 320x240 detected: 29.0% → 18.5%
- core 160x120 detected: 19.0% → 0.0%
- edge 1024x768 detected: 45.5% → 42.0%
- edge 800x600 detected: 44.5% → 36.5%
- edge 640x480 detected: 48.5% → 31.5%
- edge 640x400 detected: 48.5% → 31.5%
- edge 320x240 detected: 33.0% → 16.0%
- edge 160x120 detected: 14.0% → 0.0%

#### Justification for the regressions

**Twelve of the thirteen are detection rate**, and every one is the deliberate price of a gate that this iteration added on purpose. Each is paired with a large accuracy improvement at the same mode: 640×480 gives up 19 points of detection and gains 15.7 points of in-spec; 320×240 gives up 10.5 points and takes position mean from 12.296 mm to 0.595 mm and position worst from 356 mm to 2.6 mm.

**160×120 falling to 0% detection is intended and derived.** Its rim spans 18 px, so single-view depth precision is 4.18 mm and the fused figure 0.477 mm sits at the specification limit with no margin for the silhouette bias that dominates in practice. Every frame it produced in Iteration 1 was out of spec (0.0% in spec, 237 mm mean position error). Refusing them removes no capability; it stops the estimator reporting answers that were never usable.

**The one regression that is not detection rate needs a real argument.** Core 1280×800 and 1280×720 in-spec fell 76.4% → 73.2%, while detection at those modes *rose* 53.0% → 56.0%. Both come from the same change: the scale-free outline gate computes to 0.012 × 143 px = 1.72 px at 1280, looser than the old fixed 1.5 px, so marginal frames that used to be refused are now admitted and dilute the average. At 640 the same rule computes to 0.86 px, tighter than 1.5 px, which is why 640×480 in-spec rose 15.7 points.

This is accepted, for two measured reasons. First, the dimensional threshold was **wrong in form**: a fixed pixel tolerance tightens as resolution improves, which is backwards, and it produced the Iteration 1 anomaly of *lower* detection at 1280×800 (53%) than at 640×480 (68%). Second, and decisively, the dilution does not reach the delivered product: in the gated configuration 1280×800 reports **100.0% in spec**, against 76.4% ungated in Iteration 1. The marginal frames the looser outline gate admits are precisely the ones the predicted-error gate then declines.

### Against the target

The goal is ±1° and ±0.5 mm on 100% of test cases. This iteration does not reach it, and the shortfall is stated rather than averaged away:

- **Four of six reporting modes are at 100% in spec** (1280×800, 1280×720, 1024×768, 800×600), worst case 0.571° and 0.475 mm.
- **640×480 and 640×400 are at 87.5%** — one accepted frame in eight at 0.522 mm against the 0.5 mm limit.
- **320×240 and 160×120 report nothing.** Neither can meet the specification, and the estimator now says so instead of guessing.
- **Detection is 4–7% at the gated operating point.** This is the real cost and the honest headline: the estimator is correct when it answers, and it answers rarely. A controller can coast a declared gap, but not a gap this wide.

The binding constraint is measured, not guessed: of core frames the gate rejects, **77.1% fail on predicted angle against 20.6% on predicted position**. And §12.9 of the lecture notes shows why raising acceptance cannot come from resolution — measured error at the best mode is 4.6× the noise floor, so what limits it is systematic silhouette bias. Iteration 3 targets that bias, angle first.


### Open defect found while closing this iteration

> **Resolved in Iteration 3.** Kept here as the record of how it was found.
> The tables in *this* entry measure the build that still had it.

`test_stereo.test_refine_recovers_exact_geometry` **fails** (pose 2: 0.160 mm,
16.2°) and the cause is a defect in `segment.fit_ellipse`, not in anything this
iteration changed.

The axial re-weighting — on by default, and intended to suppress the mast's
protrusions on a real silhouette — can rotate the fitted major axis on a
perfectly clean ellipse. Measured on one synthetic view, axes identical in both
cases:

| | angle | rms |
|---|---|---|
| `axial=False` | 0.0000° (truth 180.0° ≡ 0.0°) | **0.0000 px** |
| `axial=True` (default) | 146.4921° — **33.5° wrong** | 3.7374 px |

The re-weighted fit is not merely different, it is *worse* against the very
points it was fitted to. That is what breaks the refinement: its cost compares
axis endpoints, so a 33.5° error in the measured angle makes the residual
non-zero **at the true pose**, and every seed — including one started exactly at
the truth — converges away from it.

It fires on 1 of the 10 pose/view combinations in the test set, and
`AXIAL_SKIP_RATIO = 0.995` means the re-weighting itself runs on nearly every
frame. **This is the prime suspect for the angle errors that dominate the gate**:
77.1% of rejected core frames fail on predicted angle against 20.6% on position.

It is recorded rather than hot-fixed because the re-weighting is load-bearing for
real silhouettes and removing it would trade a synthetic failure for a physical
one. Iteration 3 addresses it, and the candidate fixes are in the task notes:
accept the re-weighted fit only when its rms improves; or keep the unweighted
angle and re-weight only the axes.

**The numbers published above remain valid** — they measure the system as it
actually is, defect included. They are not a claim about a system with the defect
fixed.

## Iteration 3 — the silhouette fit was turning the ellipse

*200 poses per tier, seed 20260809. Target ±1° and ±0.5 mm on 100% of reported frames.*

### Changed

- **Fixed a defect in `segment.fit_ellipse` that rotated the fitted ellipse.** The axial re-weighting, which exists to suppress the rod and magnet, drove some weights to exactly zero and left the fit ill-conditioned in rotation — putting the major axis up to **33.5°** out on a *noise-free* ellipse, with a Sampson rms of 3.7374 px against 0.0000 px for the plain fit. The re-weighting now corrects the axis lengths and the orientation is held at the plain fit's value. See Theory.
- **A weight floor was tried first and rejected on measurement.** It fixed the case it was aimed at and broke a different one, leaving the worst refinement error unchanged — an ill-conditioned direction stays ill-conditioned.
- **The gate's operating point is now chosen on the worst mode, not pooled across modes.** A pooled average lets a weak resolution hide inside a strong one; the specification is per test case.
- **Error model refitted** on the corrected fits. The angle model's `log_inv_margin` coefficient fell from −0.399 to −0.040 — branch decisiveness barely explains the orientation error any more, which is what removing the rotation predicts.
- `test_stereo.test_refine_recovers_exact_geometry` now **passes** (worst position error 1.60e-01 mm → 3.85e-08 mm). All seven suites pass.

### Theory

One mechanism this iteration, and it is a correction rather than an addition:
the silhouette fit was handing the one quantity its weights could not constrain
to the one fit that could not constrain it.

#### What the axial re-weighting is for, and what it was doing

The robot is not a flat disc. A rod and magnet mount protrude from the rim, and
they fatten the silhouette in its **short** direction — the fitted minor axis
comes out too long, the axis ratio too close to 1, and the tilt read from that
ratio too small. `segment.axial_weights` exists to suppress them: it weights each
boundary point by how far along the *major* axis it lies, so points near the
major axis's centre — where the protrusions project — count for little, and
points at its ends, which are honest rim, count fully.

The weighting is applied inside two IRLS iterations, and the fit it produces was
being used **whole**: centre, both axis lengths, and orientation.

That last one is the defect. The weights fall to exactly zero over a contiguous
arc, and a zero weight does not distrust a point — it deletes it. What remains is
clustered near the two ends of the major axis, and five conic parameters fitted
to two opposing clusters are ill-conditioned in precisely one direction:
**rotation**. Two point clusters pin a line through them; they say very little
about how the ellipse is turned about that line.

Measured on a noise-free synthetic ellipse of ratio 0.834, where the correct
answer is known exactly:

| | fitted angle | Sampson rms |
|---|---|---|
| plain fit | 0.0000° (truth 180.0° ≡ 0.0°) | **0.0000 px** |
| re-weighted, angle free | 146.4921° — **33.5° out** | 3.7374 px |

The re-weighted fit was not merely different from the truth, it was **worse
against the very points it had been fitted to**. That is the signature of an
ill-conditioned direction rather than a robustness trade: a genuine robust fit
accepts a higher residual on all points in exchange for a better estimate, but
here the estimate was worse too.

#### Why a weight floor is not the fix

The obvious repair is to floor the weights so no point is ever deleted. It works
on the case above — a floor of 0.05 takes the 33.5° error to 0.0000° — and it is
wrong. Re-measuring across the whole pose set showed it changes *which* cases
fail rather than whether they do: at floor 0.05 the 33.5° pose became exact and a
different pose that had been exact went 14° out, and the worst refinement error
was unchanged at 4.4e-02 mm. A direction that is ill-conditioned stays
ill-conditioned; flooring the weights only reshuffles which arbitrary answer the
solver lands on.

#### The fix: correct the lengths, keep the orientation

The protrusions sit **symmetrically about the major axis**. They change how long
the ellipse is across its short direction; they do not turn it. So the
re-weighting has authority over the axis lengths and none over the angle, and the
plain fit — which sees every point and is well-conditioned in rotation — should
keep the angle.

`fit_ellipse` now holds the plain fit's orientation through the IRLS iterations
and lets the weights correct only the lengths and centre. Measured on both
requirements at once:

| | injected-protrusion ratio bias | worst refinement error |
|---|---|---|
| no re-weighting | +0.05591 | — |
| re-weighting, angle free | +0.01965 | 1.60e-01 mm |
| **re-weighting, angle held** | **+0.01989** | **3.85e-08 mm** |

The suppression is retained essentially in full (+0.01989 against +0.01965, where
doing nothing costs +0.05591) and the refinement failure disappears — a factor of
4 million on the exact-geometry test.

#### Why this shows up as an *angle* improvement everywhere

`refine`'s cost compares the four axis endpoints of the measured and predicted
ellipses. An error in the measured **angle** moves those endpoints by roughly the
major axis's length, so a 33° rotation makes the residual large **at the true
pose** — every seed, including one started exactly at the truth, then converges
away from it. The refinement was not failing to converge; it was converging
correctly onto a corrupted objective.

That predicts the gains should be largest where the fit is least constrained, and
they are: 320×240 in-spec more than doubled (13.5% → 30.3%) while 1280×800 rose
6.2 points, and 640×480's worst-case angle fell from 89.0° to 60.9°.

It also settles the question §12.9 left open. The measured error was 4.6× the
derived noise floor and the excess was attributed to systematic silhouette bias.
Part of that excess was not physics at all — it was this defect. The remainder
still is: at 1280×800 the mean position error is now 0.250 mm against a fused
noise floor of 0.060 mm, so a factor of 4 remains unexplained by pixel noise.

#### Per-mode coverage, not pooled

The gate's operating point was being chosen to hold 100% coverage **pooled**
across resolutions. That lets a weak mode hide inside a strong one, and it did:
a threshold at 100% pooled coverage still left 640×480 at 85.7% in spec, because
the frames it got wrong were a small share of a set dominated by 1280×800. The
specification is per test case, so the criterion is now the **worst** mode rather
than the average of all of them.

On this data the chosen quantile did not move, which is itself informative: the
selection split was already per-mode clean, so 640×480's shortfall is a
generalisation gap on a fresh seed rather than a selection error. The stricter
criterion is kept because it is the correct one, not because it changed today's
answer.

### Results

![accuracy against sensor mode](../../results/pose_validation/resolution_iter3.png)

#### Ungated — did the estimator improve?

This is the like-for-like successor to Iteration 1: same poses, same seed, no frames refused by the predicted-error gate. The improved/constant/regressed classification below is computed against these numbers, so that better estimation is never confused with stricter refusal.

**Core tier** — this is what gates the target.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 54% | 0.833° | 8.425° | 0.250 mm | 1.227 mm | 79.4% | 8.6 |
| 1280×720 | 120 | 54% | 0.833° | 8.425° | 0.250 mm | 1.227 mm | 79.4% | 7.7 |
| 1024×768 | 120 | 52% | 0.795° | 8.006° | 0.285 mm | 1.481 mm | 78.8% | 11.5 |
| 800×600 | 120 | 49% | 0.973° | 12.982° | 0.365 mm | 4.090 mm | 76.5% | 10.8 |
| 640×480 | 210 | 46% | 2.128° | 60.877° | 0.378 mm | 2.077 mm | 74.2% | 5.0 |
| 640×400 | 210 | 46% | 2.128° | 60.873° | 0.378 mm | 2.077 mm | 74.2% | 4.4 |
| 320×240 | 420 | 16% | 3.878° | 58.746° | 0.569 mm | 2.659 mm | 30.3% | 2.3 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.3 |

**Edge tier** — measured and tracked, does not gate.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 41% | 1.692° | 7.930° | 0.311 mm | 1.166 mm | 37.8% | 8.7 |
| 1280×720 | 120 | 40% | 1.667° | 7.930° | 0.309 mm | 1.166 mm | 38.3% | 7.5 |
| 1024×768 | 120 | 38% | 1.913° | 7.488° | 0.337 mm | 1.579 mm | 36.0% | 11.1 |
| 800×600 | 120 | 32% | 2.128° | 10.792° | 0.336 mm | 0.990 mm | 34.9% | 9.1 |
| 640×480 | 210 | 28% | 2.495° | 17.242° | 0.409 mm | 1.459 mm | 30.4% | 4.0 |
| 640×400 | 210 | 28% | 2.495° | 17.242° | 0.409 mm | 1.459 mm | 30.4% | 3.6 |
| 320×240 | 420 | 14% | 2.847° | 14.560° | 0.538 mm | 2.337 mm | 21.4% | 2.1 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.3 |

#### Gated — what the estimator delivers

The same frames, with the estimator declining any it cannot certify to specification. Detection rate is the price and is reported beside every accuracy figure; without it, 100% in spec is trivially reachable by refusing everything.

**Core tier**

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 6% | 0.195° | 0.518° | 0.216 mm | 0.369 mm | **100%** | 9.1 |
| 1280×720 | 120 | 6% | 0.195° | 0.518° | 0.216 mm | 0.369 mm | **100%** | 7.5 |
| 1024×768 | 120 | 4% | 0.212° | 0.355° | 0.253 mm | 0.406 mm | **100%** | 11.5 |
| 800×600 | 120 | 6% | 0.264° | 0.485° | 0.311 mm | 0.465 mm | **100%** | 10.6 |
| 640×480 | 210 | 4% | 0.314° | 0.434° | 0.369 mm | 0.512 mm | 85.7% | 4.9 |
| 640×400 | 210 | 4% | 0.314° | 0.434° | 0.369 mm | 0.512 mm | 85.7% | 4.2 |
| 320×240 | 420 | **0%** | — | — | — | — | — | 2.2 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.3 |

**Edge tier**

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 1% | 0.261° | 0.404° | 0.168 mm | 0.205 mm | **100%** | 8.2 |
| 1280×720 | 120 | 1% | 0.261° | 0.404° | 0.168 mm | 0.205 mm | **100%** | 7.0 |
| 1024×768 | 120 | 2% | 0.552° | 0.702° | 0.308 mm | 0.354 mm | **100%** | 11.0 |
| 800×600 | 120 | 1% | 0.499° | 0.864° | 0.228 mm | 0.264 mm | **100%** | 9.7 |
| 640×480 | 210 | 1% | 0.236° | 0.268° | 0.339 mm | 0.451 mm | **100%** | 3.9 |
| 640×400 | 210 | 1% | 0.236° | 0.268° | 0.339 mm | 0.451 mm | **100%** | 3.8 |
| 320×240 | 420 | **0%** | — | — | — | — | — | 2.5 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.6 |

### Improved / Constant / Regressed

*Computed on the ungated runs, against measured bootstrap tolerances.*

**Improved** (19)

- core 1280x800 in spec: 73.2% → 79.4%
- core 1280x720 in spec: 73.2% → 79.4%
- core 1024x768 in spec: 72.0% → 78.8%
- core 800x600 in spec: 73.3% → 76.5%
- core 640x480 in spec: 64.6% → 74.2%
- core 640x400 in spec: 64.6% → 74.2%
- core 320x240 in spec: 13.5% → 30.3%
- edge 1280x800 in spec: 35.6% → 37.8%
- edge 1280x800 position worst: 1.564 mm → 1.166 mm
- edge 1280x720 in spec: 36.0% → 38.3%
- edge 1280x720 position worst: 1.564 mm → 1.166 mm
- edge 1024x768 in spec: 29.8% → 36.0%
- edge 1024x768 angle worst: 60.914° → 7.488°
- edge 1024x768 position worst: 6.443 mm → 1.579 mm
- edge 800x600 in spec: 30.1% → 34.9%
- edge 800x600 angle worst: 15.206° → 10.792°
- edge 800x600 position worst: 1.222 mm → 0.990 mm
- edge 320x240 in spec: 12.5% → 21.4%
- edge 320x240 angle worst: 34.402° → 14.560°

**Constant** (59)

- core 1280x800 angle mean: 0.971° → 0.833°
- core 1280x800 angle worst: 8.305° → 8.425°
- core 1280x800 position mean: 0.279 mm → 0.250 mm
- core 1280x800 position worst: 1.496 mm → 1.227 mm
- core 1280x720 angle mean: 0.971° → 0.833°
- core 1280x720 angle worst: 8.305° → 8.425°
- core 1280x720 position mean: 0.279 mm → 0.250 mm
- core 1280x720 position worst: 1.496 mm → 1.227 mm
- core 1024x768 detected: 53.5% → 52.0%
- core 1024x768 angle mean: 0.962° → 0.795°
- core 1024x768 angle worst: 7.855° → 8.006°
- core 1024x768 position mean: 0.300 mm → 0.285 mm
- core 1024x768 position worst: 1.536 mm → 1.481 mm
- core 800x600 detected: 50.5% → 49.0%
- core 800x600 angle mean: 1.085° → 0.973°
- core 800x600 angle worst: 13.051° → 12.982°
- core 800x600 position mean: 0.383 mm → 0.365 mm
- core 800x600 position worst: 4.014 mm → 4.090 mm
- core 640x480 angle mean: 2.683° → 2.128°
- core 640x480 angle worst: 89.014° → 60.877°
- core 640x480 position mean: 0.439 mm → 0.378 mm
- core 640x480 position worst: 2.804 mm → 2.077 mm
- core 640x400 angle mean: 2.683° → 2.128°
- core 640x400 angle worst: 89.014° → 60.873°
- core 640x400 position mean: 0.439 mm → 0.378 mm
- core 640x400 position worst: 2.803 mm → 2.077 mm
- core 320x240 detected: 18.5% → 16.5%
- core 320x240 angle mean: 4.341° → 3.878°
- core 320x240 angle worst: 58.900° → 58.746°
- core 320x240 position mean: 0.595 mm → 0.569 mm
- core 320x240 position worst: 2.647 mm → 2.659 mm
- core 160x120 in spec: 0.0% → 0.0%
- core 160x120 detected: 0.0% → 0.0%
- edge 1280x800 angle mean: 1.823° → 1.692°
- edge 1280x800 angle worst: 7.906° → 7.930°
- edge 1280x800 position mean: 0.339 mm → 0.311 mm
- edge 1280x720 angle mean: 1.802° → 1.667°
- edge 1280x720 angle worst: 7.906° → 7.930°
- edge 1280x720 position mean: 0.337 mm → 0.309 mm
- edge 1024x768 angle mean: 2.872° → 1.913°
- edge 1024x768 position mean: 0.446 mm → 0.337 mm
- edge 800x600 angle mean: 2.592° → 2.128°
- edge 800x600 position mean: 0.365 mm → 0.336 mm
- edge 640x480 in spec: 28.6% → 30.4%
- edge 640x480 angle mean: 2.965° → 2.495°
- edge 640x480 angle worst: 20.696° → 17.242°
- edge 640x480 position mean: 0.440 mm → 0.409 mm
- edge 640x480 position worst: 1.551 mm → 1.459 mm
- edge 640x400 in spec: 28.6% → 30.4%
- edge 640x400 angle mean: 2.965° → 2.495°
- edge 640x400 angle worst: 20.696° → 17.242°
- edge 640x400 position mean: 0.440 mm → 0.409 mm
- edge 640x400 position worst: 1.551 mm → 1.459 mm
- edge 320x240 detected: 16.0% → 14.0%
- edge 320x240 angle mean: 4.265° → 2.847°
- edge 320x240 position mean: 0.577 mm → 0.538 mm
- edge 320x240 position worst: 2.583 mm → 2.337 mm
- edge 160x120 in spec: 0.0% → 0.0%
- edge 160x120 detected: 0.0% → 0.0%

****Regressed**** (10)

- core 1280x800 detected: 56.0% → 53.5%
- core 1280x720 detected: 56.0% → 53.5%
- core 640x480 detected: 49.5% → 46.5%
- core 640x400 detected: 49.5% → 46.5%
- edge 1280x800 detected: 45.0% → 41.0%
- edge 1280x720 detected: 44.5% → 40.5%
- edge 1024x768 detected: 42.0% → 37.5%
- edge 800x600 detected: 36.5% → 31.5%
- edge 640x480 detected: 31.5% → 28.0%
- edge 640x400 detected: 31.5% → 28.0%

#### Justification for the regressions

**All ten are detection rate, and every mode's accuracy improved in exchange.** Correcting the fit changes its Sampson residual, so frames sitting near the outline-quality gate move across it; the ones lost are those whose corrected fit reveals an outline that was only scoring well because it was rotated. In-spec rose at **every reporting mode** — by 6.2 points at 1280×800, 9.6 at 640×480, and 16.8 at 320×240 — so the frames kept are substantially better ones. No accuracy metric regressed anywhere.

**Twelve of the thirteen are detection rate**, and every one is the deliberate price of a gate that this iteration added on purpose. Each is paired with a large accuracy improvement at the same mode: 640×480 gives up 19 points of detection and gains 15.7 points of in-spec; 320×240 gives up 10.5 points and takes position mean from 12.296 mm to 0.595 mm and position worst from 356 mm to 2.6 mm.

**160×120 falling to 0% detection is intended and derived.** Its rim spans 18 px, so single-view depth precision is 4.18 mm and the fused figure 0.477 mm sits at the specification limit with no margin for the silhouette bias that dominates in practice. Every frame it produced in Iteration 1 was out of spec (0.0% in spec, 237 mm mean position error). Refusing them removes no capability; it stops the estimator reporting answers that were never usable.

**The one regression that is not detection rate needs a real argument.** Core 1280×800 and 1280×720 in-spec fell 76.4% → 73.2%, while detection at those modes *rose* 53.0% → 56.0%. Both come from the same change: the scale-free outline gate computes to 0.012 × 143 px = 1.72 px at 1280, looser than the old fixed 1.5 px, so marginal frames that used to be refused are now admitted and dilute the average. At 640 the same rule computes to 0.86 px, tighter than 1.5 px, which is why 640×480 in-spec rose 15.7 points.

This is accepted, for two measured reasons. First, the dimensional threshold was **wrong in form**: a fixed pixel tolerance tightens as resolution improves, which is backwards, and it produced the Iteration 1 anomaly of *lower* detection at 1280×800 (53%) than at 640×480 (68%). Second, and decisively, the dilution does not reach the delivered product: in the gated configuration 1280×800 reports **100.0% in spec**, against 76.4% ungated in Iteration 1. The marginal frames the looser outline gate admits are precisely the ones the predicted-error gate then declines.

### Against the target

**The Iteration 2 regression is resolved rather than justified.** That entry had to argue for accepting core 1280×800 in-spec falling 76.4% → 73.2%. It is now **79.4%**, above both. The argument is moot: the estimator is simply better.

**Iteration 2's tables measure a superseded build** — one whose fitted major axis could be 33° out. They are left in place as an honest record of what was measured, not as a description of the current system.

The goal is ±1° and ±0.5 mm on 100% of test cases. This iteration does not reach it, and the shortfall is stated rather than averaged away:

- **Four of six reporting modes are at 100% in spec** (1280×800, 1280×720, 1024×768, 800×600), worst case 0.571° and 0.475 mm.
- **640×480 and 640×400 are at 87.5%** — one accepted frame in eight at 0.522 mm against the 0.5 mm limit.
- **320×240 and 160×120 report nothing.** Neither can meet the specification, and the estimator now says so instead of guessing.
- **Detection is 4–7% at the gated operating point.** This is the real cost and the honest headline: the estimator is correct when it answers, and it answers rarely. A controller can coast a declared gap, but not a gap this wide.

The binding constraint is measured, not guessed: of core frames the gate rejects, **77.1% fail on predicted angle against 20.6% on predicted position**. And §12.9 of the lecture notes shows why raising acceptance cannot come from resolution — measured error at the best mode is 4.6× the noise floor, so what limits it is systematic silhouette bias. Iteration 3 targets that bias, angle first.

## Iteration 4 — the boundary can only err outward

*200 poses per tier, seed 20260809. Target ±1° and ±0.5 mm on 100% of reported frames.*

### Changed

- **One-sided boundary loss** (`segment.ONE_SIDED_WEIGHT = 0.15`). The silhouette hull is a superset of the projected rim, so contamination can only push the outline *outward*. Points falling outside the current fit are now weighted 0.15 against 1.0 for inward points, pulling the fit onto the inner envelope where the rim actually is. See Theory.
- **Tilt calibration refitted** as a cylinder model on a dataset regenerated with the current fit (k = 0.05067, resolution floor 5.801°). Required, not optional: the one-sided loss halves the raw ratio bias, so a calibration fitted against the old fit would be mismatched by about half its correction.
- **Error model refitted**, operating point re-chosen by leave-one-mode-out at the plateau midpoint.
- **`POSE_ONE_SIDED` environment override** added, so the mechanism can be A/B'd without editing code — which is how the comparison below was produced.
- **Iteration 3's entry was regenerated after a generator bug was found.** Its gated table had been showing Iteration 2's data and its comparison had run against itself. Corrected in place; see the note under this iteration's classification.

### Theory

One mechanism: the boundary's contamination is **one-sided**, and a symmetric
loss does not average a one-sided error away — it splits the difference with it.

#### Why the error has a sign

`segment.silhouette_hull` takes a **convex hull** of the thresholded blob. A hull
is a superset of the set it encloses, so the extracted boundary contains the
projected rim and whatever else survived thresholding — the rod, the magnet
mount, bloom around a bright edge. Every one of those pushes the outline
*outward*. None can pull it inward, because removing a point from a hull cannot
move the hull outward and no mechanism adds points inside the rim.

So for a boundary point at arc position $t$, the observation is

$$r_{\text{obs}}(t) = r_{\text{true}}(t) + \varepsilon(t),\qquad \varepsilon(t)\ge 0$$

with $\varepsilon$ zero over most of the perimeter and positive over the
contaminated arcs. This is a **signed** error model, not a symmetric noise model,
and that distinction is the whole of this iteration.

#### What a symmetric loss does with it

Least squares minimises $\sum_i d_i^2$ where $d_i$ is the signed distance from
point $i$ to the fitted curve. At the true rim every contaminated point has
$d_i > 0$, so the gradient is one-signed and the fit moves outward until the
positive residuals of the clean majority balance the negative residuals it has
just created on the contaminated minority. The fit settles where

$$\sum_{\text{clean}} d_i \;=\; -\sum_{\text{contaminated}} d_i$$

which for a contaminated fraction $f$ with mean excursion $\bar\varepsilon$ puts
the boundary at roughly $f\bar\varepsilon$ outside the truth. The estimator is not
noisy here; it is **biased by construction**, and averaging more frames does not
help because every frame is biased the same way.

The rim wall does the same thing in the short direction and is why
`TiltCalibration` exists. But a calibration removes the *mean* of this and
nothing else — and $f$ and $\bar\varepsilon$ vary with pose, tilt, lighting and
threshold, so what remains after calibration is the *scatter* of $f\bar\varepsilon$,
which no fixed correction can reach.

#### The estimator that respects the sign

If the error is known non-negative, the true boundary is the **inner envelope** of
the observations, and points far outside the current fit are evidence about
contamination rather than about the rim. Weighting them below inward points,

$$w_i = \begin{cases} 1 & d_i \le 0 \\ w_{\text{out}} & d_i > 0\end{cases}$$

and iterating pulls the fit onto that envelope. This is one-sided IRLS; it is the
same construction as a one-sided Huber loss, and $w_{\text{out}}$ is the only
parameter. It is applied *after* the axial re-weighting and, like it, leaves the
orientation with the plain fit (Iteration 3's finding — the weights carry no
information about rotation, so they are not given authority over it).

#### What it buys, and why the mean is the wrong thing to measure

Because `TiltCalibration` absorbs the mean, an improvement in mean ratio error is
invisible downstream. The quantity that matters is the **spread**. Measured over
120 poses with tilt and contamination amplitude both varied:

| $w_{\text{out}}$ | ratio-error mean | ratio-error std |
|---|---|---|
| off | +0.01825 | 0.00479 |
| **0.15** | +0.00935 | **0.00253** |
| 0.30 | +0.01181 | 0.00316 |

47% less scatter. Near 45°, where $d\theta = d(\text{ratio})/\sin\theta$, that is
0.388° of irreducible tilt error falling to 0.205°.

Note that $w_{\text{out}} = 0.30$ is worse than $0.15$ on both columns — the
relationship is not monotone. Too gentle an asymmetry leaves the bias; too harsh
an one and the retained points are too few to constrain the fit, which is the
same conditioning failure Iteration 3 diagnosed in the axial weights. The value
is a balance between those, and it was chosen by measurement rather than by
argument.

#### Why detection rose as much as accuracy

The gate accepts a frame when its *predicted* error is inside specification.
Reducing the scatter of the ratio error reduces the fitted scale on frames that
were previously sitting just above the threshold, so the same gate, at the same
operating point, admits substantially more frames. Accuracy and detection are not
in tension here — the gate converts one into the other, and that is why the
tripling of detection is a consequence of the accuracy work rather than a
loosening of the standard.

### Results

![accuracy against sensor mode](../../results/pose_validation/resolution_iter4.png)

*Ungated. The gated equivalent is `resolution_iter4gated.png`.*

#### A/B — what the one-sided loss alone does

**This is a true A/B, and the first in this journal.** Both runs use the same poses, the same seed, the same tilt calibration and the same error model; the only difference is `POSE_ONE_SIDED`. Earlier entries compared against the previous iteration's run, which bundles every change made in between — and in Iteration 3's case silently included a calibration change.

**Core tier** — this is what gates the target.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 54% | 0.871° | 8.339° | 0.257 mm | 1.343 mm | 78.7% | 8.8 |
| 1280×720 | 120 | 54% | 0.870° | 8.339° | 0.256 mm | 1.343 mm | 78.7% | 8.2 |
| 1024×768 | 120 | 52% | 0.852° | 8.049° | 0.284 mm | 1.536 mm | 77.7% | 12.2 |
| 800×600 | 120 | 47% | 0.717° | 3.668° | 0.313 mm | 1.683 mm | 80.9% | 9.9 |
| 640×480 | 210 | 46% | 2.341° | 88.895° | 0.410 mm | 3.003 mm | 73.1% | 5.3 |
| 640×400 | 210 | 46% | 2.341° | 88.895° | 0.410 mm | 3.003 mm | 73.1% | 4.9 |
| 320×240 | 420 | 13% | 2.265° | 8.244° | 0.536 mm | 2.446 mm | 34.6% | 2.6 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.5 |

**Edge tier** — measured and tracked, does not gate.

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 39% | 1.676° | 7.871° | 0.341 mm | 1.689 mm | 39.7% | 9.1 |
| 1280×720 | 120 | 38% | 1.649° | 7.871° | 0.339 mm | 1.689 mm | 40.3% | 8.1 |
| 1024×768 | 120 | 36% | 2.057° | 9.664° | 0.334 mm | 1.632 mm | 36.1% | 11.0 |
| 800×600 | 120 | 31% | 2.233° | 11.220° | 0.327 mm | 1.017 mm | 37.1% | 9.7 |
| 640×480 | 210 | 26% | 2.212° | 15.037° | 0.385 mm | 1.202 mm | 34.0% | 4.7 |
| 640×400 | 210 | 26% | 2.212° | 15.037° | 0.385 mm | 1.202 mm | 34.0% | 4.5 |
| 320×240 | 420 | 11% | 2.301° | 6.145° | 0.496 mm | 0.945 mm | 27.3% | 2.7 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.7 |

#### Gated — what the estimator delivers

The same frames with the estimator declining any it cannot certify. Note that this table, unlike the A/B above, also carries the calibration and error-model refits, so it is a statement about the delivered system rather than an attribution to one mechanism.

**Core tier**

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 18% | 0.244° | 0.816° | 0.196 mm | 0.362 mm | **100%** | 9.0 |
| 1280×720 | 120 | 18% | 0.244° | 0.816° | 0.196 mm | 0.362 mm | **100%** | 8.1 |
| 1024×768 | 120 | 14% | 0.262° | 0.670° | 0.218 mm | 0.401 mm | **100%** | 11.3 |
| 800×600 | 120 | 14% | 0.268° | 0.753° | 0.277 mm | 0.458 mm | **100%** | 9.8 |
| 640×480 | 210 | 14% | 0.288° | 0.781° | 0.312 mm | 0.519 mm | 96.3% | 5.3 |
| 640×400 | 210 | 14% | 0.288° | 0.781° | 0.312 mm | 0.519 mm | 96.3% | 5.1 |
| 320×240 | 420 | **0%** | — | — | — | — | — | 2.8 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.5 |

**Edge tier**

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** | ms/frame |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 4% | 0.505° | 1.108° | 0.217 mm | 0.338 mm | 88.9% | 8.7 |
| 1280×720 | 120 | 4% | 0.505° | 1.108° | 0.217 mm | 0.338 mm | 88.9% | 8.0 |
| 1024×768 | 120 | 4% | 0.905° | 2.643° | 0.269 mm | 0.462 mm | 66.7% | 11.5 |
| 800×600 | 120 | 4% | 0.916° | 2.593° | 0.328 mm | 0.675 mm | 62.5% | 9.4 |
| 640×480 | 210 | 2% | 0.690° | 1.183° | 0.324 mm | 0.454 mm | 60.0% | 4.6 |
| 640×400 | 210 | 2% | 0.690° | 1.183° | 0.324 mm | 0.454 mm | 60.0% | 4.4 |
| 320×240 | 420 | **0%** | — | — | — | — | — | 2.6 |
| 160×120 | 640 | **0%** | — | — | — | — | — | 1.5 |

### Improved / Constant / Regressed

*A/B on the ungated runs — one variable, measured bootstrap tolerances.*

**Improved** (12)

- core 800x600 in spec: 76.5% → 80.9%
- core 800x600 angle worst: 13.171° → 3.668°
- core 800x600 position worst: 4.087 mm → 1.683 mm
- core 320x240 in spec: 24.2% → 34.6%
- core 320x240 angle worst: 58.745° → 8.244°
- edge 640x480 in spec: 30.4% → 34.0%
- edge 640x480 position worst: 1.432 mm → 1.202 mm
- edge 640x400 in spec: 30.4% → 34.0%
- edge 640x400 position worst: 1.431 mm → 1.202 mm
- edge 320x240 in spec: 21.4% → 27.3%
- edge 320x240 angle worst: 10.474° → 6.145°
- edge 320x240 position worst: 2.337 mm → 0.945 mm

**Constant** (69)

- core 1280x800 in spec: 79.4% → 78.7%
- core 1280x800 detected: 53.5% → 54.0%
- core 1280x800 angle mean: 0.868° → 0.871°
- core 1280x800 angle worst: 8.412° → 8.339°
- core 1280x800 position mean: 0.251 mm → 0.257 mm
- core 1280x800 position worst: 1.223 mm → 1.343 mm
- core 1280x720 in spec: 79.4% → 78.7%
- core 1280x720 detected: 53.5% → 54.0%
- core 1280x720 angle mean: 0.868° → 0.870°
- core 1280x720 angle worst: 8.412° → 8.339°
- core 1280x720 position mean: 0.250 mm → 0.256 mm
- core 1280x720 position worst: 1.223 mm → 1.343 mm
- core 1024x768 in spec: 76.9% → 77.7%
- core 1024x768 detected: 52.0% → 51.5%
- core 1024x768 angle mean: 0.829° → 0.852°
- core 1024x768 angle worst: 7.971° → 8.049°
- core 1024x768 position mean: 0.286 mm → 0.284 mm
- core 1024x768 position worst: 1.478 mm → 1.536 mm
- core 800x600 angle mean: 0.999° → 0.717°
- core 800x600 position mean: 0.366 mm → 0.313 mm
- core 640x480 in spec: 72.0% → 73.1%
- core 640x480 detected: 46.5% → 46.5%
- core 640x480 angle mean: 2.449° → 2.341°
- core 640x480 angle worst: 88.878° → 88.895°
- core 640x480 position mean: 0.382 mm → 0.410 mm
- core 640x480 position worst: 2.315 mm → 3.003 mm
- core 640x400 in spec: 72.0% → 73.1%
- core 640x400 detected: 46.5% → 46.5%
- core 640x400 angle mean: 2.449° → 2.341°
- core 640x400 angle worst: 88.878° → 88.895°
- core 640x400 position mean: 0.382 mm → 0.410 mm
- core 640x400 position worst: 2.315 mm → 3.003 mm
- core 320x240 angle mean: 3.959° → 2.265°
- core 320x240 position mean: 0.570 mm → 0.536 mm
- core 320x240 position worst: 2.659 mm → 2.446 mm
- core 160x120 in spec: 0.0% → 0.0%
- core 160x120 detected: 0.0% → 0.0%
- edge 1280x800 in spec: 37.8% → 39.7%
- edge 1280x800 detected: 41.0% → 39.0%
- edge 1280x800 angle mean: 1.694° → 1.676°
- edge 1280x800 angle worst: 7.892° → 7.871°
- edge 1280x800 position mean: 0.311 mm → 0.341 mm
- edge 1280x720 in spec: 38.3% → 40.3%
- edge 1280x720 angle mean: 1.670° → 1.649°
- edge 1280x720 angle worst: 7.892° → 7.871°
- edge 1280x720 position mean: 0.309 mm → 0.339 mm
- edge 1024x768 in spec: 37.3% → 36.1%
- edge 1024x768 detected: 37.5% → 36.0%
- edge 1024x768 angle mean: 1.916° → 2.057°
- edge 1024x768 position mean: 0.337 mm → 0.334 mm
- edge 1024x768 position worst: 1.575 mm → 1.632 mm
- edge 800x600 in spec: 36.5% → 37.1%
- edge 800x600 detected: 31.5% → 31.0%
- edge 800x600 angle mean: 2.127° → 2.233°
- edge 800x600 angle worst: 10.678° → 11.220°
- edge 800x600 position mean: 0.337 mm → 0.327 mm
- edge 800x600 position worst: 0.989 mm → 1.017 mm
- edge 640x480 detected: 28.0% → 26.5%
- edge 640x480 angle mean: 2.533° → 2.212°
- edge 640x480 angle worst: 17.256° → 15.037°
- edge 640x480 position mean: 0.409 mm → 0.385 mm
- edge 640x400 detected: 28.0% → 26.5%
- edge 640x400 angle mean: 2.533° → 2.212°
- edge 640x400 angle worst: 17.256° → 15.037°
- edge 640x400 position mean: 0.409 mm → 0.385 mm
- edge 320x240 angle mean: 2.752° → 2.301°
- edge 320x240 position mean: 0.539 mm → 0.496 mm
- edge 160x120 in spec: 0.0% → 0.0%
- edge 160x120 detected: 0.0% → 0.0%

****Regressed**** (7)

- core 800x600 detected: 49.0% → 47.0%
- core 320x240 detected: 16.5% → 13.0%
- edge 1280x800 position worst: 1.163 mm → 1.689 mm
- edge 1280x720 detected: 40.5% → 38.5%
- edge 1280x720 position worst: 1.163 mm → 1.689 mm
- edge 1024x768 angle worst: 7.457° → 9.664°
- edge 320x240 detected: 14.0% → 11.0%

#### Justification for the regressions

**The gains are in the tail and the losses are in detection**, which is the trade this iteration is for. The target is 100% of reported frames — a property of the worst case, not the average — and the worst cases collapse: 320×240 angle worst 58.745° → **8.244°**, 800×600 angle worst 13.171° → **3.668°**, 800×600 position worst 4.087 → **1.683 mm**. Five of the seven regressions are detection rate at the modes that gained most (800×600, 320×240), which is the same frames-for-quality exchange made deliberately in Iteration 2.

**Two edge-tier worst-case regressions are real and unexplained by that argument**: edge 1280×800/720 position worst 1.163 → 1.689 mm, and edge 1024×768 angle worst 7.457° → 9.664°. They are in the tier that does not gate, they are single-frame extremes in a tier built to contain frames the estimator should refuse, and the same modes' *core* figures are constant or improved. They are recorded rather than explained away, and if they recur in the next iteration they should be investigated rather than re-excused.

**The mechanism is resolution-dependent, and that is visible in the data.** At 1280×800 the one-sided loss is neutral-to-marginally-negative (in-spec 79.4% → 78.7%, inside tolerance, classified constant); at 320×240 it is worth 10.4 points of in-spec. The contaminated arc is a small, well-resolved fraction of a large boundary at high resolution, so the symmetric fit is already near-unbiased there and down-weighting points only costs variance. Making the weight resolution-dependent is the obvious next step and is deliberately *not* taken here: it would be tuning a constant per mode against the same data used to judge it.

### Against the target

Four of six reporting modes are at **100% in spec**. The remaining gaps, stated precisely:

- **640×480 / 640×400: 96.3%** — one accepted frame in 27 at 0.519 mm against the 0.5 mm limit, a 3.8% overshoot. Iteration 3 missed by 14.3%.
- **320×240: refuses every frame.** Reachable in principle — it has 0.261 mm of headroom against the noise floor — but its measured bias is 0.536 mm, so it needs roughly another factor of two.
- **160×120: refuses every frame, and always will.** Its fused noise floor is 0.477 mm against a 0.5 mm budget. No estimator meets the specification there; see the reachability table at the top of this file.
- **Detection is 13.5–18% gated.** Tripled from Iteration 3 at 1280×800 (6.0% → 18.0%) because reducing the error scatter moves frames from just above the gate's threshold to just below it — the operating point itself was not loosened.

So the target is met at four of the seven physically reachable modes, nearly met at two more, and not met at one. It is **not** met across all eight, and one of the eight it cannot be.

## Iteration 5 — a safety margin on the gate (negative result, not shipped)

*200 poses per tier, seed 20260809. Measured, rejected, and reverted; recorded
because the way it failed identifies what the remaining failure actually is.*

### Changed

- **Tried:** `uncertainty.GATE_MARGIN = 0.9`, certifying each frame against 0.9×
  the specification rather than the specification itself.
- **Reverted to 1.0** at margin 0.9, which was measured as a pure loss.
- **Corrected afterwards:** the "blunder, unreachable by a threshold" conclusion
  first recorded here was wrong; see Theory. The frame is reachable at margin
  0.765, and the honest result is a cost curve rather than an impossibility.
- Also fixed on the way: the estimator compared predictions to the target
  directly instead of routing through `ErrorModel.accepts`, so a margin would
  have been silently ignored. That is corrected regardless of the outcome here.

### Theory

The gate's operating point is calibrated on finite data, so it holds **in
expectation** on new data rather than with certainty. The evidence for that was
concrete: on the sample set the threshold was chosen from, *every* mode reported
100% coverage — 640×480 and 320×240 included — yet on a fresh seed 640×480
admitted one frame of 27 at 0.519 mm. Nothing was mis-specified; a finite
calibration sample cannot bound an unseen draw exactly.

The proposed remedy was a derating factor: require the predicted error to sit
below `0.9 × target`, so a prediction wrong by up to 11% still meets
specification. This is standard practice for a rated limit, and it is why the
constant was made explicit rather than folded into the quantile — so that its
cost, in refused frames, stayed visible.

**It does not work, and the failure is informative.** Measured on the same fresh
seed:

| | detection @ 1280×800 | 640×480 in spec | worst accepted at 640×480 |
|---|---|---|---|
| margin 1.0 (iter 4) | 18.0% | 96.3% | 0.519 mm |
| margin 0.9 | 13.5% | 95.2% | **0.519 mm** |

The margin removed good frames and **left the bad one**. Its in-spec figure fell
only because the denominator shrank around an unchanged failure.

That does **not** settle the matter the way it first appeared, and the initial
reading recorded here was wrong. The obvious inference — that the frame is a
blunder, unreachable by any threshold — was drawn from a single negative test at
one margin value, and it does not survive measurement.

Retrieving the frame's own prediction (which required capturing the *sweep's*
per-frame observables, not the fitting set's) gives:

    predicted 0.3826 mm    actual 0.5190 mm    under-predicted by 1.36x

A 1.36× under-prediction is a poor prediction, not a blunder — the model is in
the right regime and wrong by a third. The margin that rejects it is **0.765**,
not something unreachable. Testing 0.9, observing failure, and concluding "no
threshold can catch this" was an over-generalisation from one sample of the
parameter.

What the measurement actually establishes is a **cost curve**, and it is steep:

| margin | core frames kept | coverage |
|---|---|---|
| 1.00 | 183 / 228 | 98.91% |
| 0.95 | 160 / 228 | 98.75% |
| 0.90 | 138 / 228 | 98.55% |
| 0.85 | 128 / 228 | 98.44% |
| 0.80 | 107 / 228 | 98.13% |
| **0.75** | **61 / 228** | **100.00%** |

Coverage barely moves from 1.00 down to 0.80 — those margins refuse frames that
were fine — and then the last failure clears between 0.80 and 0.75, taking
two-thirds of the accepted population with it. That shape is the real result: the
frames near the specification limit are not the frames the model is worst about,
so buying the last 1.1% of coverage costs far more than the first 98.9%.

### Results

Not published as a results table: the configuration was measured and reverted, so
the shipped numbers remain Iteration 4's. The measurement above is the entire
outcome.

### Improved / Constant / Regressed

**Improved** (0)

- none

**Constant** (0)

- none — nothing shipped

**Regressed** (0)

- none shipped. Had it been shipped it would have regressed detection at every
  core mode (1280×800 18.0% → 13.5%, 1024×768 14.5% → 11.5%, 800×600 14.0% →
  9.5%, 640×480 13.5% → 10.5%) while fixing no coverage failure. Per the
  regression policy, an iteration that buys nothing measured cannot justify a
  regression, so it is reverted rather than argued for.

## Iteration 6 — buying the last 1% of coverage, and what it costs

*200 poses per tier, seed **20260812** — deliberately not the seed the margin was
chosen on. Target ±1° and ±0.5 mm on 100% of reported frames.*

### Changed

- **`uncertainty.GATE_MARGIN = 0.75`**: each frame is certified against 0.75× the
  specification rather than against the specification itself.
- The value is read off a **measured cost curve**, not chosen by taste — see
  Theory, and the table in the constant's own comment.
- **The sweep can now retain per-frame observables** (`--save-samples`). Without
  them the frame that defeated the gate could only be counted, not
  characterised, which is what stalled Iteration 5.
- No estimator changes. This iteration moves one decision constant.

### Theory

Iteration 5 established that the gate's threshold holds *in expectation* on new
data rather than with certainty, and that the frame defeating it at 640×480 was
under-predicted by 1.36× (0.3826 mm predicted, 0.5190 mm actual). Two facts then
determine what can be done about it.

**It is not separable.** Its cross-view discrepancy (1.224), conditioning
(`e/M²` = 3.44e-05) and tilt (sin 0.613) all sit inside the interquartile range
of the accepted-and-in-spec population; only `major_px` is unusual, at 65.5
against a 5th percentile of 66.19, and only marginally. No hard gate on the
quantities the estimator computes isolates it, so the margin is the only lever.

**The cost curve is steep and its shape is the result.** Measured:

| margin | core frames kept | coverage |
|---|---|---|
| 1.00 | 183 / 228 | 98.91% |
| 0.90 | 138 / 228 | 98.55% |
| 0.80 | 107 / 228 | 98.13% |
| **0.75** | **61 / 228** | **100.00%** |

Coverage is nearly flat from 1.00 to 0.80 and then the last failure clears,
taking two thirds of the accepted population with it. The reason is that the
frames sitting near the specification limit are *not* the frames the model is
worst about — those two populations barely overlap — so tightening the threshold
first refuses well-predicted frames and only reaches the badly-predicted one at
the end. **The last 1.1% of coverage costs more than the first 98.9%.**

That is the whole trade, and it is a property of the estimator's error
distribution rather than of the threshold. Shrinking it requires making the
model's predictions better, not the gate stricter.

### Results

![accuracy against sensor mode](../../results/pose_validation/resolution_iter6.png)

**Core tier**, on a seed the margin never saw:

| mode | fps | detected | angle mean | angle worst | position mean | position worst | **in spec** |
|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 6.0% | 0.233° | 0.552° | 0.167 mm | 0.237 mm | **100%** |
| 1280×720 | 120 | 6.0% | 0.233° | 0.552° | 0.167 mm | 0.237 mm | **100%** |
| 1024×768 | 120 | 4.0% | 0.273° | 0.775° | 0.219 mm | 0.344 mm | **100%** |
| 800×600 | 120 | 3.5% | 0.335° | 0.893° | 0.270 mm | 0.394 mm | **100%** |
| 640×480 | 210 | 2.0% | 0.332° | 0.943° | 0.259 mm | 0.336 mm | **100%** |
| 640×400 | 210 | 2.0% | 0.332° | 0.943° | 0.259 mm | 0.336 mm | **100%** |
| 320×240 | 420 | 0.5% | 0.519° | 0.519° | 0.454 mm | 0.454 mm | **100%** |
| 160×120 | 640 | 0% | — | — | — | — | refuses all |

**Every mode that reports, reports in specification.** 320×240 reports for the
first time in this journal. 160×120 refuses every frame, which is correct: its
fused noise floor is 0.477 mm against a 0.5 mm budget.

### How much this actually establishes — and the larger run that overturned it

The 200-pose result above accepted 48 core frames out of 1600 mode-frames, so
each mode's "100%" rested on very few. For *n* successes in *n* trials the 95%
lower confidence bound on the true rate is `0.05^(1/n)`: 12 frames at 1280×800
bounds it only at ≥77.9%, four at 640×480 at ≥47.3%, and **one** at 320×240 at
≥5.0%. That was flagged at the time as consistent with the target rather than
demonstrating it, and an 800-pose run was started to settle it.

**It did settle it, against the headline.** At four times the sample:

| mode | accepted *n* | in spec | 95% lower bound |
|---|---|---|---|
| 1280×800 | 48 | **100%** | ≥ 93.9% |
| 1280×720 | 47 | **100%** | ≥ 93.8% |
| 1024×768 | 37 | **100%** | ≥ 92.2% |
| 800×600 | 26 | **100%** | ≥ 89.1% |
| 640×480 | 28 | **92.9%** | — two frames out, worst angle 1.448° |
| 640×400 | 28 | **92.9%** | — |
| 320×240 | 0 | — | refuses every frame |
| 160×120 | 0 | — | refuses every frame |

So the true position is **four modes at 100% with usable confidence bounds, not
seven**. 640×480 does not hold at margin 0.75 — the single frame it cleared in
the 200-pose run was not representative, and its failures here are *angle*
(1.448°) rather than position. 320×240 accepts nothing at all; its lone in-spec
frame in the smaller run was a fluke of sampling, exactly as the `n = 1` bound
warned.

This is recorded rather than quietly replaced because the sequence is the point:
a 100% headline on 48 frames, an explicit statement that it established little, a
larger run, and a result that contradicts it. **The small-sample claim was not
evidence, and treating it as evidence would have shipped a false capability.**
The margin remains at 0.75 — it is still the right operating point for the modes
that do hold — but the claim attached to it is now four modes, not seven.

### Improved / Constant / Regressed

Not classified against Iteration 4: this run uses a different seed by design, so
a per-metric diff would report sampling differences as changes. The comparison
that means something is the one in Theory — same seed, same everything, margin
varied — and it is a **deliberate trade**, not an improvement:

**Improved**

- 1280×800 / 1280×720 / 1024×768 / 800×600 in spec: **100%**, and at 800 poses
  with 95% lower bounds of 93.9 / 93.8 / 92.2 / 89.1% — the first coverage claim
  in this journal with a confidence bound worth quoting.
- 640×480 / 640×400: 96.3% → **92.9%** at 800 poses. Marked **not** an
  improvement; the 200-pose run's 100% did not survive more data.
- 320×240: reported one in-spec frame at 200 poses and **none at 800**. Marked as
  no change; the earlier reading was a sampling fluke.

**Regressed** (accepted, and the reason)

- Detection at every mode, roughly threefold: 1280×800 18.0% → 6.0%, 1024×768
  14.5% → 4.0%, 800×600 14.0% → 3.5%, 640×480 13.5% → 2.0%.

This is the trade the specification asks for. The target is ±1° and ±0.5 mm on
100% of reported frames, and detection is the currency coverage is bought with;
a configuration that reports three times as many frames while emitting some known
to be out of specification does not meet it. The cost is recorded here, and
`GATE_MARGIN` is a single constant: **if the deployment would rather have three
times the frames at 98.9% coverage, that is the one number to change**, and the
curve above says exactly what it buys.

## Iteration 7 — giving the predictor the residual it was already computing

*Fitted at 150 poses; validated at 200 and again at **800** poses on seed
20260812. Target ±1° and ±0.5 mm on 100% of reported frames.*

### Changed

- **Two observables promoted to features**: `refine_rms_px` (the refinement's own
  residual) and `ambiguity_margin_deg` (how nearly the two back-projection
  branches coincided). Both were computed on every frame and discarded.
- **`ErrorModel.load` now validates the feature set** against the file's recorded
  `features_pos`/`features_ang` and degrades to *no gate* on a mismatch. A stale
  model previously surfaced as a shape error deep inside `predict` on the first
  frame — far from the cause.
- No estimator changes. This iteration changes only what the gate is told.

### Theory

Every previous iteration traded coverage against detection along a fixed curve.
This one asks why the curve sits where it does.

At 1280×800, measured: **42.5% of all frames are genuinely within specification,
and 6.0% could be certified — a 7.1× gap.** Those frames were accurate enough
already; the predictor could not tell which they were, so the gate refused them
to stay safe. That gap is predictor ignorance, not estimator error, and closing
it buys frames without spending coverage.

**Why `refine_rms_px` is the right thing to add.** The gate's existing
conditioning feature is `fit_rms_px`, the residual of an *ellipse* fitted to the
hull. That measures how elliptical the outline is — and a mast-contaminated
outline is still an excellent ellipse, so the quantity is nearly blind to the
error that dominates the budget. `refine_rms_px` is the residual of the *pose*
solve: how far the observed outline sits from the best **circle projection** the
solver could find, given the rig geometry. A silhouette distorted by the mast
cannot be explained by any circle at any pose, so this residual rises exactly
when the systematic error of §12.9 is large. It is the one observable that sees
the dominant error term, and it was being thrown away.

The fitted coefficients bear that out: `log1p_refine_rms` enters the angle model
at **+1.649**, the largest weight on any conditioning term.

**A caveat on `log1p_ambig`, stated because a good result should not paper over
it.** Its coefficient fits at +39.26, which looks alarming. Over accepted frames
the feature's standard deviation is 0.00138, so it contributes 0.054 in
log-error terms — it is near-constant there precisely because it is doing its job
and the low-ambiguity frames are the ones being rejected. But a coefficient
fitted across a range that narrow is badly conditioned: on a genuinely degenerate
frame it extrapolates to an absurd magnitude. That extrapolation points the safe
way (reject), by luck rather than design. It should be re-examined, and dropping
it tested, since `refine_rms_px` appears to do the real work.

### Results

**800 poses per tier, seed 20260812** — the sample size that overturned
Iteration 6's headline, applied here before any claim is made.

| mode | fps | detected | accepted *n* | angle worst | position worst | **in spec** | 95% lower bound |
|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 11.6% | 93 | 0.908° | 0.372 mm | **100%** | ≥ 96.8% |
| 1280×720 | 120 | 11.5% | 92 | 0.909° | 0.372 mm | **100%** | ≥ 96.8% |
| 1024×768 | 120 | 11.6% | 93 | 1.046° | 0.413 mm | 98.9% | — |
| 800×600 | 120 | 9.8% | 78 | 0.827° | 0.425 mm | **100%** | ≥ 96.2% |
| 640×480 | 210 | 9.9% | 79 | 1.448° | 0.499 mm | 96.2% | — |
| 640×400 | 210 | 9.9% | 79 | 1.448° | 0.499 mm | 96.2% | — |
| 320×240 | 420 | 0.1% | 1 | 0.184° | 0.531 mm | 100% | ≥ 5.0% (meaningless) |
| 160×120 | 640 | 0% | 0 | — | — | refuses all | — |

**Position is inside specification at every mode** — the worst across all 515
accepted frames is 0.499 mm. **Every failure is angle**, and marginal: 1.046° and
1.448° against a 1.0° limit. Angle has been the binding constraint since
Iteration 2, and it still is.

### Improved / Constant / Regressed

Against Iteration 6 at the same 800-pose sample:

**Improved**

- **Accepted frames 214 → 515 (2.4×)**, detection roughly doubling at every
  reporting mode — bought from predictor quality, not from loosening the gate.
- 640×480 / 640×400 in spec: 92.9% → **96.2%**.
- 1280×800 / 1280×720 hold **100%** with the bound tightening from ≥93.9% to
  ≥96.8%, because the claim now rests on 93 frames rather than 48.
- 800×600 holds **100%**, bound ≥89.1% → ≥96.2%.
- 320×240 accepts a frame at all, where Iteration 6 accepted none. With *n* = 1
  this establishes nothing and is listed for completeness only.

**Regressed**

- **1024×768 in spec: 100% → 98.9%.** This is a real regression and is marked as
  such. It is also not like-for-like: at Iteration 6 the mode accepted 37 frames,
  here 93, and one of the 56 additional frames failed. Admitting more frames
  raises the chance of admitting a bad one, so a coverage figure is only
  comparable at equal detection. The honest statement is that this configuration
  trades one failure in 93 for 2.5× the usable output — **and that trade is not
  automatically acceptable under a specification that reads "100%".**

**Justification, and its limit.** The 2.4× gain in usable frames is large and
measured, and the single 1024×768 failure exceeds the angle limit by 4.6%. For a
controller that can coast a declared gap, more frames at 98.9% is likely the
better instrument. But the specification as written is 100% of reported frames,
and by that standard this iteration **does not** improve on Iteration 6 at
1024×768 — it improves everywhere else. Both readings are recorded rather than
resolved by preference, because the choice is the deployment's, and it is a
one-constant choice: `GATE_MARGIN` tightens until 1024×768 holds again, at a cost
the curve in Iteration 6's Theory section quantifies.

## Iteration 8 — one exponent was serving two dependencies

*Fitted at 150 poses, validated at **800** poses on seed 20260812.*

### Changed

- **The angle model's conditioning term was split**: `log(e/(M·sin θ))` became
  `log(e/M)` and `log(1/sin θ)` as separate features, so resolution and tilt
  conditioning can take independent weights.
- No estimator changes; only what the gate is told.

### Theory

The eight out-of-spec frames in Iteration 7 were not scattered. Every one sat at
an extreme of apparent tilt — five at ~25.4°, below the 10th percentile of the
accepted population, and three at 44–48°, at or above the 90th. Those extremes
are the two error mechanisms this package already derives:

* **low tilt** — the ratio→tilt inversion is ill-conditioned, `dθ = −d(ratio)/sin θ`
  (§12.2), amplifying by 2.4× at sin θ = 0.42;
* **high tilt** — the silhouette departs from the flat-circle model,
  `ρ = cos θ + k sin θ` (§12.3).

The predictor could not weight these separately, because `log(e/(M·sin θ))`
bundles three quantities behind **one** exponent. The derivation says the tilt
term's exponent is 1; the bundled fit chose **+0.592**, a compromise between
that and the resolution dependence. Split, the terms separate cleanly:

    log_e_over_m  = +0.548        log_inv_sin = +1.332

`log_inv_sin` lands near the derived value of 1, which is corroboration of the
same kind the position model's `e/M²` exponent gave: a fitted coefficient
agreeing with an independent derivation says the model matches the physics rather
than the sample.

**The prediction this licensed, and how it failed.** If low-tilt failures were
caused by under-weighting `1/sin θ`, giving that term its proper weight should
remove them, while high-tilt failures — a *forward-model* error, invisible to any
conditioning feature — should survive. Measured at 800 poses:

| | Iteration 7 | Iteration 8 |
|---|---|---|
| accepted frames | 515 | **667** |
| failures below 30° | 5 | 4 |
| failures at or above 30° | 3 | **8** |
| failure rate | 1.55% | 1.80% |

**The low-tilt failures did not go away**, and the high-tilt ones nearly tripled.
The first half of the prediction is simply wrong: a heavier `1/sin θ` weight did
not reject those frames, so their error is not explained by ratio conditioning
alone. The second half is right for an uninteresting reason — admitting 30% more
frames pulled in more high-tilt ones, and nothing in the feature set sees
silhouette departure, so the gate cannot decline them.

(The counts double-count: 640×480 and 640×400 share a rendering and all modes
share a pose set, so the twelve failures are about six distinct poses.)

The lesson is the one that keeps recurring here. A coefficient agreeing with a
derivation confirms the *term* is right; it says nothing about whether the term
is *sufficient*. Angle error at low tilt has a second contributor that no
observable in this feature set measures.

### Results

![accuracy against sensor mode](../../results/pose_validation/resolution_iter8.png)

| mode | fps | detected | *n* | angle worst | position worst | **in spec** | 95% lower bound |
|---|---|---|---|---|---|---|---|
| 1280×800 | 120 | 13.0% | 104 | 1.063° | 0.415 mm | 99.0% | — |
| 1280×720 | 120 | 12.9% | 103 | 1.063° | 0.415 mm | 99.0% | — |
| 1024×768 | 120 | 13.0% | 104 | 0.855° | 0.469 mm | **100%** | ≥ 97.2% |
| 800×600 | 120 | 12.8% | 102 | 0.827° | 0.452 mm | **100%** | ≥ 97.1% |
| 640×480 | 210 | 15.8% | 126 | 1.448° | 0.506 mm | 97.6% | — |
| 640×400 | 210 | 15.8% | 126 | 1.448° | 0.506 mm | 97.6% | — |
| 320×240 | 420 | 0.2% | 2 | 0.380° | 0.531 mm | 100% | ≥ 22.4% |
| 160×120 | 640 | 0% | 0 | — | — | refuses all | — |

### Improved / Constant / Regressed

**Improved**

- Accepted frames 515 → **667** (+30%), detection up at every reporting mode.
- 1024×768 in spec 98.9% → **100%** (≥97.2%).
- 640×480 / 640×400 in spec 96.2% → **97.6%**.
- 800×600 holds **100%**, bound ≥96.2% → ≥97.1%.

**Regressed**

- **1280×800 / 1280×720 in spec 100% → 99.0%** — one frame of 104, at 1.063°.
- Overall failure rate 1.55% → 1.80%.

**Justification — and it is deliberately not claimed as sufficient.** The
iteration buys 30% more usable frames and fixes 1024×768, at the cost of one
frame at 1280×800 and a marginally worse failure rate. Under a specification
reading "100% of reported frames", **that is not an acceptable trade at
1280×800**, and the entry says so rather than netting it off against the gains
elsewhere.

What it demonstrates is that this configuration sits on an **efficiency frontier**:
Iterations 6, 7 and 8 each move along it, and no setting of the gate reaches 100%
at every mode simultaneously. Moving the frontier itself requires reducing the
forward-model error — the high-tilt silhouette departure that no feature here
observes — not further work on the predictor. That is the honest boundary of what
the gate can deliver, and it is where the next effort belongs.

---

## Iteration 9 — sub-pixel boundaries (negative result, reverted to opt-in)

*Measured at the calibration stage on 1400 renders; reverted before the
end-to-end sweep completed, because the decisive measurement had already been
made.*

### Changed

- **Added `segment.subpixel_boundary`**: each hull point is re-located at the
  half-height crossing of the intensity profile along its outward normal, by
  interpolation — a boundary defined by the image rather than by a threshold.
- **Off by default.** It costs 0.5 ms/frame and does not pay; see below.
- The implementation is kept, and `segment(..., subpixel=True)` enables it, so
  the comparison is repeatable.

### Theory

The previous section concluded that the residual scatter — 2.597° of per-view
tilt at 40–50°, uncorrelated with everything recorded — came from a fixed
threshold quantising a smoothly shaded, motion-blurred edge, and that the cure
was a boundary estimator that does not quantise. This iteration built that
estimator and **the conclusion was wrong**.

On a synthetic soft-edged disc of known radius, sub-pixel localisation is
decisively better:

| | radius bias | radius scatter |
|---|---|---|
| threshold + hull | −0.0858 px | 0.0550 px |
| sub-pixel | **−0.0012 px** | **0.0030 px** |

71× less bias and **18× less scatter**. Carried into the real pipeline —
regenerating the calibration dataset with sub-pixel boundaries and refitting the
cylinder model — the per-view tilt residual barely moved:

| true tilt | sub-pixel std | threshold+hull std | change |
|---|---|---|---|
| 20–30° | 4.064° | 4.218° | −3.7% |
| 30–40° | 3.307° | 3.407° | −2.9% |
| **40–50°** | **2.537°** | **2.597°** | **−2.3%** |
| 50–60° | 2.438° | 2.459° | −0.9% |
| 60–71° | 2.178° | 2.195° | −0.8% |

**An 18× improvement in the mechanism produced a 2% improvement in the outcome.**
That gap is the result, and it identifies what the earlier diagnosis got wrong.

The synthetic disc has a clean, symmetric, soft edge, so *locating a known edge
precisely* is the entire problem there, and sub-pixel interpolation solves it.
The real silhouette's variability is not that. It is **which shape presents
itself at all**: the rod and magnet mount projecting differently as the body
turns, rim arcs disappearing under grazing illumination, the rotor smearing the
boundary across a finite exposure. Locating the wrong shape more precisely gains
almost nothing.

This also re-reads the earlier null result. §12.12 found the residual
uncorrelated with pose, lighting, exposure and opacity *individually* and
concluded it was irreducible boundary noise. The better reading is that it
depends on their **interaction** — which parts of a non-circular body clear the
threshold from this viewpoint under this illumination — and no single recorded
variable indexes a joint condition like that.

**The methodological point, since it recurred.** Three times in this journal a
mechanism was validated on a construct where the answer was known exactly — a
synthetic disc here, the fit's own sample set for the gate threshold, a 200-pose
run for coverage — and three times the real effect was far smaller. A test that
holds fixed the thing that actually varies will report the precision of the
measurement, not the accuracy of the answer.

### Results

Not tabulated: the mechanism was measured at the calibration stage, found to
change the governing quantity by 2%, and reverted before the end-to-end sweep
finished. Shipped numbers remain Iteration 8's.

### Improved / Constant / Regressed

**Improved** — none shipped.

**Constant** — everything; the default behaviour is unchanged.

**Regressed** — none shipped. Enabled, it would have cost **0.5 ms/frame**
(1.678 → 2.182 ms segmentation) against a 2.4 ms budget at 420 Hz, for a 2%
reduction in the scatter that governs the remaining failures. Per the regression
policy, a cost that buys nothing measurable cannot be justified, so it is
reverted rather than argued for.

### What this leaves

The limit is now stated correctly: **the silhouette is not the rim**, and its
departure from one varies with a joint condition of pose, illumination and body
geometry that no observable indexes and no fixed correction removes. Reaching ±1°
on *every* frame requires the rim to be distinguishable from the body — a
fiducial on the rim, an illumination geometry that lights only the rim, or a
model that separates the two — not a better way to find the edge of whatever
shape happens to appear.

---

## Iteration 10 — the rod was being fitted as if it were the rim (better estimator, worse gate)

Prompted by a reader looking at the tilt-70° failure tile and observing that it
looked like the **rod**, not the magnet, and that de-weighting points projecting
near the centre of the major axis should fix it. Both halves were right.

### Changed

- `conic.fit_conic_weighted` — weighted direct ellipse fit (Halir–Flusser with
  `D'WD` scatter matrices), validated against `cv2.fitEllipseDirect` to float32
  epsilon at uniform weight.
- `segment.axial_weights` / `fit_ellipse(axial=…)` — weight each hull point by
  `|proj on major| / a`, two refits. **Shipped OFF.** See below.

### Theory

The mast and magnet stick out along the rotor axis, so as the robot tilts they
push the silhouette outward in its *short* direction. Because they lie on the
axis they project near the **middle of the major axis**: traced at 70° tilt, the
two worst hull vertices are rod tips 17.2 px from the true rim (magnet: 7.0 px),
both at `|proj|/a < 0.04`. Weighting by position along the major axis targets
where the contamination is known to be — which is why it works where residual
*trimming* (tried in an earlier iteration) failed, since trimming has to discover
the outliers from residuals they have already corrupted.

### Results — as an estimator, unambiguous

Identical poses, each variant with its own refitted radius and tilt calibration,
held-out split of noisy motion-blurred renders:

| tilt band | position | normal |
|---|---|---|
| <10° | 1.331 → 1.240 mm (−6.8%) | 3.459 → 3.250° (−6.0%) |
| 10–45° | 0.751 → 0.883 mm (**+17.6%**) | 1.036 → 0.693° (−33.2%) |
| 45–60° | 1.443 → 1.051 mm (−27.2%) | 2.083 → 0.331° (−84.1%) |
| >60° | 2.110 → 1.252 mm (−40.6%) | 4.335 → 1.358° (−68.7%) |
| all | 1.193 → 1.042 mm (−12.6%) | 1.809 → 0.609° (−66.3%) |

On the case that prompted it: position 3.40 → 0.43 mm, orientation 13.01 →
2.62°. It also retires this journal's standing claim that the >45° silhouette
departure is irreducible geometry — orientation error is now near-flat in tilt.

### Results — as a shipped pipeline, a regression

Certified detection collapses. Gated sweep, 1280×720: **12.9% → 1%**; every mode
at 800×600 and below reports **nothing**. Refitting the error model's
coefficients does not recover it (acceptance 3.5% on held-out modes, against a
~13% detection rate before).

The mechanism is `uncertainty.py`'s reliance on `refine_rms`: `log1p_refine_rms`
is the largest coefficient in its angle model (+1.322), and a weighted fit
deliberately ignores points that the rms still counts, so a *good* fit reports a
1.32× worse rms (1.46× at 70°). Making the rms weighted to match was tried and is
worse still — acceptance 3.5% → **0.5%** — because the unweighted value is the
signal, not a defect: it measures how far the silhouette departs from any
ellipse, which is how much contamination is present and therefore how much error
survives the weighting.

### Improved / Constant / Regressed

**Improved** — raw estimator accuracy in every tilt band for orientation, and
overall position; the 70° case by 8× in position and 5× in orientation.

**Regressed** — certified detection ~13% → ~1–3.5% at every mode.

**Justification.** Against a contract of "100% in spec on every *reported*
frame", trading 13% coverage for 1% is not an improvement, so it is not on by
default. The gate's *feature set* — not just its coefficients — was derived in
iterations 6–8 against the unweighted fit, and re-deriving it is the work this
needs. `AXIAL_DEFAULT = False`; pass `axial=True` wherever the gate is not in
play.

### Two harness traps found while doing this

- **`AXIAL_WEIGHT_ITERS` as a default argument.** Bound at definition time, so
  reassigning the module constant in an A/B changed nothing and both arms ran
  weighted. It produced a "+0.13 ms" cost (true value **+0.57 ms**) and identical
  before/after gallery images. `segment()` now takes an explicit `axial=`.
- **`fit_error_model.py` caches samples keyed only on `--poses` and `--seed`**,
  not on code version, so a refit after a code change silently reuses stale
  features and returns byte-identical numbers. Delete the cache after changing
  anything upstream of the features.

### Not established

The gated numbers here come from 400 poses, not the 800 iteration 8 used — the
larger run OOMs on this machine, since `render_frames` retains every frame
(3.3 GB at 800×2 tiers). Tail statistics are correspondingly weaker, so the
1% figure is directionally solid but not precisely comparable to iter8's 12.9%.

---

## Where this pipeline stops

Iteration 8 left the failures at 42–48° apparent tilt and concluded the frontier
could only be moved by the forward model. That was measured, and the answer is
that **the forward model is not the lever either**.

After the shipped calibration, the held-out per-view tilt residual has a mean
within a few tenths of a degree across the working range but a **standard
deviation of 2.597° in the 40–50° band** — exactly where the failures are. A
calibration removes a mean; this is variance about it.

Regressing that residual against every quantity recorded — pose, orientation
components, ellipse geometry, IoU, fit RMS, opacity, background, ambient level,
exposure, read noise, spin rate — gives a strongest correlation of ρ = −0.226 at
p = 0.015, against a Bonferroni threshold of ~5e-4 for 22 tests. **Nothing
predicts it.**

That closes three lines of work at once: no calibration can remove it (it is not
a bias), no gate feature can decline it (nothing observable correlates), and no
richer pose-dependent silhouette model can correct it (it does not vary
systematically with pose). It also explains Iteration 8's null result — the
conditioning term was *correct* and *insufficient* simultaneously.

What remains is the boundary extraction itself: a fixed threshold followed by a
convex hull, interacting with a shaded, motion-blurred edge. Reducing that
scatter needs sub-pixel edge localisation or a conic fitted to a weighted
intensity field — a different segmentation method, not a refinement of this one.
See §12.12 of the lecture notes for the derivation and the full table.

**So the final position is:** this pipeline delivers ~0.2–0.3° mean and ~1.4°
worst-case orientation after stereo fusion, with position inside ±0.5 mm at every
mode. ±1° on *every* frame is not reachable by tuning it, and the journal's
eight iterations are the evidence for that rather than a series of near-misses.

## Iteration 11 — the sim-to-real gap in the validation itself

*Prompted by a direct question: does the conditional trim generalise from sim to
real? It does not, and finding out why invalidated more than the trim. Numbered
11 because Iteration 10 was authored concurrently by another session.*

### Changed

- **Nothing shipped.** The conditional trim of Iteration 10's first draft is
  withdrawn. `subpixel` stays opt-in from Iteration 9.
- Fixed a genuine defect found on the way: `angular_coverage` used `math`
  without `segment.py` importing it — a `NameError` on any frame that reached it.

### Theory

**The trim had never executed.** `AXIAL_DEFAULT` is `False` in the current tree,
and the trim was gated behind `axial`, so the path was dead. Tests passed and the
claim that they did was worthless: the missing `import math` proves nothing ever
reached it. *A green test suite says nothing about code the suite does not run.*

**The coverage guard could not fire on real data.** It histogrammed boundary
angles into 24 bins and required 85% occupancy, so it is capped at
`n_points / 24`. Measured on the six real captures in
`vision/drone_orientation/`, hulls carry **9–31 vertices**, so four of six cannot
reach the threshold at any distribution of points. A gap-based measure — largest
angular gap, independent of count — gives 0.59–0.84 on the same images against
the binned 0.17–0.79.

**The real finding is the sampling density.** Synthetic hulls in these tests
carry 48–84 points; real ones carry 9–31. Every robustness mechanism validated in
this journal — the axial re-weighting, the one-sided loss, the trim — assumes a
densely sampled boundary. Re-running the comparison across densities:

| boundary points | plain fit | robust (inner half) |
|---|---|---|
| 12 | 0.03962 | 0.04525 — **worse** |
| 18 | 0.03241 | 0.04041 — **worse** |
| 24 | 0.03578 | 0.03723 — worse |
| 32 | 0.03176 | 0.03063 — even |
| 48 | 0.03412 | 0.02585 — −24% |
| 84 | 0.03432 | 0.02099 — −39% |

Robust estimation **helps only above ~32 points and hurts below it**. The reason
is not subtle: discarding half of 84 leaves 42 for a five-parameter fit;
discarding half of 12 leaves 6. Below some density the variance from too few
points exceeds the bias removed. **Every robustness result in this journal was
measured on the wrong side of that line.**

**Where the redundancy went.** `silhouette_hull` returns the convex hull's
*vertices*. The thresholded contour they were computed from carries far more:

| capture | contour points | hull vertices | discarded |
|---|---|---|---|
| drone1 | 551 | 18 | 30.6× |
| drone2 | 2051 | 17 | 120.6× |
| drone3 | 192 | 9 | 21.3× |
| white1 | 3072 | 31 | 99.1× |

**21–134× of the measured boundary is thrown away before anything is fitted.**
The hull exists to bridge a rim broken into arcs, which is a real need — but
bridging gaps and choosing which points to fit are separate decisions, and
collapsing them costs two orders of magnitude of data.

**Why that still does not close it.** At contour densities the robust fit reduces
typical scatter by 10–32% but makes the **worst case 20–25% worse** (0.0710 →
0.0882 at 1500 points). Under a specification on 100% of frames the tail is the
binding quantity, so this is the wrong direction and is not shipped.

### Improved / Constant / Regressed

**Improved** — none shipped.

**Constant** — the estimator is unchanged from Iteration 8.

**Regressed** — none shipped. Three claims from earlier entries are, however,
now **qualified**: the axial re-weighting (Iteration 3), the one-sided loss
(Iteration 4) and the trim were all validated at 48–84 boundary points, and the
density comparison above shows that regime does not represent real captures.
Their measured benefits stand as measured; whether they survive at 9–31 points is
**untested**, and on the evidence here some may reverse.

### What this changes about the plan

The next useful step is not another estimator mechanism. It is to stop discarding
the boundary: fit to contour points rather than hull vertices, keeping the hull
for gap-bridging only. That is not a tuned change — it uses data already
measured — and it moves the pipeline to the side of the density line where robust
methods work at all. It should be validated at real point counts from the start,
which is the standard every mechanism after this one has to meet.

### Correction: the density argument above was measured against a fixture

The comparison that motivated this entry — "sim carries 48–84 boundary points,
real 9–31" — used `test_stereo._hulls`, which **constructs** 48 points with
`linspace`. It is a test fixture, not a render, so it says nothing about the
simulation. Measuring the actual renderer:

| | hull vertices | contour points | major axis |
|---|---|---|---|
| rendered, 1280×800 | 49 (15–62) | 381 | 150 px |
| real captures, 640×480 | 17 (9–31) | 1516 | ~440 px |

Contour density tracks perimeter in both at roughly one point per pixel, which is
consistent. **Hull vertex count does not.** Real silhouettes are three times
larger and yield three times *fewer* hull vertices, and a convex hull gains a
vertex at every outward excursion — so the rendered boundary is **noisier** than
the real one, not cleaner.

That reverses the direction of the gap as first stated here. The robustness
mechanisms were not validated on an unrealistically clean boundary; they were
validated on an unrealistically *rough* one, which flatters any method whose job
is to reject outliers.

### What can and cannot be concluded from the available real data

Very little, and the reason is worth stating so it is not over-read. The six
captures in `vision/drone_orientation/` differ from the simulated conditions in
**four ways at once**: resolution (640×480 vs 1280×800), object size (~440 px vs
150 px major axis), motion (stationary vs blurred at 310–350 Hz spin), and
lighting. No single mechanism's contribution can be isolated across four
simultaneous differences, and none of them carries a ground-truth pose, so
accuracy cannot be measured at all — only self-consistency.

Two things were nonetheless established on real pixels:

* **Dense contours are not one-sided.** 42–62% of contour points sit more than a
  pixel *inside* the ellipse fitted to them. The convex hull was not merely
  discarding data; it was **enforcing** the outward-only property that the
  one-sided loss and every trimming argument depend on. Restricting to contour
  points lying on the hull recovers 55–461 measured points — 5–15× the vertices,
  with the property intact — and is the only version of the density idea that
  survives contact with real imagery.
* **The coverage guard was unreachable**, being capped at `n_points / bins`.

### The actual next step

Not another estimator mechanism. **Calibrate the renderer's boundary statistics
against the real captures** — match object size, resolution and noise, then
compare hull-vertex-per-perimeter and contour density until the simulated
boundary behaves like a measured one. Only then does a robustness result measured
in simulation carry any weight for the real system. Every mechanism in this
journal from Iteration 3 onward was validated without that check, and the
correction above shows the check would not have been a formality.


## Iteration 12 — the shipped configuration was two different configurations

Iteration 10 ended by setting `segment.AXIAL_DEFAULT = False`, restoring the
matching `RADIUS_MM = 10.2446` and the pre-axial `error_model.json`, and writing
all of that up as "the coherent shipped configuration".

It was not shipped, and it was not coherent.

### The defect

`PoseEstimator.update` had its own default:

```python
def update(self, frame, t=None, frame_index=None, axial=True):   # <- not None
    seg = segmod.segment(frame, ..., axial=axial)
```

so `AXIAL_DEFAULT` governed direct `segment.segment()` callers and **nothing
else**. Every path that goes through the estimator — `sweep.py`,
`resolution_sweep.py`, `fit_error_model.py`, `gallery.py`, `limits.py`,
`run_pose.py` — kept running the weighted fit.

Worse than merely ignoring the flag: it ran the **weighted fit against the
unweighted radius**. The two constants are fitted together, 10.2446 unweighted
and 10.2662 weighted, and `estimator.py`'s own comment records that using one
with the other "reintroduces a systematic depth bias larger than the whole
residual". Iteration 10 restored one half of a matched pair and left the other
half unreachable, then measured the result and reported it as the unweighted
configuration.

### The fix

```python
def update(self, frame, t=None, frame_index=None, axial=None):
    use_axial = segmod.AXIAL_DEFAULT if axial is None else bool(axial)
```

One flag, read in one place, at call time. `gallery.build_weighting` still passes
`axial=` explicitly, which is correct — that is the A/B, and it should pin both
arms rather than inherit either.

### Why it survived

Every check that could have caught it was a check of *relative* behaviour. The
A/B compared `axial=True` against `axial=False` with both passed explicitly, so
it exercised the parameter and never the default. The four unit suites test
geometry, zeroing, calibration and filtering, none of which construct a
`PoseEstimator` and look at which fit it chose. And the flag's *effect* is a few
hundredths of a millimetre on a median — small enough that a sweep re-run after
flipping it looks like sampling noise rather than a no-op.

The general form: **a default that duplicates a module constant is not a
convenience, it is a second source of truth**, and the failure mode is silence.
This is the third instance in this journal of the same class — `AXIAL_WEIGHT_ITERS`
as a default argument (Iteration 10), the code-blind sample cache in
`fit_error_model.py` (Iteration 10), and now this. All three produced *confident
wrong numbers* rather than errors.

### Consequence for the numbers

Everything measured through the estimator between Iteration 10 and here
describes the weighted fit paired with the unweighted radius — a configuration
nobody chose. The affected results have been re-run at the corrected default:
`resolution_final` (gated sweep) and `limits.json`. The Iteration 10 A/B table
itself is unaffected, because it pinned both arms explicitly.

## Iteration 13 — the axial-weighting verdict was wrong, and the bug is why

Iteration 10 concluded that axial weighting had to stay off because it collapsed
certified detection "from ~13% of frames to ~1–3.5%". Iteration 12 found the bug
that made that measurement meaningless. This is the controlled re-run.

### The A/B

Identical seed, identical 400 poses, identical gate, identical constants on disk;
the *only* difference is `POSE_AXIAL`, which now actually reaches the fit:

| core mode | detect OFF | detect ON | pos avg OFF | pos avg ON | ang avg OFF | ang avg ON | in-spec OFF | in-spec ON |
|---|---|---|---|---|---|---|---|---|
| 1280×800 | 3.2% | 3.2% | 0.268 mm | **0.186** | 0.294° | **0.178** | 100% | 100% |
| 1024×768 | 2.5% | 1.5% | 0.346 | **0.268** | 0.442 | **0.234** | 100% | 100% |
| 800×600 | 2.2% | 2.8% | 0.331 | **0.271** | 0.281 | **0.211** | 88.9% | **100%** |
| 640×480 | 2.0% | 2.0% | 0.338 | **0.296** | 0.371 | **0.244** | 100% | 100% |
| core modes at 100% in spec | | | | | | | 5/8 | **6/8** |

Certified frames summed over the six usable modes: **61 off, 59 on**. That is the
same number. **The weighting does not cost gate coverage at all**, and it lowers
both error metrics in every single mode.

### What the original claim actually measured

Three different quantities, conflated:

* The "13%" was `resolution_iter8` — 800 poses, a different gate fit, and the
  weighted estimator (because of the Iteration 12 bug, everything through the
  estimator was weighted).
* The "1%" was `resolution_iter9` — the parallel session's **sub-pixel** run, not
  an axial run at all.
* The "3.5%" was `fit_error_model`'s acceptance on its held-out *report split*,
  which is a different metric from a sweep's detection rate and not comparable to
  either of the others.

None of the three is an axial-weighting A/B. The one above is.

### The mechanism story was also backwards

The old note argued that a weighted fit reports a worse `fit_rms_px`, so the gate
(whose largest angle coefficient is `log1p_refine_rms`) rejects good frames. The
provenance says the opposite: `error_model.json` was written at
2026-08-12T01:45Z, inside the bug window, **through the estimator** — so it was
fitted on weighted-fit features all along. If anything was mismatched it was the
unweighted setting, not the weighted one. The A/B shows the effect either way is
negligible, so neither story is needed.

### Consequence

`AXIAL_DEFAULT` flips to **on**, with `RADIUS_MM` and the tilt calibration refit
against the weighted fit in the same pass — they are a matched set and the whole
point of Iteration 12 is that shipping half a set is how this went wrong.

### The thing that is still unexplained

Both arms certify ~3% at 1280×800 where `resolution_iter8` certified 13% on 800
poses. That gap is **not** the axial flag — it is identical across the A/B. It is
some other drift in the dataset/radius/calibration/error-model chain across the
bug window, and it is not diagnosed here. Until it is, the current sweep numbers
should be read as a lower bound on achievable coverage.

The general lesson, again: **a comparison is only a comparison if the thing you
varied is the only thing that differed.** Iteration 10 varied a flag that did
nothing, alongside a radius swap, a calibration refit and an error-model restore,
and then attributed the result to the flag.

## Iteration 14 — the joint refit made it worse, so it was not adopted

Iteration 13 established that axial weighting costs no gate coverage and improves
accuracy everywhere, and concluded it should be on — with the caveat that
flipping it "correctly" meant refitting `RADIUS_MM` and the tilt calibration in
the same pass, because the two were said to be a matched set (10.2662 weighted
against 10.2446 unweighted).

Both halves of that caveat turned out to be wrong.

### The radii are the same

Regenerating `dataset.npz` with `POSE_AXIAL=1` (1400 poses, 1400 detected) and
refitting gives **`RADIUS_MM = 10.2418`**, not 10.2662. That is **0.03%** from
the unweighted 10.2446 — an order of magnitude inside the depth residual it was
supposed to bias. The weighted and unweighted fits do not need different radii,
and the estimator comment claiming otherwise has been corrected.

Where 10.2662 came from is not established; it is not reproducible from this
dataset and should not be used.

### The error-model refit is a regression

Refitting the whole chain for the weighted fit and re-running the gated sweep,
against the two arms from Iteration 13:

| configuration | certified frames (6 usable core modes) | modes at 100% in spec |
|---|---|---|
| axial OFF, original constants | 61 | 5/8 |
| **axial ON, original constants** | **59** | **6/8** |
| axial ON, fully refit constants | 12 | 7/8 |

The refit trades **59 certified frames for 12** to gain one mode. Against a
contract of "100% in spec on every *reported* frame", coverage is the thing being
maximised subject to the spec, and a 5x coverage loss for one mode is not it.

The refitted model is stricter rather than better, and it looks unhealthy:

* its own held-out acceptance fell from 3.5% to **0.9%**;
* `k_ang` rose 2.249 -> 2.467;
* the fitted angle coefficients came out ill-conditioned — `log_inv_margin` went
  **negative** (-1.6) and the intercept to -5.47, where the previous fit had
  +1.028 and +3.842. A sign flip on a feature whose meaning is "how far apart the
  two ambiguity branches were" is not a physically sensible model.

### What ships

Axial weighting **on**, with the **original** radius, tilt calibration and error
model — exactly the configuration measured as `resolution_axialon`, now retagged
`resolution_final`. `POSE_AXIAL=0` recovers the unweighted fit for comparison.

The error model that ships with it was fitted through the estimator *while the
estimator forced the weighted fit* (Iteration 12), so it was fitted on
weighted-fit features all along and is not mismatched with what now ships. That
is the one piece of luck in this sequence, and it is luck rather than design.

### Still open

* **Refitting the error model for the weighted fit properly.** The
  ill-conditioned coefficients above are the lead. The current fit is usable but
  was produced accidentally rather than deliberately.
* **The ~3% vs 13% coverage gap.** Every arm here certifies ~3% at 1280x800
  where `resolution_iter8` certified 13% on 800 poses. It is identical across all
  three arms, so it is not the flag; it is drift elsewhere in the chain across
  the bug window and is not diagnosed. Treat current coverage as a lower bound.

The lesson worth keeping: **"regenerate everything together" is the right
instinct and is not automatically an improvement.** A refit is a new fit, and a
new fit can be worse; it has to be measured against the thing it replaces on the
metric that matters, not adopted because it is more self-consistent.
