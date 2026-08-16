# Real-time 5-DOF pose estimation

Position and orientation of the flying robot from a single camera, fast enough
for a 240–420 fps loop, with a live six-panel plot, CSV logging, and a synthetic
harness that measures its own accuracy against known ground truth.

Existing vision code ([visual_servo/servo.py](../visual_servo/servo.py)) recovers
**height only** — it PCA-fits a blob and throws the orientation away on purpose.
This recovers the full observable state.

## Headline numbers

Measured on a **held-out test split** the tuning never saw (700 poses, rendered
with sensor noise and motion blur), plus a live-camera latency run. Details and
caveats below.

| | value |
|---|---|
| lateral residual (x, y) | **0.135 mm** = **0.06% of range** |
| depth residual (z) | **0.864 mm** = **0.39% of range** |
| ‖position‖ residual | **0.883 mm** = **0.40% of range**, 4.3% of the robot's 20.4 mm body |
| orientation residual (normal) | **0.693°** median |
| tilt residual (θ) | **0.497°** median |
| azimuth residual (φ) | **0.57°** median — but degrades below ~10° tilt, see below |
| latency, grab → pose | **2.47 ms** median, 3.91 ms p95 |
| compute, 640×480 | **1.10 ms** → 911 Hz sustained |
| detection rate | 21,504 / 21,504 cells across all opacity, background and lighting |

Position residuals are for the well-conditioned band (tilt 10–45°). Over the
full 0–70° range: ‖position‖ **1.042 mm**, normal **0.609°** — and unlike before,
orientation barely varies with tilt at all, because the axial weighting described
below removed what used to be a fourfold degradation past 45°.

**Rate verdict: 240 fps is comfortable at any usable resolution; 420 fps needs
≤640×480.** No camera on this rig produces either yet — see below.

---

## What it estimates, and what it cannot

The robot's duct ring is a circle. Measured off
[flyingrobot_thick _rod2.STL](../vision/flyingrobot_thick_rod2.STL): **20.409 mm
across, radial standard deviation 0.108 mm, planar to 1.5 mm**. A circle of known
radius and its image ellipse determine the circle's 3-D pose analytically, so the
estimator is a closed-form solve, not a search.

| | |
|---|---|
| **Recovered** | position `(x, y, z)` in mm, rotor normal on `S²` (tilt `θ`, azimuth `φ`) |
| **Not recovered** | roll about the spin axis — the robot spins at 310–350 Hz against a camera an order of magnitude slower, so blade position aliases beyond rescue |
| **Sixth plot channel** | `ψ`, the image ellipse's major-axis angle. Directly measured, tied geometrically to `φ`, carried separately as a consistency check. Not spin. |

This matches the 5-DOF `R³ × S²` state in
[docs/pose_localization_project_context.md](../../docs/pose_localization_project_context.md).

### The two-fold ambiguity is irreducible

A tilted circle and its mirror image about the viewing axis project to the
**identical** ellipse. Every frame yields two poses and no single frame can
choose. The estimator picks the one nearest the previous frame (falling back to
the datum, then to a face-on prior) and records `ambiguity_margin_deg` — how far
apart the candidates were — on every row, so the size of the bet is always in
the log.

Resolving it properly needs outside information: a second camera (already
planned in the pose-localisation doc) or a hard prior on which way the robot
leans. Using the mast as an asymmetry cue was tried and rejected — the signal is
only 3–6 px and is contaminated by a fixed offset from blade asymmetry.

Because of this, the validation reports **two** normal errors: `normal_err_deg`
(the branch chosen) and `normal_err_best_deg` (the better of the two). When they
diverge, the fit was fine and the branch was wrong — a different problem with a
different fix.

---

## Measured performance

All numbers below are from a **held-out test split**: 700 randomised poses the
calibration was never fitted on (`validation/tune.py`, trained on a disjoint
700). Poses span tilt 0–70°, range 140–360 mm, random opacity 0.7–1.0,
background 0–0.3 grey, and randomised lighting.

### 5-DOF residual, per degree of freedom

Within the well-conditioned band (tilt 10–45°, n = 298):

| DOF | median | p95 | max | % of range (median) |
|---|---|---|---|---|
| x | 0.078 mm | 0.203 | 0.323 | **0.03%** |
| y | 0.067 mm | 0.230 | 0.364 | **0.03%** |
| z (depth) | 0.857 mm | 2.726 | 6.300 | **0.36%** |
| **‖position‖** | **0.874 mm** | 2.732 | 6.313 | **0.37%** |
| tilt θ | 1.114° | 2.650 | 4.635 | — |
| azimuth φ | 0.482° | 4.535 | 39.42 | — |
| normal (θ, φ combined) | 1.238° | 2.854 | 7.776 | — |

‖position‖ is **4.3% of the robot's own 20.4 mm diameter**.

Across the full tilt range 0–70° (n = 700): ‖position‖ **1.348 mm median
(0.60% of range)**, p95 4.606 mm; normal **2.055°** median, p95 7.415°.

**φ degrades as tilt → 0, definitionally.** The azimuth of the tilt is which way
the robot leans; when it barely leans, that direction is barely defined, and the
error goes as `θ_err / sin θ`:

| tilt | \|φ error\| median | \|θ error\| median |
|---|---|---|
| 0–5° | 30.84° | 2.92° |
| 5–10° | 14.32° | 2.48° |
| 10–20° | 1.81° | 2.07° |
| 20–35° | 0.57° | 1.04° |
| 35–50° | **0.23°** | 1.24° |
| 50–70° | 0.34° | 3.06° |

This is not a defect to fix — a zero-tilt vector has no azimuth. It does mean φ
should not be fed to a controller below ~10° of tilt.

### Independently confirmed on a different grid

The numbers above come from 700 held-out random poses. `validation/sweep.py`
then re-measures the shipped configuration on a completely separate
**21,504-cell structured grid** (4 opacities × 4 backgrounds × 14 lighting rigs
× 96 poses). Adequately lit and in-envelope, 4,800 of those cells:

| | random held-out (n=700) | structured grid (n=4800) |
|---|---|---|
| ‖position‖ median | 0.883 mm | **0.823 mm** |
| relative depth | 0.39% | **0.391%** |
| normal median | 0.693° | **1.279°** |
| segmentation IoU | — | 0.892 |

Above 40° tilt on the structured grid the axial weighting shows most clearly:
normal error is **0.640°**, against 4.43° with the unweighted fit — better than
the low-tilt figure, because tilt is well conditioned there and the protrusions
are no longer contaminating it.

Two independent samplings of the condition space agree to within ~20%, which is
the real check that neither is an artefact of how the poses were drawn.

### Depth is the weak axis, by 7.6×, and structurally so

Lateral position comes from the ellipse **centre** — a well-localised centroid.
Depth comes from its **size**: `z = f·2R/major`, so `dz/z = −d(major)/major`. A
one-pixel error in a 130 px major axis is 0.77% of range; the same pixel on the
centre is 0.16 mm.

| range | lateral | depth | depth % of range | major axis |
|---|---|---|---|---|
| 140–190 mm | 0.158 mm | 1.002 mm | 0.60% | 178 px |
| 190–240 mm | 0.159 mm | 1.221 mm | 0.60% | 134 px |
| 240–290 mm | 0.172 mm | 1.420 mm | 0.53% | 107 px |
| 290–360 mm | 0.183 mm | 1.682 mm | 0.54% | 90 px |

**Relative depth error is constant at ≈0.57% of range** — the scale-free way to
state it. Working back, the pipeline measures the major axis to **0.67 px**,
i.e. sub-pixel. That is the floor; nothing improves depth except more pixels
across the rim (closer, longer lens, or higher resolution).

The exact penalty is **`g(θ)·z/(2R)` with `g(0) = √3`** — the range over the
robot's *diameter*, times a factor between 1.73 and 2.18. The √3 is the price of
having to estimate the tilt from the same ellipse; quoting `z/(2R)` alone (as
this README did, and as the lecture notes did through §12) understates it by 73%.
At 250 mm that is 21–25× lateral depending on tilt, not 12×. Noise level, point count, focal length
and resolution all cancel. Derivation and Monte-Carlo verification:
`bounds.py` / `test_bounds.py`, lecture notes §13.4.

### How much of the error is even removable?

`bounds.py` derives the Cramér–Rao floors for the whole chain — photons → edge
position → ellipse parameters → pose — and `validation/limits.py` places the real
system against them (484 frames, medians, oracle ambiguity branch):

| level | status | ‖position‖ | angle |
|---|---|---|---|
| photon limit | bound | 0.026 mm | 0.013° |
| pixel-quantised boundary | bound | 0.153 mm | 0.080° |
| + convex hull (31 of ~354 points) | bound | 0.490 mm | 0.249° |
| noise-equivalent | *prediction* | 2.358 mm | 1.374° |
| **measured** | actual | **1.820 mm** | **2.668°** |

The three levers people reach for first are all closed off, with numbers:

- **The solver is at the bound.** Statistical efficiency 0.96–1.01 on the five
  ellipse parameters and 0.94–1.06 end-to-end through the back-projection. A
  maximum-likelihood fit recovers at most 4%.
- **Sub-pixel refinement has ≤1.5× left.** `subpixel_boundary` achieves 0.076 px
  on a clean disc against an edge CRLB of 0.052 px — and the measured boundary
  scatter on real renders is 1.087 px, 14× worse. The curve is being located
  precisely; it is the wrong curve.
- **A perfect sensor changes nothing.** The photon floor is 0.048 px against
  1.087 px measured — under 5% of the error.

What *is* available: the convex hull keeps 31 vertices of a ~354 px perimeter,
and `1/√N` says that costs **3.2×**. That is a design choice, not a limit.

The whole gap factors as `6.0 × 3.2 × 3.7 = 71×`. Only the middle factor is
untouched; the axial weighting is a partial software attack on the third, and the
first was never available. Lecture notes §13.8.

### What tuning bought

Two parameters, both fitted on train and scored on test:

On the held-out random split (n = 700):

| | before | after | change |
|---|---|---|---|
| ‖position‖ median | 1.486 mm | 1.348 mm | **−9.2%** |
| normal median | 3.223° | 2.055° | **−36.2%** |
| tilt bias (systematic) | −4.918° | **−0.255°** | eliminated |

And re-confirmed on the independent 21,504-cell grid:

| | before | after |
|---|---|---|
| well-lit ‖position‖ | 0.887 mm | **0.751 mm** |
| in-envelope normal | 3.320° | **2.851°** |
| in-envelope tilt bias | −2.660° | **+0.390°** |
| beyond-envelope tilt bias | −7.840° | **−1.250°** |

- **Effective radius** 10.2065 → 10.2245 mm. Depth scales linearly with the
  assumed radius, so a scalar error appears as a constant *relative* depth bias
  (measured −0.177%) and inverts directly.
- **Tilt calibration** (`tilt_calibration.json`) removes the mast-induced bias.
  See `calibration.py` — only the mean is recoverable; the scatter is the 3-D
  silhouette interacting with viewing geometry, which no factor explained above
  R² = 0.06.

### Sensor noise barely matters; background matters only at one point

Swept on their own, 300 fixed poses per level (`validation/sensitivity.py`):

| read noise σ | median | RMSE | normal | seg IoU | failures |
|---|---|---|---|---|---|
| 0 | 0.957 mm | 1.75 | 0.427° | 0.895 | **0%** |
| 10 | 0.929 | 1.76 | 0.472° | 0.886 | 0% |
| 20 | 0.916 | 1.84 | 0.527° | 0.872 | 0% |
| 40 | 0.913 | 2.08 | 0.663° | 0.824 | 0% |
| 80 | 1.146 | 2.86 | 1.017° | 0.710 | **0%** |

**Not one frame failed at any noise level**, and the median grows only 18% from
σ=0 to σ=80 — a third of full scale. A fixed threshold on a bright object is
inherently noise-immune; the convex hull then averages 30–60 boundary points, so
per-pixel noise is gone before the fit sees it.

Background is flat until it is catastrophic:

| background grey | median | failures |
|---|---|---|
| 0.00 – 0.333 | 0.77 – 1.06 mm | 0% |
| 0.417 | 1.47 | 0% |
| **0.50** | **230 mm** | **100%** |

The cliff is not mysterious: grey 0.5 is level 128, which *is* the threshold, so
the entire frame becomes foreground. Anything below ~0.4 is fine. This is the
one place where "background doesn't matter" stops being true, and it stops
completely rather than gradually.

### Three independent harnesses agree

The same quantity measured by three different samplings of the condition space,
restricted to a common tilt envelope:

| harness | ‖position‖ median |
|---|---|
| structured 21,504-cell sweep, well-lit in-envelope | 0.751 mm |
| held-out random split (700 poses), tilt 10–45° | 0.751 mm |
| sensitivity σ=0, tilt 10–40° / 10–45° | 0.718 / 0.796 mm |

Agreement within ~6% across three harnesses that share only the estimator is the
real check that none of them is an artefact of how its poses were drawn.

### The calibration generalises across resolution

It is fitted at 1024×768 but the model is built on a dimensionless axis ratio,
so it should transfer. Checked on 160 fresh poses per resolution:

| resolution | raw tilt bias | calibrated bias | \|tilt err\| | ‖position‖ |
|---|---|---|---|---|
| 1024×768 (fitted here) | −2.97° | **+0.12°** | 1.27° | 0.89 mm |
| 768×576 | −3.08° | **−0.00°** | 1.34° | 1.06 mm |
| **640×480** (operating point) | −3.25° | **−0.19°** | 1.36° | 1.31 mm |
| 512×384 | −3.55° | **−0.53°** | 1.52° | 1.79 mm |

Residual bias stays inside ±0.6° everywhere, so one curve covers the range. The
position error growth is the expected pixel-count effect, not a calibration
failure.

### The residual is not noise, and that changes what a filter is for

The obvious next move is a Kalman filter: motion is sub-Hz, capture is 240–420
fps, so there ought to be enormous averaging available. **It does not work, and
the reason is worth knowing.**

The per-frame error is not white. Measured along a rendered trajectory, the
depth error autocorrelates at **r = 0.966 after one frame** and stays above 0.5
for **408 ms**. It is a smooth function of pose — tilt, azimuth, range — not
measurement noise, and averaging cannot remove a bias that persists. Filtering
position was swept across process noise from 15 to 4000 mm/s²: the best result
was **1.01×**, and tighter settings made it monotonically worse.

Blade spin does not rescue it. Re-rendering with the rotor turning at 330 Hz, so
blade phase is fully aliased as on the real robot, moved the one-frame
autocorrelation only from 0.966 to **0.886** — because the segmenter takes a
convex hull, the hull is set by the outermost feature, and that feature is the
**rotationally symmetric duct rim**. Spinning blades inside the ring barely
change the silhouette.

**But the filter earns its place on velocity**, and dramatically. Against
analytic truth on a trajectory whose true speed is 15–22 mm/s:

| frame rate | finite difference | + 5 Hz IIR (what `simulate_hover.py` does) | Kalman |
|---|---|---|---|
| 60 fps | 20.8 mm/s | 6.96 | **3.97** |
| 240 fps | 55.3 mm/s | 6.21 | **3.64** |
| 420 fps | **102.8 mm/s** | 6.35 | **3.56** |

Finite differencing gets **worse the faster you sample** — at 420 fps its error
is 570% of the true speed — because it divides a nearly-constant error by an
ever smaller `dt`. This is the same correlation, biting from the other side. The
Kalman is flat across the range.

So: **take position raw, take velocity from the filter.** `filter.py` also
coasts through dropouts and offers `predict_ahead()` to compensate the measured
2.5 ms latency. `online_camera.ipynb` enables it by default and logs velocity to the
CSV; `--no-filter` disables it.

### What tuning was tried and rejected

Each of these was implemented and measured before being dropped. They are
recorded so nobody spends the afternoon again.

| tried | result |
|---|---|
| **Sub-pixel edge refinement** — walk each hull vertex to the interpolated threshold crossing | **No change**: depth scatter 0.661% vs 0.662%, lateral slightly worse. The hull already averages 30–60 points, so quantization has averaged out before the fit. |
| **Morphological opening 5×5** (the value `servo.py` uses) | **Worse**: depth scatter 1.10% vs 0.64% at 3×3. A 5×5 open erodes the thin rim wall away. |
| **Closing kernel 5 / 7 / 9** | **No difference** to three decimals. 7 kept arbitrarily. |
| **Retuning the threshold** to 112 (lowest synthetic scatter, 0.52% vs 0.64%) | **Not adopted**: the optimum depends on exposure, so a level fitted to renders would not transfer. 128 is already validated on the bench. |
| **Contrast-relative (auto) threshold** | **Worse**: did not fix the unlit-rim failure (58 mm vs 64 mm) and degraded every well-lit case. |
| **Robust trimmed ellipse refit** | **Worse**: with only 20–60 hull points, trimming destabilises the fit. |
| **Mast asymmetry as an ambiguity cue** | **Too weak**: 3–6 px, contaminated by a fixed blade-asymmetry offset. |

The pattern: the remaining scatter is **not** measurement precision. It is the
3-D silhouette varying with blade phase and lighting, which a 2-D ellipse cannot
represent. Reducing it requires more information — a second camera — not a
sharper edge.

### What actually drives error — and it is not what the sweep was aimed at

Spread in median position error across each axis:

| axis | spread |
|---|---|
| material alpha (0.7 → 1.0) | **1.07×** |
| background (black → 0.3 grey) | **1.14×** |
| ambient light (0.05 → 0.6) | **3.40×** |

Opacity and background are, to a good approximation, irrelevant — the robot
stays segmentable at 70% opacity on a 30% grey background. **Lighting is the
whole story**, and specifically the ambient component. The worst four rigs are
all hard lateral lights at low ambient (≈4.5 mm median against ≈1.1 mm for
dome-lit).

### Latency

Latency is not throughput. Throughput is how many frames a second the estimator
can chew through; latency is how stale the answer is when a controller sees it,
and that is what bounds loop bandwidth — `ai/design_hover_lqr.py` hard-fails on
closed-loop poles above `rate/6` for exactly this reason.

The headline: **queueing adds only 0.1–0.6 ms, so latency is essentially
compute.** That is what the drop-oldest single slot in `sources.CameraSource`
buys. Across every run, zero frames were dropped and the camera's own 33.5 ms
frame interval was the only bottleneck.

Compute, and therefore latency, is strongly **scene dependent** — segmentation
cost scales with how many bright blobs there are, not just with resolution:

| scene | resolution | compute median | grab → pose |
|---|---|---|---|
| rendered robot on black | 640×480 | **1.10 ms** | ~1.2 ms |
| repo 240 fps footage | 408×720 | 2.13 ms | — (file, no capture) |
| repo 240 fps footage | 544×960 | 3.01 ms | — |
| laptop webcam, cluttered room | 640×480 | 2.4–4.8 ms | 2.5–5.3 ms |

The cluttered-room figures are a **worst case and not representative of the
rig** — a laptop webcam pointed at a room has dozens of bright regions, where
the bench has one white robot on black. They are included because measuring only
the favourable case would overstate the result. Expect the rig to behave like the
first row.

Live stage breakdown at 640×480 (webcam, 250 frames):

| stage | mean | median | p95 |
|---|---|---|---|
| segmentation | 2.131 ms | 1.934 | 3.353 |
| back-projection | 0.448 | 0.446 | 0.540 |
| compute total | 2.592 | 2.394 | 3.792 |
| **grab → pose ready** | **2.688** | **2.475** | **3.913** |

**Not included**: sensor exposure and USB transfer. `cv2` stamps a frame when
the driver hands it over, which is already after both, so they are unmeasurable
from the host without hardware timestamps. On a UVC camera expect one to two
extra frame periods — likely larger than everything measured above.

**Throughput** — compute per frame, clean synthetic imagery:

| resolution | segment | back-project | total | sustained | 240 fps | 420 fps |
|---|---|---|---|---|---|---|
| 1280×960 | 2.73 ms | 0.28 ms | 3.01 ms | 333 Hz | OK | over |
| 1024×768 | 2.07 | 0.24 | 2.32 | 432 Hz | OK | OK |
| 800×600 | 1.38 | 0.22 | 1.60 | 626 Hz | OK | OK |
| **640×480** | **0.90** | **0.19** | **1.10** | **911 Hz** | OK | OK |
| 480×360 | 0.61 | 0.19 | 0.80 | 1255 Hz | OK | OK |
| 320×240 | 0.38 | 0.20 | 0.58 | 1732 Hz | OK | OK |

Position error stays under ~1.5 mm down to 480×360 and then degrades sharply
(6 mm at 320×240 — too few pixels across the rim).

**640×480 is the operating point**: 911 Hz sustained, about 2× headroom over
420 fps, with no accuracy cost.

Segmentation dominates — the back-projection is a fixed ~0.2 ms regardless of
resolution, because it works on five ellipse parameters, not pixels. Any further
speed work belongs in `segment.py`.

On real footage ([writeup/two_channel.mp4](../../../writeup/two_channel.mp4),
544×960, shot at 240 fps), from `validation/latency.py`:

| scale | resolution | compute median | sustained | over 240 fps | over 420 fps |
|---|---|---|---|---|---|
| 1.0 | 544×960 | 2.67 ms | 375 Hz | 0.25% | 100% |
| 0.75 | 408×720 | 1.92 ms | 521 Hz | 0.00% | 5.0% |
| 0.5 | 272×480 | — | — | — | no detections |

Axial weighting costs **+0.57 ms** at 1024×768 and **+0.61 ms** at 640×480 —
two extra weighted conic fits plus their Sampson evaluations. Decomposed against
a base segmentation of 1.85 ms / 1.04 ms:

| | 1024×768 | 640×480 |
|---|---|---|
| threshold + morphology + hull | 1.85 ms | 1.04 ms |
| + sub-pixel boundary | +0.76 | +0.16 |
| + axial weighting | +0.57 | +0.61 |
| **total** | **3.09 ms** | **1.91 ms** |

> An earlier revision of this file claimed +0.13 ms. That was wrong. The A/B
> switched the weighting by reassigning `AXIAL_WEIGHT_ITERS`, which was a
> *default argument* and therefore bound when the function was defined — so both
> arms ran weighted and the "delta" was noise. `segment()` now takes an explicit
> `axial=` parameter so the comparison is real. The lesson generalises: a module
> constant used as a default argument cannot be monkeypatched.

> Do **not** read throughput off `validation_results.csv`. That sweep renders a
> frame before each estimate, so its timing columns carry interleaved GL work and
> read 30–50% high; one stretch of the last run recorded a 10.1 ms median against
> a 3.0 ms tail purely because another render job was running. `latency.py` is
> the measurement that means something.

Real imagery is messier than renders and costs more per pixel. At 0.5 the robot
falls below the minimum blob size in *this* footage — a target-size limit, not a
timing result.

**Bottom line on rate: 240 fps is comfortable at any usable resolution; 420 fps
needs ≤640×480.**

> No camera on this rig can currently produce 240–420 fps. The bench C270 is
> measured at ~28 fps flat at every resolution, and only the built-in FaceTime
> camera is attached. The numbers above are what the *software* sustains;
> `sources.py` has the camera backend ready for when hardware arrives.

---

## Where it breaks

The sweep exists to find this, and it found three things — none of which are
opacity or background, which is what it was originally pointed at.

**Past ~45° of tilt the fit breaks down — and it is mostly our fault, not the
robot's.** The fix below is measured and available, but is **off by default**
because it regresses the certification gate; see the end of this section. The mast and magnet stick out along the rotor
axis, so as it tilts they push the silhouette outward in its short direction and
the minor axis inflates. This was recorded here as irreducible geometry. It was
not.

Because the protrusions lie *on* the rotor axis, they land near the **middle of
the major axis** — traced at 70° tilt, the two worst hull vertices are rod tips
17.2 px off the true rim, against the magnet's 7.0 px, both at
`|projection|/semi-major < 0.04`. So each hull point is weighted by how far along
the major axis it sits, `w = |proj|/a`, and the fit is reweighted twice.

Measured on identical poses, each variant with its own refitted radius and tilt
calibration, on a held-out split of realistic (noisy, motion-blurred) renders:

| tilt band | position | normal |
|---|---|---|
| <10° | 1.331 → **1.240** mm (−6.8%) | 3.459 → **3.250°** (−6.0%) |
| 10–45° | 0.751 → 0.883 mm (**+17.6%**) | 1.036 → **0.693°** (−33.2%) |
| 45–60° | 1.443 → **1.051** mm (−27.2%) | 2.083 → **0.331°** (−84.1%) |
| >60° | 2.110 → **1.252** mm (−40.6%) | 4.335 → **1.358°** (−68.7%) |
| **all** | 1.193 → **1.042** mm (−12.6%) | 1.809 → **0.609°** (−66.3%) |

Orientation improves in every band and transforms above 45°, where the error
used to quadruple. **The one real cost is position between 10° and 45°, 18%
worse.** That band is also the one this rig spends least time in: a side-on
camera sees the rotor near edge-on, which is the high-tilt end where weighting
gains 27–41%.

Three things that did not work, worth knowing:

- **Weight power 2** (rather than 1). Judged on minor-axis recovery alone it
  looks better (9.4 px vs 13.8 at 70° tilt), but it also discards the mid-range
  points that constrain the ellipse's *shape*, and depth is sensitive to that:
  position at 10–45° degrades 24% instead of 18%. Suppression of the rod does
  not need the extra power — at `|proj|/a = 0.03` both give effectively zero.
- **Tilt-adaptive weighting** — scale the correction by estimated tilt so low
  tilt is untouched. Chicken-and-egg: the blend must read tilt from the
  provisional ellipse, but that is the contaminated one, so it under-reads tilt
  and under-applies the fix. Every variant landed at 12–13° of normal error at
  high tilt where unconditional weighting gives under 1.4°.
- **Any non-zero weight floor** — even 0.05 restores most of the error (68.1 px
  minor instead of 54.4 at 70°). The rod points dominate a fit that gives them
  any weight at all.

A binary *skip* below a 0.995 axis ratio is applied, because contamination can
only push the ratio up and it never exceeds 0.9896 above 10° of true tilt. That
recovers the near-face-on regime and changes nothing above it.

### Why it is off by default

**This section previously said the weighting collapses certified detection from
~13% to ~1–3.5%. That claim is withdrawn — it was measured under the bug in
journal Iteration 12, and the controlled A/B refutes it (Iteration 13).**

Same seed, same 400 poses, same gate, same constants, `POSE_AXIAL` the only
difference:

| core mode | detect off | detect on | pos avg off | pos avg on | ang avg off | ang avg on |
|---|---|---|---|---|---|---|
| 1280×800 | 3.2% | 3.2% | 0.268 mm | **0.186** | 0.294° | **0.178** |
| 800×600 | 2.2% | 2.8% | 0.331 | **0.271** | 0.281 | **0.211** |
| 640×480 | 2.0% | 2.0% | 0.338 | **0.296** | 0.371 | **0.244** |

Certified frames over the six usable modes: **61 off, 59 on** — the same. Modes
at 100% in spec: **5/8 off, 6/8 on**. The weighting costs no coverage and
improves both metrics everywhere, so it *should* be on.

The mechanism story was backwards too. It argued that a weighted fit reports a
worse `fit_rms_px`, so the gate — whose largest angle coefficient is
`log1p_refine_rms` — rejects good frames. But `error_model.json` was written
inside the bug window, *through the estimator*, so it was fitted on weighted-fit
features all along. The A/B shows the effect is negligible either way, so neither
story is needed.

It is still off, and the only reason is that flipping it correctly means
refitting `RADIUS_MM` (10.2662 mm, not 10.2446) and the tilt calibration in the
same pass — they are a matched set, and shipping half a set is exactly what went
wrong in Iterations 10–12. `POSE_AXIAL=1` overrides it per process for
experiments. **The outstanding work is: regenerate dataset → radius → tilt
calibration → error model with the weighting on, flip the default, re-run the
gated sweep, re-render the galleries.**

`AXIAL_DEFAULT` is the single switch, and it did not used to be: until journal
Iteration 12, `PoseEstimator.update` carried its own `axial=True` default, so the
flag governed direct `segment.segment()` callers and nothing that went through
the estimator — which is every sweep, the error-model fit, the galleries and the
runner. Those all ran the weighted fit against the *unweighted* radius, the one
pairing the comment on `RADIUS_MM` specifically warns costs more depth bias than
the entire residual. `update` now reads the module flag at call time.
**It matters for the current rig**: a side-on camera views the rotor nearly
edge-on, which is the worst case. A top-down or oblique view sits inside the
envelope.

**Tilt is ill-conditioned near face-on, and always will be.** Tilt is read from
foreshortening, `θ = acos(minor/major)`, so `dθ/d(ratio) = -1/sin θ` — which
blows up as `θ → 0`. The projection is *stationary* at face-on: a circle tilted
by 2° is only 0.06% narrower than one tilted by 0°.

| true tilt | error from a 1% axis-ratio error | sensitivity |
|---|---|---|
| 0.5° | 0.5° (i.e. total) | 115× |
| 5° | 5° (total) | 11.5× |
| 20° | 1.6° | 2.9× |
| 45° | 0.6° | 1.4× |
| 60° | 0.3° | 1.2× |

Any estimator that reads tilt from foreshortening has this property; it is not
specific to this implementation. Position is unaffected — it stays sub-mm at
face-on, because it depends on the ellipse *size*, which is well conditioned.

Practical consequence: **do not choose a dead-on reference pose for zeroing.**
The datum is a rotation, so its axis error tilts every later reading. Measured
datum quality against reference tilt:

| reference tilt | axis error | position error |
|---|---|---|
| 0° | **10.4°** | 0.45 mm |
| 5° | 2.7° | 0.71 mm |
| **10°** | **2.3°** | **0.19 mm** |
| 20° | 2.7° | 0.42 mm |
| 30° | 2.8° | 0.95 mm |
| 50° | 2.8° | 2.27 mm |

Axis error collapses as soon as you leave face-on; position error then grows
with tilt as the mast inflates the minor axis. **Zero at roughly 10–20° of
tilt.**

Taken with the >45° limit above, the good band for **orientation** is about
**10°–45°**. Position is accurate throughout, and best below 30°.

**Face-on under a hard side light with little ambient.** The duct's outer wall
is nearly parallel to the view, catches almost no directional light, and the
silhouette breaks up; the hull then collapses onto the blade cross. In the
sweep this is a single sharp corner of the space — median normal error by
(tilt, ambient):

| ambient \ tilt | 0° | 10° | 20° | 40° |
|---|---|---|---|---|
| 0.05 | **63.4°** | **40.2°** | 5.8° | 3.1° |
| 0.15 | **46.7°** | 9.4° | 4.0° | 3.2° |
| 0.25 | 1.7° | 3.0° | 3.1° | 3.1° |
| 0.60 | 2.0° | 3.4° | 3.1° | 3.0° |

Everything outside that corner is ≈3°. `seg_iou` drops to ~0.39 against a 0.89
median when it happens, so it is detectable in the log without ground truth.

Ambient light is the knob that matters here — not intensity. Measured as the
fraction of the true silhouette clearing the threshold, face-on:

- intensity 3 → 80 at ambient 0.05: **0.21 → 0.39**
- ambient 0.05 → 0.30 at fixed intensity: **0.21 → 0.89**

**The fix is lighting, not thresholding.** A contrast-relative threshold was
implemented and removed: it did not repair the unlit case (58 mm instead of
64 mm) and made every well-lit case slightly worse.

---

## Rig appearance: white-on-dark, and a red robot on any neutral backdrop

Everything above was measured on the original bench setup — a **white robot on a
dark ground** — where a fixed brightness threshold is enough. A **red robot** is
also supported, and it is not the same problem with the sign flipped.

**Brightness does not separate a red robot from its backdrop reliably.** Over
surface patches spanning specular highlight to deep shade:

| backdrop | channel | robot | backdrop | margin |
|---|---|---|---|---|
| white | luminance | 34–172 | 63–252 | **−218** (overlapping) |
| white | `R − max(G, B)` | 48–135 | 0–4 | **+44** |
| black | luminance | 34–172 | 4–73 | **−39** (overlapping) |
| black | `R − max(G, B)` | 48–135 | 1–3 | **+45** |

On white the polarity inverts *and* the ranges overlap, so no luminance threshold
works. On black the polarity is right and a lowered threshold (~40) does detect —
which is the trap, because it fails the moment the backdrop picks up a sheen and
the whole frame passes. Measured: brightness@40 detects on velvet and matte black,
and returns nothing on a sheeny black.

**Chroma is background-invariant, and that is the real result.** `R − max(G, B)`
responds to the paint, and every neutral surface has R = G = B however it is lit,
so black, grey and white all read 0–4. Across seven backdrops — velvet, matte
black lit and bright, sheeny black, nominal white, blown white, shadowed white —
the fitted rim varies by **0.21 %**. One threshold and one calibration serve all
of them, so you can change the backdrop without refitting anything.

So `segment.score_channel` returns *a single channel in which the robot is
bright*, and everything downstream — morphology, hull, ellipse fit, sub-pixel
refinement — is appearance-agnostic. Select with `segment.APPEARANCE` or
`POSE_APPEARANCE=red`; `test_appearance.py` pins both claims.

Chroma also keeps the property the fixed threshold was chosen for: an empty
backdrop contains no red, so it yields no detection. An adaptive threshold would
find the most extreme thing present and report it confidently.

**If you get to choose, red on black is the better rig.** Same chroma margin as
white, plus a dark backdrop suppresses stray light and inter-reflection onto the
robot, and it leaves luminance as a weak second cue rather than an inverted one.

### Measured, per rig

Each appearance was rendered on its own 1400-pose dataset and refitted; medians
on a held-out split, plus a fresh 120-pose run never used for tuning:

| | bright (white on dark) | **red on white** | **black on white** |
|---|---|---|---|
| `RADIUS_MM` | 10.2446 mm | 10.2616 mm | 10.1106 mm |
| detected, tuning set | 1400/1400 | 1400/1400 | 1375/1400 |
| ‖position‖, tilt 10–45° | 0.717 mm | **0.567 mm** | 1.772 mm |
| normal, tilt 10–45° | 0.826° | 1.132° | 2.795° |
| fresh run, ‖position‖ median | — | **0.675 mm** | 1.707 mm |
| fresh run, ‖position‖ p95 | — | **2.59 mm** | 50.8 mm |

**Black on white is the weakest of the three, by about 3×**, and the p95 is an
order of magnitude worse. The cause is not the threshold level — it was swept,
and detection is 100 % at every level tried. It is that a black body has bright
regions of its own: on the ~5–10 % of frames that fail, segmentation IoU is
**0.064** against 0.344 on the rest, i.e. a lit side of the body failed the
darkness test and the hull lost that whole side. A larger morphological closing
does not bridge it (identical failure count from 7×7 to 41×41) because these are
not small gaps.

Chroma cannot rescue it either, and that is the structural point: **a black body
and a white ground are both neutral**, so only brightness separates them, and
brightness is exactly what illumination changes. Across the lighting sweep the
body's brightest decile reaches 206 counts while the ground's dimmest drops to
178 — genuinely overlapping. Red on white does not have this problem, because
paint colour is invariant to how it is lit.

If the body can be coloured rather than black, **it is worth roughly 3× in
position accuracy**. If it cannot, keep the backdrop evenly lit and above ~145
counts, and prefer diffuse over directional light — the failures are specular.

> **The renders have no cast shadows** (pyrender's shadow flags are not enabled),
> so a real black body on a white backdrop will be *worse* than these numbers: it
> throws a shadow that is dark and neutral, which is to say indistinguishable
> from the robot by either cue. Worth checking on the bench before trusting the
> table above.

### The drive coils

The bronze/orange coils beside the robot are **dark against a white ground**
(44–163 counts, against the body's 4–73), so an inverted threshold picks them up
— and `silhouette_hull` pools every blob into one convex hull, so a coil near the
rim does not add a stray contour, it swallows the rim. Measured on a scene with
coils flanking the rotor: the fitted major axis goes to **270 px against a true
120**, and the fit still returns an ellipse, just a meaningless one.

Chroma excludes them, hue-agnostically (`max − min` over BGR, so "orange" and
"bronze" under changing light are one case): every neutral surface reads ≤ 8,
every coil ≥ 48. With the gate the fit is unchanged by their presence
(126.7 → 127.2 px). End to end they still cost something — 1.71 → 2.03 mm median
— but that residual is not at their boundary: growing the mask to cover the
blend zone changed the median by nothing and doubled the cost, so it was not kept.

`segment.clutter_mask()` returns the coil mask on its own. They are fixed to the
rig, so their image position is a direct check on whether the camera has moved.

### How the appearances are kept apart

The renderer emits the body colour it is given (`render.RED_BODY`,
`render.BLACK_BODY`, `render.COIL_BODY`) on a ground of any level, so each rig is
measured rather than merely run: `make_dataset.py --appearance {bright,dark,red}`.

Each appearance carries its own matched constant set, resolved automatically by
`estimator.RADIUS_BY_APPEARANCE` and `calibration.calibration_path()`
(`tilt_calibration.json` / `tilt_calibration_red.json`). Shipping half a pair is
the exact failure of journal Iterations 12–14, so `test/test_appearance.py`
asserts every appearance has a complete one.

### What needs refitting for a new appearance

`RADIUS_MM`, the tilt and centre calibrations and the error model are all fitted
against a particular channel: a chroma threshold cuts a red edge at a different
fraction of its transition than a luminance threshold cuts a white one, and depth
is read from the hull's size. For the red rig that shift is 0.17 % of radius —
0.4 mm of depth at 240 mm, larger than the depth residual.

Done for `red`:

```
uv run python controller/pose/validation/make_dataset.py --appearance red \
    --out results/pose_validation/dataset_red.npz
POSE_APPEARANCE=red uv run python controller/pose/validation/tune.py \
    --data results/pose_validation/dataset_red.npz --write
```

then paste the printed radius into `estimator.RADIUS_BY_APPEARANCE`. Still
outstanding for `red`: the gated sweep and the error model, so `uncertainty.py`
should not be relied on under this appearance yet.

Capture also has to change: `CameraSource` and `VideoFileSource` default to
`grayscale=True`, which discards the only usable signal. Ask for colour
explicitly (`grayscale=False`) under `red`; a grayscale frame reaching the red
path raises rather than guessing.

`dark` is the exception and needs no such change, because its camera has no
colour to give. See "Rig appearance" below.

## Usage

```bash
cd ESP32_PMW

# the live path: open the camera, measure its rate, run the loop with the
# overlay (fitted ellipse, rotor axis, ignored area shaded, sustained fps),
# and optionally record. Resolution is the main throughput knob and is set in
# section 1: 1280x800 costs 7.9 ms/frame against an 8.3 ms camera period,
# 640x400 costs 2.5 ms against 3.7 ms and stays camera-bound.
uv run jupyter lab controller/pose/online_camera.ipynb

# offline, against the 240 fps footage already in the repo
uv run jupyter lab controller/pose/offline_video.ipynb

# set the datum, then report relative to it (section 3 of the notebook)
uv run python controller/pose/calibrate_zero.py --source camera --frames 30

# tuning: render a train/test set once, fit on train, score on held-out test
uv run python controller/pose/validation/make_dataset.py
uv run python controller/pose/validation/tune.py --write

# diagnostic page: overlays + interactive curves, one self-contained HTML file
uv run python controller/pose/validation/sensitivity.py
uv run python controller/pose/validation/visualise.py

# filter: noise-vs-lag and velocity quality on a rendered trajectory
uv run python controller/pose/validation/trajectory.py

# latency (stage timings; --camera adds the grab->pose figure)
uv run python controller/pose/validation/latency.py
uv run python controller/pose/validation/latency.py --camera --width 640 --height 480

# stereo: geometry check first, then the solver, then the rig sweep
uv run python controller/pose/validation/selftest_stereo.py
uv run python controller/pose/test_stereo.py
uv run python controller/pose/validation/sweep_stereo.py --quick
uv run python controller/pose/validation/sweep_stereo.py --elev 35 45 55 --mixed

# the visual report: stereo pairs, projection lines, pose vs truth in 3-D
uv run python controller/pose/validation/visualise_stereo.py

# what a candidate rig buys, before rendering anything
uv run python controller/pose/rig.py --elev 45 45 --azim 0 90 --scale 0.488

# stereo extrinsics: open this from controller/pose/ and run it top to bottom.
# Capture pairs with the board SWEPT through many poses, angled to bisect the two
# cameras. World frame is camera A. Sections 12-13 (self-test, intrinsics
# regression) run with no hardware, so start there to check the install.
uv run jupyter lab controller/pose/stereo_calibration.ipynb

# then a live stereo run against the measured rig (untested against hardware):
# section 7 of online_camera.ipynb, which is guarded and reports capture skew

# validation
uv run python controller/pose/validation/sweep.py --quick     # 48 cells, ~5 s
uv run python controller/pose/validation/sweep.py             # 21504 cells, ~27 min
uv run python controller/pose/validation/report.py results/pose_validation/validation_results.csv

# tests
uv run python controller/pose/test_conic.py
uv run python controller/pose/test_zeroing.py
uv run python controller/pose/test_calibration.py
uv run python controller/pose/test_filter.py
uv run python controller/pose/validation/selftest.py
```

### The CSV

`# key, value` provenance comments (intrinsics, radius, source, datum) then a
header row — the convention from `ai/picoscope_capture.py`. Read it with
`pandas.read_csv(path, comment="#")`.

Lost frames are written as blank rows rather than dropped. A gap is a real
event, and silently omitting it would make a run look cleaner than it was.

Beyond the six plotted channels, the columns that matter when something looks
wrong:

| column | use |
|---|---|
| `ambiguity_margin_deg` | how far apart the two candidate poses were — the size of the bet taken this frame |
| `jump_deg` | how far the normal moved since the last frame; larger than the margin suggests a branch flip rather than motion |
| `fit_rms_px` | how elliptical the silhouette actually was; rises when segmentation grabs the wrong thing |
| `area_px`, `major_px`, `minor_px` | raw measurements, for sanity-checking scale |
| `t_seg_ms`, `t_est_ms` | where the time went |

### Zeroing

`calibrate_zero.py` estimates the pose in a reference view and stores it in
`pose_zero.json`. Every later pose is reported in that frame, so re-running the
estimator on the reference reads **exactly zero on all six channels** — asserted
in `test_zeroing.py`.

The datum is a full rotation, not just a translation: it is built from the
reference normal *and* the reference in-plane direction, so azimuth is pinned
too. Zeroing only position would leave `φ` reading whatever the camera mounting
happened to be. `--clear` restores identity; averaging several frames is worth
it live, since datum noise becomes a fixed bias on the whole run.

---

## Files

| file | role |
|---|---|
| `conic.py` | the maths: image ellipse + known radius → two 3-D circle poses |
| `segment.py` | threshold → morphology → convex hull → ellipse fit |
| `estimator.py` | per-frame `Pose`, branch disambiguation, datum application |
| `zeroing.py` | the `Zero` datum: build, apply, save, load |
| `sources.py` | `FrameSource`: images, video, threaded camera grabber |
| `recorder.py` | CSV, in the repo's `# key, value` header convention |
| `online_camera.ipynb` | the live path: camera, loop, overlay, recording, stereo |
| `calibrate_zero.py` | set the datum |
| `calibration.py` | the fitted tilt correction: fit, apply, save, load |
| `filter.py` | constant-velocity Kalman — for velocity, coasting and prediction, not smoothing |
| `rig.py` | `StereoRig`: where the cameras are, and the geometry predictions that follow |
| `stereo.py` | `match` → `fuse` → `refine`, and `StereoPoseEstimator` |
| `stereo_calibration.ipynb` | intrinsics + extrinsics from a swept ChArUco board; world = camera A; refuses to write a rig above 0.5 px RMS |
| `test_stereo.py` | the solver against analytic geometry, no renderer |
| `validation/render_stereo.py` | one GL context, N views per pose, plus occluders |
| `validation/selftest_stereo.py` | rendered pair vs. the rig's own transform — run this first |
| `validation/sweep_stereo.py` | the rig-geometry grid, scored per world axis |
| `validation/scene3d.py` | the 3-D panel: cameras, back-projection cones, pose vs truth |
| `validation/page_stereo.py` | the stereo report's markup, reusing `page.py`'s CSS |
| `validation/visualise_stereo.py` | builds `stereo_diagnostics.html` |
| `validation/overlay.py` | draws fitted vs true ellipse, orientation vectors, both ambiguity branches, amplified residual |
| `validation/gallery.py` | the condition and failure tile sets the page shows |
| `validation/sensitivity.py` | fine noise and background curves |
| `bounds.py` | Cramer-Rao floors: edge localisation, ellipse Fisher matrix, pose covariance, the depth law |
| `test_bounds.py` | Monte-Carlo verification of every bound in `bounds.py` (60+ checks) |
| `validation/limits.py` | the real system placed against those floors; where the gap comes from |
| `validation/visualise.py`, `page.py` | build the self-contained diagnostic page |
| `run_tests.py` | runs every `test_*.py` suite as a subprocess; `-v <name>` to rerun one |
| `test_appearance.py` | bright-on-dark vs red-on-white segmentation |
| `calibration.ipynb` | the five constants, in dependency order |
| `offline_video.ipynb` | run the estimator over a recording; plots, CSV, per-frame diagnosis |
| `online_camera.ipynb` | open the camera, measure its real rate, capture a burst |
| `stereo_calibration.ipynb` | ChArUco intrinsics + camera-to-camera extrinsics |

Machine-generated working files live in `results/pose_validation/_ai_scratch/`
and `validation/_ai_scratch/` — per-iteration sweeps, feature caches, superseded
datasets, one-shot journal generators. Nothing there is on a shipping path and
each folder has a README saying what it holds; both are safe to delete.
| `validation/render.py` | pyrender wrapper: pose, alpha, lighting, background → image + ground-truth mask |
| `validation/make_dataset.py` | render a train/test pose dataset once, so tuning need not re-render |
| `validation/tune.py` | fit calibration on train, report per-DOF residuals on held-out test |
| `validation/latency.py` | stage timings and grab→pose latency |
| `validation/trajectory.py` | filter evaluated on a rendered path with known ground truth |
| `validation/sweep.py` | the grid, scored against truth |
| `validation/report.py` | summary and figures |
| `validation/selftest.py` | renderer vs analytic projection — run this first if anything looks wrong |

### Design notes worth knowing before editing

**Segmentation hulls the silhouette rather than taking the largest contour.**
The duct is a thin ring; lighting routinely breaks it into arcs and the largest
connected blob becomes the blade cross in the middle. Face-on, largest-contour
fits an 83 px ellipse where the rim is 131 px — a 37% underestimate landing
straight on depth. The hull gets 129.9 px against 130.7 px analytic.

**No robust refit on the hull.** Trimming outliers was tried and measurably hurt:
the hull carries only 20–60 points, so discarding a meaningful fraction
destabilises the fit (at 70° tilt it pushed the major axis *away* from truth,
129.5 → 139.5 px). The hull already does what trimming was meant to do.

**The radius is the outer rim (10.204 mm), not the mean rim (9.965 mm)**, because
a hull rides the outermost surface. Using the mean biases every distance by 2.4%
— a systematic 5 mm at 200 mm that no filtering removes.

**Open small, close large** (3×3 then 7×7). The projected rim wall is a few
pixels thick and a 5×5 opening erodes it away.

### Renderer

pyrender, with `trimesh` and **`pyglet<2`** (pyrender 0.1.45 predates the
pyglet 2 API break).

Open3D was the first choice and does not work here. `OffscreenRenderer` cannot
run at all — the macOS wheel is built EGL-headless-only and fails with *"EGL
Headless is not supported on this platform"*; calling
`gui.Application.instance.initialize()` first does not help. The legacy
`Visualizer` does render, but `RenderOption` exposes no material opacity, which
makes the alpha sweep impossible.

**One `Renderer` per process.** pyglet's Cocoa backend cannot build a second
NSOpenGL pixel format, so constructing another — even after `close()` — raises
inside pyglet. To compare resolutions, render once and resize.

`validation/selftest.py` checks the rendered silhouette against
`conic.project_circle`, an independent analytic projection. This is the test that
catches a flipped axis convention, which otherwise produces images that look
perfectly plausible and ground truth that is silently mirrored.

---

## Stereo

Two cameras. `stereo.py`, `rig.py`, and the harness under `validation/`.
Everything above still describes the single-camera path, which is unchanged.

### Headline

Measured on 900 rendered pairs at **500×375** — the operating point the rate
target implies — against world-frame ground truth, with realistic read noise,
motion blur and 310–350 Hz rotor spin.

| | mono (oracle branch) | stereo | |
|---|---|---|---|
| worst position axis | — | **0.217–0.375 mm** | gate is 0.5 mm |
| ‖position‖ | 1.28–1.58 mm | **0.33–0.52 mm** | 2.5–3.9× |
| normal | 1.61–2.04° | **0.81–1.11°** | 1.8–2.5× |
| branch ambiguity | wrong on ~50% of frames | decided geometrically | |
| catastrophic frames (>5 mm) | 9.1% | **0.00%** | see the gate, below |

**The 0.5 mm position gate is met on every axis, in every rig configuration
except the worst one.** The 0.5° orientation target is not met; the best
measured is 0.81°, and the reasons are structural — see below.

Mono is scored on whichever of its two branches is closer to truth, an estimator
that cannot exist, so that stereo has to beat the geometry rather than the coin
toss.

### Rig geometry: the model beats the triangulation

Both cameras at elevation `e` above horizontal, 90° apart in bearing. Raising
`e` moves the optical axes *closer together* (worse triangulation) but shows each
camera a *less tilted* rotor (better flat-circle model), and it was not obvious
in advance which wins.

| elevation | axis sep | tilt seen | worst axis | ‖pos‖ | normal | detected |
|---|---|---|---|---|---|---|
| **+55 / +55** | 48° | 35° | **0.217 mm** | 0.332 | 0.840° | 83% |
| +45 / +45 | 60° | 45° | 0.352 | 0.483 | 1.109° | **93%** |
| +45 / −45 | 60° | 45° | 0.375 | 0.523 | **0.807°** | 89% |
| +35 / −35 | 71° | 55° | 0.430 | 0.639 | 1.115° | **99%** |
| +55 / −55 | 48° | 35° | 0.433 | 0.535 | 1.084° | 65% |
| +35 / +35 | 71° | 55° | **0.819** | 1.014 | 1.848° | 98% |

The ranking is **by tilt-seen, not by axis separation** — the exact opposite of
what the triangulation arithmetic predicts on its own. Purely geometrically,
elevation 35° should win: its 71° separation gives a predicted fused σ of
0.183 mm against 0.273 mm at 55°. It comes last. Silhouette model error
dominates triangulation conditioning by enough to invert the order, which is
worth remembering the next time a rig decision looks like a geometry problem.

**45° is the recommended default**: 0.352 mm worst axis with a 93% detection
rate. 55° is more accurate and drops 10 more points of detection; 35° holds
almost every frame but fails the gate on depth when both cameras sit high-tilt.

Mixed hemispheres (+e/−e) were expected to cost nothing optically — ±elevation
are mirror images for a single view — and mostly do not. They give the best
orientation of any configuration (0.807°), because the mast protrudes on
opposite sides of the rim plane in the two views so the silhouette biases oppose
rather than reinforce. They cost detection rate at high elevation (65% at ±55°)
and they cost a much harder calibration: a ChArUco board is one-sided, so
cameras straddling the rotor plane need a double-sided board or the wand method.

### The cross-view gate is the most valuable thing the second camera buys

This was not the expected answer. Before any gate, 9.1% of frames were
catastrophic — position error above 5 mm, sometimes above 50 mm — and on every
one of them **the monocular estimate was already wrong by ~23 mm**, because
segmentation had grabbed the wrong thing in one view. Stereo cannot repair that.
What it can do is *notice*:

| | good frames | catastrophic frames |
|---|---|---|
| cross-view discrepancy | 5.4 mm median | **349 mm** median |

A 60× separation, with no ground truth involved — just the two cameras
disagreeing about where the robot is. Rejecting frames above **25 mm**
(≈1.25 body diameters: two views that disagree by more than the robot's own size
have not both seen the robot) takes the catastrophic rate to **0.00%** and
brings p95 ‖position‖ from 5.59 mm to 1.81 mm, for 5–15% of frames. A controller
is far better served by a declared gap than by a confident 50 mm error, and
`estimator.py` already treats a lost frame as normal.

The gate is also the only per-frame health check that works on the bench, where
there is no ground truth at all. A rising `discrepancy_mm` means the extrinsic
has drifted, a view is occluded, or segmentation is failing — and it says so
before the poses look wrong.

### Rate

| configuration | compute | sustained | 240 fps | 420 fps |
|---|---|---|---|---|
| `--no-refine` (match + fuse) | **1.71 ms** | **584 Hz** | OK | **OK** |
| full (+ orientation refinement) | 3.22 ms | 311 Hz | OK | over |
| — of which segmentation, 2 views | 1.64 ms | | | |

**420 Hz is met today with `--no-refine`.** Position is identical either way —
it comes from the closed-form fusion in both — so the whole cost of the rate
target is orientation, which degrades from ~0.8–1.1° to ~1.0–1.8°.

Segmentation is 96% of the `--no-refine` budget, so it is the only thing worth
optimising next, exactly as the monocular timing section concluded. ROI tracking
off `filter.predict_ahead()` is the obvious lever and is not implemented: the
robot is ~115 px across in a 500 px frame, so a tracked 200×200 window is ~20%
of the pixels.

### How it works, and what each layer is worth

Three layers, each usable alone.

**`match`** — 60 µs. Both cameras see the same circle, so of the four
combinations of the two views' two branches, one agrees and three do not.
Closed form, no temporal prior. `estimator.py`'s continuity heuristic is
replaced by a measurement.

The agreement test is **Mahalanobis, not Euclidean**, and the difference decides
frames. Each view's error is anisotropic ~11:1, so two views that agree
perfectly still differ by millimetres along their respective depth axes. Scoring
that as evidence against the pair left the true pair ahead by only 1.6× the
noise. Weighting by `inv(Σa + Σb)` asks the right question — not "how far apart
are these answers" but "how surprised should I be, given how each view is
allowed to be wrong" — and the winning margin went from 1.6× to ~50× the noise.

**`fuse`** — 35 µs. Information-form combination. Each camera's bad axis is the
other's good one, so the fused worst axis lands near the *lateral* number rather
than the depth one. It attacks **bias**, not just noise, and that is the part
that matters: the single-view residual autocorrelates at 0.966 after one frame,
so no filter removes it, but along camera A's depth axis camera B measures
laterally and carries ~120× the weight. A 3 mm systematic depth error in A
enters the fused answer as 0.025 mm.

**`refine`** — 1.5 ms. Joint fit on ℝ³ × S², five parameters, never six.
**Orientation only**, holding the centre at the fused value — because measured
against ground truth the joint fit improves the normal ~1.7× and makes position
slightly *worse*. Information weighting is already the right estimator for a
centre; a reprojection fit weights both views' pixels alike.

### What was tried and rejected

Recorded so nobody spends the afternoon again. The pattern is that three of the
four things expected to matter did not, and the thing that did was not on the
list.

| tried | result |
|---|---|
| **Robust loss (`soft_l1`) to reject the mast** | **No effect**: 0.524° vs 0.509° for plain least squares. The robustness argument only ever applied to per-point residuals, and there it loses outright (below). Default is `linear`. |
| **Sampson residuals on every hull point** | **Worse**: 1.55° against 0.53° for eight axis-endpoint residuals. `tilt_calibration.json` was fitted against `cv2.fitEllipseDirect` output, so it is a statement about *that statistic*; applying it while residualising raw hull points compares the model against a quantity the correction was never fitted for. |
| **Modelling the silhouette (`TiltCalibration.unapply`)** | **This was the whole thing.** Without it, refinement improved position on 75% of frames and degraded orientation on 78%. With it, both improve. Isolated on synthetic data with injected mast distortion: 3.74° → 0.13°, a 28.9× reduction. |
| **Five-parameter refinement** | Position 0.242 → 0.258 mm, i.e. worse, and 2.4× the cost. Orientation-only is both faster and better. |
| **Euclidean branch agreement** | Left the correct pair ahead by 1.6× the noise; a few percent of frames picked wrong, each a catastrophic outlier. |

One real bug worth recording: the axis-endpoint residual was not invariant to
the 180° ellipse-angle wrap. An ellipse axis is a line, so 179° and −1° describe
the same shape but put the endpoints at opposite ends, producing a discontinuous
cliff in the residual that the optimiser lands on as a spurious minimum. Fixing
it took the five-parameter solve from 0.94 mm to 2.4e-06 mm on clean data.

### Background subtraction: not needed, and RPCA is the wrong tool

Considered and dropped, with the reasoning kept because the question recurs.

Robust PCA (`min ‖L‖* + λ‖S‖₁`) works on video because the redundancy is
**temporal** — columns of `D` are frames and a static background is low-rank
across time. On a stereo *pair*, `D = [vec(I_A) vec(I_B)]` has two columns, and
two views 48–71° apart share no pixel-level content. There is nothing for a
nuclear norm to find; the cross-view redundancy here is geometric, not
linear-algebraic, and exploiting it is the reprojection gate above. Batch RPCA
also needs an SVD per iteration — GRASTA manages ~57 fps in MATLAB, four orders
off this loop.

The decisive argument is simpler. **The drone starts stationary on the takeoff
stand.** Any background built from temporal statistics — MOG2, KNN, GRASTA,
online eigenbackground — absorbs a stationary object into the background and
then loses it. And if instead you capture the background once with the volume
clear and freeze it, the low-rank component is literally one stored image, rank
one: RPCA has degenerated into `cv2.absdiff` and the machinery evaporates.

Meanwhile the fixed threshold is measured noise-immune (0% failures at σ=80 read
noise) and fails only above background grey 0.4, which a controlled backdrop
never reaches. `segment.py` is unchanged, so `RADIUS_MM` and the 21,504-cell
monocular validation both remain valid.

### Why tilt hurts, and what actually fixes it

Measured, not inferred. `seg_iou` is **flat at 0.89** from 0° to 71° while
`fit_rms_px` goes **0.15 → 1.90 px**: the segmenter traces the silhouette
correctly at every tilt, and the silhouette stops being an ellipse. Nothing in
segmentation is left to win — which the seven rejected variants above already
suggested, and this confirms from a second direction.

The cause is **not** what this README previously said. Rendering sliced copies of
the mesh separates two mechanisms:

| | mechanism | onset | evidence |
|---|---|---|---|
| below ~55° | the rim is a **4 mm-tall cylinder wall**, not a circle | all tilts | the rim ring *alone* reproduces the contamination exactly |
| above ~55° | the **mast** | 52.5° (`tan θ > R/h`) | deleting the mast removes it entirely: +23.1 px → +3.2 px at 70° |

The magnet body contributes nothing at any tilt. The old attribution ("the mast
and magnet") was wrong for the regime that matters.

Both act on the **short** direction only. Measured on held-out renders, the major
axis holds to ~0.5 % at every tilt while the minor axis scatter reaches **26 %**
past 60° — a 25:1 split, and the reason position survives tilts that destroy
orientation. Full derivation in
[theory/lecture_notes.md §12](../vision/theory/lecture_notes.md).

Two consequences now shipped:

- **The tilt calibration is a physical model, not a fitted curve.** The wall gives
  `ρ = cos θ + k sin θ`, which inverts in closed form; one parameter `k = h/R`
  replaces the old two-parameter quadratic. Held-out median tilt error
  **2.43° → 1.97°**, bias **+0.807° → −0.038°**. It must be fitted over 20–50°:
  fitted over the full range the mast drags `k` from 0.043 to 0.098 and the
  result is *worse than no correction at all*.
- **A resolution floor falls out of it.** `ρ` exceeds 1 on `(0, 2·atan k)`, so
  tilts inside that band are indistinguishable at any resolution. With the
  fitted `k` that is **4.96°** — and it explains the previously-empirical
  observation that azimuth collapses below ~10° of tilt.

**The centre is displaced too, and it resists correction.** The hull grows on one
side, so the fitted ellipse's centre moves — and lateral position is read straight
off it. Measured: 0.185 → 0.274 mm through 10–55°, then the sign flips near 57°
as the mast takes over from the wall, reaching **0.867 mm at 70°** — past the
position target on its own. Three corrections were built and measured; all lose. Applied to the
*measured* ellipse it removes 22–68% of the 2-D displacement and still makes 3-D
position worse (0.397 → 0.533 mm), because back-projection consumes all five
ellipse parameters jointly and one cannot be corrected in isolation. Moved into
the forward model it is neutral or worse. `centre_calibration.json` is fitted and
shipped but **not applied**; see the notes' §12.7.

**Two gates, because one is blind to correlated failure.** The cross-view gate
catches views that *disagree*. It cannot see both views failing the same way,
which is what happens near face-on where the rim's outer wall goes unlit in both
cameras at once — those frames agree to within 5.6 mm on a wrong answer. A second
gate on fit quality (`fit_rms_px ≤ 1.5`, about 5× the sub-pixel boundary
precision) catches them: orientation p95 **25.7° → 3.15°**, median 1.87° → 0.96°,
for 29% of frames.

`validation/tune_weighting.py` tests re-weighting the hull; see the notes' §12.7
table for what won.

### The report

`validation/visualise_stereo.py` writes a self-contained
`results/pose_validation/stereo_diagnostics.html`: for a typical case and a hard
one, the stereo pair the estimator saw, the back-projection geometry in 3-D, and
the answer against ground truth — **truth black, estimate amber**.

Three things it shows that a table cannot:

- **One pose reprojected into both views.** Each frame carries the fitted
  ellipse, the true rim, and the joint 3-D estimate projected back into that
  camera. If the third lands on the robot in *both*, one pose explained two
  images — which is the entire claim of the joint solve.
- **The two cones meeting.** Rays leave each camera through pixels on that
  view's fitted ellipse and stop at the estimated disk's plane, so the ring of
  endpoints *is* the observed rim back-projected. Its spread off the drawn
  circle is the reprojection residual, in millimetres, in space.
- **What 0.4 mm actually looks like.** Nothing is amplified. On the typical case
  the two disks separate by a hairline; on the hard case (91st percentile) they
  come visibly apart. Cases are re-rendered and re-measured rather than replayed
  from the sweep CSV, because those rows record the pose but not the lighting or
  noise draw — so a replay would caption a new image with an old number.

### Occlusion

The takeoff stand — a black 8 mm rod below the robot — hides **6.8% of the
robot's lit pixels from a camera looking up and none from one looking down**, as
expected. It costs essentially nothing: the fitted ellipse moves by 0.1 px
(major 115.7 vs 115.7 px), because `silhouette_hull` takes a convex hull and
broken arcs hull to the same circle. That is the design rationale in
`segment.py` doing exactly what it was built for.

Worth knowing: a "black" rod is not black. glTF pins dielectric specular at
F0 = 0.04, so under a hard light the stand peaks around grey 155 and crosses the
128 threshold — physically right, a real anodised rod has a sheen. Its 13 lit
pixels fall under the 2% blob-keep fraction and are rejected, but a larger or
closer occluder would not be.

### Not done

- **Closing the loop.** `hover_controller_runner.py:103` still has a
  `CameraSource` raising `NotImplementedError`. Both estimators satisfy that
  interface but nothing here modifies the controller.
- **The live path has never seen two cameras.** There is one webcam on the
  bench, measured at ~28 fps flat, and no high-speed camera. `--stereo`,
  `sources.StereoSource` and `stereo_calibration.ipynb` are written, exercised
  against rendered pairs and files, and geometrically verified — but first
  contact with hardware should be treated as a debugging session, not a
  measurement.
- **ROI segmentation**, the one clear path to 420 Hz *with* refinement.
- **0.5° orientation.** Best measured is 0.807°. Reaching 0.5° needs the
  silhouette model error to fall further; a per-view correction indexed on
  viewing geometry rather than a single scalar curve is the next thing to try.
- **Sub-pixel boundary refinement**, requested and not built: the monocular
  README already records the vertex-walk variant as measured-and-dead
  (0.662% → 0.661%), and the stereo results say the same thing from a different
  direction — the residual is silhouette model error, not edge precision.

---

## Results on disk

In `ESP32_PMW/results/pose_validation/`, following the repo's convention of
keeping run artifacts out of the source tree (the full CSV is ~8 MB):

- `validation_results.csv` — the full 21,504-cell sweep
- `validation_results_figures/summary.png` — heatmaps of position error over (alpha × background)
  and normal error over (tilt × ambient), error-vs-tilt curves showing the
  ambiguity cost, and the compute-time histogram against the 240/420 fps budgets
