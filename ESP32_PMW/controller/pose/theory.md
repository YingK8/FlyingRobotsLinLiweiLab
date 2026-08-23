# Chapter 3. Pose: frame in, five degrees of freedom out

*Stage 3 of the pipeline. Consumes: frames from [chapter 1](../camera/theory.md)
and the frames fixed by [chapter 2](../calib/theory.md). Produces: a `Pose` per
frame. Consumed by: [chapter 4, control](../control/theory.md).*

This is the largest chapter because it is the one that turns an image into a
number, and there are three separable questions in that: what a camera can *see*
of the robot (§12), what precision is *possible* at all (§13), and how to find the
robot in a cluttered frame in the first place (§15).

## Reading order

| # | file | what it does |
|---|---|---|
| 1 | `segment.py` | frame -> silhouette -> fitted rim ellipse. Start here; §15 is about this file |
| 2 | `conic.py` | ellipse -> circle pose. The geometric core, §12 |
| 3 | `estimator.py` | ties 1+2 together, applies the calibrations, resolves the ambiguity |
| 4 | `stereo.py` | the same from two views, which kills the ambiguity outright |
| 5 | `filter.py` | constant-velocity Kalman: velocity, coasting, latency compensation |
| 6 | `uncertainty.py` | predicts this frame's error so the estimator can decline |
| 7 | `bounds.py` | the Cramer-Rao floors of §13: what no estimator can beat |
| 8 | `render.py`, `render_stereo.py` | the renderer: ground truth to measure all of the above against |

The dependency runs 1 -> 2 -> 3 -> 4; `bounds.py` and the renderer are how the
claims in this chapter were measured rather than asserted.

**Read §15 first if you are debugging a live frame**, and §12 first if you are
debugging a number. §13 is the one to read when something seems *too* good.

## 12. The observation model: what a camera can see of the robot

[§11.6](../control/theory.md) (ch.4) ends by making the tilt states observable through "the disk's normal vector" and
treats that measurement as given. It is not given. This section derives what a camera can
and cannot recover from the silhouette, why the recoverable part is anisotropic by an order
of magnitude, and where the model breaks. Implementation: `controller/pose/`.

**Assumptions (B1–B4):**

- **B1, Known circular feature.** The duct rim is a circle of known radius $R$. Measured
  off `flyingrobot_thick _rod2.STL`: $R = 10.204$ mm, radial standard deviation 0.108 mm.
- **B2, Calibrated pinhole.** Intrinsics $K$ known; the contour is undistorted before use.
- **B3, Rigid.** The rim does not deform in flight.
- **B4, Spin aliased.** At 310–350 Hz against $\le$ 420 fps, blade phase is temporally
  aliased ([§0](../control/theory.md) (ch.4), A5) and is treated as an unknown nuisance, not a state.

### 12.1 Circle to ellipse, and back

A circle of radius $R$, centre $C$, unit normal $n$, viewed from the origin, generates a
cone. With the circle's plane written $n\cdot X = d$, $d = n\cdot C$, the ray $sX$ meets it
at $P = dX/(n\cdot X)$; demanding $|P - C| = R$ and clearing the denominator gives a
quadratic form $X^\top Q X = 0$ with

$$Q = d^2 I - d\,(n C^\top + C n^\top) + (|C|^2 - R^2)\, n n^\top .$$

In pixels, $p = KX$ up to scale, so the image conic is $K^{-\top} Q K^{-1}$: an ellipse.
Running it backwards, $Q$'s eigendecomposition has signature $(+,+,-)$ and a quadric cone
admits exactly **two** families of circular cross-section, so a fitted ellipse yields two
poses $(C, n)$ and $(C', n')$. This is not a numerical degeneracy: a tilted circle and its
mirror image through the viewing axis project to the *identical* ellipse, so no single view
can choose. Two views can (§12.6). Code: `pose/conic.py`, after Safaee-Rad et al.,
IEEE T-RA 8(5), 1992.

### 12.2 Conditioning: why depth is the weak axis by construction

Let $a, b$ be the image ellipse's semi-axes. Two scalars carry almost all the information:

$$z = \frac{2 f R}{2a} \quad\Longrightarrow\quad \frac{\mathrm{d}z}{z} = -\frac{\mathrm{d}a}{a},
\qquad\qquad
\theta = \arccos\frac{b}{a} \quad\Longrightarrow\quad \frac{\mathrm{d}\theta}{\mathrm{d}(b/a)} = \frac{-1}{\sin\theta}.$$

Two consequences, both structural rather than implementation defects:

- **Depth is measured from the ellipse's *size*, lateral position from its *centre*.** A
  one-pixel error on a 130 px major axis is 0.77 % of range; the same pixel on the centroid
  is 0.16 mm. Measured, the ratio is $\approx 11:1$ (0.078 mm lateral against 0.857 mm
  depth), and it is constant in *relative* terms: 0.57 % of range at every distance.
  The clean statement of this ratio is $z/2R$, range over diameter, and that is what the
  rest of §12 uses. It is the **tilt-known** case. §13.4 derives the version that applies
  here, where tilt must be estimated from the same ellipse: the penalty is
  $g(\theta)\,z/2R$ with $g(0)=\sqrt3$, i.e. **73 % larger**, and closer to $z/R$ in
  practice.
- **Tilt is singular at face-on.** $\mathrm{d}\theta/\mathrm{d}(b/a)$ diverges as
  $\theta \to 0$: a circle tilted 2° is 0.06 % narrower than one tilted 0°. Any estimator
  reading tilt from foreshortening inherits this.

### 12.3 The silhouette is not the rim

B1 says the rim is a circle. It does not say the *silhouette* is the rim's projection, and
it is not. Two distinct mechanisms, separated by rendering sliced copies of the mesh:

**(a) The wall.** The rim is a short cylinder of radius $R$ and half-height $h$, not a
zero-thickness circle. Tilt it and the near-top and far-bottom edges project outside the
mid-plane circle, so the short half-extent becomes $R\cos\theta + h\sin\theta$ while the
long extent stays $R$. The observed axis ratio is therefore

$$\rho = \frac{b}{a} = \cos\theta + k\sin\theta = \sqrt{1+k^2}\,\cos(\theta - \arctan k),
\qquad k = h/R,$$

which inverts in closed form: $\theta = \arctan k + \arccos\!\big(\rho/\sqrt{1+k^2}\big)$.
Fitted on held-out data, $k = 0.0433$, i.e. $h = 0.44$ mm. Rendering the rim ring **alone**
reproduces this contamination exactly, so it is the wall and not the mast.

**(b) The mast.** The mast extends $h_+ = 7.84$ mm along $+n$ (radius 0.51 mm) and the
magnet body $h_- = -4.84$ mm (radius up to 3.67 mm). Either first protrudes past the rim
silhouette when $\tan\theta > R/h$, i.e. at

$$\theta_+ = \arctan\frac{10.204}{7.84} = 52.5^\circ, \qquad
  \theta_- = \arctan\frac{10.204}{4.84} \approx 64.6^\circ \ \text{(reduced to} \approx 53^\circ \text{by its 3.67 mm radius).}$$

Measured, the mast's contribution is unmeasurable below 55° and then explodes: peak outward
deviation +1.9 px at 55°, +16.6 px at 65°, +23.1 px at 70°, and it **vanishes entirely if
the mast is deleted from the mesh**. The magnet body contributes nothing at any tilt.

> **Correction to earlier notes.** `pose/theory.md` and `calib/shape.py` originally
> attributed *all* of this to "the mast and magnet". That is wrong for the regime that
> matters: below ~55° the mast contributes nothing and the error is entirely the wall.

### 12.4 Contamination is one-sided

The rim is part of the object, and the projection of a circle is convex. The segmenter
returns the **convex hull** of the silhouette (`pose/segment.py`, chosen because lighting
routinely breaks the thin ring into arcs). Therefore

$$\text{hull} \ \supseteq\ \text{projection of the rim} \quad\Longrightarrow\quad
\text{every hull point lies on or outside the true rim ellipse.}$$

The fitted ellipse can only ever be **too big**, never too small. Measured against the true
rim, the signed radial deviation is positive at essentially every position around the
perimeter and at every tilt.

This has a direct consequence for estimator design: a **symmetric** robust loss cannot
exploit it, because it has no notion of which direction is suspicious. Measured, `soft_l1`
against plain least squares changed the normal error from 0.524° to 0.509°: nothing. A
one-sided loss, penalising points inside the current estimate and tolerating points outside,
is the form that matches the geometry.

### 12.5 Channel reliability: the major axis survives, the minor does not

Parameterise position around the rim by the ellipse parameter $t$, measured from the
major-axis tip. The contamination of §12.3 acts along the rotor axis, which projects onto
the image's **short** direction, so it concentrates at $t \approx \pm 90^\circ$ and leaves
$t \approx 0, 180^\circ$ nearly clean. Measured on held-out renders:

| tilt | major length | major direction $\psi$ | minor axis (after correction) |
|---|---|---|---|
| 20–40° | +0.2 to +0.5 % ± 0.7 % | 0.33–0.68° | +1.0 to +1.2 % ± 0.5 % |
| 40–60° | +0.8 to +0.9 % ± 0.5–1.1 % | **0.20°** | −0.3 to −3.1 % ± 1.1 % |
| 60–71° | −0.6 % ± 1.0 % | **0.48°** | **+12.7 % ± 26.5 %** |

The major axis holds to ~0.5 % everywhere; the minor axis scatter reaches 26 %. **That
25:1 split is why position survives tilts that destroy orientation**: depth comes from the
major axis (§12.2), tilt from the ratio.

It also constrains what can be done about it. The sensitivity of a boundary point to the
semi-minor axis is $\partial(b\sin t)/\partial b = \sin t$, so the points at
$t \approx \pm 90^\circ$ carry **all** the information about $b$. De-weighting them
symmetrically to reject the contamination removes precisely the quantity being recovered:
the fit gets less biased and much noisier. Only the one-sided form of §12.4 avoids this.

**A resolution floor follows.** $\rho = \cos\theta + k\sin\theta$ exceeds 1 on the whole
interval $(0,\ 2\arctan k)$, it peaks at $\arctan k$ and returns to 1 at twice that, so
every tilt in that band produces a silhouette at least as wide as it is long, and none of
them can be distinguished by an axis ratio at *any* resolution. With $k = 0.0433$ that is

$$\theta_{\min} = 2\arctan k = 4.96^\circ .$$

This is a second, independent reason not to read tilt from the ratio near face-on, the
first being the singularity of §12.2.

**The centre moves too, and that is a position error, but it resists correction.** The obvious symptom of
out-of-plane structure is a fattened short axis, and everything above measures
that. But the hull grows on **one** side, so the fitted ellipse's *centre* is
displaced as well, and lateral position is read straight off the centre.
Measured against the analytic rim centre, decomposed along the ellipse's own
axes:

| tilt | along the major axis | along the minor axis | as a distance |
|---|---|---|---|
| 10–55° | −0.5 px (flat) | +1.04 → +1.68 px | 0.185 → 0.274 mm |
| 65° | −0.84 | **−3.10** | 0.501 mm |
| 70° | −0.52 | **−5.53** | **0.867 mm** |

Almost all of it is along the minor axis, and the sign **flips near 57°**: below
that the wall of §12.3a pushes the centre one way, above it the mast of §12.3b
pulls it the other. So no monotone function of tilt can correct both, for the
same structural reason a quadratic could not correct the axis ratio (§12.7).

At 70° this displacement alone exceeds the 0.5 mm position target. The one-sided
re-weighting of §12.4 only partly removes it: 0.501 → 0.341 mm at 65°, 0.867 →
0.705 at 70°, and nothing at all below 55°: because its weights are judged
against an ellipse that has already absorbed the shift, and the mast's
contribution is a coherent arc rather than sparse outliers, so an IRLS started
from a biased fit converges to a nearby biased one. **Not corrected anywhere in
the shipped estimator.** Whether stereo fusion partially cancels it is untested:
the minor direction points differently in the two views, so some cancellation is
expected, and the measured stereo lateral error (0.14–0.18 mm) is indeed below
the single-view centre bias: but that is consistent with cancellation rather
than evidence of it.

### 12.6 What a second camera adds

Three distinct gains, often conflated:

**(a) The ambiguity becomes a measurement.** Each view yields two candidate poses (§12.1).
Mapped into a shared frame, one pair agrees and three do not. Closed form, no temporal
prior. The agreement test must be Mahalanobis rather than Euclidean, weighted by
$(\Sigma_a + \Sigma_b)^{-1}$: each view's error is anisotropic 11:1 (§12.2), so two views
that agree perfectly still differ by millimetres along their respective depth axes, and
counting that as evidence against the pair leaves the true pair ahead by only 1.6× the
noise. Weighted properly the margin is ~50×.

**(b) Bias attenuation, not noise averaging.** The single-view residual autocorrelates at
0.966 after one frame: it is a smooth function of pose, so temporal averaging cannot remove
it, and a Kalman filter measured a best case of 1.01×. Fusion is different in kind: it
weights by *direction*. Along camera A's depth axis, camera B measures laterally and carries
$\approx 120\times$ the weight, so a 3 mm systematic depth error in A enters the fused
estimate as 0.025 mm.

**(c) Tilt without the minor axis.** The projected normal is perpendicular to the major
axis, so view $i$ constrains $n$ to a plane through that camera's centre: one constraint on
$n$'s two degrees of freedom. **Two views give two planes, and their intersection fixes $n$
using only the major axes and the centres.** Eight observables for five unknowns, every one
from the channel that holds to 0.2–1 % (§12.5), and it is *best* conditioned exactly where
the ratio method is worst. A single view cannot do this: it has one constraint for two
unknowns: so this capability is intrinsically stereo.

**Status of (c).** Implemented as `stereo.solve_from_major` and A/B-tested on
identical frames: the normal improves 1.65× overall and **2.25× (8.15° → 3.62°)**
where the rotor is 55–70° from the optical axis. It is nonetheless **off by
default**, because the blend weight is wrong in the 45–55° band and that band is
the default rig's operating point. The failure is in the precision model, not the
geometry: $\sigma_\psi = 0.25/\sin\theta$ was fitted to the precision of $\psi$ in
a *single view*, but the normal comes from a cross product of two of them and is
worse than either, so the channel claims 0.35° against the ratio channel's 0.45°
and outvotes it. Fitting the channel's $\sigma$ from measured normal error, rather
than deriving it from $\psi$, is the outstanding work.

**What no number of cameras adds: roll.** The rim is $\infty$-fold symmetric about $n$ and
the blades 4-fold; geometry alone carries no roll information from any viewpoint, and B4
aliases what little the blades might have offered. The observable state is
$\mathbb{R}^3 \times S^2$, five degrees of freedom, and a six-DOF solver leaves
$J^\top J$ rank-deficient along the roll direction. This is the formal statement of the
"validity monitor" role [§11.6](../control/theory.md) (ch.4) assigns the pose estimate.

### 12.7 What does not work

Recorded because the value of a model is partly the list of things it rules out. Each was
implemented and measured.

| tried | result |
|---|---|
| Sub-pixel edge refinement (walk hull vertices to the interpolated threshold crossing) | **No change**: depth scatter 0.661 % vs 0.662 %. The hull already averages 30–60 points. |
| Better segmentation generally | **Nothing to win**: `seg_iou` is flat at 0.89 from 0° to 71° while `fit_rms_px` goes 0.15 → 1.90 px. The segmenter traces the silhouette correctly; the silhouette is not an ellipse. |
| Symmetric robust loss (`soft_l1`) | **No effect** (0.524° vs 0.509°): see §12.4 for why. |
| Sampson residuals on every hull point | **Worse** than eight axis-endpoint residuals (1.55° vs 0.53°): the correction of §12.3 is defined on the *fitted ellipse*, so residualising raw hull points compares the model against a quantity it was never fitted for. |
| Quadratic-in-angle tilt correction | **Superseded.** It cannot represent §12.3's form; its residual crossed zero twice and 84 % of the leftover variance was still deterministic in tilt. Replaced by the one-parameter wall model: held-out median tilt error 2.43° → 1.97°, bias +0.807° → −0.038°. |
| **Correcting the ellipse's displaced centre** (§12.5) | **Fails, three ways.** Applied to the *measured* ellipse it removes 22–68× of the 2-D displacement and still makes 3-D position worse (0.397 → 0.533 mm): because `conic.backproject` consumes all five ellipse parameters jointly, and a centre moved independently of axes that `TiltCalibration` has already rewritten yields a conic corresponding to no real circle projection. Moved into the *forward* model, where all five move together, it is neutral with the centre held fixed and still loses with it free (0.348 → 0.462 mm). Characterised but not exploited. |
| Fitting that wall model over the full 5–71° range | **Worse than doing nothing** (3.54°, bias +3.54°): the mast regime drags $k$ from 0.043 to 0.098. The model must be fitted where it applies, 20–50°. |
| **Symmetric** hull re-weighting by position along the major axis | **Worse below 55°**: median tilt error 0.43° → 0.80°, as §12.5's $\partial/\partial b \propto \sin t$ argument requires. It does help at 55–71° (5.13° → 3.13°), where the contamination outweighs the information it destroys. |
| **One-sided** re-weighting (tolerate points outside, per §12.4) | **The form that works**: 0.43° → 0.46° below 55° (free) and **5.13° → 2.83°** above it. Costs major-axis precision at high tilt (±0.81 % → ±2.59 %), because rejecting outward points can also shave genuine rim, so it is worth enabling only where its benefit appears. |

### 12.8 Correspondence with the implementation

| Model element | Code |
|---|---|
| Cone $Q$, back-projection, the two branches | `pose/conic.py`: `cone_from_circle`, `backproject` |
| Silhouette extraction, convex hull (§12.4) | `pose/segment.py`: `silhouette_hull` |
| Wall model $\rho = \cos\theta + k\sin\theta$ (§12.3a) | `pose/calibration.py`: `TiltCalibration`, `model="cylinder"` |
| Resolution floor $2\arctan k$ (§12.5) | `pose/calibration.py`: `resolution_floor_deg` |
| Branch matching, Mahalanobis (§12.6a) | `pose/stereo.py`: `match`, `_agreement` |
| Information-form fusion (§12.6b) | `pose/stereo.py`: `fuse`; `pose/rig.py`: `position_covariance` |
| Measured tables in §12.2, §12.5 | `pose/validation/sweep_stereo.py`, `results/pose_validation/` |
| Deviation-vs-$t$ diagnostic (§12.3, §12.5) | `pose/validation/scene3d.py`: `deviation_panel` |


---

### 12.9 The noise floor is not the limit: separating precision from bias

A natural assumption is that pose accuracy is set by how precisely the boundary
is located, and therefore that more pixels buy proportionally more accuracy.
Measurement says otherwise, and the gap between the two is the single most useful
diagnostic in this package.

**What noise alone predicts.** Take a boundary located to $e = 0.3$ px. The two
observables scale differently in the rim's pixel size $M$:

$$\sigma_{\text{lat}} = \frac{e\,z}{f}\;=\;\frac{e\,z}{M}\cdot\frac{M}{f}
\;\propto\;\frac{e}{M},\qquad
\sigma_{\text{depth}} = \frac{e\,z}{M}\cdot\frac{z}{2R}\;\propto\;\frac{e}{M^{2}}$$

Depth carries **one extra power of $M$** because it is read from the ellipse's
*size*, and size is a difference of two boundary positions: $z = 2fR/M \Rightarrow
dz/z = -dM/M$. Lateral position is read from the ellipse's *centre*, which is an
average rather than a difference, and averages do not lose a power.

(The $z/2R$ factor here assumes the tilt is known. §13.4 shows that estimating it
from the same ellipse multiplies the depth term by a further $\sqrt3$, so the
noise-only depth figures in the table below are optimistic by that factor. The
conclusion the table is used for, that noise alone is not the limit, is
strengthened, not weakened, by the correction.)

Single-view depth therefore misses the target at **every** sensor mode: 0.522 mm
at 1280×800, rising to 4.176 mm at 160×120. Stereo is what rescues it: with the
optical axes 60° apart, camera B measures laterally exactly the direction along
which camera A is blind, and the fused worst axis becomes

| mode | $M$ px | $\sigma_{\text{depth}}$ (one view) | fused worst axis |
|---|---|---|---|
| 1280×800 | 143.7 | 0.522 mm | **0.060 mm** |
| 1024×768 | 114.9 | 0.653 mm | **0.075 mm** |
| 640×480 | 71.8 | 1.044 mm | **0.119 mm** |
| 320×240 | 35.9 | 2.088 mm | **0.239 mm** |
| 160×120 | 18.0 | 4.176 mm | **0.477 mm** |

Every mode, including the 640 fps one, sits inside ±0.5 mm **on noise alone**.

**What is actually measured.** At 1280×800 the estimator returns 0.277 mm mean and
1.496 mm worst: $4.6\times$ and $25\times$ the noise prediction. The discrepancy
is not a failure of the derivation; it is the derivation doing its job, by
isolating what the remaining error cannot be.

**The test that distinguishes them.** Precision and bias have different
signatures under resolution, and that is what makes them separable without ground
truth about their cause:

$$\text{precision-limited} \Rightarrow \varepsilon \propto M^{-1}\ \text{or}\ M^{-2},
\qquad \text{bias-limited} \Rightarrow \varepsilon \to \text{const}$$

Measured position error across 1280→640 changes by $1.66\times$ while $M$ changes
by $2\times$. Pure lateral noise predicts $2\times$, depth noise $4\times$, a pure
bias $1\times$. The observed 1.66 is close to neither extreme, which says the
error is a **sum** of a shrinking noise term and a floor that does not shrink:
consistent with the independently measured centre displacement of 0.185–0.274 mm,
which is a property of the projected shape and carries no resolution dependence
at all.

**Three consequences**, each of which redirects effort:

1. **Resolution is nearly free to give up.** Going from 143.7 px to 71.8 px across
   the rim costs 0.06 mm of noise against a bias floor several times larger. The
   420 fps mode is not disqualified by geometry: it is disqualified, if at all,
   by segmentation robustness, which is a different and more fixable problem.
2. **A gate built only from noise features cannot reach 100%.** The features
   $e/M^{2}$ and $e/(M\sin\theta)$ describe precision. A frame whose error is
   dominated by silhouette bias has ordinary-looking features and an
   out-of-spec error, so no monotone function of those features separates it.
   This is the structural reason the fitted quantile inflation kept exploding
   ($k = 2.4\times10^{9}$, then 280, then 674) rather than converging: the
   residual being fitted was not, in fact, noise.
3. **The productive target is the forward model.** Every millimetre removed from
   the wall/mast bias is a millimetre removed at *every* resolution
   simultaneously, whereas a doubling of resolution buys hundredths.

The general lesson is worth stating separately from this application: **fitting an
uncertainty model to residuals that are mostly systematic will always fail, and
will fail by producing an absurd bound rather than an obviously wrong one.** The
fit has no way to say "this is not noise"; it can only inflate until the worst
systematic case is covered, at which point the bound rejects everything. The
diagnostic that catches it is comparing the fitted scale against the *derived*
noise floor: if the ratio is large and roughly constant across conditions, the
residual is bias and no amount of refitting will help.

### 12.10 Certifying a measurement: the gate as an operating point

Section 12.9 established that the remaining error is mostly systematic, so a
frame's error cannot be driven below specification by better estimation alone.
The alternative is to **decline the frames that cannot be certified**, which
changes the question from "how accurate is this estimator" to "can this
particular frame be answered to specification". That is a decision problem, and
treating it as one avoids a trap that is easy to fall into.

**The construction.** Fit a scale $s(x)$ by least squares on $\log$ error against
$\log$ features, then accept when $k\,s(x) \le$ target, where $k$ is a high
quantile of the observed ratio $\varepsilon/s(x)$.

**The trap.** It is natural to read $k$ as a *bound*, "the error is below
$k\,s(x)$ with probability $q$", and therefore to raise $q$ toward 1 for safety.
This is backwards, and the sign error is instructive. $k$ multiplies every
frame's prediction by the same constant, so it induces no reordering:

$$k\,s(x_1) \le k\,s(x_2) \iff s(x_1) \le s(x_2) \quad \forall k > 0$$

The accepted set is always the sublevel set $\{x : s(x) \le \text{target}/k\}$,
and these sets are **nested in $k$**. Raising $q$ raises $k$, shrinks the sublevel
set, and accepts *fewer* frames: until at $q = 0.999$ the set is empty. So $k$
is not a bound at all; it is a monotone reparameterisation of the decision
threshold, and choosing it is choosing an operating point on an ROC curve.

Once that is seen, the calibration follows. An operating point should be chosen
against the requirement it must satisfy, on data held out for that purpose: not
computed from a training statistic that has no knowledge of the requirement.
Hence three splits rather than two:

| split | what it determines |
|---|---|
| fit | the ranking $s(\cdot)$ |
| select | the threshold $k$, as the most permissive value holding 100% coverage |
| report | the quoted acceptance and coverage |

Two splits are insufficient for the specific reason that a threshold chosen on
the same data it is reported against is guaranteed to look good; the measured gap
here was 37.9% acceptance claimed against 30.6% cross-validated.

**Searching in the right direction.** Since acceptance rises as $q$ falls, a scan
that stops at the first qualifying $q$ returns the *most conservative* operating
point: which satisfies any coverage requirement trivially by accepting almost
nothing. The scan must retain the *last* qualifying value. This produced a 1.8%
acceptance figure before it was corrected, at the same 100% coverage, and the two
are indistinguishable on the coverage metric alone. **Coverage without acceptance
beside it is not a measurement**, which is why both appear in every table in the
journal.

**Precision and blunders are separate populations.** The features
$e/M^{2}$ and $e/(M\sin\theta)$ describe precision. A blunder: the outline closed
around the wrong object: has ordinary-looking features and an error two orders of
magnitude out, so no monotone function of them predicts it. Including blunders in
the fit lets them set the quantile: measured, $k$ reached $2.4\times10^{9}$, then
280, then 674. They belong in the *evaluation* but not in the *fit*, and that
asymmetry is deliberate: a blunder the hard gates miss must surface as a coverage
failure rather than be excused by a model that was allowed to expect it.

The general form of this result is worth separating from the application:
**a quantity that enters a decision rule as a global scale factor is a threshold,
not a confidence level, and must be calibrated against the decision's requirement
rather than estimated from the residuals.**

### 12.11 An ill-conditioned direction in a robust fit

Section 12.9 attributed the error above the noise floor to systematic silhouette
bias. Part of it was not bias at all: it was a defect in the fit, and the shape
of that defect is worth recording because the failure mode is general.

**The construction.** The rod and magnet fatten the silhouette in its short
direction, so boundary points are re-weighted by how far along the *major* axis
they lie: near zero at the major axis's centre where the protrusions project,
one at its ends where the boundary is honest rim. Two IRLS iterations, and the
resulting ellipse was used whole.

**The failure.** A weight of exactly zero does not distrust a point, it removes
it. Removing a contiguous arc leaves the points clustered at the two ends of the
major axis, and the information those clusters carry about the five conic
parameters is not isotropic. Write the ellipse as centre, two axis lengths, and
orientation $\phi$. Two opposing clusters at $\pm a\hat u$ pin the line through
them and the length along it, but the derivative of a point's position with
respect to $\phi$ is

$$\frac{\partial}{\partial\phi}\big(a\cos t\,\hat u + b\sin t\,\hat v\big)
= a\cos t\,\hat v - b\sin t\,\hat u$$

which at the retained points ($\cos t \approx \pm 1$, $\sin t \approx 0$) is
$\pm a\hat v$: perpendicular to the cluster's own extent, and therefore almost
unconstrained once those points are the only ones with weight. The Fisher
information in $\phi$ is carried by the points near $\sin t = \pm 1$, which are
exactly the ones the weighting deletes.

Measured on a noise-free ellipse of ratio 0.834: the major axis came out
**33.5° out**, with a Sampson rms of 3.7374 px against 0.0000 px for the plain
fit. That inequality is the diagnostic. A robust fit trades residual on all
points for a better estimate; a fit that is worse *on the points it was fitted
to* has lost a direction rather than gained robustness.

**Why flooring the weights does not fix it.** Flooring stops the deletion, and on
the case above it restores the exact answer. But re-measuring across a pose set
showed it moved the failure rather than removing it: at floor 0.05 the 33.5°
case became exact and a different pose went 14° out, with the worst downstream
error unchanged. Conditioning is a property of the weighted design matrix, and a
small floor leaves the smallest singular value small.

**The fix, and the general rule.** The protrusions are symmetric about the major
axis: they change the ellipse's lengths, not its orientation. So the weights have
authority over lengths and none over $\phi$, and $\phi$ should be held at the
plain fit's value, which uses every point. Retaining the suppression cost
nothing measurable (+0.01989 ratio bias against +0.01965 with $\phi$ free, versus
+0.05591 with no re-weighting) and the downstream failure vanished, 1.60e-01 mm
to 3.85e-08 mm.

Stated generally: **when a re-weighting is designed to suppress a contamination
that affects some parameters and not others, it must not be given authority over
the parameters it does not inform.** The contamination here is symmetric in
$\phi$, so the weights carry no information about $\phi$: and a parameter
estimated from data carrying no information about it does not merely stay at its
prior, it drifts to wherever the residual is flattest.

**How it propagated.** `stereo.refine` compares the four axis endpoints of the
measured and predicted ellipses. An angle error moves those endpoints by order of
the major axis length, so a 33° rotation makes the cost large **at the true
pose**: every seed, including one initialised exactly at the truth, converged
away from it. The refinement was not failing to converge; it was converging
correctly onto a corrupted objective. The lesson for the residual design is that
a cost built from *oriented* features inherits every orientation error in its
inputs, whereas one built from the conic coefficients or from point-to-conic
distance would not have been sensitive in this way.

### 12.12 Where this approach stops: irreducible scatter in the silhouette

Sections 12.9–12.11 progressively narrowed the error budget: the noise floor is
not the limit, the systematic silhouette bias is, and part of that "bias" turned
out to be a defect in the fit. This section establishes what remains after all of
it, and it is a limit of the *method* rather than of the implementation.

**The measurement.** Take the shipped cylinder calibration, apply it to a
held-out split, and bin the residual by true tilt:

| true tilt | *n* | mean residual | **std** |
|---|---|---|---|
| 10–20° | 57 | +3.402° | 9.769° |
| 20–30° | 90 | +0.434° | 4.218° |
| 30–40° | 97 | −0.404° | 3.407° |
| 40–50° | 115 | −0.137° | **2.597°** |
| 50–60° | 143 | −0.224° | 2.459° |
| 60–71° | 174 | −1.803° | 2.195° |

The calibration does its job: the mean residual is within a few tenths of a
degree across the working range. But a calibration is a function of one variable,
so it can only remove the *mean* at each tilt. The standard deviation is what
survives, and at 40–50°, precisely where the estimator's remaining failures sit
, it is **2.6°**, more than twice the specification on its own.

Stereo fusion is what makes the estimator work at all against this: two views at
60° axis separation reduce a 2.6° per-view scatter to roughly 0.3° in the fused
normal. The failures are the tail of that distribution, not its centre.

**Is the scatter reducible?** Only if something predicts it. Regressing the
absolute residual in the 40–50° band against every quantity the dataset records
, pose (`az_deg`, `cx/cy/cz`, `nx/ny/nz`), appearance (`major`, `minor`,
`area_px`, `iou`, `fit_rms_px`), and conditions (`alpha`, `bg`, `ambient`,
`exposure_s`, `sigma`, `spin_hz`), gives Spearman correlations whose strongest
members are

$$\rho(n_y) = -0.226,\quad \rho(n_x) = +0.200,\quad \rho(\alpha) = -0.191$$

at $p = 0.015$, $0.033$, $0.041$. With 22 simultaneous tests the Bonferroni
threshold is $\alpha = 0.01/22 \approx 5\times10^{-4}$, and **none of them comes
close**. The scatter is uncorrelated with everything measured.

**What that rules out, and what it leaves.** Three consequences, each closing off
a line of work:

1. **No calibration removes it.** A calibration corrects a mean; this is
   variance about the mean at fixed tilt.
2. **No gate feature declines it.** A predicted-error model can only reject what
   its observables see, and no observable correlates. This is why Iteration 8's
   split of the conditioning term recovered the derived exponent
   ($+1.332$ against a derived $1$) and yet removed none of the failures: the
   term was right and insufficient at the same time.
3. **A richer silhouette model will not obviously help either**, because the
   residual does not vary systematically with pose. A better *forward model*
   corrects a function of the pose; there is no such function here to correct.

What is left is the silhouette's *shape*, not the precision with which its edge
is found. That distinction was tested rather than assumed, and the first answer
here was wrong.

The obvious candidate was boundary quantisation: a fixed threshold slicing a
smoothly shaded, motion-blurred edge. Sub-pixel localisation removes exactly
that, and on a synthetic soft-edged disc of known radius it is 18x better
(scatter 0.0550 -> 0.0030 px). Carried into the pipeline it reduced the tilt
residual scatter by **2%** (2.597 -> 2.537 deg at 40-50 deg). An 18x improvement
in the mechanism bought 2% of the outcome, so quantisation was not the source.

The source is that **the silhouette is not the rim**, and which non-circular
shape is presented varies frame to frame: the rod and magnet mount project
differently as the body turns, rim arcs vanish under grazing illumination, and
the rotor smears the boundary across the exposure. That also explains why the
residual correlates with no single recorded variable -- it depends on their
*interaction*, and no one variable indexes a joint condition.

Reducing it therefore requires making the rim distinguishable from the body: a
fiducial on the rim, an illumination geometry that lights only the rim, or a
model that separates rim from mast rather than suppressing the mast. Locating the
edge of the wrong shape more precisely does not help.

**The honest form of the result.** For a rotor of this size at this range, a
threshold-and-hull silhouette pipeline delivers roughly 0.2–0.3° mean and ~1.4°
worst-case orientation after stereo fusion, and the worst case is set by
irreducible boundary scatter rather than by anything the estimator chooses. A
specification of ±1° on *every* frame is therefore not reachable by tuning this
pipeline: it requires changing how the boundary is measured.

## 13. The information floor: what no estimator can beat

§12 ends by asserting that the remaining error is a property of the *method*, and
supports that with a scaling argument: the error falls too slowly with resolution
to be noise. That argument is sound but indirect: it infers a floor from an
exponent. This section derives the floor.

The tool is the Cramér–Rao bound. For an unbiased estimator of a parameter vector
$p$ from data with likelihood $L$, the covariance obeys

$$\operatorname{Cov}(\hat p) \;\succeq\; J^{-1}, \qquad
J = \mathbb{E}\!\left[\left(\frac{\partial \log L}{\partial p}\right)
\left(\frac{\partial \log L}{\partial p}\right)^{\!\top}\right]$$

which converts "how much does the data change when the parameter changes" into a
hard limit on precision. Its value here is not that it flatters the estimator:
it is that when the measured error sits far *above* the bound, the excess
provably cannot be noise, and every lever that acts on noise is ruled out at
once.

The chain has four stages, and each is separately checkable:

$$\text{photons} \;\to\; \text{edge position} \;\to\;
\text{ellipse parameters} \;\to\; \text{pose}$$

Implementation: `controller/pose/bounds.py`. Every formula below is checked
against Monte Carlo in `controller/pose/test_bounds.py` (60+ assertions), and the
real system is placed against the result by
`controller/pose/validation/limits.py`.

**Assumptions (C1–C3).** C1: additive Gaussian sensor noise of standard
deviation $\sigma_n$ per pixel, independent between pixels. C2: the boundary is a
step of contrast $C$ blurred by a Gaussian PSF of width $s$ px. C3: boundary
points are measured along the local normal only. C3 is not a simplification:
§13.2 shows the tangential component carries no information at all.

### 13.1 Locating one edge: the sub-pixel limit

Everything downstream is a function of how precisely a boundary can be placed, so
that is the first floor to establish. Take a 1-D cut across the boundary with
noiseless intensity $B + C\,\Phi\!\big((x-x_0)/s\big)$, $\Phi$ the Gaussian CDF,
and read the Cramér-Rao bound on $x_0$.

**Point samples cost $\sqrt{s}$ in blur**, $\sigma_{x_0} \ge 1.883\,(\sigma_n/C)\sqrt{s}$,
but that model breaks at its own limit: a perfectly sharp edge between two point
samples moves no sample, so the bound diverges as $s\to0$ and there is an interior
optimum near $s = 0.5$ px.

**Box pixels change the limit rather than refining it.** Real pixels integrate over
their footprint, giving

$$\frac{\sigma_n}{C} \;\le\; \sigma_{x_0} \;\le\; \sqrt{2}\,\frac{\sigma_n}{C},$$

finite for every blur width and every sub-pixel phase, and monotone: sharper is
always better. The sensor low-pass filters the scene before sampling it, and *that
filter is what makes the sub-pixel phase observable*. The two models agree to 0.3 %
at $s = 4$ px and disagree by 49× at $s = 0.15$ px.

**The quantisation floor** for a threshold-and-contour outline is
$1/\sqrt{12} = 0.2887$ px. Sub-pixel refinement pays only above $C/\sigma_n = 6.7$,
which a white rotor on a black ground clears easily.

**The shipped refinement is near the bound.** `segment.subpixel_boundary` on a
synthetic soft-edged disc at $C = 200$, $\sigma_n = 5$, $s = 1.2$ px:

| | bias | scatter |
|---|---|---|
| thresholded contour | −0.461 px | 0.269 px |
| after sub-pixel refinement | −0.038 px | **0.076 px** |
| edge CRLB | | 0.052 px |

It removes 92 % of the bias and 72 % of the scatter, landing at **1.46× the
theoretical bound**. At most a factor of 1.5 is available in edge localisation,
ever, by any method.

Hold that against the boundary scatter measured on rendered frames, **1.17 px**
(§13.7), and §12.12's negative result stops being a surprise. Sub-pixel work
improved the mechanism by 18× and bought 2 % of the outcome, because the mechanism
was never the constraint. The boundary is being located to a tenth of a pixel, and
it is a tenth of a pixel away from *the wrong curve*.

### 13.2 From boundary points to the ellipse

Parametrise the ellipse as $p = (c_x, c_y, a, b, \theta)$, $a \ge b$.

**Only the normal component of a boundary point is observable.** Displacing a point
along the tangent slides it to another point of the same curve, the aperture problem
in its cleanest form, so each point contributes one scalar and not two.

**The circle case closes analytically**, and inverts the usual intuition:

$$\sigma_{c} = \sigma_r\sqrt{2/N}, \qquad \sigma_{r_{\!\text{px}}} = \sigma_r/\sqrt{N}.$$

The centre is measured $\sqrt2$ times worse than the radius, per axis. A centre looks
like an average of opposed points and ought to be better determined, but each point
constrains it only along its own normal, diluting the information by $\cos^2\varphi$,
while every point constrains the radius fully.

Two structural facts fall out of the same matrix:

- **A circle has no orientation.** At $a = b$ the $\theta$ row and column vanish and
  the Fisher matrix has rank 4, not 5. Measured
  $\lambda_{\min}/\lambda_{\max} = 8\times10^{-30}$: exact, not ill-conditioning.
- **A short arc is catastrophic.** Keeping 40 % of the perimeter inflates the bound on
  the semi-major axis by **17.9×**. Occlusion is not a graceful degradation.

**Adjacent points are correlated**, produced by the same PSF and the same silhouette
excursion, so the honest count is $N_{\text{eff}} = N d / \max(d, L)$ for correlation
length $L$ at spacing $d$. On real renders $L = 17.5$ px against about 11 px spacing,
turning 31 hull points into 15 independent ones, a factor of 1.4. That is the second
largest source of optimism in a naive contour CRLB. The largest is hulling a ~354 px
perimeter down to 31 points at all, worth 3.4 (§13.8).

**The direct algebraic fit is statistically efficient to within 4 %.** Monte Carlo over
4000 trials with the shipped `segment.fit_ellipse` puts every parameter's efficiency
between 0.96 and 1.01. This closes off a category of work: Fitzgibbon's direct fit is
famously biased toward small ellipses, and the reflex is to replace it with an iterative
geometric or maximum-likelihood fit. At this noise level there is nothing to recover.

### 13.3 From the ellipse to the pose

The remaining link is a change of variables,
$\Sigma_{\text{pose}} = G\,\Sigma_{\text{ellipse}}\,G^\top$, with
$G = \partial(\text{pose})/\partial(c_x, c_y, a, b, \theta)$ taken by central
differences of the shipped `conic.backproject_ellipse`. Differencing the shipped code
rather than hand-linearising it keeps one implementation instead of two, and bounds the
pipeline that actually runs. The branch is re-selected at every perturbed point by
proximity to the nominal pose, so the derivative cannot jump between the two ambiguity
solutions and report a spurious infinity.

$G$ is $6\times5$ and cannot have rank above 5. That is the formal statement of §12's
claim that the estimator is 5-DOF: there is no sixth direction for it to have
information about.

**The estimator is at the bound.** End-to-end Monte Carlo through `fit_ellipse` and
`backproject_ellipse` gives efficiencies within sampling error of unity on all three
positions and all three normal components. Every stage of the solve extracts
essentially all the information in the boundary points it is given. Whatever explains
the two-orders-of-magnitude gap between these bounds and the measured residual (§13.7),
it is not the solver.

### 13.4 The depth law, corrected: $\sqrt3$ is the price of not knowing the tilt

§12.2 and §12.9 give the depth penalty as $z/2R$, range over diameter. That is
right for the problem those sections pose and wrong for the one that ships, by
73%.

**Tilt known.** Substituting the circle results into the pinhole relations
$r_{\text{px}} = fR/z$ and $X = z c_x / f$:

$$\sigma_z = \frac{z\,\sigma_r}{r_{\text{px}}\sqrt N},
\qquad
\sigma_X = \frac{R}{r_{\text{px}}}\,\sigma_r\sqrt{\frac2N}
\qquad\Longrightarrow\qquad
\frac{\sigma_z}{|\sigma_{\text{lat}}|} = \frac{z}{2R}.$$

**Tilt unknown**, which is the actual situation: the semi-axis is no longer
estimated against a one-parameter model, it must be disentangled from $b$ and
$\theta$. For a near-circular ellipse the $(a,b)$ block of the Fisher matrix is

$$J_{(a,b)} = \frac{N}{8\sigma_r^2}\begin{pmatrix} 3 & 1\\ 1 & 3\end{pmatrix}
\qquad\Longrightarrow\qquad
\sigma_a = \sqrt3\,\frac{\sigma_r}{\sqrt N},$$

three times the *variance* of the tilt-known case, while the centre block is
untouched. So

$$\boxed{\ \frac{\sigma_{\text{depth}}}{|\sigma_{\text{lat}}|}
= g(\theta)\,\frac{z}{2R}, \qquad g(0) = \sqrt3\ }$$

Both Fisher entries are verified to 2%, and $\sigma_a = \sqrt3\,\sigma_r/\sqrt N$
to 3%.

**Not knowing the tilt costs $\sqrt3$ in depth and nothing laterally.** Away from
face-on the factor rises slowly and is bounded:

| tilt | 0° | 10° | 30° | 60° | 80° |
|---|---|---|---|---|---|
| $g(\theta)$ | 1.732 | 1.740 | 1.811 | 2.038 | 2.180 |

so a serviceable form of the whole result is

$$\sigma_{\text{depth}} \;\approx\; 2\,\frac{z}{2R}\,\sigma_{\text{lat}}
\;=\; \frac{z}{R}\,\sigma_{\text{lat}}$$

: **range divided by the robot's radius**, to within 15% at any tilt. At
$z = 250$ mm and $R = 10.2$ mm the exact value runs 21.2× face-on, 22.2× at 30°
and 25.0× at 60°, against the $z/R = 24.5$ of the approximation and the
$z/2R = 12.3$ the earlier sections imply.

**What cancels is the point.** The noise level, the point count, the focal length
and the resolution all drop out. Verified over 16 combinations of $(z, \sigma_r)$
to 0.4%, and separately shown invariant to the point count between $N = 500$ and
$N = 8000$ (identical to 6 significant figures). This is a property of
perspective projection, and no camera, lens, exposure or estimator changes it.

**Finite range is a real correction, not a rounding error.** The law above is
exact in the weak-perspective limit; the correction scales roughly as
$(R/z)^2/\sin^2\theta$:

| $z$ | $R/z$ | tilt 10° | tilt 30° | tilt 60° |
|---|---|---|---|---|
| 100 mm | 0.102 | 0.649 | 0.957 | 0.995 |
| 150 mm | 0.068 | 0.789 | 0.980 | 0.998 |
| 250 mm | 0.041 | 0.907 | 0.993 | 0.999 |
| 400 mm | 0.026 | 0.961 | 0.997 | 1.000 |
| 700 mm | 0.015 | 0.987 | 0.999 | 1.000 |

Inside the operating envelope ($z$ = 150–400 mm, tilt ≥ 30°) it holds to 2%. Close
up and near face-on it does not, and the direction is favourable: perspective
gives back some of what the tilt degeneracy takes.

### 13.5 Tilt has two regimes, and the textbook one is the wrong one at hover

$\theta = \arccos(b/a)$ gives $\mathrm{d}\theta/\mathrm{d}(b/a) = -1/\sin\theta$,
so away from face-on $\sigma_\theta = \sigma_{\text{ratio}}/\sin\theta$. Verified
against Monte Carlo to 4% at 10°, 30° and 60°.

That expression diverges at $\theta = 0$, and the divergence is real but the
formula is not: at exactly face-on the linearisation fails because $\theta$ sits
at the boundary of its domain ($b \le a$ always, so a ratio *above* 1 is not
representable). Expanding instead, $\cos\theta \approx 1 - \theta^2/2$ gives
$\theta \approx \sqrt{2\delta}$ for a ratio deficit $\delta$, so

$$\mathbb{E}[\hat\theta]\Big|_{\theta=0} = \mathbb{E}\big|u\big|^{1/2}\sqrt{2\sigma_{\text{ratio}}}
= 0.8225\,\sqrt{2\sigma_{\text{ratio}}}.$$

**The error goes as the square root of the ratio noise, and the estimate is
biased away from zero.** The constant is $\mathbb{E}|u|^{1/2}$ for a standard
normal, and the law is verified to 0.1% against 200 000 Monte Carlo draws.

Feeding it the *measured* boundary statistics of §13.7, $\sigma_r = 1.035$ px,
$N_{\text{eff}} = 15$, $a = 56.3$ px, gives $\sigma_{\text{ratio}} = 0.0139$
and therefore

| configuration | $\sigma_{\text{ratio}}$ | apparent tilt of a face-on rim |
|---|---|---|
| as measured (31 hull points) | 0.0139 | **7.9°** |
| same boundary, all ~354 contour points | 0.0028 | 3.5° |
| quantisation-limited boundary, all points | 0.0008 | 1.9° |

This matters more than any other single number here, because **hover *is* the
face-on case**: the platform's attitude envelope is 1.1° RMS ([§11](../control/theory.md) (ch.4)), so the
apparent tilt at hover is dominated by noise rectification rather than by tilt.
It is also the independent confirmation of §13.7's tilt breakdown, where the
0–10° bin shows a 13.5° median angular error against 1.5–1.7° in the
well-conditioned bins: the same effect, measured rather than predicted.

Note the second row: the factor available from keeping more of the boundary is
*larger* here than anywhere else in this section, because the $\sqrt{\cdot}$
turns a $\sqrt{N}$ gain in the ratio into a $N^{1/4}$ gain in tilt but starts
from a much worse place. This is the strongest argument for the dense-contour
change in §13.8.

**Face-on also destroys lateral position, which is not in the textbook
statement.** At fixed range, resolution and noise, the CRLB on the recovered
centre is

| tilt | 1° | 2° | 5° | 10° | 20° | 40° |
|---|---|---|---|---|---|---|
| $\sigma_{\text{lat}}$ inflation | 2.5× | 4.6× | 1.4× | 1.08× | 1.00× | 1.00× |

(The 1° and 2° entries are not monotone, and should not be read as a trend: at
that tilt the Fisher matrix is within numerical reach of its rank-4 limit and the
inversion is unstable. What the row supports is the *magnitude*, several-fold,
and the *cutoff* at about 5°, both of which are stable.)

The mechanism is that an eccentricity error indistinguishable from noise is
interpreted as a tilt of the circle's *plane*, and tilting the plane slides the
recovered centre. So "face-on is bad for tilt" understates it: **face-on is bad
for everything**, and the degradation is confined to below about 5°.

### 13.6 Two structural results that no engineering removes

**Roll is exactly unobservable.** Rotating the rim about its own normal maps the
circle onto itself, so the projected conic is identical and the derivative of the
image with respect to roll is zero. Measured over 64 roll angles: the largest
image displacement is $1.6\times10^{-12}$ px, i.e. float noise. Together with
§13.3's rank-5 Jacobian this is the complete statement: the missing degree of
freedom is not weakly observed, it is absent.

**The two-fold ambiguity is a genuine bimodality, not a numerical failure.** A
quadric cone admits exactly two families of circular cross-section, so two
distinct poses generate the *identical* ellipse. Measured: the two branches'
normals differ by 49.1°, and their reprojected ellipses differ by
$4.2\times10^{-12}$ px. The single-view likelihood has two exactly equal maxima.
No amount of data from this view can prefer one, and any estimator that appears
to is using a prior. That is the formal justification for both the temporal prior
in `estimator.py` and the second camera in §12.6: and it is also why the CRLB,
a *local* bound, must be compared against oracle-branch errors, with branch
failures counted separately rather than folded into a residual.

### 13.7 What the real system achieves against these floors

391 rendered frames at 1280×800 (199 `core`, 192 `edge`), tilt 0–70°, range
170–340 mm, with the analytic ground-truth ellipse available for every one, at
the **shipped** configuration (axial weighting on: journal Iteration 14).
`controller/pose/validation/limits.py`.

This is a **harsher sample than §12's headline band**, deliberately: it spans the
full tilt range including the degenerate ends, and it includes the `edge`
condition tier. The absolute residuals below are therefore larger than the
0.87 mm quoted for tilt 10–45° under good light, and the two are not directly
comparable. What *is* comparable, and what this section is about, is the ratio
of each residual to its own floor, computed frame by frame on the same data.

**The boundary, decomposed.** Signed distance from every convex-hull point to
where the rim actually projects:

| quantity | median |
|---|---|
| photon-limited edge localisation (§13.1) | 0.048 px |
| pixel quantisation, $1/\sqrt{12}$ | 0.289 px |
| **measured scatter, robust ($1.4826\times$MAD)** | **1.087 px** |
| measured scatter, standard deviation | 1.435 px |
| contamination (std / robust) | 1.22× |
| radial bias, raw | +1.068 px |
| radial bias, after the radius calibration absorbs the mean | +0.857 px |
| correlation length along the contour | 18.3 px |
| hull points kept / independent | 31 / 14 |

**The measured scatter is 23× the photon floor and 3.8× the pixel-quantisation
floor.** It is also 14× worse than what the shipped sub-pixel refinement achieves
on a *clean* disc (0.076 px, §13.1). Three different arguments, all saying the
boundary error is not a localisation error.

**Pose, against the four levels.** Medians; pose scored against the oracle
ambiguity branch, for the reason in §13.6:

| level | status | ‖position‖ mm | depth mm | lateral mm | angle ° |
|---|---|---|---|---|---|
| photon limit | bound | 0.026 | 0.026 | 0.002 | 0.013 |
| pixel-quantised | bound | 0.153 | 0.153 | 0.011 | 0.080 |
| + convex hull (31 pts) | bound | 0.490 | 0.488 | 0.036 | 0.249 |
| noise-equivalent | *prediction* | 2.358 | 2.344 | 0.200 | 1.374 |
| **measured** | actual | **1.820** | 1.784 | 0.247 | **2.668** |

Two things to read off this table, and they point in opposite directions.

**The estimator beats the noise-equivalent prediction** (1.82 against 2.36 mm).
That is not a paradox and not an error: the prediction is not a bound. It
assumes the boundary errors are independent and Gaussian, and they are neither:
they are dominated by a common outward offset, and the fitted `RADIUS_MM` exists
precisely to absorb that common mode. Independent noise of the observed size
would be *worse* than the correlated bias actually present, because the
calibration cannot absorb what has no common mode.

**But it is 71× the photon bound and 3.7× the hull bound**, and the second of
those is the informative one. Split by tilt:

| tilt | *n* | measured mm | hull bound mm | ratio |
|---|---|---|---|---|
| 0–10° | 7 | **386.9** | 0.769 | **503×** |
| 10–25° | 56 | 3.047 | 0.395 | 7.7× |
| 25–45° | 110 | 1.528 | 0.388 | 3.9× |
| 45–71° | 218 | 1.834 | 0.567 | 3.2× |

Two things to read here. Away from face-on the ratio is a **constant multiple**
of the bound, 3.2–3.9×, which is the signature of a term that scales with the
same geometry as the bound but is not noise. And the 0–10° bin is not a
degradation, it is a **collapse**: a median of 387 mm on a 250 mm working range,
503× its own floor. That is §13.5's face-on degeneracy in its rawest form, on
seven frames. It is the single strongest argument in this document for keeping
the rotor out of the face-on geometry, or for the second camera.

Conditions move the aggregate in the expected direction: `core` gives 1.27 mm
and 2.36°, `edge` 3.15 mm and 3.95°.

**Branch selection, separately.** Prior-free, on independent frames, the shipped
selector picks the wrong ambiguity branch on **48%** of frames, with a median
margin of 103.6° between the candidates: so the shipped angular error reads
29.4° against an oracle 2.67°. Prior-free branch selection is a coin toss, which
is exactly what §13.6 says it must be. This is the worst case by construction: it is the
estimator with no temporal prior at all, and the live loop in `online_camera.ipynb` in a video stream has
one. The number is not a shipped failure rate; it is the size of the problem the
prior and the second camera exist to solve, and §13.6 is why no amount of
single-view cleverness substitutes.

### 13.8 What this rules out, and the one software lever left

The value of a floor is the work it forbids. Taking the levers in the order
people reach for them:

1. **A better sensor, brighter lighting, or a longer exposure.** The photon bound
   is 0.048 px against a measured 1.087 px. Even a *perfect* sensor changes the
   boundary scatter by less than 5%. **Ruled out.**
2. **Better sub-pixel interpolation.** The shipped refinement is already within
   1.46× of the edge CRLB (§13.1), so at most a factor of 1.5 exists there, and
   it applies to a term (0.076 px) already 14× below the one that matters
   (1.087 px). This is the
   derived version of §12.12's measured result: an 18× improvement in the
   mechanism bought 2% of the outcome. **Ruled out.**
3. **A better ellipse fit.** Statistical efficiency 0.96–1.01 on all five
   parameters, 0.94–1.06 end-to-end through the back-projection. Replacing the
   direct algebraic fit with an iterative maximum-likelihood one recovers at most
   4%. **Ruled out.**
4. **More resolution.** Bounded by the same $1/\sqrt N$ as everything else and
   acting on a term already 4× below the limiting one; §12.9 measured 1.66×
   improvement for a 2× resolution change against a predicted 2–4×, which is this
   same statement arrived at empirically. **Marginal.**
5. **Keeping more of the boundary.** This is the one that is *not* ruled out.
   The convex hull retains 31 vertices of a ~354 px perimeter, and $1/\sqrt N$
   says that costs **3.2×**: comparable to the entire remaining gap. It is a
   design choice rather than a limit: the hull is there to enforce the
   outward-only property the one-sided loss depends on, and journal Iteration 11
   found on real captures that 42–62% of dense contour points lie *inside* the
   fitted ellipse, so the hull was doing real work. The version that survives is
   dense contour points *restricted to the hull*, which recovered 55–461 points
   with the property intact. **This is where the next factor of two is.**
6. **The remaining ~3.7×.** After the hull is accounted for, what is left is
   that the silhouette is not the rim (§12.3, §12.12): the rod and magnet
   project into the outline, and which non-circular shape is presented varies
   frame to frame. Mostly this needs a change to what is imaged: a fiducial on
   the rim, illumination that lights only the rim, or a forward model that
   separates rim from mast.

   **But it is not entirely closed to software, and the axial weighting proves
   it.** Weighting each hull point by how far along the major axis it sits
   targets precisely the rod and magnet, because they sit *on* the axis and so
   land near the middle of the major axis. It is a partial forward model of the
   contamination rather than a better fit to the contaminated boundary, which is
   why it works where trimming by residual did not. The same third factor
   measures 3.7 in the shipped weighted configuration against 4.4 in an
   unweighted run: suggestive rather than controlled, since the two runs drew
   different pose samples, but pointing the same way as the controlled sweep A/B
   in journal Iteration 13.
7. **Depth specifically.** $g(\theta)\,z/2R \approx z/R$, which is 21–25× at
   250 mm depending on tilt, cancelling every sensor and algorithm parameter (§13.4). A second
   camera, or an optical axis that puts altitude *across* the image, are the only
   two answers. **Structural.**
8. **Roll, and the branch ambiguity.** Rank-deficient and bimodal respectively,
   both to machine precision (§13.6). **Structural.**

**The 71× factored, carefully.** The three bounds multiply out as
$6.0 \times 3.2 \times 3.7 = 71$, and it is tempting to read that as three
independent budgets. It is not:

| step | factor | recoverable? |
|---|---|---|
| photon → pixel-quantised | 6.0× | **No, and not for the obvious reason.** Sub-pixel refinement *does* beat quantisation, by 3.8× on a clean disc. It buys nothing here because the traced curve sits 1.087 px from the rim whether it is located to 0.289 px or 0.076 px. You cannot recover a factor on a term that is not the largest. |
| quantised → hulled | 3.2× | **Yes, and untouched so far.** Purely a consequence of fitting 31 points instead of ~354, and $1/\sqrt N$ is not negotiable but *N* is. Dense contour points restricted to the hull is the version that keeps the outward-only property (Iteration 11). |
| hulled → measured | 3.7× | **Partly.** Mostly the silhouette not being the rim, which needs a fiducial, an illumination geometry that lights only the rim, or a forward model that separates rim from mast. The axial weighting already recovered some of it in software, by modelling *where* the contamination is instead of fitting harder to it. |

So the honest summary is: **a factor of ~3.2 is sitting untouched in the hull, a
factor of ~3.7 is the shape of the silhouette and is only partly reachable from
software, and the remaining 6.0 was never available at all**: it is the price of
reading a 3-D pose off a 20.4 mm circle from 250 mm away with one camera, and it
is the smallest of the three.

None of this touches the two structural results (§13.6) or the face-on collapse
(§13.5, §13.7), which are not factors in a budget: they are geometry, and they
are why the second camera is on the plan.

## 15. Separating the robot from the room on a monochrome sensor

§12 assumes a silhouette. This section is where that silhouette comes from when the robot is
**black on a white backdrop with the rig in frame**, and the camera is the ELP OV9281: a
*monochrome* global-shutter sensor. Implemented in `controller/pose/segment.py`.

The reason it needs a section is that the obvious method is unavailable and the second
obvious method is impossible, and both failures are informative rather than incidental.

**Assumptions (C1–C4):**

- **C1, The clutter is fixed to the rig.** Coils, wires, the support box and the backdrop
  do not move relative to the camera. This is what makes §15.3 work at all, and it fails the
  moment the camera is knocked.
- **C2, The robot is the darkest thing in the working region.** Not in the frame: the room
  beyond the backdrop is darker still (§15.1).
- **C3, The backdrop is the brightest smooth surface in view.** Used only by the fallback
  of §15.4.
- **C4, The robot is one compact object.** Its pieces, rim arcs, magnet mount, rod tips,
  lie within one object radius of each other. §15.5.

### 15.1 Why brightness alone cannot work

Measured over both frames in `vision/drone_orientation/elp/`, in 8-bit counts:

| region | p5 | median | p95 | local sd (15×15) |
|---|---|---|---|---|
| white backdrop | 176–179 | **179–184** | 188–197 | **0.8** |
| drone rim | 4–26 | **42–78** | 176–182 | 26.9–27.4 |
| grey rod | 83 | **96** | 162 |: |
| coils | 8–28 | 85–104 | 170–224 | 15.7–34.4 |
| dark ambient beyond the backdrop | 6 | **7–8** | 10–11 | 1.1 |

Two facts follow, and they point in opposite directions. The coils' 85–104 sits on top of
the drone's 42–78, so no threshold separates them. And the ambient is *darker than the
drone*: so an inverted threshold does not merely admit the room, it prefers it. Because
`silhouette_hull` pools every surviving blob into a single convex hull, and the ambient
reaches the frame edge, the failure is not a stray contour but a hull spanning the image.
The fit still returns an ellipse. It is simply meaningless.

### 15.2 Why chroma cannot rescue it

The natural fix, and the one `segment.py` previously used, is that the robot is dark **and
achromatic** while the coils are dark **and coloured**: for any neutral surface $R = G = B$
by definition, whatever the illumination, so

$$c(x) \;=\; \max(R,G,B) - \min(R,G,B)$$

is near zero on the robot and the backdrop alike and large on bronze. That argument is
correct, and on a colour camera it works. On this camera it is vacuous: measured over both
captures, $c(x) \equiv 0$ at **every pixel** (max 0, mean 0.00). A mono sensor replicates one
channel, so the gate `c(x) ≤ CHROMA_MAX` passes the entire frame.

The lesson is about instruments, not about colour. The gate was not skipped for want of a
three-channel array: feeding it a replicated BGR frame passes it just as completely. A
discriminant that is identically zero on its input is not a weak test, it is *no test*, and
it had been failing open and silently.

### 15.3 Background subtraction: the whole answer, when it is available

By C1 every source of clutter is static. Let $B(x)$ be a per-pixel **median** over $N$
robot-free frames. Then

$$M_{\text{valid}}(x) \;=\; \mathbb{1}\!\left[\,|I(x) - B(x)| > \tau\,\right]$$

removes coils, wires, box, ambient and backdrop in one operation, without modelling any of
them. The median rather than the mean matters for a specific failure: a frame with the robot
half in shot. A mean spreads that across $B$ at $1/N$ contrast, where it is invisible and
permanent; a median is unmoved by it while such frames are in the minority.

Cost, measured on a 1280×800 frame: **0.056 ms**, against 2.43 ms for §15.4 at quarter
resolution and 48.4 ms at full. Since `segment()` alone costs 7.9 ms single-core against an
8.3 ms camera period at that mode, this is the difference between keeping up with the camera
and dropping every fourth frame.

Its weakness is exactly C1. A camera that has moved makes $B$ describe a scene that no longer
exists, and the subtraction reports the shifted edges as robot: confidently, with no
residual to betray it. The defence is not statistical but procedural: measure the differing
fraction and refuse when it exceeds a scene-change threshold (`background.py --check`).

### 15.4 The backdrop, when there is no background frame

Without $B$, the question changes from *what to reject* to **where to look**, and there the
table in §15.1 has a second, unused column. The backdrop's local standard deviation is 0.8
against 15.7–34.4 for the coils, a 20–40× margin, while the equally smooth ambient is 170
counts darker. Neither brightness nor texture separates the backdrop alone; their conjunction
does:

$$M_{\text{backdrop}} \;=\; \text{hull}\Big(\text{largest component of } \big[\mu_w(x) > \lambda\big] \wedge \big[\sigma_w(x) < s\big]\Big),$$

with $\mu_w, \sigma_w$ local mean and standard deviation over a $w\times w$ window, computed
as two box filters via $\sigma^2 = \mathbb{E}[I^2] - \mathbb{E}[I]^2$.

**The hull is not cosmetic, and this is the part that took measurement to find.** The robot
punches a dark hole in the backdrop, so the region must be closed over it. The obvious
closure, fill enclosed holes, *cannot work here*: the robot touches the rod, and the rod
runs off the bottom of the frame, so the robot-hole and the rod-notch form one region open to
the image border and therefore not enclosed. Both `MORPH_CLOSE` and a border flood fill
return nothing on the face-on capture. A convex hull has no such dependence on connectivity.
It also re-admits the rod, which is why the rod is removed by *brightness* instead: the rod's
p5 is 83 against the drone's median 42–78, and the threshold sits in that gap.

### 15.5 Which blobs are one object

Region gating leaves strays inside the region. Two properties distinguish the robot, and
neither works alone.

**Spread.** By C4 the robot's pieces lie within one radius of each other. Anchoring on a blob
of half-diagonal $r$ and centroid $c_0$, keep blobs with $\|c_i - c_0\| \le \kappa r$. On the
face-on capture the strays sit at 1.76–2.5 radii while a rim arc cannot exceed 1 by
definition.

**Shape, for choosing the anchor.** The anchor cannot be "the largest blob", because the rim
is a *ring* and therefore hollow: on the synthetic clutter scene the robot is 1783 px against
3225 px for a solid coil, and anchoring on area returns a confident fit to the coil. Score
each candidate grouping by its plain-fit Sampson residual relative to its own size,
$\rho = \text{rms}/a$: relative, because an absolute residual systematically prefers the
smallest group, which is the clutter.

But shape alone fails too, and symmetrically: **a solid disc is a perfect ellipse**, so round
clutter scores better than the robot ever can. Ranking on $\rho$ returns a 64 px coil in place
of the 125 px robot. The resolution is that the two criteria are not competing measures of
the same thing: $\rho$ decides *admissibility* and size decides *between* admissible groups:

$$\text{choose } \arg\max_{\,g\,:\,\rho(g)\,\le\,\rho_{\max}} a(g).$$

Every real candidate clears $\rho_{\max}$ comfortably, the robot fits at $\rho = 0.006$ on
both captures against a tolerance of 0.05, so the tolerance exists only to reject groupings
that have spanned two objects, and among what survives, the robot is the largest thing in
frame that is an ellipse at all.

### 15.6 Margins, not working points

A constant at the edge of its passing range is indistinguishable from a correct one until the
lighting moves. Swept against both real captures:

| constant | passes over | shipped | note |
|---|---|---|---|
| `DARK_THRESH` | 170–215 | 190 | below: the rod enters; above: the cut eats the drone's shading |
| `BACKDROP_LUM` | ≤100–180+ | 150 | the widest margin of the four |
| `BACKDROP_SD` | 3–20 | 12 | |
| `DARK_MAX_SPREAD` | 1.0–1.7 | 1.35 | **tightest**; re-measure if the coils move nearer |

Each shipped value is the midpoint of its range rather than a value that merely works.

### 15.7 What does not work

Recorded because the value of a method is partly the list of things it rules out. **Each was
implemented and measured** against both real captures.

| tried | result |
|---|---|
| chroma gate, $\max-\min$ over BGR | **Inoperative**: $c(x) \equiv 0$ on a mono sensor; passes the whole frame, and failed open silently |
| luminance threshold, no region gate | **Fails**: coils 85–104 overlap the drone's 42–78, and the ambient at 7–8 is darker than the drone and touches the border |
| `MORPH_CLOSE` to fill the robot-shaped hole | **Fails face-on**: cannot close a 325 px hole; no detection at all |
| enclosed-hole border flood fill | **Fails, structurally**: the drone touches the rod and the rod reaches the border, so the hole is not enclosed |
| vertical morphological opening to remove the rod | **No gain**: identical fits with and without; brightness already separates it |
| column-strip "middle third" prior | **Worse**: the drone darkens its own columns, collapsing the detected strip to 17% of frame and cutting the robot in half |
| anchoring the spread gate on the largest blob | **Fails**: the rim is hollow, so a solid coil outweighs it; returns the coil |
| ranking blob groupings by shape alone | **Fails**: a disc is a perfect ellipse; returns a 64 px coil over the 125 px robot |
| full-resolution backdrop mask | **Disqualified on cost**: 48.4 ms, i.e. 20 Hz, against an 8.3 ms camera period |

### 15.8 Correspondence with the implementation

| Model element | Code |
|---|---|
| Chroma vacuity on a mono sensor (§15.2) | recorded in `segment.py`, `score_channel` docstring; asserted by `test/test_appearance.py::test_dark_needs_no_colour` |
| Background subtraction $M_{\text{valid}}$ (§15.3) | `segment.background_mask`, `segment.load_background` |
| Median empty-rig frame, staleness check (§15.3) | `controller/elp/background.py`, `--check` |
| Bright-and-smooth backdrop, convex hull (§15.4) | `segment.backdrop_mask` |
| Region selection, and refusal when there is none (§15.4) | `segment.valid_region`, `segment.segment` returning `None` |
| Rod removed by brightness (§15.4) | `segment.DARK_THRESH` |
| Spread gate $\kappa$ (§15.5) | `segment.DARK_MAX_SPREAD`, `silhouette_hull(max_spread=)` |
| Admissibility $\rho_{\max}$ then size (§15.5) | `segment._best_group`, `segment.SHAPE_TOL` |
| Margins (§15.6) | `test/test_elp_captures.py`, swept per constant |
| Rejected area, shown to the operator | `segment.shade_rejected`, `segment.clutter_mask`, `online_camera.ipynb` [§3](../control/theory.md) (ch.4)–4 |

The region is carried on `Segmentation.valid` rather than recomputed, because the live
overlay shades it every frame and §15.4 costs 2.4 ms to derive. A wrong pose and a wrong
*rejection* are indistinguishable in a plain ellipse overlay and have opposite fixes, which
is the whole reason the ignored area is drawn at all.
