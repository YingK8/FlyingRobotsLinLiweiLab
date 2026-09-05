# Chapter 3. Pose: frame in, five degrees of freedom out

*Stage 3 of the pipeline. Consumes: frames from [chapter 1](../camera/theory.md)
and the frames fixed by [chapter 2](../calib/theory.md). Produces: a `Pose` per
frame. Consumed by: [chapter 4, control](../control/theory.md).*

This is the largest chapter because it is the one that turns an image into a
number, and there are three separable questions in that: what a camera can *see*
of the robot (§12), what precision is *possible* at all (§13), and how to find the
robot in a cluttered frame in the first place (§15). §18 closes the loop on all
three: it measures what the noise on this bench actually is, rather than bounding
it (§13) or scoring it against renders (§16).

## Reading order

| # | file | what it does |
|---|---|---|
| 1 | `segment.py` | frame -> silhouette -> fitted rim ellipse, then refitted onto the image. §15 and §16 are about this file |
| 2 | `conic.py` | ellipse -> circle pose. The geometric core, §12 |
| 3 | `estimator.py` | ties 1+2 together, applies the calibrations, resolves the ambiguity |
| 4 | `stereo.py` | the same from two views, which kills the ambiguity outright |
| 5 | `filter.py` | constant-velocity Kalman: velocity, coasting, latency compensation |
| 6 | `uncertainty.py` | predicts this frame's error so the estimator can decline |
| 7 | `bounds.py` | the Cramer-Rao floors of §13: what no estimator can beat |
| 8 | `render.py`, `render_stereo.py` | the renderer: ground truth to measure all of the above against |
| 9 | `background.py` | the empty-rig plate the mask subtracts, and its staleness check (§15.3, §15.8) |
| 10 | `fit_radius.py` | the rim radius, recovered from cross-view disagreement (§15.9) |
| 11 | `calibrate_zero.py`, `recorder.py` | the datum, and capture to disk for offline replay |
| 12 | `noise.py` | the measured noise model of §18: what the scatter is when nothing moves |
| 13 | `test_ring_fit.py`, `test_timing.py` | the ring fit and the per-stage budget of §17.5 |

The dependency runs 1 -> 2 -> 3 -> 4; `bounds.py` and the renderer are how the
claims in this chapter were measured rather than asserted.

**Read §15 and §16 first if you are debugging a live frame** -- §15 is the mask, §16 is why
the mask is now only a seed, and §12 first if you are
debugging a number. §13 is the one to read when something seems *too* good, and **§17 when
a hover bias tracks velocity** -- the two cameras do not fire together, and at hover that
is the largest lateral error there is.

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
| Wall model $\rho = \cos\theta + k\sin\theta$ (§12.3a) | `calib/shape.py`: `TiltCalibration`, `model="cylinder"` |
| Resolution floor $2\arctan k$ (§12.5) | `calib/shape.py`: `resolution_floor_deg` |
| Branch matching, Mahalanobis (§12.6a) | `pose/stereo.py`: `match`, `_agreement` |
| Information-form fusion (§12.6b) | `pose/stereo.py`: `fuse`; `calib/rig.py`: `position_covariance` |
| Measured tables in §12.2, §12.5 | `ai/validation/sweep_stereo.py`, `results/pose_validation/` |
| Deviation-vs-$t$ diagnostic (§12.3, §12.5) | `ai/validation/scene3d.py`: `deviation_panel` |


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
against Monte Carlo in `ai/tests/test_bounds.py` (60+ assertions), and the
real system is placed against the result by
`ai/validation/limits.py`.

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
`ai/validation/limits.py`.

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

### 15.8 Tuned on flight footage, not on renders

Three flights (2026-08-25 and 2026-08-26, 4295 frames) went through the stereo estimator
end to end and returned a `poses.csv` with a header and **no rows**. Four separate causes,
each measured rather than argued.

**The appearance default was the old rig.** `shape.APPEARANCE` still said `bright`, and the
rig on the bench is a black rim against white foam. `bright` thresholds luminance at 128 on
a scene whose median is 144, so the hull took the backdrop -- an ellipse covering 62% of the
frame at 14.6 px fit RMS -- and every downstream gate refused it, correctly. The default is
now `dark`.

**128 was being passed explicitly into the dark path.** `StereoPoseEstimator` defaulted
`thresh=segmod.THRESH`, so the appearance's own level never applied. Left alone, the same
600 frames returned 194 poses against 5. `segment` documents `thresh=None` as "this
appearance's level"; restating the constant defeated it.

**No background plate existed, and one file cannot serve two cameras.** S15.3 calls
background subtraction the whole answer, and `BACKGROUND_PATH` is a single path, so a stereo
pair fell back to `backdrop_mask` -- which cannot separate the rim from the shadowed gap
between the foam blocks behind it, because both are dark and they touch. The plate does not
need a trip to the bench: the rig is bolted down and the robot is not, so a temporal median
over the take *is* the empty scene (`background.from_video`). One per camera, written beside
each video, and `segment(background=...)` takes it.

**`DARK_THRESH` was cutting the rim itself.** With plates in place, every eighth stereo frame
through the full estimator:

| `DARK_THRESH` | 190 | 210 | 220 | 230 |
|---|---|---|---|---|
| 2026-08-25_135344 | 24% | 52% | 48% | 23% |
| 2026-08-26_123311 | 30% | 43% | **59%** | 41% |
| 2026-08-26_123409 | 26% | 39% | **49%** | 44% |

190 is worst on all three. 220 it is.

Together: **0 poses to 46--57% of frames**, at 27 ms per stereo frame.

### 15.9 The rim radius, measured without a renderer

`RADIUS_BY_APPEARANCE["dark"]` was marked provisional, carried from a dataset rendered at a
different threshold, with refitting "blocked on the renderer being unable to reproduce this
rig". The renderer is not needed. Two cameras $83°$ apart each back-project the rim to a
position of its own, and those agree only at the right radius: too small and each puts the
robot too close *along its own axis*, and the axes point different ways, so the error does
not cancel. Sweeping it gives a sharp V.

| radius mm | 9.8 | 10.1 | 10.3 | **10.4** | 10.5 | 10.7 | 11.0 |
|---|---|---|---|---|---|---|---|
| discrepancy mm | 11.15 | 5.99 | 3.03 | **2.19** | 2.70 | 5.61 | 10.69 |

All three flights, two days apart, minimise at 10.4 mm with near-identical curves -- three
measurements of one quantity, not a fit to one. Cross-view disagreement falls from 5.96 mm
to 2.1--2.7 mm.

Two caveats travel with the number. It is **tied to `DARK_THRESH`**, because the effective
radius is where the threshold cuts a shaded edge -- 10.4 belongs to 220, and moving one
moves the other. And its absolute scale inherits the rig's, which rests on the board pitch
([ch. 2 S14.4](../calib/theory.md)); a 1% error there is 0.1 mm here. `fit_radius.py` re-runs
the sweep on any flight already on disk.

Both caveats need re-reading after §16. The threshold no longer cuts the rim -- the direct
fit locks onto the darkness ridge instead -- so the radius is tied to a different quantity
now and wants re-measuring. §16.5 is also why a later sweep that found *no* minimum here
should not have been believed.

**Sub-pixel refinement makes it worse here**, and this is the one place that surprised.
`subpixel_boundary` measured bias $-0.0858 \to -0.0012$ px on a synthetic soft-edged disc,
and on this footage it moves the discrepancy from 2.54 to 3.01 mm at no gain in fit RMS.
That disc was filled; this rim is a thin ring with an edge on each side, so re-locating a
hull point onto "the intensity edge" can move it onto the inner one. It stays off.

### 15.10 Correspondence with the implementation

| Model element | Code |
|---|---|
| Chroma vacuity on a mono sensor (§15.2) | recorded in `segment.py`, `score_channel` docstring; asserted by `ai/tests/test_appearance.py::test_dark_needs_no_colour` |
| Background subtraction $M_{\text{valid}}$ (§15.3) | `segment.background_mask`, `segment.load_background` |
| Median empty-rig frame, staleness check (§15.3) | `controller/pose/background.py`, `--check` |
| Bright-and-smooth backdrop, convex hull (§15.4) | `segment.backdrop_mask` |
| Region selection, and refusal when there is none (§15.4) | `segment.valid_region`, `segment.segment` returning `None` |
| Rod removed by brightness (§15.4) | `segment.DARK_THRESH` |
| Spread gate $\kappa$ (§15.5) | `segment.DARK_MAX_SPREAD`, `silhouette_hull(max_spread=)` |
| Admissibility $\rho_{\max}$ then size (§15.5) | `segment._best_group`, `segment.SHAPE_TOL` |
| Margins (§15.6) | `ai/tests/test_elp_captures.py`, swept per constant |
| Rejected area, shown to the operator | `segment.shade_rejected`, `online_camera.ipynb` [§3](../control/theory.md) (ch.4)–4 |
| Empty-rig plate per camera (§15.8) | `background.from_video`, `for_flight`, `segment(background=)` |
| Rim radius from cross-view disagreement (§15.9) | `pose/fit_radius.py`, `estimator.RADIUS_BY_APPEARANCE` |
| Ellipse fitted to the image, not the mask (§16) | `segment.ring_weight`, `segment.fit_ellipse_image`, `stereo.refine(mode="image")` |
| Seed from the map, no plate or region (§16.10) | `segment.ring_seed`, `segment.segment_ring` |

The region is carried on `Segmentation.valid` rather than recomputed, because the live
overlay shades it every frame and §15.4 costs 2.4 ms to derive. A wrong pose and a wrong
*rejection* are indistinguishable in a plain ellipse overlay and have opposite fixes, which
is the whole reason the ignored area is drawn at all.

## 16. Fitting the ellipse to the image instead of to a mask

§15 ends at a mask: a level is chosen, every pixel is sorted into robot or not, and an
ellipse is fitted to whatever survived. On the recorded flights that fails, and the failure
is not a badly chosen level. It is the sorting.

### 16.1 Why no level works

Two measurements on the flight footage, and they close the question between them.

**The evidence around the rim is intrinsically uneven.** Sampling 180 points around the
fitted ring, the bottom decile carries essentially zero contrast -- in all fourteen frames
tested, in both an absolute map and an illumination-normalised one. A shadow falls across
part of the rim, a coil hides another part, the duct wall goes edge-on and unlit on a
third. A single global level must either drop those arcs or admit everything as dark as
they are, and what is as dark as they are is the shadows.

**The mask is contamination-limited, not level-limited.** Over 20 frames per camera the
mask's area swings 24k -> 154k px as shadows enter and leave. The rim is ~28k px of that.
The convex hull of §15.5 then spans the shadow with the rim, and the fitted ellipse follows.

So the level is being asked to answer a question it cannot: *is this pixel rim?* -- when
the only answerable question is *does this ellipse lie on the rim?*

### 16.2 The rim is thin and shadows are not

One separation survives that the level cannot express. Let `C_k` be a morphological
closing with a structuring element of width `k`. `C_k` fills every dark feature narrower
than `k`, so

    W = C_k(I) - I

is non-zero **only on dark structures thinner than k**. The rim is ~8 px across and scores
in full; a cast shadow is broader than the kernel, survives its own closing, and scores
zero. This is the black-hat, and it is the whole shadow/rim discriminator.

Two implementation facts are load-bearing:

- The structuring element must be `MORPH_RECT`. Rect is separable and ellipse is not:
  2.6 ms against **40.2 ms** on a 1280x800 frame. The same disqualification as the
  full-resolution backdrop finder of §15.4.
- Subtracting the plate's own response, `W = (C_k(I) - I) - (C_k(B) - B)`, removes the
  static thin dark lines between the coil formers. Those are the one thing in the scene
  shaped like the rim, so the plate is what tells them apart -- the kernel cannot.

`W` is **never thresholded**. It is a weight.

**Rejected: normalising by the local envelope.** Shadow is multiplicative, so
`(C_k(I) - I) / C_k(I)` should be illumination-invariant and should even out a rim that a
shadow has dimmed. It does not. Along-ring p10 as a fraction of the median went 43% -> 51%
on one frame and was unchanged or worse on the other thirteen. The unevenness of §16.1 is
not an illumination artefact that a better map removes; it is arcs that are genuinely not
visible, and what handles those is the loss, not the map.

### 16.3 The ellipse as the segmenter

Given `W`, fit the ellipse directly. For parameters `p` and `N` samples `x_i(p)` around
the predicted perimeter, minimise

    S(p) = sum_i rho( max(0, w_ref - W(x_i(p))) )

Four choices in that line, each measured:

**Fixed `N`, not a line integral.** A sum along arc length grows with the perimeter, so the
optimiser inflates the ellipse to collect more darkness. Sampling a fixed count evenly in
*parameter* angle makes `S` a mean, and the bias disappears.

**The residual is `sqrt` of the deficit.** Least squares then sums the deficits themselves,
so minimising `S` maximises the summed evidence -- linear in `W`. A plain `w_ref - W`
residual is quadratic and penalises a *bright* sample, which is backwards.

**`w_ref` comes from the seed's own p90, not from a constant.** The response scales with
contrast and contrast moves with the lighting. An absolute level here would be `DARK_THRESH`
again, in a new place.

**`rho` is robust, and here that is not optional.** In `refine(mode="ellipse")` a robust
loss is inert -- the per-view ellipse fit has already pooled the boundary, so there are no
outliers left. Here nothing has pooled anything: the residuals are individual samples, and
§16.1 says a tenth of them carry nothing. Cauchy bounds what a dead sample can do to five
parameters.

**Coarse to fine, because the rim is thin.** `S` is flat more than about five pixels from
the rim, and the mask seeds this replaces are 18-28 px out. One pass on a heavily blurred
copy of `W` widens the basin to the seed; the fine pass then lands it. Without it the fit
walked 35 px off a synthetic ring it had a good seed for.

The Jacobian step is the one numerical detail worth writing down. `least_squares` steps by
~1.5e-8 relative, which on a centre near 320 is 5e-6 px: `W` is bilinearly sampled, so that
reads the gradient off a single interpolation cell and the direction is noise. At 1e-3 the
step is ~0.3 px, about a third of the blur, and it converges.

### 16.4 Coverage: the blunder test this path needs

§15's gate is `fit_rms_px / a`, the hull's distance from the fitted ellipse. It cannot be
used here, because the hull is exactly what the shadow contaminated -- the gate would reject
the frames the direct fit was added to rescue.

The replacement is **coverage**: the fraction of samples carrying at least half the ring's
*own* median evidence. It is scale-free by construction, so a ring at half the contrast
scores the same, and it catches the characteristic failure of this objective -- a fit that
has slid onto scattered dark specks keeps a respectable mean and loses its coverage.

It predicts cross-view agreement sharply. Over 798 stereo fits on three flights:

| coverage floor | frames kept | median discrepancy | under the 25.5 mm gate |
|---|---|---|---|
| none | 100% | 53.8 mm | 22% |
| 0.5 | 96% | 52.3 mm | 23% |
| 0.6 | 22% | 30.0 mm | 47% |
| 0.7 | 8% | **4.1 mm** | 94% |
| 0.8 | 6% | 4.0 mm | 98% |

### 16.5 What that table overturns

The `DARK_THRESH` sweep recorded in `segment.py` ended by concluding that a ~32 mm floor in
cross-view discrepancy "is the extrinsic, and no threshold reaches it" -- `fit_radius.py`
had stopped finding the sharp V of §15.9 and the two flights disagreed about the radius.
**That was wrong.** Where the rim is genuinely covered the two views agree to 4.1 mm. The
extrinsic was sound the whole time; the floor was the mask, and it was invisible as such
because every threshold produced a confident ellipse on a contaminated hull -- including
the ones `fit_radius.py` was sweeping.

The general lesson is worth more than the number: *a statistic computed from a corrupted
intermediate cannot diagnose what corrupted it.* Cross-view discrepancy was being read as a
statement about the rig when it was a statement about the segmenter, and nothing in the
number distinguished the two. The distinguishing measurement is the one that skips the
intermediate -- here, asking the image directly whether the ellipse is on the rim.

### 16.6 Tracking, and what the level is now for

Seeded from the mask every frame, coverage reaches 0.7 on 24/26/43% of frames across the
three flights. Seeded from each view's own last accepted fit, on 56/44/49%. The mask seed
is a cold start and the coarse pass spends itself covering its error; a seed one frame old
starts on the rim.

So `DARK_THRESH` survives, with a different job. It no longer decides where the rim is --
it decides how often the direct fit starts close enough to converge. Re-swept on that
basis, end to end, as frames solved:

| `DARK_THRESH` | flight 1 | flight 2 | flight 3 |
|---|---|---|---|
| 100 | 33.3% | 12.3% | 14.7% |
| 150 | 49.1% | 34.0% | 23.6% |
| **190** | **76.0%** | **54.0%** | **43.7%** |
| 220 | 49.7% | 23.4% | 44.2% |

190 wins on every flight -- the value §15.6 already shipped, and the 100 it replaced came
from tuning the mask back when the mask was the measurement. Swept at the old radius, median
cross-view discrepancy sat at 3.3-4.0 mm at *every* level in that table: the level moves how
many frames solve and not how good the answer is, which is exactly what it means for it to
be a seed.

### 16.7 The effective radius moves with the edge

§15.9's radius is *tied to `DARK_THRESH`* because the effective radius is where a
threshold cuts a shaded edge. The direct fit cuts nothing -- it settles on the rim's
darkness ridge -- so the constant had to be re-measured, and the first paired comparison
said so before the sweep did: on frames both paths solved, position agreed to 0.14 mm while
cross-view discrepancy was 1.5 mm *worse* for the direct fit. Agreement that good with
disagreement that bad is a radius error, not a fitting error.

`fit_radius.py`, re-run with the direct fit active, is unambiguous:

| radius mm | 10.0 | 10.1 | **10.2** | 10.3 | 10.4 | 10.5 |
|---|---|---|---|---|---|---|
| 2026-08-26_184830 | 4.19 | 2.47 | **0.83** | 1.55 | 3.26 | 5.03 |
| 2026-08-26_184940 | 3.54 | 2.07 | **1.17** | 2.45 | 4.14 | 5.93 |
| 2026-08-27_090057 | 3.68 | 2.06 | **1.04** | 1.97 | 3.71 | 5.50 |

All three land on 10.2 with sharper minima than §15.9's -- and the V is back, which
§16.5 predicted: the sweep that had stopped finding one was sweeping contaminated ellipses.

### 16.8 What it is worth

Each path at its own best radius -- the mask fit at 10.4, the direct fit at 10.2 -- over
every sixth stereo frame of three flights:

| flight | solved, mask | solved, direct | discrepancy, frames **both** solve |
|---|---|---|---|
| 2026-08-26_184830 | 95 / 171 (56%) | 130 / 171 (76%) | 2.43 -> **0.79 mm** |
| 2026-08-26_184940 | 63 / 235 (27%) | 127 / 235 (54%) | 1.54 -> **0.88 mm** |
| 2026-08-27_090057 | 88 / 394 (22%) | 174 / 394 (44%) | 1.86 -> **0.90 mm** |

Roughly double the frames, at a third of the cross-view disagreement, and the direct fit
is the better of the two on 96-98% of the frames they share. The frames only it solves are
not marginal ones scraping past the gate: their median discrepancy is 1.03-1.14 mm, and all
of them are under it.

The comparison is paired on purpose. Comparing medians over each path's own solved set
compares different frames -- the direct fit admits harder ones, which pulls its median the
wrong way and reads as a regression. That is how the first version of this table was wrong.

### 16.9 Two views at once, and what is left

`stereo.refine(mode="image")` is the same objective with one pose and both views'
samples in one residual vector: five world parameters, `2N` residuals. This is the form
that covers occlusion properly. An arc missing in A contributes a bounded residual there
and is constrained by the same arc in B, because both views share the parameters. Neither
`mode="ellipse"` nor `mode="hull"` can do that -- both compress a view to a fitted ellipse
*before* the joint solve, by which point the occlusion has already corrupted what the solve
is handed.

What remains open:

- **Cost.** ~57 ms per stereo pair against a 16.7 ms period at 60 fps: `ring_weight` at
  7 ms/view, and up to two `least_squares` solves. Fine for replay, not yet for the live
  loop. The tracked path skips the coarse pass; the cold path is what costs, and most
  frames are still cold. One joint solve would replace two per-view ones.
- **Solve rate.** 44-76%. The frames that fail are largely ones where the rim is not in
  view to be found -- camera B loses it entirely for stretches of these takes -- but that
  has not been separated from fits that simply did not converge.

### 16.10 The rig-side fix, and what it deletes

On 2026-08-28 the rig changed: a **white ring against a black cloth backdrop**, the
mirror of everything above. It is the better answer to §16.1, and it is a rig fix rather
than a code fix -- *a shadow cast on a black backdrop has nothing to darken*, so it never
enters the image, let alone the map.

The code change is one line of polarity. A closing fills dark features narrower than `k`;
an opening removes *light* ones. So `ring_weight` takes the top-hat `image - opening` for
`bright` and the black-hat `closing - image` for `dark`, and nothing downstream can tell
which it was handed.

**The level is what fails on this rig, and the region is what saves it.** The bench at the
frame edge reads 252 against the robot's 240, so no level on luminance separates them --
but the plate difference alone segments the robot almost exactly. The `bright` branch of
`score_channel` was returning the frame *ungated*, discarding a region it had already been
handed, so the level was being asked to do a job the region does. Gated, the same frame
gives 58k mask pixels instead of 322k and a major axis of 449 px instead of 3242. That is a
plain defect, and four lines fix it.

Only from a plate, never `backdrop_mask`: that finder looks for the bright *smooth*
region, which on a dark backdrop is the robot itself or nothing. And unlike `dark`, no
region is not a refusal here -- the renders every `bright` constant was fitted on have no
plate and no clutter, and must keep working.

**Where there is no plate, §15 drops out entirely.** All of it exists to answer *where may
the robot be?* on a scene whose clutter resembles the robot. On the evidence map it does
not: the bench, the cloth folds and the room are all **broad**, and the top-hat keeps only
what is thin. So the map answers that question itself and `ring_seed` takes the seed
straight from it.

Which to seed from is measured. Over 456 views the three candidates -- the region alone,
the region and the level, and the evidence map -- agree on the seed ellipse to within 3 px.
The plate mask wins on the two things that then differ: it clears the ridge gate on **94%**
of views against 92%, and costs **5.4 ms** against 7.6.

**And the plate need not be captured.** A stored plate has two costs that only appear on
the bench -- someone must take it with the robot out of frame, and it is silently wrong the
moment the rig is nudged. `background.RunningPlate` estimates it from the stream instead,
as a per-pixel running median by sign steps (`bg += step * sign(frame - bg)`), O(1) memory
and no frame history. It is also the only honest option for the live loop, since
`from_video`'s median is a whole-take statistic. Scored on the ridge gate over 416 views it
matches a stored plate -- **94% against 95%** -- with a *higher* median ridge (29.4 against
19.9), because it follows drift a fixed plate cannot. 4.3 ms a view, against 6.3 ms for
`cv2.createBackgroundSubtractorMOG2`, which scores the same. It shares `from_video`'s one
failure: a true hover in one spot walks the plate onto the robot.

Two things had to be got right for the seed:

- **Not Otsu.** The map is ~95% near-zero, so Otsu puts the level down in the noise: 72k
  px on, hull major 1377 px where the ring is 455. The level is a fraction of the map's
  own 99.9th percentile instead -- a percentile and not the maximum, so one specular pixel
  cannot set the scale. 0.25-0.35 all give major 452-456 at ratio 0.91; 0.30 ships.
- **The spread limit is still required.** It was applied only to `dark` before, but its
  justification -- the rim is hollow and arrives as arcs -- is a property of the robot.
  Without it `silhouette_hull` pools every speck: same frame, major 1193 instead of 452.

Measured on `2026-08-28_092117`, every sixth stereo frame, radius re-fitted to 10.05 mm:

| | |
|---|---|
| both views seeded | 228 / 228 |
| solved, `RunningPlate` (no capture step) | **185 / 228 (81%)** |
| solved, stored plate | 181 / 228 (79%) |
| solved, evidence map alone (no plate) | 156 / 228 (68%) |
| median cross-view discrepancy | **1.8 mm** |
| cost | 66-77 ms per stereo pair |

**Optical flow was considered here and is not needed.** The background diff it would have
replaced is itself gone, and the question it answered -- *which pixels are the robot?* --
is no longer asked of any static reference. Nor would it recover the frames that still
fail: of the 43 that remain, 21 are a view whose ridge is below the gate, and inspecting
them shows the same thing every time -- **the ring has left camera B's field of view or is
cut by the frame border**, ridge 1.1-2.5 in B against 16-26 in A on the same frame. A
motion cue cannot find a ring that is not in the picture. That is a framing fix: aim B, or
widen it. The other 21 are frames both views fitted and then disagreed.

### 16.11 What the two fits do to each axis

Overlay the mask fit and the direct fit and the *major* axes differ, which looks wrong: if
the correction were only for the body's thickness it should touch the minor axis alone.
Measured over 427 views, both axes move, and by different amounts:

| | mask fit | direct fit | shrink |
|---|---|---|---|
| major | 413.9 px | 401.7 px | 10.9 px (2.6%) |
| minor | 303.2 px | 282.9 px | 20.7 px (6.9%) |

Both are correct, for two different reasons.

The **major** axis shrinks because the two fits measure different edges. The mask fit is
fitted to a convex hull, which is a superset of the rim projection (§12.4) -- the rim's
*outer* envelope. The direct fit settles on the evidence ridge, the rim's *centre-line*.
The gap between them is the wall, and it is constant across tilt bands (10.4-12.7 px) as a
constant should be. This is the same fact as §16.7, and it is why the effective radius
belongs to the fitting method and had to be re-measured.

The **minor** shrinks twice as far because of the mast and magnet. They protrude along the
rotor axis, so under tilt they push the silhouette outward in the *short* direction -- the
one-sided contamination of §12.4, which `AXIAL_WEIGHT_POWER` suppresses and does not
eliminate. The direct fit never sees them: they are not on the rim ridge.

So constraining the direct fit to keep the hull's major axis would put the wall thickness
back in as a bias. What does follow is that `TiltCalibration` no longer describes this
statistic. It was fitted against `cv2.fitEllipseDirect` output on renders (`a = 1.0736` at
`radius_mm = 10.2446`) and corrects for a widening the direct fit has already removed.
Measured it is nearly inert -- median discrepancy 1.81 mm as shipped against 1.77 mm with
an identity calibration -- so it does no harm, but it belongs to the old rig and wants
refitting rather than trusting.

### 16.12 The bulge breaks the two-fold ambiguity, but only side-on

A circle's projection determines its pose only up to a reflection: `conic.backproject`
returns two candidates and something else has to choose (§12). The robot is **not** a bare
circle -- the mast and magnet sit on one side of the rim plane -- so under tilt the
silhouette's two halves, split along the major axis, are not mirror images. The fatter half
is the one the body leans toward.

Fitting each half's minor semi-axis separately, with the centre, major axis and angle held
at the full fit's values, measures that directly. Over 281 views the asymmetry is small and
rises monotonically as the robot turns side-on, which is where the ambiguity actually bites:

| axis ratio | median \|b_upper − b_lower\| |
|---|---|
| 0.77-0.91 (face-on) | 1.8 px |
| 0.66-0.77 | 2.6 px |
| 0.39-0.66 (side-on) | 3.1 px, 1.5% of the minor |

At 1.5% of the minor axis it is far too small to *correct* the pose with -- taking the
smaller half as the true rim moves the minor by less than the fit's own scatter. As a
**sign** it is worth much more. Checked against the orientation the two-view solve resolves
independently, over 246 views:

| | agreement |
|---|---|
| all views | 57% -- a coin flip |
| side-on (ratio < 0.7) | 76% |
| side-on and \|Δ\| > 1 px | 80% |
| side-on and \|Δ\| > 3 px | **85%** |

The magnitude gate is what makes it useful: it abstains face-on, where the protrusions
point at the camera and project to a dot, and speaks up side-on. 57% to 85% is the
difference between a cue and a coin, and it is bought entirely by knowing when to say
nothing.

85% is not enough to arbitrate alone, and it is not needed as one in stereo -- two views
84 degrees apart already settle the reflection. Its value is monocular, and as a prior
where `_choose` currently has only the previous frame to go on: a dropout longer than
`dropout_s` leaves the estimator with no opinion at all, and this gives it one exactly when
the geometry is worst.

### 16.13 Which way is up, and why the scene was tilted

Two separate ambiguities get conflated under "orientation", and they degenerate in
opposite places.

**The mirror pair** is what `conic.backproject_ellipse` returns and what `_choose` and
`stereo.match` resolve. Both solutions share $|n \cdot v|$ *exactly* -- verified to 1e-16
across tilts from 0.5 to 89 degrees -- so the pair differs by a rotation about the viewing
ray and **never encodes which face is seen**. Their separation is 5.8 deg near face-on,
peaks at 84.5 deg near 45 deg tilt, and falls back to 7.5 deg at 89 deg: ill-conditioned at
*both* ends, best in the middle.

**Up or down** is $\operatorname{sign}(n \cdot v)$, and the rim carries no information
about it at all -- a circle is unoriented. `stereo.orient` does not measure it; it asserts
it, flipping the normal to agree with a reference on the assumption the robot hovers
rotor-up.

That assumption is safe here, and continuity says why. $n \cdot v$ is continuous in time,
so it can only change sign through $n \cdot v = 0$: the rotor axis perpendicular to the
viewing ray, the ring projecting to a line, axis ratio going to zero. **Edge-on, never
face-on** -- face-on is $|n \cdot v| = 1$, the state furthest from a flip, and no
trajectory reaches one without the other. Three caveats travel with that: it is a
continuous-time guarantee and not a sampling one, so a fast flip can cross between frames
at 60 fps; a dropout breaks the observed continuity; and it is per-camera, each crossing at
its own instant.

Measured, the flight never comes close. Axis ratio bottoms out at 0.395 (camera B) and
0.479 (A) -- **no frame within 23 degrees of edge-on** -- so the up/down state provably
cannot have changed, and `orient`'s assertion is not a risk on this data. This also says
the side-on failures are *not* orientation errors: at ratio 0.39-0.66 the fit is degrading,
which is a different problem.

**The world frame is not gravity aligned, and in the scene that shows up as a 90 degree
error.** `rig.meta["world_frame"]` is `camera_A` -- camera A sits at the identity extrinsic
-- so the world frame *is* a camera optical frame, where up is **-y**. `live_viz.up_direction`
reports that faithfully and viser is told `set_up_direction("-y")`. But the mesh's rotor
axis, and so the rod, points along the estimated normal, which in camera-A optical
coordinates runs roughly along the *viewing* direction. -y-up against +z-up is exactly a 90
degree rotation about x, which is what the robot lying on its side in the viewer actually
is. `up_direction`'s own docstring says as much: guessing wrong lays the robot on its side.

Neither the mesh nor the pose maths is at fault, and both were checked rather than assumed.
`render.load_mesh` leaves the rim in the x-y plane -- 1.36 mm of z spread, which is the
wall's own thickness -- with its centre on the origin and the rotor axis on +z. And
`normal -> (theta, phi) -> render.pose_matrix -> normal` round-trips to **0.0000 degrees**
over a spread of attitudes.

The datum is the fix, and the same one the tilt needs. On `2026-08-28_092117` the robot's
median tilt from world +z reads **38 degrees**, which is not the robot leaning but the
frame being tipped. `live_viz.prime_zero` builds a `zeroing.Zero` from the first stable
poses -- collected until `AUTO_ZERO_FRAMES` of them agree to `AUTO_ZERO_TOL_DEG`, so the
datum is not built from scatter -- and `Zero.apply` puts that attitude on +z. Median tilt
then reads 16.7 degrees, a hover attitude rather than a frame error, and `up_direction`
returns `+z` so the rod stands upright with the grid under it. `up_direction` and
`camera_in_datum` already follow the datum, so the rest of the scene needs no changes --
`zero="auto"` on `from_recording` and `from_stereo` is the whole of it.

**The live path must not wait for it.** A replay can prime the datum before the scene
exists, because its frames run out; a camera's do not. A session is normally started
*before* the robot is in shot, so priming there blocks `make_viz` outright -- no viser
URL, no output, nothing to interrupt, which reads as a crash rather than a wait. So
`from_stereo` brings the viewer up in whatever frame the rig provides and re-orients when
the first stable poses arrive, via `_DatumPrimer` and `LiveViz.set_zero`. `prime_zero`
carries a frame cap for the same reason, so no caller can hang on it.

**The datum's sign is a choice, not a measurement**, and `flip=True` is how to make the
other one. §16.13 above is why: a circle is unoriented, so `stereo.orient` asserts the
hemisphere rather than measuring it, and on this rig its reference points *away from
camera A* rather than up. Whether that agrees with the rod depends on which side of the
robot camera A sits, which nothing in the pipeline knows. Measured either way on
`2026-08-28_092117`, the pose normal's z-component reads +0.962 unflipped and -0.962
flipped, so 96% of frames put the rod up one way and down the other. Look at the scene; if
the rod hangs down, flip it. The bulge cue of §16.12 is what would settle it without
looking, and it is not built.

It is averaged and gated because the datum is subtracted from every later measurement, so
noise in it becomes a fixed bias on the whole run rather than something that averages out
(`calib/zeroing.py`). On this flight it settles in 12 poses at 2.63 deg spread.

**Referencing the *first* stable pose is the request, not necessarily the best datum.** The
median tilt improves but the tail does not -- p95 goes 52.4 to 69.3 deg and the maximum 76.9
to 88.3 -- because the first pose need not represent the run's mean attitude, and because
the tail contains estimation blunders as well as real excursions. A datum from the run's
median attitude would centre the distribution instead; it is a one-line change to
`prime_zero` and has not been measured.

### 16.14 Correct ellipse, wrong normal

The symptom is specific and it localises: the ellipse drawn on the frame is right, the
rotor axis in the scene is not. Everything between them is
`backproject -> branch choice -> tilt_cal -> orient`, and the radius is not in it -- the
cone fixes the normal from the ellipse's *shape*, and the radius only scales the distance
along it.

`TiltCalibration` is the culprit, for the reason §16.11 gives. It describes the gap
between a **hull silhouette** and the true rim: the mast and magnet widen the silhouette's
short direction, and the calibration widens the model to match. The direct fit produces no
silhouette. It settles on the rim's own evidence ridge and never sees the protrusions, so
the correction is applied to a widening that is already gone and the minor axis is
inflated twice. Tilt is $\arccos(b/a)$, so all of that error lands on the normal and none
of it on the drawn ellipse.

The measurement needs no ground truth: the two views each back-project their own normal,
and they should agree. Over the flight, as the angle between them:

| tilt calibration | cross-view normal angle, median | p90 |
|---|---|---|
| as shipped (`a = 1.0736`, fitted on renders) | **9.58 deg** | 23.94 |
| identity | **2.46 deg** | 17.75 |

2.46 deg is the per-view scatter floor of §12.12 (2.6 deg at 40-50 deg tilt), so identity
is not merely better, it reaches the noise. Position barely moves across the same change --
1.81 against 1.77 mm -- which is exactly why this hid: every gate in the pipeline watches
position, and the cross-view discrepancy gate passed it.

So `StereoPoseEstimator` neutralises the calibration when `direct=True`. It is not a
preference: on this path the statistic the calibration was fitted to does not exist. A
tilt calibration for the direct fit would have to be measured against the *ridge*, and
nothing has been.

### 16.15 A constant normal as ground truth

`2026-08-28_131552` was flown deliberately upright, moving only along one axis. That
makes it worth more than another flight: **the true rotor axis is constant, so the
scatter of the estimated one is the error**, with no renderer and no second instrument.
Nothing else in this chapter has that.

Run as it stood, the take exposed a failure the previous flights hid. Orientation was
bimodal -- 126 frames scattered by 9 degrees about their own mean and 39 by 68 -- and the
outliers are the **branch flip**: both views take the mirrored conic solution *together*,
so they agree with each other and are jointly wrong. §16.13 measured the two solutions as
furthest apart near 45 degrees of tilt, at 84.5 degrees, and both cameras sit at 43-45
here, so a flip costs almost a right angle.

Two things do not catch it. The branch **margin** reads 28.3 sigma on flipped frames
against 25.8 on good ones -- the matcher is confidently wrong, not undecided. And
`MAX_DISCREPANCY_MM` at 25.5 mm is far too loose, because a flip moves *position* by only
8.5 mm against 2.0 for a good frame. The gate was sized against segmentation blunders,
which separate by 60x and need no precision; this blunder separates by 4x and needs some.

Retuning it to 5 mm, and adding `MAX_JUMP_DEG_PER_S` for what survives:

| | kept | normal scatter, median / p90 | >20 deg out |
|---|---|---|---|
| as it stood (25.5 mm, no jump gate) | 77% | 9.45 / 68.85 deg | 24% |
| 8 mm + jump gate | 58% | 1.24 / 4.80 | 3% |
| **5 mm + jump gate** | **55%** | **0.57 / 2.38** | **1%** |

The jump gate is a physical bound, not a filter: 84 degrees in the 50 ms between sampled
frames is 1700 deg/s, and nothing here turns faster than 60. It stands aside after
`DROPOUT_S` so a track can still re-acquire.

**The remaining coverage loss is honest, and it is the ellipse's fault rather than the
gates'.** On rejected frames the worse view's ridge reads 5.3 median against 38.7 on
accepted ones, and 71% of them have a view below 10 against 5% of accepted. No branch
selection rescues a fit that is not on the rim -- and on those frames every pairing scores
badly, which is why the 28-sigma margin is not a tie a temporal prior could break. Coverage
here is bounded by how often both views see the ring well, which returns to §16.10's
finding about camera B's framing.

**The overlay arrow was reading the wrong frame.** `normal_segment_px` projected
`pose.xyz_mm` with the intrinsics alone, which is correct only while the world frame *is*
that camera's optical frame and no datum is set -- true when it was written, false since
`prime_zero` began installing one. So the arrow pointed somewhere plausible and wrong in
camera A, was meaningless in camera B, and was gated to view 0 for that reason rather than
fixed. It now undoes the datum to reach the world, applies each camera's own extrinsic,
and then projects, so it is drawn in both views.

Its tail is pinned to that view's fitted ellipse centre. The projected centre should land
there anyway, and pinning is still worth it: the ellipse is in distorted pixels and the
projection is an ideal pinhole, and any residual datum or extrinsic error otherwise shows
up as the arrow floating off the robot -- which reads as a direction error rather than the
position error it is. Pinned, the arrow shows only what it can actually say.

The check that it is right needs no ground truth: the rotor axis projects along the
direction of foreshortening, so the arrow must lie on the ellipse's **minor axis**.
Measured over 60 frames it sits **0.28 degrees** off in camera A and 0.42 in camera B,
with the tail exactly on the centre -- and identically with and without a datum, which is
the real test, since an image projection cannot depend on an arbitrary choice of world
frame.

For the scene, the same take makes the datum sharp: `prime_zero` settles at **0.47 deg**
spread against 2.63 on `092117`, `up_direction` reports `+z`, the mean axis comes out
`[0.017, 0.005, 0.9998]`, and 81% of poses sit within 2 degrees of vertical. An upright
take is the right thing to prime a datum from, and worth recording deliberately.

### 16.16 Perfect segmentation, no detection: it was the radius

A frame whose rings are plainly well fitted in both views, and no pose. The counters say
where: `n_rejected_fit = 0`, `n_rejected = 80`. Every one is the cross-view consistency
check, which is not an optimisation and cannot be blamed on the ellipse fit's five free
parameters. Two *correct* ellipses that disagree in 3D means the geometry between them is
wrong.

It was the radius. §16.7's method was right and its execution was not: that sweep ran the
estimator with its own gates active, so each radius was scored on the frames that survived
it -- the answer partly chose its own evidence. Re-swept with the gate open and only
frames whose fit is strong in *both* views, the two flights agree:

| radius mm | 9.90 | **9.95** | 9.975 | 10.00 | 10.05 | 10.20 |
|---|---|---|---|---|---|---|
| `131552` median discrepancy | 0.91 | **0.50** | 0.77 | 1.15 | 2.05 | 4.81 |
| `092117` median discrepancy | 1.77 | 1.08 | 0.88 | 1.02 | 1.65 | 4.21 |

Half a percent of radius mattered far more than it looks, because a radius error is a
systematic depth offset along *each camera's own axis*, and the axes point different ways,
so none of it cancels and all of it lands on cross-view discrepancy. At 10.05 only 98% of
strong-fit frames came under the 5 mm gate, at 10.20 just 66%; at 9.95, **100%** do.
Median discrepancy end to end falls 2.06 -> 0.51 mm on the upright take.

**A gate is only as good as the constant it is gating on.** Tightening
`MAX_DISCREPANCY_MM` to 5 mm was right, but with a 1% radius error it was rejecting good
frames for a fault they did not have. The two changes had to be made together, and the
lesson is that a precision gate and the constants feeding it are one system.

### 16.17 Why the fusion is Gaussian and the ambiguity is not

Position is already fused probabilistically and there is nothing to add. Each view carries
an anisotropic Gaussian about its own optical axis --
$\Sigma = \sigma_{lat}^2 I + (\sigma_{depth}^2 - \sigma_{lat}^2) dd^T$, tight across the
ray and loose along it -- `_agreement` is the Mahalanobis distance under
$(\Sigma_a + \Sigma_b)^{-1}$, `fuse` combines in information form, and `filter.PoseFilter`
is a constant-velocity Kalman on top.

**Orientation is not, and softening the gate to covariance inflation would be a mistake.**
The tempting move is to keep every frame and let a wider covariance say how much to trust
it. Measured, the band it would admit is not noisy-but-usable:

| cross-view discrepancy | share of frames | normal deviation, median | more than 20 deg out |
|---|---|---|---|
| under 5 mm (kept) | 62% | 6.2 deg | 8% |
| 5-15 mm | 13% | **70.9 deg** | **79%** |
| over 15 mm | 13% | 84.2 deg | 93% |

Four fifths of that middle band are branch flips. Their error is not Gaussian and no
covariance describes it: the posterior has **two components about 84 degrees apart**
(§16.13), and a single inflated Gaussian straddling both is wrong everywhere -- fat, and
centred where neither mode is. §13's note that the two-fold ambiguity is a genuine
bimodality rather than a numerical failure is the same statement.

The correct probabilistic treatment is a **mixture**, not a wider unimodal covariance, and
using the second component needs a consumer that can carry two hypotheses until one dies.
`StereoPose` returns one pose, so that is an architectural change rather than a tuning one.
Until then the hard gate is crude but not wrong: when the components cannot be separated,
declining beats picking.

### 16.18 Answering on every frame

The gates here decline rather than report a pose they cannot stand behind, and for a
controller that is right. For looking at footage, fitting constants, or any consumer that
would rather have a pose and a quality number than a hole, it is not.
`StereoPoseEstimator(never_reject=True)` stands all of them down -- discrepancy, fit rms,
ridge, the predicted-error model, and `require_stereo`, so a frame with one view lost still
answers from the other.

Two things had to be fixed before that was worth having, and both were real defects that
only showed up once nothing was being hidden by a rejection.

**The plate was cold on the first frame.** `RunningPlate` starts as a copy of the frame it
is given, so subtracting it leaves an empty evidence map and the segmenter finds nothing.
That cost the first frame of every run, and no amount of relaxing gates recovers it -- the
data was gone. `update` now returns ``None`` until `WARMUP_FRAMES`, which callers already
read as "no plate", and the top-hat runs unaided until then. It also lifted the *gated*
solve rate on `2026-08-28_135533` from 86.4% to 89.9%.

**Reporting policy and tracking policy are different questions.** Relaxing `min_ridge` to
zero also relaxed the test deciding whether a fit was good enough to seed the next frame,
so a collapsed fit became the next seed and the track never recovered: camera A sat at a
158 px major where the ring is 250, for hundreds of frames. Track maintenance now uses its
own fixed `_track_ridge`, independent of what is being reported. The difference is not
small:

| | frames with a pose | discrepancy median | p90 |
|---|---|---|---|
| gated | 89.9% | 0.59 mm | 1.30 mm |
| `never_reject`, tied thresholds | 100% | 1.03 mm | 269.53 mm |
| `never_reject`, separated | **100%** | **0.65 mm** | **4.94 mm** |

The third row is the point: answering on every frame costs almost nothing in the median
once the track cannot poison itself, and the frames that are genuinely poor are still
labelled as such rather than dropped. `discrepancy_mm` and the per-view `ridge` ride on
every result, so a consumer can sort by them -- which is the same information the gate
used, handed on instead of acted on.

`pose/demo_video.py` renders a whole flight with every overlay on it, running this mode by
default and marking which frames the gates would have taken, because a solve rate does not
show *which* frames were lost or why.

### 16.19 What the two cameras are actually for

The occlusion argument for stereo had never been measured, only asserted. Sampling the
**world** rim circle at 180 angles and reading each camera's evidence map at the
projection of every one of them, over 395 frames of `2026-08-28_135533`:

| fraction of the rim with evidence | p5 | p25 | p50 | p75 |
|---|---|---|---|---|
| camera A alone | 0.726 | 0.814 | 0.900 | 0.956 |
| camera B alone | 0.656 | 0.819 | 0.917 | 0.956 |
| **either camera** | **0.922** | **0.983** | **1.000** | 1.000 |
| both at once | 0.574 | 0.733 | 0.783 | 0.856 |

A view is missing a fifth of the rim or more on **36% of frames**; on exactly those the
union median is **0.989**. Both views are under 80% on 4.3%, and even there the union
median is 0.867. Union coverage against cross-view discrepancy correlates **-0.384**:
the frames that go wrong are the frames the union does not cover.

That is the 83-degree axis separation doing the one thing a second camera is uniquely
for. It is not averaging and it is not depth -- it is that the two cameras lose
*different arcs*, so between them the rim is nearly always whole.

**Which settles the shape of the fix: nothing fits a semi-circle.** The tempting reading
of the table is to fit each visible half separately and merge them, and that is strictly
worse -- a 180-degree arc pins an ellipse's axes very poorly, and merging two
badly-conditioned fits does not recover what neither had. One world circle, sampled by
world angle, with both views' samples in a single residual vector, already *is* the
fusion: five parameters, `2N` residuals, and an arc missing in A carried by the same arc
in B because they share the parameters.

Two implementation facts made that real rather than notional:

- **`refine(mode="image")` was dead code.** It had been written for exactly this and
  nothing ever called it. `update` ran `mode="ellipse"`, which compresses each view to
  one fitted ellipse *before* the joint solve -- the occlusion has already corrupted the
  measurement by then, which is the failure the mode existed to avoid. The evidence maps
  were built in `_view_candidates` and dropped on the floor.
- **Samples must be indexed by world angle, not by each view's image parameter angle.**
  Sampling each predicted ellipse independently makes sample `k` a different physical
  point in each view, which is enough to solve a pose and useless for saying which arc
  is missing. `_rim_points` builds the points once in the disk plane and both views
  project the same ones. It also samples uniformly on the disk, where parameter-angle
  sampling of a projected ellipse bunches near the minor-axis ends -- exactly where a
  side-on frame has least to spare.

`union_coverage` rides on every pose. It is reported and not gated: below roughly 200
degrees of union the five parameters are weakly determined however they are weighted,
and that is worth knowing rather than declining.

### 16.20 Most of the "occlusion" was the level, the kernel and the plate

The dead arcs of §16.19 are real and the word for them was wrong. Sampling the raw frame
along the refined rim, the luminance inside a dead arc reads **139.6** against **157.9**
where the rim is found: the rim is *bright and present*, and `ring_weight` does not
respond to it. An earlier reading of 66.8 against 153.8 was taken on a coarser pose,
before refinement, and was sampling beside the rim rather than on it.

Where it fails is specific. Dead-sample rate around the fitted ellipse, by angle from
the major axis, on `2026-08-28_131552`:

| from the major axis | 0-10 | 10-70 | 70-80 | 80-90 |
|---|---|---|---|---|
| dead | 27.0% | 9-11% | 26.7% | **49.7%** |

Half the samples die at the **ends of the projected minor axis**. That is where the near
and far arcs of the duct converge in the image: the ring stops being a thin line and
becomes a locally wide bright blob, so a 41 px opening keeps it and the top-hat returns
nothing. The 27% at the major-axis ends is the mast and duct wall doing the same thing.

**Widening the kernel fixes the coverage and breaks the estimate.**

| k | 41 | 49 | 61 | 81 |
|---|---|---|---|---|
| arc alive p50 | 0.861 | -- | 1.000 | 1.000 |
| at the minor tips | 0.567 | -- | 1.000 | 1.000 |
| discrepancy p50, mm | **0.77** | 1.92 | 2.39 | ~3 |
| under the 5 mm gate | **72.9%** | 59.8 | 55.4 | ~53 |
| more than 30 deg out | **8.8%** | 18.3 | 13.2 | ~14 |
| `ridge` p50 | **35.8** | 29.0 | 23.7 | -- |

`ridge` is the row that explains the rest. A kernel wide enough to keep the converged
arcs is wide enough to keep the body, so the map stops being a ridge along the rim and
the ellipse is no longer pinned to anything. Coverage reads 1.000 precisely because
everything nearby is bright -- the same failure `RingFit` documents for `coverage` as a
blunder test, arriving from a new direction. **41 stands**, and recovering the converged
arcs needs a growth rule at constant kernel rather than a wider kernel.

The plate is the second half. A `RunningPlate` is a running median converging on whatever
does not move, and on a *hover* rig what does not move is partly the robot --
`background.from_video` already warns that a take where the robot hovers leaves itself in
the plate, and the running form reaches that state faster, at a count a frame. Arc alive
p10 on `2026-08-28_131552`: **0.603** at full subtraction, 0.667 at half, 0.703 with no
plate at all. `RING_PLATE_WEIGHT = 0.5` splits it; the constant carries the three-flight
table and the one flight that disagrees.

The third half, and the largest of the three, is the **level itself**. `THRESH = 128` was
carried over from renders and never refitted, on the stated reasoning that a synthetic
brightness would not transfer. On a black backdrop it transfers badly in a specific
direction: there is nothing above the level except the robot, so the level is free to
fall a long way. A level of 72 keeps 80.5% of rim samples against 78.0% at 128, and 1.3%
of the backdrop against 1.1%. The rim it buys is not free rim -- it is the *shaded* rim,
whose luminance p25 is 138 with the level sitting at 128.

It only ever seeds -- the direct fit measures from the evidence map -- and it still moved
everything, because a seed missing a quarter of the rim starts the solve on a biased
ellipse:

| frames more than 30 deg out | 128 | 96 | **72** | 56 |
|---|---|---|---|---|
| `131552` | 10.2% | 4.1 | **3.1** | 5.3 |
| `135533` | 2.8% | 1.4 | **0.4** | 0.6 |
| `092117` | 4.3% | 4.0 | **2.9** | 4.6 |

`ridge` holds or rises with it (17.4 to 18.6 on `092117`), which is what says the extra
seed area is rim and not backdrop, and `092117`'s discrepancy p90 falls from 37.18 mm to
15.30. The effective radius did not need refitting with it: `RADIUS_BENCH_MM` was fitted
against the *direct* fit, which never sees this level, and discrepancy p50 moved by less
than 0.05 mm on all three flights.

Three of the four things called occlusion in §16.19 were this section. What survives as
genuine two-view cover is the arc a camera has actually lost, and §16.19's union table
still measures that -- it is simply a smaller share of the problem than it first looked.

### 16.21 The occluder has no edge to find

The natural next step is to detect the occluder rather than infer it: an object covering
the rim has a boundary, the boundary is long and straight where the rim is curved, and
curvature along a contour separates them with nothing to tune in absolute units. The
statistics of the dead arcs all point that way. Over 329 views carrying more than 10%
dead samples:

- **Contiguous.** The longest single dead run holds a median **67%** of all dead samples,
  and **100%** at p75 -- one block, not scatter.
- **Sharp.** At a run's boundary the evidence changes by a median of **0.46** of the ring
  median between adjacent samples: it halves in one or two samples, 2 to 4 degrees.
- **Dark.** Raw luminance inside a dead arc averages **66.8** against **153.8** on live
  arc. The white rim is covered, not fading out.
- Median dead fraction where present is 0.20, p90 0.34 -- a 72 to 122 degree sector.

Every one of those is consistent with an occluder, and the inference is still wrong.
Tested directly: a straight edge at least 0.6 rim-radii long, landing within 12 px of
where a dead run begins or ends, found by Canny and a probabilistic Hough over an ROI
around the rim, on the 147 views with a dead run of 50 degrees or more --

**0 of 147.**

Relaxing to any line of 30 px within 20 px gives 84%, which is not evidence of anything:
at that threshold a cluttered ROI is full of lines. The reason is structural rather than
photometric. `ring_weight` is a top-hat, so the map it produces contains **only the thin
rim**; an occluder's boundary never enters the mask a contour could be traced along, and
the rim's own silhouette is cut by the occluder over a few pixels of stroke, not a
traceable run. A curvature detector was written, measured, and deleted.

What the dead arcs are fixed to is a weaker signal than an occluder would give. Their
world direction has |mean| ≈ 0.5 over a flight -- a preferred direction, not a fixed one
-- and they sit on the near side of the rim relative to the camera by a median
$\hat{d}\cdot\hat{l} \approx -0.29$, consistently in both views. A fixed illuminant plus
the duct's own geometry fits that better than anything in the rig, and it is a lighting
fix rather than a code one.

The two-view union in §16.19 does not care. It never needed to know *why* an arc is dead.

### 16.22 The sliding window, and why it has to be told when to speak

§16.17 concluded that the residual error is a **mixture** -- the mirrored branch about 84
degrees away, taken by both views together so no measure of cross-view agreement
separates them -- and that using it "needs a consumer that can carry two hypotheses until
one dies". Time is that consumer: the true branch moves like a robot and the mirror jumps
a quarter turn between frames. `match` takes a `prior_normal` from a 15-frame median of
recent normals and scores it in the same sigmas as everything else.

**Applied to every frame it makes things worse, monotonically.** On
`2026-08-28_131552` -- flown upright, so the normal is ground truth -- the share of
frames landing more than 30 degrees from the flight's median normal went 26.8% ungated,
to 27.4% at a 15 degree prior, 37.3% at 10, and **45.0% at 8**, with the median frame's
own scatter blowing out from 0.84 to 7.50 degrees.

The reason is a loop, not a tuning error. The window is fed the estimator's output, so a
prior strong enough to change a decision is strong enough to manufacture the evidence
for its next decision. Nothing inside that loop can break it.

What breaks it is a signal from outside. The frames in question are separable *before*
anything is done about them:

| on `2026-08-28_131552` | good frames | quarter-turn frames |
|---|---|---|
| cross-view discrepancy, p50 | 0.58 mm | 23.21 mm |
| cross-view discrepancy | p90 2.86 | p10 6.20 |
| best pairing's agreement score, p50 | 1.09 | 1002 |
| union coverage, p50 | 0.94 | 0.81 |

Discrepancy separates them almost cleanly at the 5 mm that `MAX_DISCREPANCY_MM` already
sits at. So the window is asked **only about frames that gate has already flagged**, and
the three quarters that are self-consistent are left exactly as they were:

| prior sigma | off | 25 | 15 | 10 | 6 | 4 | **3** | 2 |
|---|---|---|---|---|---|---|---|---|
| more than 30 deg out | 26.8% | 26.6 | 26.2 | 25.4 | 12.4 | 8.6 | **8.1** | 7.0 |
| scatter p90, deg | 88.31 | 88.27 | 88.17 | 88.08 | 41.38 | 16.94 | **16.94** | 16.70 |
| scatter p50, deg | 0.84 | 0.84 | 0.84 | 0.84 | 0.82 | 0.83 | **0.83** | 0.83 |

The last row is what makes it honest: the median frame does not move. The whole gain is
taken on frames that were wrong, and it is worth a factor of three. Passes over 2-4;
3.0 is the midpoint and coincides with `SIGMA_NORMAL_RAD`, so the window is trusted
about as far as one camera is.

The gate that decides when to ask is `_suspect_mm`, fixed at the module constant and
deliberately *not* `max_discrepancy_mm` -- that one is reporting policy and `never_reject`
drives it to `None`. "Is this worth reporting" and "is this worth a second opinion" are
different questions, the same split as `_track_ridge` against `min_ridge` in §16.18, and
tying them together has now caused a bug twice.

**What it costs, said plainly.** On a re-arbitrated frame the estimator takes a pairing
the geometry ranked *worse*, so cross-view discrepancy on `2026-08-28_131552` goes from
p90 30.59 mm to 31.38. That is the intended trade and not a side effect: the best
agreement score on those frames runs from 134 to 1002 where a good frame reads 1.09, so
no pairing is consistent and the ranking between them carries no information. Continuity
does. The orientation is three times better and the cost is a number that was already
telling you not to trust the frame.

### 16.23 The seed is smooth and biased, the fit is sharp and noisy

Watching the overlay rather than the statistics turns up something no summary column
says: on `2026-08-28_131552` the red seed ellipse tracks the rim convincingly while the
green direct fit does not. Run end to end with each as the measurement:

| on `131552` | discrepancy p50 / p90 | under 5 mm | more than 30 deg out | frame-to-frame angle p90 |
|---|---|---|---|---|
| direct fit, r = 9.95 | **0.68** / 26.58 mm | 77.6% | 4.4% | **11.70 deg** |
| mask seed, r = 9.95 | 4.52 / **5.91** mm | 73.6% | **2.5%** | **1.92 deg** |

The seed is six times steadier and its discrepancy spread is *tight* -- 4.52 to 5.91 --
which is the signature of an offset rather than of noise. It is the effective-radius
problem of §16.7 again: the mask hulls the rim's outer edge, the evidence ridge is its
centre-line, and the two want different constants. Swept, the seed path minimises at
**10.10-10.20** and there it beats the direct fit almost everywhere:

| mask seed radius | 9.95 | 10.10 | 10.20 | 10.30 | 10.40 |
|---|---|---|---|---|---|
| discrepancy p50 | 4.52 | 1.89 | **1.15** | 1.92 | 3.71 |
| under 5 mm | 73.6% | **96.7** | **96.7** | 94.5 | 74.8 |
| more than 30 deg out | 2.5% | **2.8** | 5.2 | 7.5 | 23.5 |

96.7% under the gate against the direct fit's 77.6%, at a sixth of the jitter, for the
price of a median that is three times worse. **But it does not generalise**: on
`2026-08-28_135533` the direct fit at 9.95 gives p90 1.54 mm, 0.6% out and 2.24 deg of
jitter, and the seed at 10.10 gives 4.27, 1.5% and 2.75. The two flights want opposite
answers, so neither is simply better.

**What separates them is the seed collapsing.** The medians agree on both flights -- a
healthy refinement shrinks the major 2.8% and moves the centre 4 px -- and only the tails
differ:

| | major, p10 | axis ratio, p10 | centre move p90 / max | smallest seed major |
|---|---|---|---|---|
| `135533` | -3.7% | -0.063 | 6.8 / 108.7 px | 252 px |
| `131552` | **-11.2%** | **-0.264** | **55 / 466 px** | **14 px** |

A 14 px major on a ~400 px ring. The objective's capture radius is ~25 px, so from there
the fit cannot descend back onto the rim; it wanders, and 466 px of wandering is where
that flight's discrepancy tail and its 11.7 degrees of frame-to-frame jitter come from.

**Bounding the drift does not fix it and was removed** -- see `RING_MAX_DRIFT` in
`segment.py`, which now records the sweep as a negative result. The reason is worth
keeping: a large move is not a wrong move. A seed at 252 px on a 415 px ring needs a
+65% correction and that is the fit doing its job, so drift cannot separate recovery from
runaway, and falling back to the seed is the worse report anyway because its axis ratio
bias lands straight on the tilt. Preferring the tracked fit over a collapsed cold one by
`ridge` was also tried, and measured neutral (p90 26.58 to 26.59): when a re-acquire
fires the tracked fit is already weak, so the cold fit usually wins that comparison
honestly.

What does help is temporal, and it is a filter rather than a fit. `pose/demo_video.py`
was drawing the raw per-frame estimate where `live_viz` runs `filter.PoseFilter`, so the
overlay looked far worse than anything downstream ever sees:

| frame-to-frame angle p90 | raw | filtered |
|---|---|---|
| `131552` | 11.70 deg | **4.30** |
| `135533` | 2.24 deg | **1.89** |

Both now show the same signal. The ellipses stay per-frame and unfiltered on purpose --
the point of the overlay is to see which frames are weak, and smoothing those is exactly
what would hide them.

The open move, unmeasured: the two estimates are complementary rather than competing --
the seed is steady and biased, the fit is sharp and noisy -- so taking the centre from
one and the axis ratio from the other, or refusing to refine a seed whose own `ridge` is
already gone, are both worth a sweep.

### 16.24 Removing the per-view refinement

Watching all three flights decided it. The direct fit's advantage is on frames where the
scene has *other* contours for the mask to be confused by, and its disadvantage is on
occlusion -- which is the failure mode this rig actually has. Removed: `_view_candidates`
no longer refines each view against the evidence map, and the mask fit is what gets
reported.

The two-view joint solve stays. It is a different thing from the ellipse that was drawn
in green -- one world pose against both views' samples, which is what `union_coverage`
is measured from -- and it is what still uses `ring_weight`.

`RADIUS_BENCH_MM` moves 9.95 to **10.20** with it. The effective radius is a property of
which edge the measurement cuts (S16.7): 9.95 belonged to the darkness ridge, the rim's
centre-line, and the mask hulls its outer edge. That is not a detail -- at 9.95 only 34.7%
of `135533` clears the 5 mm gate, at 10.20 it is 95.3%.

| | poses | discrepancy p50 / p90 | under 5 mm | more than 30 deg out | ms/pair |
|---|---|---|---|---|---|
| `131552` before | 639/639 | 0.68 / 26.58 mm | 77.6% | **3.1%** | 87 |
| `131552` after | 639/639 | 1.15 / **2.60** | **95.1%** | 5.0% | **47** |
| `135533` before | 1013/1013 | 0.63 / **1.54** | 92.9% | **0.4%** | 103 |
| `135533` after | 1013/1013 | 1.21 / 2.70 | **95.3%** | 2.0% | **51** |
| `092117` before | 1365/1365 | 0.92 / 15.30 | 81.5% | **2.9%** | 167 |
| `092117` after | 1365/1365 | 1.78 / **4.22** | **91.0%** | 3.3% | **49** |

The trade is consistent and worth stating plainly: **position gets worse in the median
and much better in the tail**, orientation gets slightly worse everywhere, and the whole
thing runs at half the cost. The median column is the direct fit doing exactly what it
was built to do; the p90 column is it coming apart on the frames where the seed collapses
(S16.23), and the tail is what a controller feels. Two of the three flights more than
halve their p90 and one triples it.

Two smaller pieces went with it. The tracked-seed fallback was rebuilt without the direct
fit -- when segmentation finds nothing, the previous frame's ellipse seeds the joint
solve, which the robot's own speed justifies at 60 fps and which is worth the ten frames
a flight it recovers. And the re-arbitration count collapses (114 to 1 on `131552`),
because the window prior only fires on frames the discrepancy gate flags and there are
now far fewer of those -- the prior is still there and now has almost nothing to do.

### 16.25 A gate on the filter, not only on the estimator

`filter.PoseFilter` fused every measurement it was given. It already computed the
innovation `y` and its covariance `S = H P H' + R` and then discarded both, so the test
costs one 3x3 solve: refuse a measurement more than `GATE_SIGMA` from where the filter
expected it, in the innovation's own metric.

Refusing is not dropping. The state has been predicted forward already, so a refused
frame keeps the constant-velocity extrapolation -- which is the right answer for one
frame at 60 fps and the reason a velocity model is there at all. `MAX_GATED` bounds the
run: a manoeuvre the model did not anticipate would otherwise be refused forever, the
filter growing more confident in its own extrapolation while drifting away from the
robot. A gate that can lock on is worse than no gate, and `filter._check` is the
assertion that it cannot.

### 16.26 Where it stands

All three flights, `never_reject`, `RunningPlate`, the two-view image fit and the gated
window prior:

| | poses | discrepancy p50 / p90 | under 5 mm | more than 30 deg out | ms/pair |
|---|---|---|---|---|---|
| `2026-08-28_131552` | **639/639** | 1.15 / 2.60 mm | **95.1%** | 5.0% | 47 |
| `2026-08-28_135533` | **1013/1013** | 1.21 / 2.70 mm | **95.3%** | 2.0% | 51 |
| `2026-08-28_092117` | **1365/1365** | 1.78 / 4.22 mm | **91.0%** | 3.3% | 49 |

Every frame of every flight carries a pose, better than nine in ten of them come under
the 5 mm gate against 77-93% before, and the whole pipeline runs at roughly half its
former cost -- 47-51 ms a stereo pair against 75-167. It is still three times the 16.7 ms
period at 60 fps, so this remains a replay and viser path rather than a control one, but
the gap is now one optimisation rather than three.

The largest single contributor was none of the algorithms: `THRESH` had been carried over
from renders with a comment explaining that it had been considered and left alone, and it
was cutting through a quarter of the rim. The second was `RADIUS_BENCH_MM` describing an
edge the pipeline had stopped measuring. Both were constants with reasons attached, and
the reasons had expired.

### 16.27 The silhouette shortcut, measured live

Proposed 2026-09-04: skip the conic entirely. Threshold, keep the largest blob, take its
second-moment axes, fit a box along them, and read the centre off the box's midlines, the
diameter off its long side, and the tilt off `arccos(minor/major)`. It is a reasonable
thing to want -- it is `body_angle.py`'s primitive (20) applied to the rim robot -- and it
is worth recording exactly where it lands, because two thirds of it are right.

Measured on 60 live pairs with the robot stationary, against the *pipeline's own 3-D pose
projected back into the same camera*, so the comparison is like for like:

| | blob box vs the rim the pose says |
| --- | --- |
| major axis | **+1.0%** |
| minor axis | **+69.6%** |
| centre | **30.3 px = 3.15 mm** (the live noise floor is 0.10-0.27 mm) |
| tilt `arccos(minor/major)` | 32.3 deg against 59.8 -- **27 deg out** |
| frame-to-frame steadiness | **12.9x steadier** (sd 0.071 deg against 0.908) |

**The major axis is right and the minor is not, and the asymmetry is the whole story.**
16.11 measured a hull fit of the *rim* inflating the minor by 6.9% against the direct fit,
because the mast and magnet protrude along the rotor axis and under tilt push the
silhouette outward in the short direction. A raw threshold blob does not fit the rim at all
-- it fits the whole bright silhouette, mast included -- so the same effect arrives ten
times larger. The overlay shows it plainly: the box is dragged up and rotated by the mast.

So `arccos(minor/major)` is not measuring the rim's foreshortening, it is measuring the
silhouette's aspect ratio, and those are the same number only for a bare disc. Even on a
*perfect* projected circle the formula is the orthographic reading of a perspective image:
synthetically it is 8-11 deg out at 30 mm off the optical axis, because it measures tilt
about the line of sight rather than about anything fixed. `conic.backproject_ellipse`
inverts the same ellipse **exactly** -- 1e-12 mm and 1e-6 deg round-trip at every position
tested -- and costs 0.0033 ms in C++. There is nothing to buy by approximating it.

**The steadiness is real and is the part worth keeping.** 16.23 ends by noting the mask
seed and the direct fit are complementary rather than competing -- "the seed is steady and
biased, the fit is sharp and noisy" -- and proposes taking the centre from one and the axis
ratio from the other. This measurement says which way round: take the **major axis and its
angle** from the blob, where it is accurate to 1% and thirteen times steadier, and take
nothing else. The centre and the minor are exactly the two quantities the mast corrupts.

**On speed, the instinct was right and the target was wrong.** Threshold plus largest
component is 0.312 ms a view against the pipeline's 2.658 ms a pair, and segmentation is
82% of the live budget (22.5) -- so that *is* where the remaining time is. But it buys the
speed by segmenting the silhouette instead of the rim, which is the trade 16.1 already
priced: the level is being asked "is this pixel rim?" when the only answerable question is
"does this ellipse lie on the rim?". Measured live the background is not black enough for
it either: camera B's p90 is 146 against camera A's 67, its largest blob at threshold 120
is 37,734 px against A's 10,405, and only a narrow 140-180 window gives both cameras a
plausible disc at all.

The ambiguity half of the proposal -- default upright and flip whenever both cameras see
the disc face-on -- inverts 16.13. `sign(n . v)` can only change through `n . v = 0`, which
is **edge-on**; face-on is `|n . v| = 1`, the state furthest from a flip. And the mirror
pair, which is the ambiguity that actually bites every frame, is not the same question:
both branches share `|n . v|` exactly, so no amount of flip bookkeeping resolves it, while
two views 84 deg apart settle it geometrically for free.

### 16.29 Scoring a fit by smoothness, and what that found

2026-09-04. Every accuracy number in this chapter is a *static* one -- cross-view
discrepancy, residual rms, coverage, error against a rendered truth. None of them can see
an estimator that is right on average and jumps between frames, and the operator watching
an overlay sees exactly that and nothing else.

The prior that makes jitter measurable is that **real motion is continuous**. A trajectory
sampled at 100+ Hz has small acceleration; per-frame estimation noise does not. So the
score is the **second difference** of the trajectory -- position in mm, normal angle in deg
-- and the first difference is reported but never scored, because it contains the real
motion.

**Jitter alone is a trap and the guards are not optional.** It is minimised by an estimator
that has stopped listening to the image: 16.23 measures the mask seed as six times steadier
than the direct fit *and* biased. So every score carries `discrepancy_mm`, `refine_rms_px`,
`union_coverage` and the solve count, and a configuration that smooths by losing frames,
drifting off the rim or disagreeing across views is rejected however smooth it looks.

Swept over the bench take: ring sample count, evidence blur, ROI margin, Jacobian gradient
step, window length, iteration cap, and the stopping tolerance. **Everything was neutral or
worse except the tolerance**, which was worth a factor of two to eight on its own. Blur
looked good on jitter (0.56x) and was rejected on the guard it moved: discrepancy 1.46 ->
2.94 mm.

`REFINE_TOL_ANALYTIC`'s own comment carries the table. The shape of the finding is the part
worth keeping here: **19.12 chose that constant for speed, checked it against rms, coverage
and discrepancy, found all three unharmed, and shipped it -- and none of those three is a
measure of smoothness.** The constant was not wrong on the evidence gathered; the evidence
had a hole in it, and the hole was the thing the operator could see.

It also put a number on 21.3's warning. Below 5e-4 the solve runs past the noise floor of
its own Jacobian -- the two normal columns of `_rim_shape` are forward-differenced at
sqrt(eps) -- and the trust region's accept/reject decisions begin turning on rounding, so
the two cores diverge (p95 still 1e-4 mm, worst frame 0.18, `nfev` differing by 8) and
`native_parity` fails. **The jitter floor and the parity floor are the same floor**, and
exact derivatives for those two columns is the single change that would move both.

### 16.28 Modelling the wall instead of calibrating it away

Proposed 2026-09-04: make the fit *normal-focused*, so it handles side-on views better and
counters the disc's thickness.

Half of that is already here and half is not, and the split is worth being precise about.
`refine(mode="image")` **is** normal-parameterised -- it solves centre(3) plus normal(2)
directly against the evidence map -- and the axial one-sidedness of 12.4/12.5 is already
carried by `axial_weights` and `_outward_weights`, which 16.11 describes as suppressing the
contamination without eliminating it. What is *not* modelled is the thickness:
`stereo._rim_shape` is `R (cos(phi) u + sin(phi) v)`, a **zero-thickness circle**. The wall
is handled by calibrating the effective radius to whichever fit is in use (16.7, 16.11) -- a
scalar standing in for a geometric fact.

Measured from the mesh, the rim wall is **1.329 mm on a 10.205 mm radius, 13%**. At 60 deg
that projects to `T sin(theta) = 1.15 mm ~ 11 px`, which is 16.11's otherwise unexplained
10.4-12.7 px gap between the mask fit and the direct fit, to within the measurement. **The
wall is that gap.**

Fitting the real mesh's convex hull -- what the segmenter produces (12.4) -- with the
thin-circle model, against the same model plus the wall, thickness taken from the mesh
rather than fitted:

| source | tilt | thin: dC / dnormal | **thick**: dC / dnormal |
| --- | --- | --- | --- |
| rim only | 20 deg | 0.958 mm / 3.39 deg | 1.043 / **2.30** |
| rim only | 40 | 1.765 / 2.55 | **0.341 / 0.39** |
| rim only | 58.9 | 1.294 / 3.47 | **0.581 / 0.59** |
| rim only | 70 | 0.647 / 3.71 | **0.232 / 0.27** |
| full mesh | 58.9 | 1.375 / 3.64 | **0.399 / 0.42** |
| full mesh | 70 | 2.575 / 4.90 | **1.073 / 1.83** |

**One parameter, not fitted, takes 2.5-3.7 deg of normal error to 0.3-0.6 across 40-70 deg.**
It does least at 20 deg, as it must: the effect scales with `sin(theta)`.

Three qualifications, the first of which stops this being a shipped result.

**The `thin` column is not the shipped pipeline.** It is a hull fit at a nominal radius,
where the live path fits the evidence *ridge* at an effective radius already calibrated to
absorb this very bias, plus the axial and one-sided weights. So the table measures what
thickness costs an *uncorrected* fit, not what modelling it would buy over the pipeline as
it stands. That comparison has not been run, and it is the one to run next.

**It fixes the wall, not the mast.** Rim-only at 70 deg lands at 0.232 mm; the full mesh at
the same tilt lands at 1.073, and the difference is the mast and magnet -- the 12.3b
contamination no rim model contains. Thickness is one of the two contaminants.

**It does not help the reflection.** The wall's mass is very nearly symmetric about the rim
plane (centroid -0.025 mm of a -0.467..+0.861 span), so a thick disc is still mirror
symmetric and the two-fold ambiguity is untouched. Side-on it buys *accuracy*, measurably,
and no disambiguation; 16.12's bulge remains the only monocular cue.

What makes it structurally attractive rather than merely better is 12.5's last table. The
centre displacement is almost entirely along the minor axis and **its sign flips near 57
deg** -- the wall pushes one way below it, the mast pulls the other above -- so *no monotone
function of tilt can correct both*, which is exactly what `CentreCalibration` is. A
geometric model does not have that problem, because it computes the silhouette instead of
fitting a curve to its displacement. The live rig currently sits at **58.9 deg**, within two
degrees of that sign flip.


## 18. The static noise model: measuring the scatter instead of predicting it

§13 derives what the error *cannot* go below and §16 measures what the pipeline
achieves against rendered ground truth. Neither is a measurement of the noise this
bench actually has. Every $\sigma$ the running pipeline consumed until now was one
or the other: `filter.py` took 0.13 mm lateral and 0.36% of range in depth from a
held-out split of *renders*, `stereo.py` took its fusion weights straight from the
Cramér–Rao floor of §13, and `control/z_track.py` priced its step-out threshold
against an assumed 0.5 mm per frame that nothing had ever measured.

A stationary robot settles all of them with one experiment. If the truth does not
move, everything the estimator reports that does is error. No renderer, no ground
truth, no analytic pose: the scatter *is* the noise.

### 18.1 Why this is not simply a standard deviation

The obvious procedure -- hold the robot still, take $\sigma$ of the reported
positions, put it in the filter -- is wrong here, and §5's own module docstring
says why before this section existed: depth autocorrelates at $r = 0.966$ from one
frame to the next and stays above $0.5$ for 408 ms.

Model the residual as an AR(1) process, which is the least structure that has a
correlation time at all:

$$x_t = \rho\, x_{t-1} + e_t, \qquad e_t \sim \mathcal N\!\left(0,\ \sigma^2 (1-\rho^2)\right)$$

so that the *marginal* variance is $\sigma^2$ whatever $\rho$ is. Three consequences
follow, and each is a number the calibration has to report separately.

**The correlation time.** The envelope decays as $e^{-t/\tau}$ with

$$\tau = -\frac{\Delta t}{\ln \rho}$$

At 60 fps and $\rho = 0.966$ that is 480 ms -- far longer than any control interval,
which is what makes it a bias rather than a noise.

**The effective sample count.** For estimating a mean, $N$ correlated samples are
worth

$$N_{\text{eff}} = N\,\frac{1-\rho}{1+\rho}$$

At $\rho = 0.966$, a thousand frames are worth seventeen. This is the error bar on
$\sigma$ itself, and without it a long static run looks far more authoritative than
it is.

**The white part.** Differencing kills drift of any shape, and for a white series
$\operatorname{Var}(x_t - x_{t-1}) = 2\sigma^2$ exactly. So

$$\sigma_{\text{white}} = \frac{\operatorname{MAD}(\Delta x)\times 1.4826}{\sqrt 2}$$

answers the question the total scatter cannot: **how much of this could averaging
ever remove?** For a white channel $\sigma_{\text{white}} = \sigma$; for the depth
channel it is a small fraction of it.

MAD before standard deviation throughout, and both reported. A static run still
drops the occasional blunder frame (§16.23), one of those moves a standard
deviation and not a median, and their ratio is the same contamination figure §13.7
tabulates for the boundary.

### 18.2 Which $\sigma$ is the Kalman $R$

A Kalman filter assumes $v_k \sim \mathcal N(0, R)$ **independent between frames**.
That assumption is false here, so neither candidate is correct and the question is
only which way to be wrong.

| choice | what the filter then believes | failure |
|---|---|---|
| $R = \sigma_{\text{white}}^2$ | each frame is nearly independent | it averages, the bias does not average, and the track becomes *confidently* wrong -- the gate of §16.25 tightens around a wrong centre |
| $R = \sigma_{\text{total}}^2$ | each frame is worth less than it is | slower to converge, wider gate, no false confidence |

**The total scatter ships as $R$.** Under-trusting a correlated measurement costs
convergence rate; over-trusting one costs correctness, and the whole point of §5's
"take position raw, velocity from the filter" advice is that the correlated part
cannot be filtered away. $\sigma_{\text{white}}$ is reported beside it because it
is what sets the achievable *rate* noise, which is the quantity control actually
consumes (§18.5).

### 18.3 Depth is a fraction of range, and one station cannot show it

From §12, $z = 2fR/M$, so a boundary error $e$ propagates as

$$\mathrm{d}z = -\frac{z}{M}\,\mathrm{d}M \;\propto\; \frac{z^2 e}{2fR},
\qquad \mathrm{d}x = \frac{z}{f}\,\mathrm{d}u \;\propto\; z e$$

Depth error goes as $z^2$ and lateral as $z$: the ratio itself grows with range.
The shipped model in `filter.py` is depth $\propto z$ with lateral flat, which is a
linearisation of that around one operating point, not the law.

Two things follow. A single static station cannot separate a fraction of range from
a constant offset at all -- one measurement, two unknowns -- so the calibration
takes **three or four heights across the envelope**. And the fitted exponents are
worth reporting even though the shipped shape stays put: $\sigma = c\,z^p$ fitted in
log space gives $p$, and a $p$ far from the derived one says the shape is wrong. The
model records both, and ships the shape its consumers already speak.

### 18.4 Inverting the fusion, without an optimiser

`filter.py` built $R$ as $(\sigma_{\text{lat}}, \sigma_{\text{lat}}, \sigma_{\text{depth}})$
on the world axes. That is the *monocular* geometry. This rig's cameras sit 90°
apart in azimuth (ch. 2), and `rig.position_covariance` combines their anisotropic
per-view covariances in information form:

$$\Sigma_{\text{fused}} = \left[\sum_i \left(\sigma_{\text{lat}}^2 I +
(\sigma_{\text{depth}}^2 - \sigma_{\text{lat}}^2)\, d_i d_i^{\!\top}\right)^{-1}\right]^{-1}$$

which is not diagonal in world coordinates for any pair of $d_i$ that are not axis
aligned. On the shipped rig the $x$–$z$ term is real and not small. So the filter
now takes the full $3\times3$ when a rig is available, and the triple only on the
monocular path.

Going the other way -- measured fused scatter to per-view scales -- needs no
optimiser. The map is homogeneous of degree one,

$$\Sigma_{\text{fused}}(k\sigma_{\text{lat}}, k\sigma_{\text{depth}}) = k^2\,
\Sigma_{\text{fused}}(\sigma_{\text{lat}}, \sigma_{\text{depth}})$$

so the *shape* of the eigenvalue triple fixes the ratio and one scalar then fixes
the size. A two-parameter fit becomes two one-parameter ones, neither of which can
fail to converge. A scan rather than a solver, because nobody has shown the shape
error is unimodal, and because a scan can be asked whether it stopped at its own
edge -- which is how an anisotropy no rig can produce gets reported instead of
fitted.

### 18.5 What the rate filters can and cannot remove

Control consumes rate, not position, and rate comes from a first difference through
a one-pole low-pass: `z_track.TAU_ZDOT_S`, `predictor.TAU_VEL_S`, and
`simulate_hover.VelocityEstimator`. With $a = \Delta t/(\tau + \Delta t)$ and a
white position component $\sigma_w$,

$$\sigma_v = \frac{\sigma_w \sqrt 2}{\Delta t}\sqrt{\frac{a}{2-a}}$$

and it is $\sigma_w$ that appears, **not** $\sigma_{\text{total}}$. The correlated
part of the position error is common to both ends of the difference and cancels out
of it exactly. This is the useful half of §18.1's bad news: the drift that no filter
removes from position never reaches the rate in the first place.

It also bounds what $\tau$ is worth. Raising $\tau$ past the measured $\tau_{\text{corr}}$
starts averaging over samples that are not independent, so it buys lag and almost no
noise. An 80 ms low-pass against a 400 ms correlation time is inside that regime.

### 18.6 What does not work

| tried | result |
|---|---|
| Standard deviation of a static run as $R$ | **Right number, wrong quantity.** It is the total wander, most of which is bias. Correct as a conservative $R$ (§18.2), wrong as "the noise", and useless for predicting rate noise. |
| One station, at hover height | **Underdetermined.** Depth $\sigma$ is a fraction of range; one height cannot separate the fraction from a constant. |
| Fitting a two-parameter line $\sigma_{\text{depth}} = a + bz$ | **Fitted then discarded.** The shipped consumers have no slot for an intercept, so it would be estimated and silently dropped. The exponent in §18.3 is where a bad shape is meant to show. |
| Per-view scales from a single fused number | **Not possible.** The ratio is a property of where the cameras are; the fused scatter constrains shape and size together, and needs the rig to be separated. |
| Recording with the coils energised | **Not done here, and it matters.** The shipped calibration is coils-off, so it is the *vision* noise floor and excludes drive-induced vibration and EMI. The artifact records `condition` so a coils-on set can be taken later and differenced against this one. |
| An LQG observer in `control/` | **Redundant.** The full state is already measured and filtered in `filter.py`; §11 of ch. 4 says no further observer is needed. The measured noise goes into the gain file as provenance, not as a second Kalman gain. |

### 18.7 Correspondence with the implementation

| Model element | Code |
|---|---|
| Robust scatter, $1.4826\times$MAD (§18.1) | `noise._robust_sigma` |
| $\rho_1$, $\tau$, $N_{\text{eff}}$, $\sigma_{\text{white}}$ (§18.1) | `noise._channel` |
| One station, and the fused $3\times3$ (§18.1) | `noise.measure` |
| Total scatter ships as $R$ (§18.2) | `noise.NoiseModel.sigma_pos`, `filter.PoseFilter.update` |
| Depth as a fraction of range, exponents (§18.3) | `noise.NoiseModel.fit`, `noise._powerlaw` |
| Inverting the fusion (§18.4) | `noise.per_view_scales`, against `rig.position_covariance` |
| Full $3\times3$ $R$ on the stereo path (§18.4) | `filter._ConstantVelocity.update`, `filter.PoseFilter(rig=...)` |
| Rate noise from $\sigma_{\text{white}}$ (§18.5) | `noise.NoiseModel.velocity_sigma_mm_s`, `.tau_for_velocity_sigma` |
| Step-out threshold, checked not asserted (§18.5) | `control/z_track.check_stepout_margin`, asserted in its `demo()` |
| Capture, one take per height | `camera/record.py --out results/static`, `noise.station_from_recording` |
| The artifact, and the fallback when it is absent | `pose/noise_model.json`, `noise.NoiseModel.load`, announced in `live_viz._stereo_estimator` |

The measured numbers are not in this document because the recording has not been
taken yet. `python pose/noise.py --show` prints them against the §13 floors once it
has, and the artifact carries its own provenance -- condition, station count,
sample count, date -- so a number read here can always be traced to the run that
produced it. **Until then the pipeline runs on the rendered fallbacks and says so
on every start**, which is the state this section exists to make visible rather
than comfortable.

## 19. Making the pipeline fast enough to close a loop around

2026-08-30. Section 16 ends by noting the joint image-mode solve is "fine for replay, not
yet for the live loop" at 47-51 ms a pair. This is how that gap was closed to 15.4 ms
(20.8 -> 64.8 Hz on `results/flights/2026-08-29_231418`, 246 of 250 frames solved at every
step, so none of it was bought by dropping frames). The per-step table and the rejected
options are in `control/theory.md` 19.2 and 19.5, with the control-side consequences; what
belongs here is why two of the wins were sitting in this directory unclaimed.

### 19.1 A cache keyed on the one thing that changes every frame

`ring_weight` caches the plate's 41x41 opening -- 2.6 ms a view -- under the comment "the
plate does not change ... the estimator passes the same array every frame". Neither half
held. `RunningPlate.update` returns `self.bg.astype(np.uint8)`, a **fresh array every
call**, and a `RunningPlate` is the live default. Keyed on `id(img)`, the cache therefore
missed 100% of the time and paid in full the cost it existed to remove, while holding a
reference to every dead plate so the ids could not be recycled.

The tempting repair -- return a stable buffer from `update` -- is worse than the bug. A
running plate *does* change: it walks a count a frame. A stable id would have pinned a
response to a plate that had moved, silently. The fix has to version the content, not the
buffer, so the key is now the plate's own frame counter over `PLATE_REFRESH_FRAMES`.

**The general lesson is the one this file keeps relearning: a cache whose invalidation
premise is written in a comment rather than checked in code is a measurement waiting to
be wrong.** Nothing failed. It was 5 ms a pair, every pair, for as long as the running
plate has been the default.

### 19.2 The ROI was written, documented, measured, and never called

`ring_weight` has taken an `roi` argument since it was written, with `_clamp_roi` behind
it and a measured "0.37 ms on 450x450 against 2.64 ms full-frame" in its own docstring.
A repo-wide search found no caller passing one. Meanwhile `StereoPoseEstimator._prev_ellipse`
was already carrying the previous frame's ellipse per camera -- used only as a fallback
seed when segmentation returned nothing, never to say where to look.

So the live loop segmented the full frame twice a frame to find a rim whose position it
already knew to a few pixels. The window is the previous ellipse's major axis times
`ROI_MARGIN = 1.6`, squared rather than fitted: the rim rotates between frames and a box
that hugs the minor axis clips it when it does. No previous ellipse, or a failed solve,
falls back to the whole frame -- **a tracker that cannot re-acquire is worse than a slow
one**, and this is the only branch that guarantees re-acquisition.

### 19.3 Resolution: the prediction was right

13's fused-worst-axis table puts 1280x800 at 0.060 mm and 640x480 at 0.119, and 351 says
outright that resolution is nearly free to give up because the error is bias-dominated.
Measured against the same 246 frames, 640x400 moves the pose by a per-axis spread of
0.110 / 0.119 / 0.065 mm. **The 0.119 mm prediction landed on the nose.** That sits under
the 0.185-0.274 mm centre-displacement bias which was already the dominant term, and it
bought 23.2 -> 15.4 ms.

640x400 specifically, not 640x480: it is a true 0.5x decimation of the native mode, so
`fx, fy, cx, cy` halve exactly and the distortion coefficients are untouched.
`StereoPoseEstimator._match_scale` does that rescale off the frame it is handed, comparing
against the rig's own `image_size`, so the live loop and `from_recording` cannot disagree
about it and no caller has to remember. The pixel constants scale with it -- `RING_KSIZE`
and `MIN_BLOB_AREA_PX` are quoted at the calibration resolution and are silently wrong at
any other.

320x240 was measured and rejected: 6 Hz more, for 0.205/0.169 mm of bias, which is over
the floor rather than under it.

### 19.4 Two views, two threads -- and what that broke

`update` looped over the views serially. Every stage inside is cv2 or numpy, which release
the GIL, so a two-worker pool took segmentation 16.0 -> 8.9 ms. The pool is built once per
estimator, not per frame: at these rates thread creation would cost more than the work.

They share less than it first appears -- separate plates, separate `_prev_ellipse` keys,
separate response-cache keys -- but "separate keys" was not the same as "separate state".
`ring_weight`'s plate-response cache was one global dict with a `clear()` at four entries,
so one view could evict the other's live entry, and the response would then be recomputed
against a *later plate generation*. **The pose stopped being reproducible run to run, by up
to 0.34 mm**, with no crash and nothing obviously wrong. The eviction rule was what coupled
two callers that had been carefully given distinct keys. Each caller owns a slot now, with
no eviction at all. Full account in `control/theory.md` 19.7; the lesson that belongs here
is that **a cache is shared state even when its keys are not.**

### 19.5 Where the remaining time is

At 640x400 the pair costs 9.4 ms: 3.2 segmentation, 6.2 the joint solve. `refine` is the
pipeline now, and its cost is the *numerical* Jacobian -- five parameters differenced
two-point turn each reported evaluation into six.

**The arithmetic is not what is slow.** Project-and-distort on this rig is 22.9 us for 45
points, 40.9 for 180, 132.6 for 900: about 15 us of fixed per-call overhead plus ~120 ns a
point. So the solve is bound by the number of numpy and cv2 calls on small arrays, and the
lever is to make fewer, larger calls rather than faster ones. Three were taken:

- the five perturbations are evaluated in **one** pass over `5n` points instead of five
  passes over `n` (`evidence_many` plus a supplied `jac`), 7.6 -> 6.2 ms;
- three of the five move only the centre, which shifts the rim rigidly, so the
  normal-dependent part of `_rim_points` is cached -- exact, verified bit-identical over
  3000 random poses;
- `_tangent_basis` drops `np.cross`, whose axis bookkeeping dominates its own arithmetic
  for a 3-vector.

The same measurement is why **a GPU does not belong here**: 3 kB of points per call against
a 5-10 us launch, evidence maps that would have to be uploaded every frame, and a
sequential trust region with nothing to run wide. The one data-parallel part of the pipeline
is segmentation, and segmentation is now 3.2 ms of 9.4. Full argument in
`control/theory.md` 19.4.

An analytic Jacobian -- image gradient of the evidence map times the pixel-vs-pose
derivative -- removes the extra evaluations instead of batching them. **Done 2026-09-01;
the full result is `control/theory.md` 19.12.** It is not the speed win this paragraph
expected: at the shipped tolerance it is *slower* and *more accurate*, because scipy's
default finite-difference step is below the float32 resolution of the pixel coordinate it
perturbs and had been differencing rounding. On a synthetic scene the analytic Jacobian
recovers a planted pose to 0.030 mm where those forward differences land 2.33 mm out. The
speed arrives by re-tuning the stopping tolerance against the better gradient
(`REFINE_TOL_ANALYTIC = 1e-3`): 6.9 -> 4.1 ms with `refine_rms_px` and `union_coverage`
both improved and `discrepancy_mm` unmoved.

## 20. Segmenting a rotor that has no rim

The tilt-sweep robots (`control/theory.md` 23) are a propeller on a mast. The rim
extractor of 16 correctly finds nothing on them, so `pose/disc_pose.py` swaps only the
segmentation and leaves undistortion, `conic.backproject_ellipse` and `fuse` alone. Three
things went wrong in turn, each visible only once there was a video to look at.

### 20.1 A threshold takes the lit half

The disc is lit from one side. Above the seed level that separates it from the scene
(110 grey) only its lit half survives; the shadowed half reads 65-100 and the fitted
ellipse was a half-moon on every robot in both views. The fix is hysteresis: the largest
component above the seed level, grown into every component above a lower level that
touches it. The lower level is bounded below by the foam blocks (60-90), which touch the
disc in some views: 50 leaked into them on all three robots, 65 did not. Drone 3's rim is
a thin bright ring round a dark interior, which the 11 px opening deletes outright, so
roundish enclosed holes are filled before the opening. Only roundish: at rest the frozen
blades, mast and guy wires enclose a sliver (42x113, 51x133 px) that must not be filled,
while disc interiors ran 78x42 to 89x62.

### 20.2 The background plate is a low percentile, not an average

Take 205012 opens with a hand and a sheet across camera B, and a bright patch on the
foam wins the threshold whenever the robot is out of view. The obvious remedy, subtract
an average frame, fails here for a reason particular to this rig: the robot never leaves.
It sits on its mast for the whole take and spins for about 70% of it, so the mean and the
median at a disc pixel *are* the spinning disc (152-202 grey), and subtracting either
left the disc 0-2 grey above the background. Between frozen blades at rest the backdrop
shows through, so a per-pixel 10th percentile over 300 frames spread across the take is
the backdrop (19-38) everywhere but the hub. The brightest 20% of frames by mean grey are
dropped first, which is what keeps a hand across the lens (108-150 mean against 37) out
of the plate and does nothing on a clean take. On the difference image the thresholds
become 90 / 35.

### 20.3 What the plate cannot do

A hand is transient; no plate removes it, and it segments as a 100 000 px component with
a confident ellipse. Anything over 40 000 px is refused instead (the largest real disc,
drone 3 at 60 Hz, is 21 912). The ellipse's sign is still ambiguous -- rotor-up and
rotor-down project alike -- and `control/tilt_report.py` resolves it against the mast.

### 20.4 The mast is the rotor axis, and the estimator should know it

The disc normal and the mast direction are two readings of one physical axis, and until
2026-09-02 the estimator used only the first and the report compared them afterwards.
Two things follow from treating the mast as what it is.

First, the branch. The ellipse's two-fold ambiguity (16.13) puts the mirrored pair about
84 deg from the truth, and the two views agree with each other on the wrong branch as
readily as on the right one, so `stereo.match` settles it with a prior. The base class
uses a sliding median of recent normals (`_window_normal`), which needs five frames and,
once it has settled on the wrong branch, follows it. `disc_pose.DiscStereoEstimator`
overrides that prior with the mast triangulated from the same frame (`mast_world`),
and arbitrates every frame rather than only when the two centres disagree -- the disc's
centres are scaled by a rim radius it does not have (above), so their discrepancy says
nothing. On drone 1 the disc's hold scatter at 160 Hz was 16 deg median and 72 at p90
before this; those are branch flips, not motion.

Second, the estimate. With both readings in hand the report blends them by inverse
variance (`stereo.blend_normals`), at the scatter each showed about its hold mean on
drone 1 with the plate segmenter: disc 2.3-3.4 deg, mast 1.9-4.0, so 3 and 2. A frame
where they disagree by more than 15 deg is one of them being wrong -- frozen blades at
20 Hz (29 deg at p90), a blade outscoring the mast -- and is left out rather than drawn.
What the blend does not settle is a systematic 5 deg between the two, the same at every
frequency from 40 to 140 Hz. That is a bias in one channel -- the hub-to-bead line not
passing through the ellipse centre, or the ellipse's tilt calibration -- and it is
reported per frame as `agree_deg` until it is measured.

Two viewing geometries need separate treatment, and only one of them needs new code.
A view looking straight down the axis sees a near-circle whose *orientation* is
meaningless -- the major axis of a circle is whichever way the noise leans -- but whose
normal is still the axis, which is that camera's own optical axis; `stereo.fuse` already
weights each view's normal by $\sin^2$ of the tilt it sees, so that view contributes
almost nothing to tilt and could not mislead if it did. A view seeing the disc edge-on
is the opposite: the geometry is at its most sensitive and the segmentation at its
worst, since the opening erodes a sliver and the hub and mast are much of what is left.
Measured on drone 1 (13587 frames with a mast), disc-vs-mast runs 5.3-5.9 deg median
while the thinner view's minor/major is 0.25-0.50 and 15.5 under 0.25. The report
divides the disc's sigma by a quality $q$ that is 1 above 0.25, falls linearly to 0.12,
and floors at 0.05 -- at which point the mast carries the frame -- and the 15 deg
disagreement gate applies only to a disc worth believing ($q > 0.5$). Above 0.5 the
disagreement also rises (19-29 deg), but on this rig, which never sees the disc
face-on, those are frozen blades reading round, and the gate handles them.

The rod was being found in only 51% of drone 1's hold frames with both views, 84% with
at least one, and in every frame it was missed the rod and bead were plainly visible.
The cause was the finder searching only the disc's own connected component: the hub
between rod and disc is dim, so the rod is usually its own component at the threshold.
`find_mast` now searches every bright component within 50 px of the disc -- drone 3's rod is dark and only its bead passes the threshold, 30-50 px above the rim, while a wire that far out is still refused by the distance and direction gates -- and the gates that
keep blades and the guy wires out never depended on connectivity. And a single view is
no longer wasted. The image line of the rod and the lens centre span a plane the axis
must lie in (`disc_pose.mast_plane`), so with one view the report projects the disc
normal into that plane -- a hard constraint on one component, the disc keeping the other
-- under the same disagreement gate. Only with neither view is the frame disc-only.

The same `fused_axis` serves the live loop in `control/tilt_servo.py` (`control/theory.md` 24), and `DiscStereoEstimator.update` now takes the centre from the two views' undistorted ellipse centres by ray triangulation rather than from the rim radius: on 300 frames of `2026-09-01_210758` the position is identical to the digit at an assumed radius of 10.24 and 25 mm, with a ray gap of 1 mm median.

## 21. The pipeline in C++, and what holding it to the Python taught

Written 2026-09-03. Everything `StereoPoseEstimator.update` does per frame -- the evidence
map, the segmenter and its blob grouping, the three reweighted ellipse fits, undistortion,
the cone back-projection, `match`, `fuse`, the sliding-window prior, the gates, and the
image-mode `refine` with its analytic Jacobian and scipy's trust region -- now exists a
second time, in C++ (`controller/native`, the `pmw_pose` module). `stereo_native.py`
wraps it as a drop-in `StereoPoseEstimator`, `live_viz._stereo_estimator` picks it when
it is built, and every consumer downstream reads the same `StereoPose` it always did.

The decision to do this was taken knowing `control/theory.md` 19.14: pose latency is
not what limits the controller. What the port buys is measured in 21.4. What it *found*
is in 21.2 and 21.3, and that is most of the chapter, because a port that has to agree
with its reference to rounding is the most searching test the reference has ever had.

The rule that made it possible is the one `control/theory.md` 19.7 set: reproducibility
is the instrument. The C++ is a statement-for-statement port that keeps numpy's
evaluation order, and `native_parity.py` holds the two implementations to each other
stage by stage on recorded frames. Nothing in `controller/native` holds a tuning
constant: `stereo_native.native_config` reads every one from the Python module it lives
in and the C++ constructor refuses a missing key, so a number still has exactly one home.

### 21.1 Step zero: what the published segmentation numbers had measured

`control/theory.md` 19.2 warned that every replay benchmark ran with running plates, on
which `segment()` returned `None` on 100% of frames and the pose came from the tracked
ellipse seeded during the plate's warm-up. Before porting, that was re-measured with
every plate available for the 250-frame take:

| plate | built by | `segment()` | frames solved | why |
|---|---|---|---|---|
| running (`RunningPlate`) | the stream itself | `None`, every frame | **246** | tracked-ellipse fallback, as 19.2 says |
| median of the take | `background.for_flight` | `None` | 0 | `plate_holds_still_subject` fires: the robot is in the plate |
| 10th percentile of the take | `disc_pose.plates_for_flight` | `None` | 0 | same; the robot never leaves its spot |
| the saved rig plates | `background_{A,B}.png`, same day | **runs**, both views | **0** | view B's hull fit is 1.6% of its major against the 1.2% `MAX_FIT_RMS_REL` gate, every frame |

So the mask path has never produced a pose on this recording, and the running-plate
path -- the one that works -- acquires through `segment_ring` for the five warm-up frames
and tracks from there. Both cores now reproduce both behaviours exactly (the segmenter is
bit-identical on the saved plates, 21.3), which is the honest statement of where the
segmenter stands; making it segment this take is a separate job and was not attempted.
The reference numbers below are therefore the running-plate path at `scale=0.5`, twice,
bit-identical run to run: **246/250, 3.2 ms segment + 4.1 ms solve, 8.8 ms wall,
113 Hz** through the Python estimator. (The 19.12 table's 3.2 + 4.1 ms was this same
number; its CLAUDE.md bench command lacked `scale=0.5` and would have run at 1280x800.)

### 21.2 Three things the reference was doing that nobody had written down

Each of these showed up as a parity failure and turned out to be the Python, not the port.

**The cv2 wheel is OpenCV 5, and its `remap` no longer quantises.** 19.12 explains the
analytic Jacobian's image gradient as a central difference *of the sampled field* because
`cv2.remap` reads bilinearly with coordinates quantised to 1/32 px (`INTER_TAB_SIZE`).
That was true of OpenCV 4. The wheel in `uv.lock` is `opencv-contrib-python 5.0.0.93`,
and measured against a hand-written bilinear read its `remap` on float maps is exact
(3e-6 rms against exact, 0.80 against the 1/32-quantised form), while Homebrew's 4.11
quantises (5.6e-6 against quantised, 0.80 against exact). The pipeline changed its
objective when the wheel moved, and nothing recorded it. The C++ does not call `remap`
at all: `sample_map` is the wheel's kernel written out -- a lerp of lerps, each a fused
multiply-add in float32, which is the only association that matches it bit for bit
(`evidence.cpp`). It is also what makes the port independent of which OpenCV it links.

**The Jacobian is a float32 computation and the residual is not.** `sample_map` returns
float32, and under NEP 50 a Python float is weak, so inside `jac_analytic` the gradient
`(g1 - g2) / (2 hg)`, the residual `sqrt(max(ref - g0, 0))` and the `live` mask are all
evaluated in float32 -- while `residual()` writes the same samples into a float64 array
first and is float64 throughout. A port that did the Jacobian in float64 agreed with the
reference on the residual to the last bit and disagreed on the Jacobian by one part in
10^3, because `-dE / (2r)` on a sample with `r ~ 0.04` amplifies a float32 rounding of
`ref` (76.5426147 -> 76.54261) by 1/r. The C++ mirrors the float32 arithmetic exactly
(`refine.cpp`). It is not a bug in the Python -- a Jacobian good to 1e-3 is plenty for a
trust region -- but it is a precision the design never chose.

**`cv2.fitEllipseDirect` differs between OpenCV builds by one float32 ulp, and the solve
turns that into 0.4 mm.** With everything else exact, the two estimators still disagreed
by 0.1-0.4 mm and 1-4 deg on ~5% of frames, with different iteration counts. Feeding the
Python side the C++'s undistorted ellipse collapsed that to 2e-6 mm at p95 with identical
iteration counts on all 246 frames; the whole difference was `undistort_ellipse`'s refit.
The undistorted *points* are identical between the two OpenCVs (their five-iteration
`undistortPoints` agrees to the last bit), but `fitEllipseDirect` returns a float32
`RotatedRect` and its internals moved between 4.11 and 5.0, so on 180 sub-pixel points
the two builds land on different sides of a float32 rounding on 59% of ellipses (177 of
300 random ones). That is 1.5e-5 px. The trust region, stopping at
`REFINE_TOL_ANALYTIC = 1e-3`, is sensitive to a seed perturbation of that size on the
frames where a trial step sits near the acceptance threshold. **`undistort_ellipse` now
refits with `conic.fit_conic_weighted` in double on both sides**, which either build
reproduces to 1e-10 px; the segmenter's other `fitEllipseDirect` calls, on integer-valued
hull points, agreed to 3e-12 across 500 views and were left alone. The reference moved by
a float32 ulp of ellipse, i.e. by less than it already moved with the last wheel upgrade.

A fourth, smaller one: `_rim_shape`'s two normal columns are forward-differenced at
`sqrt(eps)`, and clang fuses multiply-adds by default where numpy rounds every product.
That is a one-ulp difference in the rim points, which the 1.5e-8 step turns into 1e-5 of
those columns. The native build compiles with `-ffp-contract=off`.

### 21.3 Parity, stage by stage

`uv run python controller/pose/native_parity.py --stage all`, 250 frames of the bench
take at `scale=0.5`, running plates for the solve, the saved rig plates for the segmenter
(so it runs):

| stage | compared | agreement |
|---|---|---|
| evidence | `ring_weight` on the frames' own ROIs, 500 views; `sample_map` at 900 random points each | max 0.0 and 0.0 |
| segment | hull point sets, `area_px`, ellipse, 500 views | 0.0, 0, 2.9e-12 px |
| refine | the solve on the Python estimator's own captured inputs, 246 solves | `nfev` identical on all; centre p95 1.6e-6 mm, max 3.9e-4 |
| solve | both estimators end to end on identical frames, stamps and motion | same 246 frames; `discrepancy_mm` 6.5e-13; xyz p95 4.4e-7 mm, max 9.3e-3; angle p95 4.6e-6 deg |

The spread that remains -- a few times 1e-4 mm on the odd solve, with identical iteration
counts -- is the trust region's own sensitivity to the `sqrt(eps)` forward-difference
noise in the two normal columns, which no two IEEE implementations round identically.
That is the floor, and it is also a statement about the reference: at tolerance 1e-3 the
solve's answer is defined to about 1e-4 mm and no better, and the 0.4 mm swings of 21.2
show how far a slightly different seed can carry it within that tolerance. Exact
derivatives for those two columns would remove the noise on both sides; not done, because
it changes the reference's descent direction and the diff was already large.

Two state details the port had to reproduce to reach this: `ring_weight`'s plate-response
cache is keyed on the running plate's frame counter // `PLATE_REFRESH_FRAMES`, so the
plate's own top-hat is up to 30 frames stale by design (19.1); and with saved plates the
cache lives in one slot shared by both cameras and they evict each other every frame, so
those are never stale. The C++ carries the same cache, keyed the same way.

### 21.4 What it cost, and what it bought

The same bench as 21.1, both cores, two runs each, all four bit-identical run to run
(`results/bench/py_running_*.csv`, `native_*.csv`):

| core | segment | estimate | other | wall | rate |
|---|---|---|---|---|---|
| Python | 3.2 | 4.1 | 1.6 | 8.8 | 113 Hz |
| native | 2.4 | 0.4 | 2.4 | 5.3 | 189 |

The solve went 4.1 -> 0.4 ms: `control/theory.md` 19.4's diagnosis that it was bound by
the *number* of numpy and cv2 calls on small arrays, not by arithmetic, was exactly
right, and removing the calls removed the time. Segmentation barely moved, because it was
never Python: morphology, blur, connected components and contours are OpenCV in both
cores, Homebrew's build (TBB) is if anything a little slower per call than the wheel's
(GCD), and the ROI is too small for threads to help. `other` is mp4 decoding, the filter
and the viser push, which the live loop does not pay for the decode; the wrapper's own
cost -- two mask copies and a dict -- is under 0.2 ms.

Against 19.1's table, 8.8 -> 5.3 ms of pipeline is a fraction of a degree of
phase at the 0.78 Hz closed loop. 19.14 still stands: the loop is bound by `k_lat`, not
by this.

### 21.5 Correspondence with the implementation

- `controller/native/src/pmw.h` declares everything; one `.cpp` per Python module
  (`evidence`, `conic`, `segment`, `refine`, `stereo`) plus `trf.cpp`, scipy's
  `trf_no_bounds` for the exact solver with the Cauchy loss, ported call for call.
- `bind.cpp` is the only file that touches Python. Frames and plates cross as numpy
  buffers wrapped in `cv::Mat` headers, never as `cv::Mat` objects: the wheel and
  Homebrew's OpenCV are different builds with different ABIs and both live in the process.
- `stereo_native.NativeStereoPoseEstimator` subclasses the Python estimator so every
  configuration decision (`noise`, `centre_cal`, `error_model`, the rescale on the first
  frame) is made once, by the Python constructor, and refuses anything outside the live
  configuration with `NotImplementedError`. `_gate_predicted` was split out of `update` so
  the error-model gate is one function for both.
- `background.RunningPlate.update` takes its sign step from `pmw_pose.running_plate_update`
  when it is built, same float32 arithmetic, 0.57 -> 0.05 ms a view.
- `uv sync --extra native` builds it (scikit-build-core, nanobind, Homebrew `opencv`
  and `eigen`); without the extra nothing changes and the Python estimator runs.

## 22. Capture in C++, and buying observations from phase instead of pixels

Written 2026-09-04. The brief was a 300 Hz observation rate with the low-level camera
capture and the pose processing in C++. The pose half was already done (21). The capture
half turned out not to be the interesting part: the interesting part is that **300 Hz is
above what one of these cameras can deliver, and the way past it is the pairing, not the
language.**

### 22.1 The ceiling is the sensor, and nothing on the host moves it

Three measurements decide the whole shape of this chapter.

| what | measured | source |
| --- | --- | --- |
| ELP at 640x400, both cameras, real flights | **205-209 fps**, 0 dropped | five takes' `meta.json`, 2026-09-01 |
| the native pose core, one pair | **2.9-4.3 ms** | `results/bench/native_*.csv`, and 22.5 |
| the only ELP mode above 300 fps, 320x240 | 422 fps, but a **crop** | `elp_camera.json` |

The five flights are the load-bearing row. `fps_measured` of 204.96, 208.53, 208.50 and
208.02 with **`dropped` zero** is not a consumer figure: a drop-oldest slot that loses
nothing is a consumer that saw every frame the camera produced, so those numbers are the
*camera*. One ELP at 640x400 delivers about 208 fps and no arrangement of software asks
it for more.

**Downsampling does not help, and the reason is worth stating because it is the natural
first idea.** A 2x2 bin of a 640x400 frame is a real operation -- `_match_scale` already
rescales the intrinsics for any uniform factor, and 640x400 is itself an exact 0.5x of
the calibrated 1280x800 -- but it happens **after the USB transfer**. It buys compute,
and compute is not what is short: the pair already solves in 2.9 ms against a 4.8 ms
frame period. What it costs is in 19.3, which measured that exact downscale at 0.205 /
0.169 mm of per-axis bias against 640x400's 0.110 / 0.119, for 6 Hz. Twice the error to
relieve the one constraint that was not binding.

The 320x240 sensor mode is the same trade with an extra penalty: it is a **crop**, not a
rescale (NCC 0.836 against a resize, 0.9994 for 640x400), so it needs its own calibration
and sees less of the scene.

So the target as posed is unreachable at this resolution, and reachable at 320x240 only
by paying for it in the currency 13 says is already dominant. The rate had to come from
somewhere else.

### 22.2 Two cameras, uncorrelated phase, and an observation nobody was collecting

1.4 models the two free-running cameras as independent uniform phases on a period $T$,
which is why their skew is triangular on $[-T, T]$ with $\mathbb{E}|\Delta| = T/3$. That
model has always been treated as a *cost* -- the thing `fuse` has to correct for. Read the
other way it says something useful: **the two cameras' frames are interleaved in time,
and half of the instants at which the rig learns something new were being thrown away.**

`sources.StereoCamera.read` throws them away by construction. `MonoCamera.read` *consumes*
its slot and blocks until the next frame arrives, so a stereo read waits for a fresh frame
from **each** camera. A pair therefore costs the slower camera's period and the
observation rate can never exceed one camera's rate, however fast the pipeline behind it
runs.

Make the slot never-consumed -- always holding the newest frame, with `seq` rather than
emptiness saying "new" -- and a pose can fire on **every frame from either camera**,
paired with the other's most recent. At 208 fps a camera that is a period of 4.8 ms wide,
the partner view is then somewhere in $[0, T]$, mean $T/2 = 2.4$ ms old.

Three reasons that staleness is affordable, in decreasing order of how much they matter:

- **It is not new.** The live path runs `max_skew_s=None` (only `calib/capture.py` sets a
  limit). Today's pairs already carry a skew uniform on $[0, T]$; interleaving doubles how
  many such observations arrive without widening the distribution one bit.
- **`fuse` already prices it.** It takes per-view `stamps` and advances each view to the
  pair's mean instant, with the velocity and its covariance (17). That machinery was
  written for exactly this and has been running unused-for-its-purpose ever since.
- **The number is small.** 2.4 ms at hover's 15-22 mm/s is about 0.05 mm, against a
  centre-displacement bias of 0.185-0.274 mm that 19.3 says already dominates.

What it is *not* affordable to ignore is the failure mode a never-consumed slot
introduces and a consuming one cannot have. `MonoCamera.read` returns `None` after a 2 s
timeout, so a camera that stops delivering stops the loop. A slot that is never consumed
would instead pair its frozen last frame forever, producing a confident wrong pose stream
at full rate -- the worst shape a failure can take here. **`max_skew_s` is what replaces
the timeout**: a stopped camera makes the pair's skew grow without bound, every pair is
refused, poses stop, and the caller's existing "the camera was unplugged" path fires.
Defaulted to 1.5 frame periods, above the one period a healthy pair can reach and far
under anything a stopped camera produces. `tracker._self_check` asserts it fires and that
the refusal is *counted*, because a guard that drops frames silently is how a bench gets
misdiagnosed for a week.

**Consecutive interleaved observations share a view, so their errors are correlated** and
a filter that treats them as independent will slightly over-trust them. Not measured yet;
the honest statement is that the rate doubled and the information did not.

### 22.3 Not `videoio`, and the accident that turned out to be a better design

The plan was `cv::VideoCapture(CAP_AVFOUNDATION)` in C++: one dependency already linked,
no new code. It does not work on this machine, for a reason worth recording because it
will recur. Homebrew's `libopencv_videoio.4.11.0.dylib` links
`/opt/homebrew/opt/ffmpeg/lib/libavcodec.61.dylib`; the installed ffmpeg is **8.1.1**,
whose avcodec is **62**. The library does not load at all, so linking it would have broken
the import of a pose core that works.

**The repair is worse than the fault.** `brew reinstall opencv` installs **5.0.0** -- the
exact version bump 21.2 records as moving `remap` and `fitEllipseDirect`, the latter by a
float32 ulp that the trust region turns into 0.4 mm on 5% of frames. Pinning `ffmpeg@7`
means `install_name_tool` on a brew-managed dylib that any upgrade silently reverts. So
the capture went straight onto AVFoundation (`native/src/capture_avf.mm`), linking only OS
frameworks.

Measured first, in a throwaway probe, because the whole design rested on it: an
`AVCaptureSession` **creates, configures, starts and delivers frames entirely from a
worker thread with no `NSRunLoop`.** No main-thread requirement anywhere.

Having to write it turned out to buy three things the wrapper could not:

1. **A and B are resolved by identity rather than by probing.** `identify.py`'s whole
   index dance exists because "neither macOS listing enumerates in OpenCV's order" and a
   unique-id "cannot be tied to an OpenCV index", which is true of OpenCV and not of
   AVFoundation: its `uniqueID` is the string `system_profiler` reports. The rig already
   stores the calibrated pair in order (`elp_ids`), so **which camera is A becomes a fact
   about the calibration** instead of about probe order, and `identify.py`'s "as long as
   neither cable moves" caveat goes away.
2. **The format is chosen, not requested.** `AVCaptureDeviceFormat` enumerates exact
   size, fourcc and max rate up front, so `activeFormat` is *set* rather than asked for,
   and `open_camera` refuses outright rather than taking the nearest size.
   **This paragraph originally went on to say the silent size substitution of 1.2
   therefore "cannot happen". That was wrong, and 22.6 is what it cost to find out**: a
   session preset overrides `activeFormat` while leaving it reading correctly, so the
   check has to be on the delivered buffer and choosing the format is not by itself
   enough. Left here rather than quietly corrected, because the mistake is instructive --
   the guarantee was assumed from the API's shape instead of measured, which is the same
   error 19.1 records about a cache whose invalidation lived in a comment.
3. **The Y plane is the grayscale.** The sensor is monochrome (1.5), so for a biplanar 420
   buffer plane 0 *is* the image. Taking it is not a conversion but a projection onto the
   only channel carrying anything, which removes the `cvtColor(BGR2GRAY)` that
   `MonoCamera._grab_loop` pays per frame per camera.

One thing it cost, and it is the thing to weigh if this is ever revisited: `VideoCapture`
opens an mp4 as readily as a camera, which would have made the whole threading path
testable offline for free. `Tracker.push_frame` / `pump` buy that back deliberately --
the self-check drives slots, pairing, the staleness guard, the view cache and the solve
from the bench recording with no camera and no AVFoundation involved.

### 22.4 The view cache, and why "exact" is the only acceptable answer

At the interleaved rate one of the two views is unchanged on every call, so re-segmenting
it is pure waste. `Estimator::update` takes an optional frame `seq` per view and reuses
the cached `ViewResult` when it has not moved.

It is safe because `view_candidates` touches only `prev_ellipse[ci]` and
`plate_cache_[ci]`, both per-view, and writes `prev_ellipse[ci]` from **its own
segmentation** rather than from the refined pose. A view with no new frame has nothing to
advance; its evidence map is its own pixels, so reusing it is reuse and not staleness.

It is keyed on the grabber's sequence number and **never on the `Mat`'s data pointer**.
19.1 is the whole reason: `ring_weight`'s plate cache keyed on `id(img)` against an array
reallocated every frame, missed 100% of the time, and paid in full the cost it existed to
remove -- for as long as the running plate had been the default. Version the content, not
the buffer.

**A cache that changes the answer is a bug in the cache, not a speed-up**, so that is an
assertion rather than a claim. `tracker._self_check` runs two trackers on identical
pixels, one allowed to reuse a view and one forced to recompute it by pushing the same
frame again under a new seq, and requires the pose to agree **exactly** -- 0.0 mm, 0.0 in
the normal, not a tolerance.

The default is off (`seq = 0` recomputes), so `stereo_native`, `native_parity` and every
existing caller are untouched and 21.3's parity numbers still mean what they meant.

### 22.5 What it measured, on the bench

Replay first, `pump()` on the bench take, 240 pairs after a 10-frame plate warm-up, one
fresh view per step for interleave and two for `both`:

| pairing | fresh views | ms a pose | rate |
| --- | --- | --- | --- |
| `both` | 2 | 4.20-4.34 | 231-238 Hz |
| `interleave` | 1 | 2.49-2.68 | 373-401 Hz |

Both solve the same **246 of 250** in every run, which is the number that says none of the
rate was bought by losing the robot. (Those were taken on a machine at load average 17 and
repeat runs swung as far as 11.13 / 5.69 ms; the ratio held across all seven runs. They are
kept only for the ratio.)

**Live, 2026-09-04, both ELPs on a stationary robot** -- which is the measurement that
counts, because `pump()` times compute in isolation while the real worker competes with two
capture queues and the interpreter:

| | measured |
| --- | --- |
| frames grabbed, per camera | **197-200 fps** each, so ~395 arriving a second |
| poses, `pair_mode="interleave"` | **221-240 Hz**, 0 lost over 34,700 |
| segment / estimate a pose | **2.70 / 0.68 ms** = 3.31 ms, a **302 Hz** compute ceiling |
| skew | median **0.46 ms**, p90 4.2 |
| pairs refused on skew | ~14% under sustained load |
| position scatter, robot still | **0.10 / 0.16 / 0.27 mm** |
| through `stereo_frames(tracker=True)`, viser rendering | **239 Hz** sustained, 0 lost |

Three things follow, and the third is the one that matters.

**The rate roughly doubled**: the same core through `sources.py` ran the loop at 65-100 Hz.

**It is not 400 Hz, and the reason is the feedback loop between the worker and the cache.**
The view cache only pays when the worker is keeping up: at 395 frames/s against a 302 Hz
ceiling the worker is behind, so by the time it wakes *both* cameras have usually advanced,
both views recompute, and it pays the `both` cost -- which keeps it behind. The 14% skew
refusals are the same fact seen from the other side: a worker that lags pairs a fresh frame
with a partner more than one period old, and the guard correctly throws those away.

**Compute is now the binding constraint, at 302 Hz against 395 arriving.** That is a
reversal. For the whole of 19.x the pipeline was far below the camera and the question was
always how to make the solve cheaper; 21 answered it and this chapter's pairing change spent
the answer. Segmentation is **2.70 of the 3.31 ms, 82%** -- so the next real gain is there
and nowhere else, and 22.8 is about what that costs.

### 22.6 Three things macOS does to a capture that nothing warns you about

All three were found by measurement, and all three fail *silently* -- which is the reason
`Tracker::start` checks the delivered buffer rather than trusting any of the configuration.

**A session preset overrides `activeFormat`, and the format keeps reading correctly.**
Asking for 640x400 and configuring the device before adding the input delivered **1280x800
at 94 fps** while `dev.activeFormat` still reported 640x400. This is exactly 1.2's
undetectable size substitution -- every distance downstream wrong by a fixed factor -- and a
check on the *format* cannot see it. Only the delivered buffer can.

**`activeFormat` does not take until the session is running.** Measured across all four
orderings, asking 640x400 of the ELP:

| where `activeFormat` is set | delivered |
| --- | --- |
| inside `beginConfiguration` | 1280x800 @ 119 fps |
| after `commitConfiguration` | 1280x800 @ 118 |
| **after `startRunning`** | **640x400 @ 207** |
| `videoSettings` width/height keys | 640x400 @ 200, but **scaled** from 1280x800 |

The last row is the trap inside the trap: it produces the right *size* by resampling, not by
putting the sensor in its own mode, so it is a soft image at a rate that is not the mode's.
`AVCaptureSessionPresetInputPriority`, which states "the format is mine", is iOS-only.

**Each size is offered twice, at rates that differ by 7x.** `camera_formats` on the ELP:

| size | `yuvs` | `420v` |
| --- | --- | --- |
| 640x400 | 30 fps | **210** |
| 640x480 | 30 | 210 |
| 320x240 | 120 | 420 |
| 1280x800 | 10 | 120 |

Picking a format by size alone lands on the 30 fps variant seven times out of ten. This also
settles two of 1.3's three anomalies: 160x120 really does offer 640 fps, so the 285.4 fps
measured there was the consumer ceiling 1.3 suspected; and 640x400 caps at 210, so the
271.3 fps that "exceeds the asked rate" was the bad warm-up, not the sensor.

**`setSampleBufferDelegate:` keeps only a WEAK reference.** Passing a freshly allocated
delegate as a temporary lets ARC release it before the first frame arrives, and the session
then runs forever delivering nothing -- no error, no warning, just a camera that appears
dead. A diagnostic probe written that way reported "both cameras deliver zero frames" while
the cameras were in fact fine at 208 fps, which is a wrong conclusion drawn from a broken
instrument. `AvfCamera` holds its `PmwGrabber` as a member, so the shipped path is safe;
the trap is for anything written alongside it.

**`activeFormat` after `startRunning` is a RACE, and losing it looks like a scene fault.**
The ordering above is necessary but not sufficient: the call succeeds, `activeFormat` reads
back correctly, and the camera streams at the session's size anyway because the stream was
still coming up. Measured on this bench: one camera of the pair silently running 1280x800
while the other ran 640x400. The C++ estimator has **no `_match_scale`** -- its intrinsics
are fixed at construction -- so that view's ellipse is in the wrong pixel coordinates, the
two views' poses disagree, and `n_rejected`, the cross-view discrepancy gate, refuses every
frame. It presents as "the estimator suddenly stopped solving", which is what sent a whole
afternoon chasing lighting and running plates. `AvfCamera::start` now retries until the
*delivered* size matches (10/10 opens correct afterwards), and the grabber drops a
wrong-size frame rather than handing it to a solver that cannot know it is wrong
(`n_wrong_size`).

**The sensor cannot always fill the frame period, and the shortfall is silent.** In a dim
scene the ELP returns *empty buffers* at the requested rate rather than slowing down.
Measured on one scene, changing only the frame period:

| mode | period | empty frames, camera A / B |
| --- | --- | --- |
| 640x400 @ 210 | 4.76 ms | **53% / 40%** |
| 1280x800 @ 120 | 8.33 ms | **10.8% / 6.5%** |

Same light (frame mean ~19-21 either way), same code, same cameras. This is exposure time
against frame period, not bandwidth: at 210 fps the exposure available is 4.76 ms and the
scene did not fill it. For reference, this bench solving happily at 240 Hz earlier the same
day read **mean 59 / 82 with max 235**; at **mean 20 with max 170** it solved nothing.
**The rate the pipeline can run at is therefore a property of the lighting, not only of the
sensor** -- 1.3's mode table is an upper bound that assumes enough light to reach it.

**And an Objective-C exception is not a `std::exception`.** An `NSException` crossing into
C++ reaches Python as `exception could not be translated!` with the cause discarded. Every
AVFoundation call that can raise goes through `objc_guard`.

### 22.7 Addressing a camera by identity, and what it was hiding

`identify.py` finds the ELPs by opening each OpenCV index and checking the delivered size,
because "neither macOS listing enumerates in OpenCV's order" and a unique-id "cannot be tied
to an OpenCV index". Both are true of OpenCV and neither is true of AVFoundation, whose
`uniqueID` is exactly the string `system_profiler` reports. Confirmed on this bench:
`0x11000032e49281` and `0x12000032e49281`, matching the rig's `elp_ids` character for
character.

**AVFoundation lists them in the opposite order** -- `0x12...` first. So a tracker that took
the first two devices it found would have run with A and B swapped, which `identify.py`'s
own docstring describes as producing "poses that are smooth, plausible and wrong". The rig
records which camera is A; asking the rig is the fix, and it also retires the "as long as
neither cable moves" caveat that the probe order carried.

### 22.8 What the rate change found, and the one mistake behind half of it

Tripling the pose rate broke four things, and **three of them were the same mistake**: a
constant expressed in *frames* whose meaning is a *duration*. Every one of them was correct
at the rate it was written for and silently wrong at 220-400 Hz, and none of them failed
loudly. The general form is worth more than any of the three:

> **A per-frame constant is a duration in disguise, and this pipeline's frame rate moved.**

Anything counted in frames -- a window length, an adaptation step, a warm-up, a refresh
interval -- should be derived from the rate at construction rather than written down, and
`pose/tracker.py` now does that for the two that mattered.


**`WINDOW_FRAMES` is a duration written as a count.** `stereo.py:253` calls 15 frames "a
quarter second at 60 fps, matching `DROPOUT_S`", and `window_normal` does gate on
`DROPOUT_S` in *seconds* -- but the deque is trimmed by *count*. So the window is a
quarter second only at 60 Hz. It has been ~150 ms at the 100 Hz the pipeline actually
runs, and at 400 Hz it would be 36 ms, leaving the branch-flip prior with almost no
memory. `tracker.py` derives it from the rate; `stereo.py`'s constant is deliberately left
alone so nothing moves under `native_parity`. The reference's own drift is recorded here
rather than silently fixed, because fixing it changes answers on every existing path.

**The viser threshold slider was dead on the native core.** `live_viz` sets
`est.thresh = viz.thresh` every iteration, but `thresh` reaches C++ as `Config.level`,
read once at construction, and `_ensure_native` only rebuilds on a frame-scale change --
so the slider had done nothing since the native core became the default, on the live path
and on the replay path that exists to be tuned with it. Both now push it through per
frame. This is the same shape of fault as 19.1: a value whose propagation was assumed
rather than checked.

**`RunningPlate.step` is counts per frame and means counts per second.** (This one is a
real defect that was fixed; it is **not** the explanation for the failure that found it --
see the end of this entry.) Its own docstring
says so -- "``step`` is in counts per frame. At 1.0 and 60 fps the plate can follow a 60
count/s drift" -- and it ends with "it cannot see a robot that never moves ... lower `step`
buys time". At 220 Hz an unchanged `step = 1.0` walks the plate at **220 counts/s instead of
60**, so it buys 3.7x less time, on the exact failure the docstring names. Measured: a live
preview on a deliberately stationary robot ran about five minutes and then lost **2915
consecutive frames** while both cameras stayed fresh (4 ms and 1 ms old) and only 213 pairs
were refused on skew. `tracker.py` now sets `step = 60 / pose_rate`, which holds the plate
at the counts per second it was tuned for.

**And the fix did not fix the failure, which is the part worth recording.** The obvious
reading -- the plate adapts 3.7x too fast, so it absorbs the still robot 3.7x sooner -- is
wrong, or at least not sufficient. Corrected to 60 counts/s the preview ran **63,322 ticks
(260 s)** before the same cliff, against **73,724 ticks (306 s)** uncorrected. No meaningful
change.

Two observations that the plate story does not explain:

- **It is a cliff, not a slope.** `lost` sat at 4 -- the plate warm-up frames, and nothing
  since -- for the entire run, and then 2651 consecutive losses. A plate slowly walking onto
  a subject degrades the evidence gradually; this does not degrade at all and then stops.
- **The two runs died at different tick counts and different elapsed times**, so it is not
  a fixed counter either.

**The cause was the light.** Chased with the gate counters, the losses turned out to be
`n_rejected` -- the cross-view discrepancy gate -- firing on every frame: 180/s, then 622/s,
then everything. And the frame statistics say why. During the healthy preview the two views
had means of **59.4 and 82.0** with a max of 235; by the end of the session the same scene,
same cameras, same code, read **19.7 and 22.0** with maxima of 165 and 144. The bench
lighting fell about threefold while the investigation was running. A segmenter built on a
brightly lit rim against a dark room does not survive that, and the first view to lose the
rim disagrees with the second, which is exactly the gate that fired.

That also fits the post-mortem image, in which camera A was nearly black while camera B
still had signal -- one side of the rig losing its light first, not a camera fault. Both
cameras were measured healthy throughout: 208-210 fps standalone, 1026-1050 frames per five
seconds, frames 1-7 ms old at every failure.

**What is not established** is that the earlier degradations had the same cause. They were
not measured against frame statistics at the time, and by the time the instrumentation
existed the lighting had already moved, so there is no clean before-and-after. The honest
statement is that the failure reproduced at the end of the session is a lighting failure,
and the earlier ones are unexplained and were investigated with tools that could not have
told the difference. What it is not: the cameras (both delivering at full rate, frames 1-4 ms old at
the failure) or the skew guard (311 refusals against 2651 losses). Saved plates are not an
available control -- the ones on disk are dated six days before this test, mean 21.5/18.8
against a live scene at 43.7/58.2, which is the "silently wrong the moment the lighting
drifts" cost the class docstring opens with. The next measurement is the per-gate breakdown
(`n_rejected`, `n_rejected_fit`, `n_rejected_mono`) across the silence, which now reaches
`stats()`, plus the first and last frames side by side.

**A counter that could only ever read zero, and the wrong diagnosis it bought.** On the
tracker path `stereo_frames` kept `lost += pose is None` from the `sources.py` branch. It
is dead there: a lost frame never reaches that line, because `read` returns `None` and the
miss branch above it `continue`s. So `lost` sat at 0 while the estimator solved nothing,
and a five-minute live preview reported `lost 0` for 73,724 consecutive ticks right up to
the moment it stopped. The estimator's own counter is the true one.

That mattered because it made the *stop message* lie. The message differenced nothing: it
saw both cameras still delivering and asserted "the pairs are being refused on skew, so the
two have drifted apart" -- while the two slots were 2 and 3 ms fresh, a skew of about 1 ms,
nowhere near the 7.14 ms guard. The real cause was the estimator losing every frame, which
the message had no way to say and the zeroed counter had hidden.

Both are the same mistake in two places: **a number was reported without checking it could
move.** The message now differences `n_lost`, `n_skew_dropped` and `n_grabbed` across the
silence and names whichever one actually moved -- estimator, skew guard, stopped camera, or
none of the three, which points at the worker itself. `_why_no_pose` in `live_viz.py`.

The suspected cause of that particular silence is worth stating separately, because it is
not a bug in any of this: the preview ran `backgrounds="running"` on a robot that was
deliberately **stationary**, and `background.RunningPlate` walks itself onto anything that
stops moving -- its own docstring says so, and 21.1 measured it returning `None` on 100% of
frames for exactly that reason. A still robot is the one case the running plate cannot
survive, and a preview is the one place a robot is likely to sit still.

### 22.9 Correspondence with the implementation

| Model element | Code |
| --- | --- |
| Capture, AVFoundation directly (22.3) | `native/src/capture.h`, `capture_avf.mm` |
| Device by `uniqueID`, A/B from the rig (22.3) | `capture_avf.mm` `find_camera`, `pose/tracker.py` `camera_ids` |
| Exact format or refuse (22.3) | `capture_avf.mm` `best_format`, `open_camera`'s closing assert |
| Y plane as grayscale (22.3) | `capture_avf.mm` `gray_of` |
| Never-consumed slot, `seq` says new (22.2) | `native/src/tracker.h` `deliver`, `ready` |
| Interleaved vs both (22.2) | `tracker.h` `PairMode`, `pose/tracker.py` `pair_mode` |
| The staleness guard (22.2) | `tracker.h` `step`, `pose/tracker.py` `SKEW_LIMIT_PERIODS` |
| `RunningPlate` in the worker (22.2) | `tracker.h` `RunningPlate`, warmup and version semantics |
| The view cache and its exactness (22.4) | `Estimator::set_frame_seq`, `stereo.cpp` `update`, `tracker._self_check` |
| Velocity pushed in, not passed (22.2) | `tracker.h` `set_motion`, `pose/tracker.py` `set_motion` |
| Window as a duration (22.6) | `pose/tracker.py` `cfg["window_frames"]` |
| Threshold, live (22.6) | `Estimator::set_thresh`, `stereo_native.update`, `tracker.set_thresh` |
| One switch for the live loop (22.2) | `live_viz.stereo_frames(tracker=True)` |
| Format chosen by size AND rate (22.6) | `capture_avf.mm` `best_format`, `camera_formats` |
| `activeFormat` applied at start (22.6) | `capture_avf.mm` `AvfCamera::apply_format` |
| Delivered-buffer check (22.6) | `CameraSource::delivered_size`, `Tracker::start` |
| NSException never escapes (22.6) | `capture_avf.mm` `objc_guard` |
| A and B from the rig's ids (22.7) | `pose/tracker.py` `camera_ids`, `capture_avf.mm` `find_camera` |
| Which camera stopped (22.6) | `TrackerStats::age_ms`, `stereo_frames`'s stop message |
| Why the poses stopped (22.8) | `live_viz._why_no_pose`, differenced across the silence |
| Plate step held per-second (22.8) | `pose/tracker.py` `PLATE_STEP_REF_HZ`, `plate_step` |
