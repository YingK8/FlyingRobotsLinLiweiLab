# Chapter 2. Calibration: fixing the frames everything else assumes

*Stage 2 of the pipeline. Consumes: frames from [chapter 1](../camera/theory.md).
Produces: intrinsics, `stereo_rig.json`, the tilt and centre corrections, and the
datum. Consumed by: [chapter 3, pose](../pose/theory.md).*

Every gain the estimator claims is stated in some frame, and this stage is where
those frames come from. The chapter's real subject is not how to run a
calibration: it is **which errors the residual can see and which it structurally
cannot**, because the second kind are the ones that survive to become confident
wrong answers.

## Reading order

| # | file | what it does |
|---|---|---|
| 1 | `calibrate.py` | a bag of pairs -> `calibrate_intrinsics` -> `calibrate_extrinsics` |
| 1a | `capture.py` | the refusing shutter, writing one bag per session |
| 1b | `results.py`, `plots.py`, `sheet.py`, `test_calibrate.py` | the gate and rig file, the figures, the printable board, the checks |
| 1a | `calibrate_camera.ipynb` | the worked run and the plots, over `calibrate.py` |
| 2 | `rig.py` | `StereoRig`: where the cameras are, and how to carry a pose between frames |
| 3 | `shape.py` | the tilt/centre bias correction, and `APPEARANCE` |
| 4 | `zeroing.py` | `Zero`: report pose relative to a reference instead of the lens |
| 5 | `calibrate_shape.ipynb` | fits the corrections in (3) |

`rig.py` is the data model the rest of the pipeline reads; run `calibrate.py` to produce
it and `rig.StereoRig.load` to consume it.

§14 is the extrinsic itself; **§16 is how the frames that feed it are captured**, and is the
one to read when a calibration fails its gate for no visible reason.

## 2.0 Three calibrations, not one

They are independent and fail differently, which is why they are separate files:

| calibration | fixes | lives in | failure if wrong |
|---|---|---|---|
| **intrinsics** $K$, distortion | pixels -> rays | `assets/camera_intrinsics.npz` | every distance scales |
| **extrinsics** $T_{B \leftarrow A}$ | two cameras -> one world | `stereo_rig.json` | smooth, plausible, wrong triangulation |
| **shape** (tilt, centre) | flat-circle model -> real robot | `tilt_calibration*.json` | systematic tilt bias, ~3 degrees |

The first two are geometry and are checked by reprojection. The third is not
geometry at all, it corrects for the robot not being the flat circle
[chapter 3](../pose/theory.md) assumes, and no reprojection residual can see it.

**`APPEARANCE` lives here**, in `shape.py`, not with the segmenter that uses it
most. It is the key the calibration files are named by: each rig appearance
thresholds the boundary in a different channel, so each carries its own fitted
constants. `pose/segment.py` re-exports it.

## 14. Extrinsic calibration: fixing the frame the two cameras share

[§12.6](../pose/theory.md) (ch.3) assumes a shared frame: every gain it claims is stated in one. This section is how
that frame is obtained, and, more usefully, which of its errors the calibration residual can
see and which it structurally cannot. Implemented in
`controller/calib/calibrate.py`.

**A7: Rigid stereo mount.** The two cameras hold their relative pose across both
calibration and use. Nothing here estimates a time-varying extrinsic, and nothing downstream
would notice one drifting: [§12.6](../pose/theory.md) (ch.3)'s fusion would keep reporting a confident wrong pose. A
knock to the rig invalidates this section's output, not just degrades it.

### 14.1 Why the board's motion cancels

A ChArUco view gives each camera the board's pose in its own frame, $T_{A\leftarrow b}$ and
$T_{B\leftarrow b}$, for one physical placement $b$. Composing,

$$T_{B\leftarrow A} \;=\; T_{B\leftarrow b}\, T_{b\leftarrow A} \;=\; T_{B\leftarrow b}\,
T_{A\leftarrow b}^{-1}.$$

The board pose appears once forward and once inverted, so it cancels **identically**, not
approximately. Each placement is therefore an independent estimate of the same fixed
quantity, and the board should be *swept* through many poses rather than held still: the
spread across placements is a direct, assumption-free estimate of the extrinsic's
uncertainty, and an outlier identifies a bad view rather than a moved board.

The obvious alternative, hold the board **stationary**, average
$T_{\text{cam}\leftarrow b}$ per camera, and adopt the board as the world frame, was
implemented first and then removed. It is a valid estimator of a different thing, but it
requires the board fixed, it cannot separate a board-placement bias from an extrinsic error,
and it inherits whatever single placement happened to be chosen. Sweeping is strictly
better: more information, an uncertainty estimate for free, and one less thing to keep still.

**The world frame is camera A**, $T_{\text{world}\leftarrow A} = I$. Anchoring the world to
the *board* would tie the rig to a placement that exists only during calibration; anchoring
it to the *robot* would make the extrinsic depend on the thing being measured. Camera A is
the only datum that is both physically persistent and independent of the measurement.
Rebasing onto the robot's disk frame at rest is a separate, later step, taken when the
estimator is wired into visual servoing ([§11.6](../control/theory.md) (ch.4)): and it is a pure change of basis, so it
cannot disturb anything derived here.

### 14.2 Averaging rotations

Translations average componentwise; rotations do not. The elementwise mean of rotation
matrices is not a rotation, and projecting it back to $SO(3)$ discards exactly the accuracy
the averaging was for. The chordal mean

$$\bar R \;=\; \arg\min_{R \in SO(3)} \sum_i \lVert R - R_i \rVert_F^2$$

is used instead (`scipy.spatial.transform.Rotation.mean`), with dispersion reported as
$\max_i \lVert \log(R_i \bar R^{-1}) \rVert$ in degrees. Translation uses the **median**, not
the mean, so one bad placement moves the answer by nothing rather than by $1/N$ of its error.

### 14.3 Conditioning: incidence, and how far apart the cameras may be

Let $\theta$ be the board's incidence, $0$ face-on and $90°$ edge-on. The board's projected
extent perpendicular to the tilt axis is compressed by $\cos\theta$, so the corner
displacements that constrain rotation about that axis shrink with it and the angular
uncertainty goes as

$$\sigma_{\text{angle}} \;\sim\; \frac{\sigma_{\text{px}}}{D \cos\theta},$$

with $D$ the board's projected diagonal. The penalty is $1.5\times$ at $50°$, $2.9\times$ at
$70°$, $11\times$ at $85°$: which is why views past $70°$ are rejected outright rather than
downweighted (`MAX_INCIDENCE_DEG`).

This settles **open item 3** of `docs/pose_localization_project_context.md`. For an axis
separation $\Delta$, a flat board placed with its normal on the **bisector** of the two
optical axes is seen at $\Delta/2$ by both. $\Delta$ here is the **directed** angle between
the optical axes, and that distinction is not pedantry -- see §14.3d, where it was measured
the hard way. The single-board method is therefore viable while

$$\Delta \;<\; 2\theta_{\max} \;=\; 140°,$$

which covers the default $60°$ rig with enormous margin and a $90°$ rig comfortably. A
ChArUco cube is needed only past that, not at $90°$ as previously assumed.

This also corrects a belief carried in the earlier prose, that a mixed-hemisphere rig needs
a double-sided board. The binding constraint is $\Delta$, not which hemisphere each camera
occupies: both cameras sit on the same side of a bisector-oriented board whenever
$\Delta < 180°$, and the measured round trip below includes a $+45°/-45°$ pair. What does
fail is a board constrained to lie *horizontal* in the flight volume, which is the case that
warning was really about.

Measured, with $\pm18°$ of random 3-axis jitter about the bisector:

| rig | directed $\Delta$ | incidence at bisector | pairs accepted |
|---|---|---|---|
| $45°/45°$, azimuth $0/90$ | $60°$ | $30°$ | 24 / 24 |
| $45°/-45°$, azimuth $0/90$ | $120°$ | $60°$ | 0 / 24 |

The jitter budget, not the nominal geometry, is what runs out as $\Delta$ grows. The second
row was recorded here as $\Delta = 90°$ and 13 of 24 accepted, which was wrong on both
counts: `axis_separation_deg` reports the *undirected* angle and calls that rig $60°$, so
the number was read off the wrong quantity. Measured, the board sits at exactly $60°$ to
both, and nothing survives §14.3c.

### 14.3a Posing a view and calibrating from it are different questions

`MIN_CORNERS = 6` is enough to pose the board: `solvePnP` needs four, and six noisy ones
give an angle. It is not enough to calibrate from. On a 4x12 board only three corners wide,
a 12-corner view is a thin strip, and the homography that strip determines is badly
conditioned in the direction it is thin. The bundle does not reject such a view; it lets
its residual balloon and drags the fit with it.

Measured on a 63-view set (camera A, 33 corners on the board):

| corners required | views kept | intrinsics RMS | worst view |
|---|---|---|---|
| 6 | 63 | 0.620 px | 1.91 px |
| 16 (half the board) | 41 | 0.533 px | 1.91 px |
| 19 | 31 | 0.466 px | 0.77 px |

Every view over 1.2 px had between 11 and 18 corners of 33. `MIN_FIT_FRAC` is therefore a
second, higher threshold -- half the board -- applied where a view enters the fit, and by
the shutter so such a frame is never saved. On the same set it moved the residual
anisotropy from 0.60/0.49 to 0.81/0.92, both inside the gate: the thin strips were what
made the residuals directional.

The cost is real and is the other half of the trade. Corners are lost to *tilt*, so
requiring more of them requires flatter views, and the orientation spread §14.3 wants
shrinks with it: 14.1 deg at 6 corners, 10.8 deg at 19. The threshold cannot buy both. What
buys both is the board being **larger in frame** -- closer, or a board with more corners
across -- so that a tilted view still delivers half of it.

**What it was not.** Two hypotheses were measured and rejected before this one. Motion:
the skew median was 4.0 ms against the 2 ms gate, which looks damning, but the pair gate
had already refused every frame where it mattered. Defocus: the correlation between
per-corner edge width and per-view reprojection RMS is +0.08 for camera A and +0.17 for
camera B, and the four worst views were among the sharpest in the set. Neither is the
limiter here, and no focus gate was added -- there is nothing yet for it to catch.

**Which board.** The $4 \times 12$ shape was chosen for its backing -- a $25 \times 75$ mm
microscope slide is the flattest cheap thing there is -- and it costs conditioning. Same 40
views, same 0.3 px noise, corners outside the frame dropped, at 119 mm:

| board | corners seen | reprojection RMS | $\sigma_{f_x}$ |
|---|---|---|---|
| $4 \times 12$, 6 mm (3 across) | 18.0 / 33 | 0.381 px | 7.92 px |
| $9 \times 6$, 6 mm (8 across) | 36.3 / 40 | 0.406 px | **3.41 px** |

The residual is identical and the focal length is $2.3\times$ better determined, which is
the trap: the acceptance gate reads RMS, and RMS cannot see this. It needs a backing that
is not a slide.

### 14.3b Tilt spread was measuring the wrong angle

The shutter refuses shots until the saved set has enough orientation spread (§16.5), and it
measured that as the standard deviation of `board_incidence_deg`. Incidence is *unsigned*:
tilting the board left $40°$ and right $40°$ score identically, and so does tilting it up.
The metric collapses a direction on the sphere into a magnitude, and cannot tell a set that
covers the hemisphere from one that only ever tilts one way.

What it does to a good set:

| set | incidence std | normal spread |
|---|---|---|
| face-on plus $40°$ tilted N/S/E/W -- textbook | $16.0°$ | $35.8°$ |
| a real 65-view session | $16.1°$ | $48.3°$ |

Both would be told to tilt more against a $20°$ threshold, and the second one had a median
angle of $54.8°$ between its board normals across the full $360°$ of tilt direction. The
gate is now `orientation_spread_deg`: the RMS angle of the board normals about their mean,
threshold $25°$, which the textbook set clears and a one-direction set does not.

### 14.3c The incidence limit is the intrinsics' too

`MAX_INCIDENCE_DEG` was applied in `pair_views` and nowhere else, so solo views at $77°$
walked straight into `calibrate_intrinsics`. Incidence correlates $+0.58$ (camera A) and
$+0.44$ (camera B) with per-view reprojection RMS on an 87-view set. Capping it:

| cap | camera A | camera B |
|---|---|---|
| none | 65 views, 1.075 px | 42 views, 0.863 px |
| $70°$ | 47, 0.790 px | 34, 0.676 px |
| $60°$ | 30, **0.404 px**, spread $31°$ | 15, **0.465 px**, spread $32°$ |

§14.3 derives $70°$ from rotation uncertainty, on a board whose projected diagonal $D$ is
large. This board's is not: a $24 \times 72$ mm slide at 120 mm subtends little, and the
$\sigma_{\text{angle}} \sim \sigma_{\text{px}} / (D\cos\theta)$ penalty bites earlier.
The constant is now $60°$, measured, and it is one limit for both calibrations rather than
two: `calibrate_intrinsics` drops steep views in a second pass, since incidence needs a pose,
a pose needs $K$, and $K$ is what is being solved.

### 14.3d Undirected is right for triangulation and wrong for a plane

`rig.axis_separation_deg` treats the optical axes as **undirected lines**, which is correct
for what it was written for: two cameras facing each other triangulate like two facing the
same way, so a mixed-hemisphere rig conditions depth as well as its supplement does.

A flat board is not indifferent to that sign. It faces one hemisphere, so the best it can
present to both cameras is half the **directed** angle -- `rig.bisector_incidence_deg`.
The two agree whenever the cameras share a hemisphere and diverge completely when they do
not: at $45°/-45°$ the rig reports a comfortable $60°$ while the board is pinned at $60°$
incidence, at the limit of §14.3c, and every pair is refused.

The bench rig measures $83.2°$ directed, so its bisector is $41.6°$ and a board held facing
*between* the cameras clears the limit with $18°$ to spare. Held square to either camera it
does not, which is why sessions kept landing at $62$--$75°$.

### 14.4 Scale: the one error no residual can see

Suppose the printed square is wrong by a factor $\lambda$, so the assumed object points are
$\tilde X_i = \lambda X_i$ while the true ones are $X_i$. Take $\tilde R = R$ and
$\tilde t = \lambda t$. Then

$$K(\tilde R \tilde X_i + \tilde t) \;=\; \lambda\, K(R X_i + t),$$

and the pinhole projection $\pi$ is invariant under positive scaling of its argument, so
$\tilde x_i = x_i$ **exactly**, for every point and every view.

The consequences are worth stating separately:

- The mis-scaled problem attains an *identical* reprojection residual. No criterion built on
  reprojection, RMS, per-view error, the residual-structure test of §14.5, the bundle's own
  cost, can detect $\lambda$. It is not that the check is weak; the quantity is exactly
  invariant.
- $R$ is untouched, so every **angle** the rig reports is correct.
- Every **length** scales by $\lambda$: the baseline, the depths, and therefore the robot
  position that [§12](../pose/theory.md) (ch.3) recovers from a known 20.4 mm radius. A 3 % print error is a 3 % position
  error, silently, with a clean 0.3 px residual.
- $K$ and the distortion coefficients are untouched, the latter because they are defined on
  normalised coordinates.

The last point is testable, and was tested: building the board in millimetres rather than
metres, $\lambda = 1000$, changes the recovered $K$ by **exactly zero** to all printed
digits. That is this theorem at an absurd $\lambda$, and it is why the calibration can be
carried out in millimetres to match the rest of the package (§14.7) with no correction
anywhere.

The only remedy is metrological, outside the optics entirely: measure the printed square with
calipers. Hence the ruler bar printed beside the board, and the recorded incident in which
`charuco_6x8.pdf` was exported with a $1080\times1400$ pt MediaBox and rescaled by every
printer that touched it.

### 14.5 What the residual *can* see, and the trap in measuring it

The natural residual, re-solve each camera's board pose by PnP and reproject, is worthless
for extrinsics: $T_{B\leftarrow A}$ never enters it, so an arbitrarily wrong extrinsic still
yields a clean, isotropic, structureless residual. It measures intrinsics and nothing else.

The residual must come from a **joint** fit: one board pose per placement, required to explain
both views *through* $T_{B\leftarrow A}$,

$$\min_{T_{A\leftarrow b}} \; \sum_i \big\lVert \pi_A(T_{A\leftarrow b} X_i) - x^A_i
\big\rVert^2 + \big\lVert \pi_B(T_{B\leftarrow A} T_{A\leftarrow b} X_i) - x^B_i \big\rVert^2 .$$

Camera B's residual then carries the extrinsic error, which is the gated quantity. Combined
with the acceptance criteria of [§6](../control/theory.md) (ch.4) of the project context, RMS $< 0.5$ px, residuals
isotropic and structureless, this is what the calibration is allowed to certify.

The board pose per placement is not re-solved here: `cv2.stereoCalibrateExtended` returns
the `rvecs`/`tvecs` of exactly this minimisation alongside $T_{B\leftarrow A}$, so the
residual is measured against the fit that produced the extrinsic rather than a second solver
layered on top.

A second, free check exists because OpenCV 5 **refuses** `CALIB_USE_EXTRINSIC_GUESS` in
`stereoCalibrate`. The closed-form per-placement estimate of §14.1 therefore cannot seed the
bundle, which makes the two genuinely independent; their agreement is evidence rather than a
tautology. Measured: identical to printed precision on the noise-free round trip, and
$0.008°$ / $0.054$ mm under $0.2$ px of corner noise.

### 14.5a What the gate is allowed to be strict about

A reprojection residual converts. At working distance $z$ with the rays $\theta$ apart, a
residual of $\sigma$ px is

$$\sigma_{3D} \;\approx\; \frac{z\,\sigma}{f\sin\theta},$$

which on this bench -- 119 mm, $f = 2748$ px, $83°$ -- is 22 um at 0.5 px and 44 um at 1.0.
The robot is 10 mm across and the hover loop runs at 0.8 Hz. Everything in that table is
free, so **0.5 px was a round number, not a requirement**, and it is now 1.0 px with the
conversion printed beside the value.

That relaxation applies only to the precision gates. The rest do not move, because they
guard against a different failure: a bad extrinsic is smooth, plausible, self-consistent
and wrong, and nothing downstream will notice it. The closed-form seed agreement (§14.5),
the radial trend, and the pair-to-pair rotation agreement are the only things standing in
its way. Scale is worse still and none of them can see it (§14.4).

**Isotropy is a warning, not a gate.** Residuals twice as large in $y$ as in $x$ name a
noise source with a preferred direction -- a board that is not flat, a drift during the
pair. Synthetic isotropic noise on this same $4 \times 12$ board returns $s_x/s_y = 0.98$,
so the measure is not reporting the target's shape and a real reading of $0.60$ is real
structure. It still cannot mean a *wrong* rig while the seed agrees to $0.25°$ and the
radial trend is flat, and at 30 um of total uncertainty, doubling one axis changes nothing
a 10 mm robot can feel. It prints as `[WARN]` and is worth chasing on its own time.

### 14.6 Measured behaviour

Synthetic round trip against a known rig, board on the bisector, $300$ mm range,
$960\times720$:

| case | placements | rotation error | translation error |
|---|---|---|---|
| noise-free | 18 / 18 used | $<10^{-6}$ deg | $2\times10^{-6}$ mm |
| $0.2$ px corner noise | 24 / 24 used | $0.0038°$ | $0.019$ mm (0.006 % of a 300 mm baseline) |
| disjoint corner subsets, 35 % dropped, $90°$ rig | 13 / 24 used | $<10^{-6}$ deg | $2\times10^{-6}$ mm |

The noise case is the informative one: $0.2$ px of corner noise buys $0.006$ % of baseline,
so the extrinsic is nowhere near the limiting error in [§13](../pose/theory.md) (ch.3)'s budget. Print scale (§14.4) is
larger by three orders of magnitude and is the only term worth attention.

### 14.7 What does not work, and what turned out not to matter

| tried | result |
|---|---|
| Seeding `stereoCalibrate` with the closed-form extrinsic | **Not available.** OpenCV 5 raises "stereoCalibrate does not support CALIB_USE_EXTRINSIC_GUESS". Turned into an asset, see §14.5. |
| Board in metres vs millimetres | **Exactly zero** change in $K$, $\mathrm{dist}$, or RMS. §14.4 predicts this. |
| Flat `(N,2)` vs `(N,1,2)` correspondence arrays | **Exactly zero** change. The OpenCV 4→5 shape change is cosmetic at this layer. |
| `cv2.imread(..., IMREAD_GRAYSCALE)` vs `cvtColor(imread(...), BGR2GRAY)` | **Real, and small**: the two JPEG grayscale paths differ by up to 2 grey levels (mean 0.005), moving $f_x$ by $0.34$ px, $c_y$ by $0.18$ px. Neither is more correct; the notebook uses `IMREAD_GRAYSCALE` to match how `camera/sources.py` feeds the live pipeline. |
| Reproducing the checked-in `vision/camera_intrinsics.npz` from `board_images/9x6/` | **Not reproducible** by either decode path, $f_x$ 1411.14 / 1410.80 against the file's 1408.78. That file predates the current image set or came from a different OpenCV, so it is a fact about the reference, not a drift. |
| Per-camera PnP residual as an extrinsic check | **Blind by construction** (§14.5). |
| Averaging $T_{\text{cam}\leftarrow b}$ over a stationary board | Valid but weaker: cannot separate placement bias from extrinsic error (§14.1). |

Also recorded: the 29 shots in `board_images/9x6/` span only $0.8$–$55.8°$ of incidence with a
$16.5°$ spread, below the $20°$ the notebook warns at. Coplanar targets at near-identical
angles leave focal length and radial distortion poorly separated, which is the likeliest
reason that set's $k_3$ reaches $-2.5$.

### 14.8 Correspondence with the implementation

| Model element | Code |
|---|---|
| Board geometry in mm, print with ruler bar (§14.4) | `calibrate.CharucoSpec`, `sheet.generate_pdf` |
| Board-pose cancellation (§14.1) | `seed_extrinsic`, via `pair_views` |
| Chordal mean and dispersion (§14.2) | `seed_extrinsic`, `Rotation.mean` |
| Incidence gate (§14.3, §14.3c) | `board_incidence_deg`, `MAX_INCIDENCE_DEG`, in `pair_views` *and* `calibrate_intrinsics` |
| Directed vs undirected separation (§14.3d) | `rig.bisector_incidence_deg` vs `rig.axis_separation_deg` |
| Corners required to join the fit (§14.3a) | `calibrate.MIN_FIT_FRAC`, `fit_corners`, used by `_load_dir` and `capture.gates` |
| Joint two-view residual (§14.5) | `results.stereo_residuals`, on `cv2.stereoCalibrateExtended`'s own `rvecs`/`tvecs` |
| Acceptance gate (§14.5) | `results.acceptance`, `results.structure_report` |
| Extrinsic from views plus fixed $K$ | `calibrate_extrinsics`, the peer of `calibrate_intrinsics` |
| Round trip and negative results (§14.6, §14.7) | `test_calibrate.py` `self_test`, `regression_9x6` |
| Resulting rig, world = camera A | `calib/stereo_rig.json`, consumed by `rig.StereoRig.load` |

The world frame written here is **camera A**, not the $+z$-up frame `rig.from_spherical`
builds. $\Delta$ and the baseline are frame-independent and remain valid; elevation-flavoured
readings such as `tilt_seen_deg` do not, until the rig is rebased onto the robot's disk frame
at rest when the estimator is wired into visual servoing ([§11.6](../control/theory.md) (ch.4)).

## 16. Capture synchronisation: why the shutter refuses

Everything in §14 assumes the two views describe **one** board position. Two USB cameras do
not guarantee that, and the failure is quiet: a moving board photographed a few milliseconds
apart produces a consistent, plausible, wrong extrinsic.

### 16.1 Where the skew comes from, and why frame rate does not fix it

The cameras free-run. Each holds its own frame clock, and the host receives whatever the
grabber has most recently stored, so the two timestamps differ by an offset that drifts
slowly and is otherwise uniform over one sensor frame period $T$:

$$\Delta \sim U[0, T], \qquad \mathbb{E}[\Delta] = T/2$$

Measured here at $1280\times800$, $T = 1/119 = 8.4$ ms and the median skew is **7.71 ms**.
The distribution leans toward $T$ rather than staying uniform, because a dropped frame pushes
an offset past one period. At $640\times400$ the sensor runs at 217 fps, $T = 4.6$ ms, and the
median falls to **2.39 ms**.

The instinct is to record faster. It does nothing. Each grabber hands over its *newest*
frame whatever the consumer's rate, so a loop at 13 Hz and a loop at 119 Hz see the same
skew distribution. Only the sensor's own rate $T$ moves it. Measurement rather than
assumption: at a 13 fps write rate the median skew was 4.5 ms, and at 25 fps it did not
change within noise.

### 16.2 What skew costs

A board moving at $v$ (px/s in the image) breaks the one-position assumption by

$$\varepsilon_{\text{pair}} = v\,\Delta$$

At $f_x = 2765$ px on the 6 mm board, one millimetre is 11 px, so a hand sweep at
40 mm/s = 440 px/s with $\Delta = 4.5$ ms gives $\varepsilon = 1.98$ px, four times the
0.5 px the whole calibration is allowed. This is exactly what the first real run measured:
1.08 px joint RMS, and per-pair disagreement of 1.08 mm for the blurriest third of pairs
against 0.38 mm for the sharpest (correlation $-0.52$ between image sharpness and pair
disagreement).

Blur is the second, independent cost. Within one exposure $\tau$ the corner smears by $v\tau$,
and no amount of synchronisation removes it because it happens inside a single frame.

### 16.3 Rejection sampling on phase

The offset drifts, so pairs that happen to land close arrive on their own. `StereoCamera`
re-reads until one does. Measured yield against threshold, at $1280\times800$:

| limit | delivered | median skew |
|---|---|---|
| none | 119 pairs/s | 7.71 ms |
| 2 ms | 16 pairs/s | 1.02 ms |
| 1 ms | 1.8 pairs/s | 0.80 ms |
| 0.5 ms | 0.7 pairs/s | 0.48 ms |

The knee is at 2 ms: a $7.6\times$ improvement for a rate a keypress cannot perceive. Below
it the yield collapses, because the acceptance probability is $\approx \Delta_{\max}/T$ and
the offset must drift into the window rather than being drawn afresh.

This is why rejection sampling is a **calibration-only** technique. It costs re-reads, and
§17 of ch. 3 explains why the live loop cannot pay them.

### 16.4 Setting $v$ to zero beats estimating $v\Delta$

The error is a product. Two ways to make it small:

1. **Correct it.** Estimate each corner's velocity from neighbouring frames and shift both
   views to a common instant, leaving the acceleration residual $\tfrac12 a \Delta^2$. This
   was built and it worked: epipolar disagreement fell $6.2\times$ (1.334 → 0.215 px) on a
   board sweeping at 550 px/s, and the same 40 frames went from 0.821 px FAIL to 0.388 px
   PASS with the baseline recovered to 0.5 mm.
2. **Rest the board.** $v = 0$ makes both $v\Delta$ and $v\tau$ vanish exactly, with no
   estimator, no neighbouring frames, and no assumption that motion is locally linear.

This chapter takes the second and deletes the first, which is worth recording because the
first was the more interesting engineering. A calibration target is *static by nature*. The
only reason it ever moved is that a hand held it. A correction for self-inflicted motion is
machinery that somebody must maintain, test and understand forever. A block of wood is
free. The general lesson: when an error is a product of two terms and one of them is a
choice, set it to zero rather than modelling it.

### 16.5 The gates

The shutter therefore refuses rather than corrects. A press opens a request, and the shutter
saves the first pair that satisfies every condition. If none does so before the timeout, the
shutter names the condition that blocked:

| gate | threshold | what it protects |
|---|---|---|
| board detected in both, and posed | `MIN_CORNERS` | there is something to measure |
| shared corners | `MIN_COMMON_CORNERS` | the pair has common points to relate |
| pair consistency | $v\Delta \le 0.1$ px | one board position, §16.2 |
| blur | $v\tau \le 0.5$ px | corner localisation, which sync cannot help |
| pose novelty | $8°$ or 80 px from every saved pose | the orientation spread of §14.3 |

The novelty gate deserves its place. Every run to date has warned at 13–16° of orientation
spread against the 20° the intrinsics need, and shooting *more* frames never fixed it.
A pose held while frames stream past produces copies, not information.

### 16.5a One session, two calibrations

Two of those gates -- shared corners, and pair consistency -- are about the *other* camera.
A frame that fails them still constrains its own camera's $K$ perfectly well, and refusing
it is what starved the intrinsics. Pairs exist only where both cameras see the board, and
on a $90°$ rig that is a narrow band of tilts: a board angled well for A is angled badly for
B (§14.3). Every run solved from pairs alone has come out at 12--16° of orientation spread
against the 20° `calibrate_intrinsics` asks for, with per-camera RMS near 1.0 px against a
0.5 px gate. The pairs were not too few. They were all the same shape.

So the shutter now saves two things from one session:

| saved as | when | used for |
|---|---|---|
| `pair_NNN.png` in both `A/` and `B/` | every gate in §16.5 passes | intrinsics **and** the extrinsic |
| `solo_A_NNN.png`, `solo_B_NNN.png` | that one camera has the board, posed, unblurred, at a new pose | that camera's intrinsics only |

The separate stem is not cosmetic. `pair_views` matches the two cameras by filename, so two
solo frames caught in the same instant must not be able to collide into a pair the skew gate
refused -- which is the exact failure §16.2 exists to prevent. `solo_A_007` and `solo_B_007`
never match.

Pose novelty is then judged per camera, and a saved pair counts against both cameras' pose
lists as well as the pair list. The alternative -- a second, single-camera capture session
per camera -- gives the same conditioning and costs three sessions instead of one, with the
extra hazard that the lens must not be touched between them.

### 16.5b The bag

A session writes one **bag**: a directory holding `A/`, `B/`, `pairs.csv` and a `meta.json`
naming the board, the cameras, the mode and the counts. The meta file is what marks a bag
finished, and `capture` reuses a finished bag instead of re-shooting it.

That default is not convenience. The shutter clears the old set on its first new save,
which fires seconds after the window opens and long before the operator can tell the run is
going badly -- and it destroyed a 26-pair set that way. Re-shooting is now something you
ask for: ``override=True``, or ``append=True`` to add to what is there.

### 16.6 What does not work

* **Hardware trigger.** The OV9281 has an FSIN pin and a documented external-trigger mode,
  but the ELP board does not break it out and UVC has no host-side exposure trigger. Vendors
  that do expose it (Arducam, Innomaker) ship a separate mode-switch utility, and it is
  Windows/Linux only.
* **$640\times400$ to halve the skew.** It works: 2.39 ms median, and 46% of pairs already
  under 2 ms. But corners are then twice as coarse in millimetres, and the trade is bad once
  the board rests and $\Delta$ multiplies by zero anyway.
* **Recording continuously and choosing afterwards.** Correct, and about 270 lines including
  a lossless codec, a writer thread and a timestamp sidecar. §16.4 is why it is gone.

### 16.7 Correspondence with the implementation

| Model element | Code |
|---|---|
| Skew distribution over one frame period (§16.1) | `sources.StereoCamera.last_skew`, `skew_stats` |
| Rejection sampling and its yield (§16.3) | `sources.StereoCamera.read`, `max_skew_s` |
| Pair-consistency and blur limits (§16.2) | `MAX_PAIR_ERROR_PX`, `MAX_BLUR_PX`, `EXPOSURE_S` |
| The gate set (§16.5) | `capture.gates`, drawn by `_overlay` |
| The refusing shutter (§16.5) | `capture.capture` |
| Solo frames for intrinsics (§16.5a) | `capture.solo_ok`, `calibrate.load_views` |
| One bag per session, reused not re-shot (§16.5b) | `capture.write_meta`, `capture(..., override=)` |
| Live-loop counterpart, deliberately different | [§17](../pose/theory.md) (ch. 3) |
