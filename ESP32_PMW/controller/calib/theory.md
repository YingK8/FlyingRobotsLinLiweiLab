# Chapter 2 — Calibration: fixing the frames everything else assumes

*Stage 2 of the pipeline. Consumes: frames from [chapter 1](../camera/theory.md).
Produces: intrinsics, `stereo_rig.json`, the tilt and centre corrections, and the
datum. Consumed by: [chapter 3, pose](../pose/theory.md).*

Every gain the estimator claims is stated in some frame, and this stage is where
those frames come from. The chapter's real subject is not how to run a
calibration — it is **which errors the residual can see and which it structurally
cannot**, because the second kind are the ones that survive to become confident
wrong answers.

## Reading order

| # | file | what it does |
|---|---|---|
| 1 | `calibrate_camera.ipynb` | board -> detect -> intrinsics -> extrinsics -> `stereo_rig.json` |
| 2 | `rig.py` | `StereoRig`: where the cameras are, and how to carry a pose between frames |
| 3 | `shape.py` | the tilt/centre bias correction, and `APPEARANCE` |
| 4 | `zeroing.py` | `Zero`: report pose relative to a reference instead of the lens |
| 5 | `calibrate_shape.ipynb` | fits the corrections in (3) |

`rig.py` is the data model the rest of the pipeline reads; run
`calibrate_camera.ipynb` to produce it and `rig.StereoRig.load` to consume it.

## 2.0 Three calibrations, not one

They are independent and fail differently, which is why they are separate files:

| calibration | fixes | lives in | failure if wrong |
|---|---|---|---|
| **intrinsics** $K$, distortion | pixels -> rays | `assets/camera_intrinsics.npz` | every distance scales |
| **extrinsics** $T_{B \leftarrow A}$ | two cameras -> one world | `stereo_rig.json` | smooth, plausible, wrong triangulation |
| **shape** (tilt, centre) | flat-circle model -> real robot | `tilt_calibration*.json` | systematic tilt bias, ~3 degrees |

The first two are geometry and are checked by reprojection. The third is not
geometry at all — it corrects for the robot not being the flat circle
[chapter 3](../pose/theory.md) assumes — and no reprojection residual can see it.

**`APPEARANCE` lives here**, in `shape.py`, not with the segmenter that uses it
most. It is the key the calibration files are named by: each rig appearance
thresholds the boundary in a different channel, so each carries its own fitted
constants. `pose/segment.py` re-exports it.

## 14. Extrinsic calibration: fixing the frame the two cameras share

[§12.6](../pose/theory.md) (ch.3) assumes a shared frame — every gain it claims is stated in one. This section is how
that frame is obtained, and, more usefully, which of its errors the calibration residual can
see and which it structurally cannot. Implemented in
`controller/pose/stereo_calibration.ipynb`.

**A7 — Rigid stereo mount.** The two cameras hold their relative pose across both
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

The obvious alternative — hold the board **stationary**, average
$T_{\text{cam}\leftarrow b}$ per camera, and adopt the board as the world frame — was
implemented first and then removed. It is a valid estimator of a different thing, but it
requires the board fixed, it cannot separate a board-placement bias from an extrinsic error,
and it inherits whatever single placement happened to be chosen. Sweeping is strictly
better: more information, an uncertainty estimate for free, and one less thing to keep still.

**The world frame is camera A**, $T_{\text{world}\leftarrow A} = I$. Anchoring the world to
the *board* would tie the rig to a placement that exists only during calibration; anchoring
it to the *robot* would make the extrinsic depend on the thing being measured. Camera A is
the only datum that is both physically persistent and independent of the measurement.
Rebasing onto the robot's disk frame at rest is a separate, later step, taken when the
estimator is wired into visual servoing ([§11.6](../control/theory.md) (ch.4)) — and it is a pure change of basis, so it
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
$70°$, $11\times$ at $85°$ — which is why views past $70°$ are rejected outright rather than
downweighted (`MAX_INCIDENCE_DEG`).

This settles **open item 3** of `docs/pose_localization_project_context.md`. For an axis
separation $\Delta$, a flat board placed with its normal on the **bisector** of the two
optical axes is seen at $\Delta/2$ by both. The single-board method is therefore viable while

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

| rig | $\Delta$ | incidence at bisector | pairs accepted |
|---|---|---|---|
| $45°/45°$, azimuth $0/90$ | $60°$ | $30°$ | 24 / 24 |
| $45°/-45°$, azimuth $0/90$ | $90°$ | $45°$ | 13 / 24 |

The jitter budget, not the nominal geometry, is what runs out as $\Delta$ grows.

### 14.4 Scale: the one error no residual can see

Suppose the printed square is wrong by a factor $\lambda$, so the assumed object points are
$\tilde X_i = \lambda X_i$ while the true ones are $X_i$. Take $\tilde R = R$ and
$\tilde t = \lambda t$. Then

$$K(\tilde R \tilde X_i + \tilde t) \;=\; \lambda\, K(R X_i + t),$$

and the pinhole projection $\pi$ is invariant under positive scaling of its argument, so
$\tilde x_i = x_i$ **exactly**, for every point and every view.

The consequences are worth stating separately:

- The mis-scaled problem attains an *identical* reprojection residual. No criterion built on
  reprojection — RMS, per-view error, the residual-structure test of §14.5, the bundle's own
  cost — can detect $\lambda$. It is not that the check is weak; the quantity is exactly
  invariant.
- $R$ is untouched, so every **angle** the rig reports is correct.
- Every **length** scales by $\lambda$: the baseline, the depths, and therefore the robot
  position that [§12](../pose/theory.md) (ch.3) recovers from a known 20.4 mm radius. A 3 % print error is a 3 % position
  error, silently, with a clean 0.3 px residual.
- $K$ and the distortion coefficients are untouched, the latter because they are defined on
  normalised coordinates.

The last point is testable, and was tested: building the board in millimetres rather than
metres — $\lambda = 1000$ — changes the recovered $K$ by **exactly zero** to all printed
digits. That is this theorem at an absurd $\lambda$, and it is why the calibration can be
carried out in millimetres to match the rest of the package (§14.7) with no correction
anywhere.

The only remedy is metrological, outside the optics entirely: measure the printed square with
calipers. Hence the ruler bar printed beside the board, and the recorded incident in which
`charuco_6x8.pdf` was exported with a $1080\times1400$ pt MediaBox and rescaled by every
printer that touched it.

### 14.5 What the residual *can* see, and the trap in measuring it

The natural residual — re-solve each camera's board pose by PnP and reproject — is worthless
for extrinsics: $T_{B\leftarrow A}$ never enters it, so an arbitrarily wrong extrinsic still
yields a clean, isotropic, structureless residual. It measures intrinsics and nothing else.

The residual must come from a **joint** fit: one board pose per placement, required to explain
both views *through* $T_{B\leftarrow A}$,

$$\min_{T_{A\leftarrow b}} \; \sum_i \big\lVert \pi_A(T_{A\leftarrow b} X_i) - x^A_i
\big\rVert^2 + \big\lVert \pi_B(T_{B\leftarrow A} T_{A\leftarrow b} X_i) - x^B_i \big\rVert^2 .$$

Camera B's residual then carries the extrinsic error, which is the gated quantity. Combined
with the acceptance criteria of [§6](../control/theory.md) (ch.4) of the project context — RMS $< 0.5$ px, residuals
isotropic and structureless — this is what the calibration is allowed to certify.

A second, free check exists because OpenCV 5 **refuses** `CALIB_USE_EXTRINSIC_GUESS` in
`stereoCalibrate`. The closed-form per-placement estimate of §14.1 therefore cannot seed the
bundle, which makes the two genuinely independent; their agreement is evidence rather than a
tautology. Measured: identical to printed precision on the noise-free round trip, and
$0.008°$ / $0.054$ mm under $0.2$ px of corner noise.

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
| Seeding `stereoCalibrate` with the closed-form extrinsic | **Not available.** OpenCV 5 raises "stereoCalibrate does not support CALIB_USE_EXTRINSIC_GUESS". Turned into an asset — see §14.5. |
| Board in metres vs millimetres | **Exactly zero** change in $K$, $\mathrm{dist}$, or RMS. §14.4 predicts this. |
| Flat `(N,2)` vs `(N,1,2)` correspondence arrays | **Exactly zero** change. The OpenCV 4→5 shape change is cosmetic at this layer. |
| `cv2.imread(..., IMREAD_GRAYSCALE)` vs `cvtColor(imread(...), BGR2GRAY)` | **Real, and small**: the two JPEG grayscale paths differ by up to 2 grey levels (mean 0.005), moving $f_x$ by $0.34$ px, $c_y$ by $0.18$ px. Neither is more correct; the notebook uses `IMREAD_GRAYSCALE` to match how `pose/sources.py` feeds the live pipeline. |
| Reproducing the checked-in `vision/camera_intrinsics.npz` from `board_images/9x6/` | **Not reproducible** by either decode path — $f_x$ 1411.14 / 1410.80 against the file's 1408.78. That file predates the current image set or came from a different OpenCV, so it is a fact about the reference, not a drift. |
| Per-camera PnP residual as an extrinsic check | **Blind by construction** (§14.5). |
| Averaging $T_{\text{cam}\leftarrow b}$ over a stationary board | Valid but weaker: cannot separate placement bias from extrinsic error (§14.1). |

Also recorded: the 29 shots in `board_images/9x6/` span only $0.8$–$55.8°$ of incidence with a
$16.5°$ spread, below the $20°$ the notebook warns at. Coplanar targets at near-identical
angles leave focal length and radial distortion poorly separated, which is the likeliest
reason that set's $k_3$ reaches $-2.5$.

### 14.8 Correspondence with the implementation

| Model element | Code |
|---|---|
| Board geometry in mm, print with ruler bar (§14.4) | `pose/stereo_calibration.ipynb` [§1](../control/theory.md) (ch.4), `CharucoSpec`, `generate_pdf` |
| Board-pose cancellation (§14.1) | `seed_extrinsic`, via `pair_views` |
| Chordal mean and dispersion (§14.2) | `seed_extrinsic`, `Rotation.mean` |
| Incidence gate $\theta < 70°$ (§14.3) | `board_incidence_deg`, `MAX_INCIDENCE_DEG` |
| Joint two-view residual (§14.5) | `stereo_residuals` |
| Acceptance gate (§14.5) | `acceptance`, `structure_report` |
| Round trip and negative results (§14.6, §14.7) | `stereo_calibration.ipynb` [§12](../pose/theory.md) (ch.3)–13 |
| Resulting rig, world = camera A | `pose/stereo_rig.json`, consumed by `rig.StereoRig.load` |

The world frame written here is **camera A**, not the $+z$-up frame `rig.from_spherical`
builds. $\Delta$ and the baseline are frame-independent and remain valid; elevation-flavoured
readings such as `tilt_seen_deg` do not, until the rig is rebased onto the robot's disk frame
at rest when the estimator is wired into visual servoing ([§11.6](../control/theory.md) (ch.4)).
