# Vision-Based Pose Localization of the Subcentimeter Flying Robot — Project Context & Setup

> Working record. Captures decisions, rig geometry, estimation approach, and the
> calibration plan for optically localizing the magnetically-driven flying robot.
> Status: pre-implementation. Open derivations are flagged as **[OPEN]** — they are
> deliberately left for the engineer to work through, not yet solved here.

---

## 1. Platform under observation

The target is the untethered subcentimeter rotating-wing robot (Sui et al., *Sci. Adv.* 2025),
driven by a single-axis alternating magnetic field from the coil cluster. Key facts that
constrain the perception problem:

| Property | Value | Why it matters to tracking |
|---|---|---|
| Wingspan / largest dim. | **9.4 mm** | Sub-cm target → calibration target and working volume are centimeter-scale, not desktop-scale. |
| Mass | 21 mg | No onboard sensing/markers of any weight → **outside-in** (external cameras) is forced. |
| Drive frequency | **310–350 Hz** (9.4 mm prototype) | Spin rate is ~10× any affordable camera frame rate → instantaneous roll is temporally aliased. |
| Visible structure | 4 propeller blades + **circular balance ring** | Ring = rotationally symmetric feature; carries position + attitude but **zero roll info**. |
| Attitude stability | 1.1° RMS, 5.2° peak-to-peak pitch | Sets the resolution the estimator must reach: **sub-degree tilt**. |
| Vertical dynamics | ~1.4 m/s² up, 4 cm in 138 ms | Fast motion along the vertical axis → affects camera choice for the depth axis. |

Actuation context: coils are driven by the two-channel VNH5019 PWM amplifier board
(locked-antiphase bipolar drive, series-resonant coil cap). Not part of the perception loop,
but the **physical coil assembly occludes some viewing angles** — a hard constraint on camera
placement (see §4).

---

## 2. What "pose" we estimate — and what we do not

Decision: estimate **5 DOF, excluding instantaneous roll.**

### State space
The pose lives in **ℝ³ × S²**, not SE(3):

- `c ∈ ℝ³` — ring center position (3 DOF)
- `n ∈ S²` — spin-axis unit normal / attitude (2 DOF)

This is SE(3) with the `SO(2)` roll subgroup quotiented out. **Consequence for the solver:**
do *not* carry a 6-DOF state and let a generic PnP/LM optimizer chew on roll — roll is
unobservable, so that direction of `JᵀJ` is rank-deficient and the normal equations go
singular. Remove roll from the state; optimize/filter directly on `ℝ³ × S²` (retract attitude
updates onto the sphere, position updates in the tangent plane).

### Why roll is excluded (two independent walls)
1. **Symmetry.** The balance ring is ∞-fold rotationally symmetric (zero roll info); the four
   blades are 4-fold symmetric (roll ambiguous mod 90°). Geometry alone can never give absolute
   roll without an added symmetry-breaking mark.
2. **Temporal aliasing.** ~300 Hz spin vs ~30–120 fps cameras. Nyquist needs >~600 Hz to not
   alias the fundamental. Undersampled by ~10× → blade lands at a near-random phase each frame
   (stroboscopic/wagon-wheel). A rolling shutter additionally smears one rotation *across the
   frame*. Roll is observability-limited by the hardware, not by the algorithm.

Dropping roll is the correct engineering call: it is the one DOF the hardware cannot observe
and the body does not physically carry meaningfully.

---

## 3. Sensors and their roles

Two fixed, external cameras watching the coil workspace:

| Sensor | Measures well | Role in the estimate |
|---|---|---|
| **Intel RealSense (RGB-D)** | Metric depth of ring centroid → `c` directly, no scale guess. | **Position anchor.** Independent of the size-vs-known-radius method, so robust to blade occlusion of the ring rim. Note: depth is the noisiest channel (~1–2 mm close range, degrades ~distance²). |
| **RGB camera** | High lateral resolution, ellipse orientation/eccentricity. | **Attitude engine** (ellipse fit → `n`) and disambiguation/redundancy. No depth. |

Note: RealSense D4xx already carries stereo IR imagers + IR projector and is **factory-calibrated**
— pull its intrinsics and depth↔color extrinsics from the SDK rather than re-deriving. (IR won't
help as a *shared* feature since the RGB camera can't see it.)

---

## 4. Camera placement

### Current plan: front-on + top-down
- **Front-on (θ ≈ 90°, ring edge-on):** maximum attitude sensitivity — this is the attitude camera.
- **Top-down (θ ≈ 0°, ring face-on):** maximum lateral (x, y) position quality, cleaner ellipse
  (blades sweep *within* the silhouette, not across the rim) — this is the position camera.

`θ` = angle between the spin axis and the camera's viewing ray.

### The governing sensitivity law
Attitude observable = ellipse minor axis ∝ `r·cos θ`. Differentiating:

```
d(minor axis)/dθ  ∝  − r · sin θ
```

Sensitivity ∝ **sin θ** → **zero at θ=0 (top-down), maximal at θ=90° (side).**
Worked numbers (100 px ring, 1° tilt):

- Top-down: `100·(1 − cos 1°)` ≈ **0.015 px/deg** → below fit noise (~0.1–0.3 px). Effectively blind to tilt.
- Side: `100·sin 1°` ≈ **1.7 px/deg** → usable. ~100× more sensitive than top-down.

### Two degeneracies to respect
- The **two-fold normal ambiguity** (a single view's conic yields two candidate normals — ring
  tilted toward vs away) is **worst at θ=90°.** So the front-on camera is highly sensitive but
  maximally ambiguous; it is **unusable alone.** Resolved here by the top-down view + RealSense
  depth (near rim reads closer than far rim) + temporal continuity.
- Top-down alone is blind to tilt (sensitivity → 0).

The pair works because each camera covers the other's blind spot. They are complementary, not redundant.

### Alternative to evaluate on the bench
Consider pulling **both** cameras to ~45–60° elevation, separated ~90° in azimuth:
`sin 45° = 0.71` (71% of peak sensitivity), `sin 60° ≈ 0.87`. This keeps most of the attitude
sensitivity, avoids *both* degeneracies, and yields a real triangulation baseline. A single
oblique view already recovers both tilt components through two channels: lean toward/away →
ellipse **thickness**; lean sideways → ellipse **orientation**. **[OPEN]** Confirm whether the
coil assembly leaves these angles unoccluded — empirical, decide at the bench.

---

## 5. Estimation approach

### Core mechanism: circle → ellipse conic pose recovery
A 3D circle of known radius projects to an image ellipse; run backwards to recover the ring's
3D center direction, range, and plane normal from the fitted ellipse.

Pipeline:
1. Segment ring, fit ellipse → conic matrix `C` (pixels).
2. Normalize into the camera frame with intrinsics `K`.
3. Eigendecompose → plane normal (up to the two-fold sign choice) + center direction; known
   radius `r` fixes metric scale.

### Full pipeline stages
1. **Intrinsics** per camera (ChArUco). *[mechanical — to script]*
2. **Extrinsics** — both cameras into one world frame. *[mechanical, see §6]*
3. **Time sync** — hardware trigger strongly preferred; else timestamp alignment, and quantify
   residual sync error (× 300 Hz sets the roll-noise floor, which is why roll is dropped anyway).
4. **Front end** — segment robot / fit ring ellipse / locate blades. *[mostly mechanical]*
5. **5-DOF solve** — reprojection minimization on `ℝ³ × S²` across both views; RealSense depth
   anchors position, ellipses supply attitude, second view disambiguates.

### Sensor fusion for the two-fold ambiguity — three independent resolvers (rig has all three)
- Second camera: only one normal is consistent across both views.
- RealSense depth gradient across the ring rim: near vs far tells which way it leans.
- Temporal continuity: normal cannot physically flip between consecutive frames.

---

## 6. Calibration plan

**Key distinction:** a calibration board and tracking balls solve *different* problems.
- **Board** → camera parameters (K, distortion, camera↔camera transform). Known geometry, solve for cameras.
- **Balls/markers** → object pose at runtime (unknown-position points on the moving robot).
  (Mocap also uses a *ball on a wand* as a distinct **extrinsic** technique — see method 3 below.)

### The wide-baseline problem
Front-on + top-down are ~90° apart. A **planar** board visible to one at good incidence is
nearly **edge-on** to the other, where it calibrates terribly (compressed corners, soft depth).
This is the most common wide-baseline calibration failure. Three ways out:

1. **Angle the board ~45°** so both cameras see it at ~45°, sweep through many poses. *Simplest — start here.*
2. **3D target** — a ChArUco **cube** or two boards rigidly joined at ~90°; each camera sees a face square-on. *Right answer for perpendicular cameras.*
3. **Wand / point-based** — wave a single distinct blob through the shared volume, collect thousands
   of synchronized 2D–2D correspondences, solve relative pose (essential matrix → bundle-adjust).
   Works *because* a sphere is view-invariant — this is exactly why mocap uses balls, and where the
   blue-ball idea is correct.

Use **ChArUco** (checkerboard subpixel corners + ArUco identity + occlusion tolerance), via
`cv2.aruco`. Not a plain checkerboard, not loose ArUco tags.

### Blue-ball marker — viable, with caveats
- Threshold in **HSV**, not RGB (hue separates from illumination).
- Choose a color present **nowhere else** in the scene (check coil, bench, background before committing to blue).
- Control lighting; prefer **matte** over glossy (speculars shift the centroid).
- Subtlety: under perspective a sphere's projected centroid ≠ projection of its center (offset
  silhouette ellipse). Small, but non-negligible against a sub-degree budget on a 9.4 mm object.

### Centimeter-scale pitfalls (these silently wreck calibration)
- **Measure printed squares with calipers**; enter measured size, not nominal (printers rescale; 3% scale error → 3% in every reported distance).
- **Mount the print rigid** (glass / foam board); paper curl is a *systematic* bias, worst kind.
- **Size the target to the working volume and distance** (~cm), calibrated at the operating focus — don't extrapolate distortion into an unmeasured regime; watch depth of field up close.
- **Use RealSense factory intrinsics / depth↔color extrinsics** from the SDK.

### Acceptance criteria (calibration is done when the *error* is acceptable, not when code runs)
- Reprojection RMS **< 0.5 px**, and residuals **isotropic and structureless** (a radial or
  edge-worse pattern = underfit distortion = systematic error hiding under a decent average).
- **Independent check:** triangulate a caliper-measured distance between two marks in the volume;
  compare. Uses none of the machinery that produced the calibration → a real check.

---

## 7. Open items — engineer's kernels (deliberately unsolved)

These are the load-bearing derivations/decisions. Left open on purpose.

1. **[OPEN] Conic → normal derivation.** Given fitted conic `C` and intrinsics `K`:
   (a) normalize `C` into the camera frame (`x̂ = K⁻¹x`); (b) predict the **eigenvalue sign
   signature** of the normalized conic for a *circle* projection and justify it from the surface
   `x̂ᵀC′x̂ = 0` in ray space; (c) show the normal is a weighted combination of two eigenvectors and
   that the sign choice = the two-fold ambiguity.
2. **[OPEN] Azimuthal separation.** What does 90° azimuth separation buy over 0°? Which *component*
   of the tilt does each side view measure? (A tilt is a 2-vector — what can one side view not distinguish?)
3. **[OPEN] Bench geometry.** Does a 45°-angled board reach good incidence at *both* cameras given
   coil occlusion, or is a cube / wand method required?
4. **[OPEN] Filtering.** Per-frame estimates are noisy (undersampled spin). Design a filter on
   `ℝ³ × S²` (manifold EKF/UKF or complementary filter using known drive dynamics) — only after
   the per-frame solve works. Simplest thing first.

### Verification invariants (independent of ground-truth mocap)
- **Rigid-radius check:** solved ring radius must be constant frame-to-frame. If it "breathes" with motion, the pose is wrong.
- **Triangulated known distance** matches the caliper measurement (see §6).

---

## 8. Reference

- Sui, Yue, Behrouzi, Gao, Mueller, Lin. "Untethered subcentimeter flying robots." *Sci. Adv.* 11, eads6858 (2025).
- Coil driver: two-channel VNH5019 PWM amplifier (`PWM_amp.kicad_sch`, Rev 1, 2026-03-04).
