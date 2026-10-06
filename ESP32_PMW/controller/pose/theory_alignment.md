# Alignment experiment notes

These chapters retain their original numbering from the alignment_rate branch.
Shared background remains in [theory.md](theory.md); hover/native chapters with
overlapping numbers remain there.

## 20. The disc axis: an axis-ratio fit, and what actually limits it

The tilt robots have a propeller and a mast and no rim, so the rim pipeline of 16 solves
none of their frames. What has been flying instead — `controller/pose/disc_axis.py`, and the
missing producer behind the 56 takes in `results/rim/` — reads the rotor axis from the
foreshortening of the disc silhouette. This section establishes what that costs, because the
obvious upgrade turns out not to be one.

### 20.1 The shipped estimator is `acos(minor / major)`

The producer of `results/rim/*/tilt_A.csv` is in no branch of this repository; commit
`7f638cf` landed 56 takes of its output and no code. Its method can nevertheless be read off
its own columns. Over all 7065 frames of `2026-09-08_205108`,

$$\theta_{\text{deg}} \;=\; \arccos\!\left(\frac{d_1}{d_2}\right)$$

to within $8\times10^{-4}$ degrees, where $d_1, d_2$ are the fitted ellipse's minor and major
axes. That is the weak-perspective foreshortening relation for a circle: an inclined circle
projects to an ellipse whose axis ratio is $\cos\theta$, exactly, when the circle is centred
on the optical axis of a pinhole camera.

### 20.2 The exact alternative, and why it buys nothing here

`conic.backproject_ellipse` does the same job without the approximation: the image conic
$C$ maps to a cone $K^{\mathsf T} C K$, whose eigendecomposition yields the circle's normal
in closed form, always two-fold ambiguous. It uses the full intrinsics and is correct
off-axis.

Both were run on identical synthetic contours at this rig's measured intrinsics
($f \approx 1375$ px, disc at the offsets the real takes show), with Gaussian pixel noise
added to the contour before fitting, 300 trials per level:

| contour noise | axis-ratio bias / sd | conic bias / sd |
|---|---|---|
| 0.2 px | +0.250 / **0.020** deg | −0.004 / **0.020** deg |
| 0.5 px | +0.224 / **0.050** deg | −0.029 / **0.050** deg |
| 1.0 px | +0.129 / **0.099** deg | −0.125 / **0.100** deg |
| 2.0 px | −0.253 / **0.198** deg | −0.509 / **0.200** deg |

**The uncertainties are identical to three digits.** They must be: both estimators read the
same physical quantity, the foreshortening, and both inherit the same sensitivity

$$\frac{d\theta}{d(b/a)} \;=\; -\frac{1}{\sin\theta},$$

which at the operating point $\theta \approx 69^\circ$ is 1.07 rad/rad. The conic buys
**bias** — 0.25 deg down to 0.004 — and nothing else.

The bias itself is small because the geometry is benign: $f \approx 1375$ px at 640 px wide
is a 26 deg field of view, the disc subtends only $\pm 2.8$ deg, and it sits 2.2 deg (camera
A) and 5.8 deg (camera B) off axis. Simulated at $\theta = 69$ deg, the axis-ratio estimator
reads +0.04 and +0.06 deg high respectively, of which −0.16 deg on B is lens distortion
(its $k_1,k_2,k_3$ are −0.338, 0.652, −5.38). The **gain** on a differential tilt swing is
0.997 — a 3.4 deg cone measures as 3.39 deg.

### 20.3 Segmentation is the limit, by two orders of magnitude

Against those numbers, the jitter actually present in the shipped takes. Scored by second
difference, since real motion at 190 Hz has small acceleration and noise does not, over a
settled window of `2026-09-08_205108`:

```
view A theta   total 2nd-diff rms  2.203 deg
               coherent 1-3x part  0.962 deg   <- real motion, removed
               residual            2.025 deg   <- noise
view B theta   total               6.049 deg
fused axis                         2.318 deg
```

2.03 deg of residual corresponds, through the table above, to roughly **20 px** of contour
error — not 0.2. The ellipse is being fitted to a materially different shape frame to frame.
The silhouette area modulates 5% / 4% at the spin frequency in the two views, which is the
same statement.

So the ranking is not close: **segmentation dominates the estimator choice by about 100x.**
Swapping `acos(b/a)` for exact conic backprojection would improve this pipeline by 0.25 deg
of bias and by nothing at all in jitter.

`disc_axis.py` therefore keeps the axis-ratio form for the per-view `theta_deg` column — it
costs nothing and it keeps the 56 existing takes numerically comparable — and spends the
effort on `disc_pose.segment_disc` instead: hysteresis threshold, round-hole fill, an
opening that deletes the mast and guy wires, blade recovery, an area cap. Measured on
`2026-09-09_194514`, that moves the second difference from 2.03 deg to **0.46 deg** fused,
0.63 / 0.58 deg per view. A 4.5x improvement, from the term that was actually large.

### 20.4 The negative result: better segmentation made the two views agree LESS

The same change moved A-vs-B disagreement the wrong way — 6.8 deg to 9.9 deg on the same
take, and 14.8 deg on `2026-09-09_194514`. This is not the calibration: the pre- and
post-recalibration rig files give 14.92 and 14.81 deg on the same frames. It is not a 180 deg
rotation mismatch either; flipping either view makes it worse (16.7 deg). It tracks
`segment_disc` finding a systematically larger silhouette (10224 against 9723 px) and
shifting the two views by different amounts.

Two caveats keep this from being a clean conclusion. The shipped `agree_deg` may not be
view-A-vs-view-B at all — in `control/tilt_report.py` that column name means disc-vs-mast,
which is a different quantity with no shared failure mode and a different scale. And the
disagreement is nearly constant (spread ~1.5 deg), so it behaves like a fixed offset rather
than added noise.

For a measurement that is a *difference* in axis over a few seconds — the alignment rate of
23 — a constant view offset largely cancels, and the jitter improvement is what matters. For
absolute attitude it does not cancel and the number should not be trusted. The next
measurement is per-view mask overlays at a few instants, to see which view's silhouette is
wrong.

### 20.5 The mast projects along the minor axis, and inflates it

The rotor axis and the mast are the same physical direction, so the mast projects into the
image along the ellipse's **minor** axis — the very axis the foreshortening measurement reads.
Anything thin left attached to the silhouette therefore stretches the ellipse along that axis,
makes $b/a$ too large, and makes $\arccos(b/a)$ too **small**. It under-reports tilt, and it
does so in a way no residual or fit-quality gate can see, because the inflated ellipse is a
perfectly good ellipse.

Sweeping a morphological opening applied to the mask before the fit, on
`2026-09-09_194514`:

| kernel px | $\theta_A$ | $\theta_B$ | 2nd-diff A | A-vs-B world |
|---|---|---|---|---|
| 0 | 41.85 | 49.64 | 1.052 | 15.15 |
| 3 | 41.97 | 49.52 | 1.051 | 15.69 |
| 5 | 46.15 | 49.33 | 0.876 | 13.61 |
| **7** | **46.46** | 49.26 | **0.474** | **13.38** |
| 9 | 46.41 | 49.05 | 0.529 | 13.61 |
| 13 | 46.37 | 49.16 | 0.757 | 13.44 |

View A was reading **4.5 deg low** and its frame-to-frame noise halves at $k=7$. View B barely
moves, so only A carried the contamination.

**The plateau is the evidence, not the improvement.** $\theta$ jumps between $k=3$ and $k=5$
and then sits flat from 5 to 13. An isotropic erosion of the disc itself could not do that —
it would walk $\theta$ monotonically, since $(b-2r)/(a-2r) \neq b/a$ for $b \neq a$. A jump
followed by a plateau is the signature of a distinct feature about 4-5 px wide being removed
and nothing else changing. `disc_axis.OPEN_PX = 7`.

Note `disc_pose.segment_disc` already opens at 11 px to delete the mast, and then calls
`_recover_blades` to give thinned blade tips back — and it is that recovery which returns the
thin structure. The opening therefore lives in `disc_axis`, downstream, so `disc_pose.py`
stays the upstream file it was cherry-picked as.

### 20.6 Direction-only reconstruction: the minor axis as a projected normal

The stronger form of the same idea is to stop using the minor axis's **length** at all.

An inclined circle's normal projects into the image along the minor axis. That image line,
together with the lens centre, spans a plane which contains the true axis whatever the other
camera says; two views give two planes, and the axis is their intersection —
$\hat n \propto n_1 \times n_2$ for plane normals $n_i = R_i K_i^{\mathsf T} l_i$. This is
`disc_pose.mast_direction`, applied to the ellipse's minor axis instead of the mast's line.

The division of labour is then clean, and each ellipse parameter does the job it is actually
good at:

| quantity | source | why |
|---|---|---|
| axis direction | minor axis **direction**, two planes | cannot be biased by an over-long minor axis |
| disc size | **major** axis length | the one direction that is not foreshortened |
| 3-D position | ellipse **centre**, triangulated | the axes' intersection is a point on the rotor |

**The radius is a free validation**, and it passes: 10.46 ± **0.02** mm over 545 frames
spanning a whole take, 0.2% stability on a quantity that is physically constant. Nothing in
the construction forces that, so it is real evidence that the major axis and the triangulated
depth are both right.

The two reconstructions disagree by 6.4-6.7 deg median. An earlier note here claimed the
plane construction was the worse of the two on the grounds that its second difference is
larger (0.919 against 0.530). **That claim is withdrawn.** It is not a fair comparison: the
conic path averages two branch normals before the number is taken, so part of its smoothness
is that averaging rather than better information. And the obvious referee — the angle to the
mast — is not one here, because both methods sit ~29 deg from it with only 0.8 deg of scatter,
which on inspection of the footage is far more likely to mean the thin line `find_mast` locks
onto is a tether than that two independent estimators are both 29 deg wrong.

Both are written per frame — `axis.csv` and `axis_minor.csv`, the same pairing
`results/rim/` already uses for `axis.csv` and `axis_b3.csv` — and the question of which to
believe is left to a take with a known answer.

### 20.7 Radial and azimuth are one cone, not two channels

Decomposing the rotor axis about its pre-cut attitude into a radial (polar) and an azimuthal
angle, and running a DFT on each — on the ACTUAL timestamps, since the pose stream is
non-uniform and an FFT misreads it (20.1) — gives, on the coil-cut takes:

| drive | radial peak | radial amp | azimuth peak | azimuth amp |
|---|---|---|---|---|
| 20 Hz | 20.00 Hz | 0.97 deg | 20.00 Hz | 7.88 deg |
| 40 Hz | 40.00 Hz | 2.28 deg | 40.05 Hz | 19.98 deg |

Both peak **exactly at the drive frequency**, so the axis cones at the spin rate, locked to it.

The azimuth's eight-fold larger amplitude is not eight times more motion. Azimuth carries a
$1/\sin\theta$ singularity, and $\theta$ here is only 8.7 and 11.9 deg:

$$A_\phi \sin\theta \;=\; 1.20,\; 4.10 \text{ deg} \quad\text{against}\quad A_\theta \;=\; 0.97,\; 2.28 \text{ deg}$$

Once descaled the two agree. They are two coordinates of one motion.

The tangent-plane components settle what that motion is. Projecting onto an orthonormal
$(e_1, e_2)$ perpendicular to the reference gives 1x amplitudes of **1.08 / 1.08 deg** at
20 Hz and **3.17 / 3.20 deg** at 40 Hz — equal in quadrature, which is a circular cone of
half-angle 1.53 and 4.50 deg. The cone grows with drive.

**So do not deproject the azimuth against the radial peak.** It was proposed as a way to
separate precession from whatever else the azimuth is doing, but the two angles are not
independent channels: subtracting the drive frequency from the azimuth removes the same
physical cone a second time, in the worse-conditioned of the two coordinates. Measured, it
takes 8.5 deg out of the azimuth and moves its scatter only 13.67 -> 12.30 deg, because most
of what remains is $1/\sin\theta$-amplified noise rather than signal.

The tangent plane is the right coordinate for this throughout: no singularity, and the
co-rotating / counter-rotating lock-in there separates a circular cone from a linear
oscillation directly (9.6:1 in favour of circular on `2026-09-08_205108`).

**A sampling caveat that bit once.** `disc_axis --stride` exists because a 3 s transient does
not need 210 Hz, and stride 4 leaves ~47 Hz — ample for the step, and Nyquist 23.75 Hz. That
is fine for the STEP and useless for the SPECTRUM: at stride 4 the 40 Hz take put its radial
peak at 12.60 Hz, an alias, and reported an azimuth line above its own Nyquist. Spectra want
the full-rate solve; step metrics do not.
