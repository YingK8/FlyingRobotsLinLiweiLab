# Rig appearance — what was tried, what it cost, what is still open

Working notes for the `bright` / `dark` split in `controller/pose/segment.py`.
Kept out of the module itself so the shipping code carries the decision and not
the argument. Everything here is a measurement, with the command that produced it.

## The chroma path (removed)

A `red` appearance existed and is gone. The argument was sound: on a white or
black ground a red body separates by chroma rather than brightness, because every
neutral surface has R = G = B however it is lit. Measured over plausible patches,
`R - max(G, B)` gave a +44 count margin where luminance overlapped by −218.

It does not survive contact with the hardware. **The high-speed cameras are mono
sensors.** On `controller/pose/assets/captures/elp/` chroma is exactly 0 in every
pixel — max 0, mean 0.00 — so a chroma test passes the whole frame. It is a no-op
whether the frame arrives with one channel or three, and it fails *silently*: the
segmentation looks gated when nothing is gated.

Removed with it: `REDNESS_THRESH`, `CHROMA_MAX`, the chroma form of
`clutter_mask`, `render.RED_BODY`, `render.COIL_BODY`, the renderer's BGR output
path, `RADIUS_BY_APPEARANCE["red"]`, `tilt_calibration_red.json`,
`dataset_red.npz`, and the red half of `ai/tests/test_appearance.py`.

Worth keeping in mind only if a colour camera is ever added, which the frame-rate
requirement makes unlikely.

## The `dark` constants are NOT fitted, and should not be trusted

`RADIUS_BY_APPEARANCE["dark"] = 10.1106` and `tilt_calibration_dark.json` come
from a dataset rendered at `DARK_THRESH = 110`. The shipped threshold is now
**190**, retuned against the real ELP captures. Constants and pipeline do not
match, which is the Iteration 12–14 failure class exactly.

Refitting was attempted and **failed**, which is the useful result:

| BLACK_BODY albedo | rendered body median | frames detected |
|---|---|---|
| 0.09 | 93 counts | 1375/1400, but rim fitted at 63 px against a true 120 |
| 0.055 | 78 counts | **471/1400** |

At 0.09 the body is *brighter* than the dividing luminance (`255 - 190 = 65`), so
most of it fails the darkness test and the hull collapses onto the darkest
fragments — a 47 % underestimate of the major axis that detection rate does not
reveal. Darkening it to match the real body (which reads a median of 42–78)
recovers the rim at one lighting condition but loses two thirds of the dataset,
because the dataset sweeps ambient 0.25–0.6 and intensity 6–20 and the render's
body brightens across that range far more than the real one does.

**So the renderer cannot currently express this rig**, and no constant fitted
through it means anything. The single real capture is one lighting condition and
cannot substitute for a sweep. What is needed is real captures at several known
tilts and lighting levels, or a body material whose response to lighting matches
the measured one — not another albedo guess.

Until then `dark` segments correctly on real frames and its *metric* output is
uncalibrated.

## The round-trip guard

`test_constants_match_the_shipped_threshold` renders a known pose under the
current constants and checks the fitted rim comes back. It is the only check that
catches a calibration fitted through a different pipeline than the one running,
because that failure never raises — it biases. It caught the 63 px case above,
which every other test in the suite passed.

## Open

- Real captures at known tilt/lighting, to replace the render for `dark`.
- `test_stereo::test_speed` still fails: the full two-view solve is 3.40 ms
  against a 4.17 ms budget at 240 Hz. Axial weighting costs 433 → 283 Hz; two
  thirds of that is the one-sided and trim passes, whose accuracy contribution
  has never been measured on its own.

## Axial weighting — the history that used to sit in `segment.py`

Two claims lived in the module and are withdrawn (Iterations 12–14):

* that the weighting collapses certified detection. It does not. That was
  measured while `PoseEstimator.update` ignored the module flag entirely, so both
  arms of the A/B ran weighted.
* that the weighted fit needs its own radius, 10.2662 mm. Refitting gives
  10.2418 — 0.03 % from the unweighted value. The same radius serves both.

The controlled A/B (same seed, 400 poses, same gate, `POSE_AXIAL` the only
difference) gives 59 certified frames against 61, and better error in every
sensor mode — 1280×800 position 0.268 → 0.186 mm, angle 0.294 → 0.178°, modes at
100 % in spec 5/8 → 6/8.

Regenerating the whole constant chain for the weighted fit was tried and rejected
on measurement: it certifies **12** frames where the shipped set certifies 59, and
its refitted error model came out ill-conditioned (`log_inv_margin` went negative
at −1.6, intercept −5.47, held-out acceptance 3.5 % → 0.9 %).
