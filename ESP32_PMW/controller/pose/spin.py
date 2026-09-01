"""
Is the rotor turning at all? The one question the 5-DOF pose cannot answer.

`estimator.py` recovers position and axis and stops there, because roll about the
rotor's own axis is *structurally* unobservable from the rim: `bounds.roll_null_space`
measures the derivative of the image with respect to roll at 1.6e-12 px. The rim is a
circle; rotating a circle about its normal maps it onto itself.

The four blades are a different feature, and they are not symmetric under roll -- they
are symmetric under a quarter turn, which is a much weaker statement. That 4-fold
structure survives thresholding and dies at exactly one line: the `cv2.convexHull` that
ends `segment.silhouette_hull`. A hull of a 4-bladed rotor fills in the gaps between the
blades, which is the whole signal. So this reads the **raw frame** and borrows only the
fitted ellipse, for a centre and a scale.

Measured on `results/flights/2026-08-29_153356`, a real failed takeoff: the 4th angular
harmonic carries 34% of in-band power, about 5x either the 1st or the 2nd, and its phase
holds to 0.023 rad per frame -- roughly 0.3 degrees of physical rotation.

**What this can and cannot say, and the trap in between.** Four-fold symmetry means an
unambiguous *rate* needs under a quarter turn between frames, so spin < fps/8: about
3.2 Hz at the pipeline's ~25 fps, 15 Hz off the camera at its native 120, 52 Hz at
320x240@420.

Above that the phase is aliased, and aliasing here is not merely noisy -- it is
*actively deceiving*. With four blades the pattern repeats every quarter turn, so a rotor
spinning at any multiple of fps/4 (6.5, 13, 19.5 Hz at 25 fps) presents an identical
image frame after frame. It looks exactly, and stably, like a rotor that is stopped. This
is the wagon-wheel effect, and on this rig it is not hypothetical: the blades are
routinely seen "stationary" while spinning.

So the two verdicts are not symmetric, and this class refuses to pretend otherwise:

* **"turning"** is safe at any speed. Phase that moves means something moved.
* **"stopped"** is only ever claimed below `fps/8`, where a strobe cannot masquerade as
  stillness. Above it the answer is `None` -- unknown -- no matter how still the phase
  looks.

Pass the commanded field frequency to `update` so the class knows which regime it is in.
Without it, "stopped" is never claimed at all.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

BLADES = 4  # what makes the 4th harmonic the one to read
N_SAMPLES = 720  # points round the ring; 0.5 deg, far finer than the phase noise
R_FRAC = 0.6  # of the rim's semi-major axis: inside the rim, out on the blades
BAND = slice(1, 20)  # harmonics the strength is measured against
MIN_STRENGTH = 0.15  # below this the ring is not looking at blades
TURN_RAD = 0.15  # per-frame phase change that counts as motion, ~6x the noise


def blade_phase(gray, ellipse, r_frac=R_FRAC, n=N_SAMPLES):
    """
    The blades' angular phase, and how much to believe it.

        Samples the image on a ring about the ellipse centre and takes the angular
        FFT. Returns ``(phase_rad, strength)``, where strength is the 4th harmonic's
        share of in-band power -- a blank or blade-free ring gives a low share rather
        than a confident wrong angle, which matters because the caller is deciding
        whether the rotor moved.

        The phase is in *harmonic* radians: one full turn of it is a quarter turn of
        the robot. Nothing here unwraps it; that is `SpinWitness`'s job.
    """

    (cx, cy), (major, _minor), _angle = ellipse
    r = r_frac * major / 2.0
    if r < 2.0:
        return 0.0, 0.0

    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = np.clip(np.rint(cx + r * np.cos(th)).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip(np.rint(cy + r * np.sin(th)).astype(int), 0, gray.shape[0] - 1)

    prof = gray[ys, xs].astype(np.float64)
    prof -= prof.mean()
    spec = np.fft.rfft(prof)            # once: the phase and the power are one transform
    power = np.abs(spec) ** 2
    band = power[BAND].sum()
    if band <= 0.0:
        return 0.0, 0.0
    return float(np.angle(spec[BLADES])), float(power[BLADES] / band)


class SpinWitness:
    """
    Watches the blade phase and says whether the rotor is moving.

        Deliberately a latch on *motion*, not a rate estimate. Above fps/8 the rate is
        unrecoverable and pretending otherwise would put a confident wrong number next
        to a drive frequency, which is worse than an honest "moving".
    """

    def __init__(self, min_strength=MIN_STRENGTH, turn_rad=TURN_RAD, history=12,
                 fps=25.0):
        self.fps = float(fps)
        self.min_strength = float(min_strength)
        self.turn_rad = float(turn_rad)
        self.history = int(history)
        self._phase = None  # last raw phase, radians
        self._t = None
        self._total = 0.0  # unwrapped harmonic radians since the first sample
        self._steps = []  # recent |per-frame phase change|
        self.strength = 0.0
        self.n = 0
        self.field_hz = None      # last commanded field frequency, if the caller knows

    def update(self, t, gray, ellipse, field_hz=None):
        """
        Fold in one frame. Returns ``turning``, or ``None`` when there is no signal.
        """

        if field_hz is not None:
            self.field_hz = float(field_hz)
        phase, strength = blade_phase(gray, ellipse)
        self.strength = strength
        if strength < self.min_strength:
            return None

        self.n += 1
        if self._phase is not None and self._t is not None and t > self._t:
            # Wrapped into (-pi, pi]: the smallest rotation consistent with the two
            # samples. Past fps/8 that is not the true one, which is why only its
            # magnitude is used, and only to answer "did it move".
            d = math.remainder(phase - self._phase, 2.0 * math.pi)
            self._total += d
            self._steps.append(abs(d))
            del self._steps[: -self.history]
        self._phase, self._t = phase, t
        return self.turning

    @property
    def alias_limit_hz(self):
        """Spin rate above which a strobe can imitate a standstill: ``fps / 8``."""

        return self.fps / 8.0

    @property
    def turning(self):
        """
        ``True`` moving, ``False`` stopped, ``None`` unknown.

                Deliberately asymmetric. Motion is provable at any speed, but stillness
                is not: at a multiple of ``fps/4`` a spinning rotor renders an identical
                frame every time, so a still phase is evidence of nothing. Unless the
                commanded field is known to be below `alias_limit_hz`, a still phase
                returns ``None`` rather than a confident and possibly false ``False``.
        """

        if len(self._steps) < 3:
            return None
        if float(np.median(self._steps)) > self.turn_rad:
            return True
        if self.field_hz is not None and self.field_hz <= self.alias_limit_hz:
            return False
        return None

    @property
    def drift_rev(self):
        """
        Physical revolutions since the first frame -- only meaningful below fps/8.
        """

        return self._total / (2.0 * math.pi * BLADES)

    def summary(self):
        state = {None: "unknown (aliased)", True: "TURNING", False: "STOPPED"}[
            self.turning if self._steps else None
        ]
        if not self._steps:
            state = "no blade signal"
        return (f"{state}  (h{BLADES} {self.strength:.2f}, "
                f"{self.drift_rev:+.2f} rev over {self.n} frames)")


def from_recording(rec_dir, tag="A", limit=None, stride=1):
    """
    Replay a take through the witness. Returns the `SpinWitness`.

        The offline twin of the live path, and the only way to test this against a
        real rotor without energising anything. `results/flights/2026-08-29_153356`
        is the known-bad fixture: a takeoff whose ramp ran while the rotor sat still.
    """

    import sys

    import cv2

    from controller.pose import segment as segmod

    rec = Path(rec_dir)
    cap = cv2.VideoCapture(str(rec / tag / f"{tag}.mp4"))
    stamps = None
    csv = rec / "frames.csv"
    if csv.exists():
        rows = [r for r in csv.read_text().splitlines() if r and not r.startswith("#")]
        cols = rows[0].split(",")
        if f"t_{tag.lower()}" in cols:
            k = cols.index(f"t_{tag.lower()}")
            stamps = [float(r.split(",")[k]) for r in rows[1:]]

    w = SpinWitness()
    i = 0
    try:
        while limit is None or i < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                g = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     if frame.ndim == 3 else frame)
                seg = segmod.segment(g)
                if seg is not None:
                    t = stamps[i] if stamps and i < len(stamps) else i / 25.0
                    w.update(t, g, seg.ellipse)
            i += 1
    finally:
        cap.release()
    return w


def probe(index=None, mode="640x400@210", seconds=6.0, roi=None, grayscale=True):
    """
    High-rate blade-phase capture: is the rotor rotating, or just oscillating?

        The pose pipeline runs at ~25 fps because segmenting a stereo pair costs
        ~50 ms, which puts Nyquist at 12.5 Hz -- below the drive frequency, below the
        19 Hz phase mode, below everything worth seeing. A spectrum taken there is
        empty no matter what the rotor does.

        This drops the pose estimate entirely. One camera, one segmentation to fix the
        ring, then nothing per frame but sampling intensity around it. That runs at
        camera rate: 210 fps at 640x400 (Nyquist 105 Hz), 420 at 320x240.

        Returns ``(t, phase)`` arrays. Feed them to `classify`.
    """

    import sys

    from controller.camera import elp
    from controller.pose import segment as segmod

    if index is None:
        from controller.camera import identify

        index = identify.elp_indices()[0]
    cam = elp.open_elp(index=index, mode=mode, grayscale=grayscale)
    try:
        # One segmentation to place the ring. The robot is on the pad and does not
        # translate meaningfully during a spin-up, so re-fitting per frame would only
        # buy jitter at the cost of the frame rate this exists for.
        if roi is None:
            for _ in range(40):
                item = cam.read()
                if item is None:
                    continue
                seg = segmod.segment(item[1])
                if seg is not None:
                    roi = seg.ellipse
                    break
            if roi is None:
                raise RuntimeError("no segmentation; cannot place the sampling ring")

        t, ph = [], []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            item = cam.read()
            if item is None:
                continue
            angle, strength = blade_phase(item[1], roi)
            if strength >= MIN_STRENGTH:
                t.append(item[0])
                ph.append(angle)
    finally:
        cam.close()
    return np.asarray(t), np.unwrap(np.asarray(ph))


def classify(t, phase, field_hz=None):
    """
    Rotating, oscillating, or neither, from a high-rate phase record.

        Two independent statistics, because they fail differently:

        * **coherence** = ``|sum d| / sum|d|`` over per-frame phase steps. A rotor
          turning steadily accumulates phase, so this approaches 1. One rocking back
          and forth returns to where it started, so this approaches 0. Aliasing
          weakens it but does not invert it.
        * **spectrum** (Lomb-Scargle, because frame times are not evenly spaced). An
          oscillating rotor is a periodic signal and shows a peak -- at the drive
          frequency if it is being shaken by the field, or near the phase mode if it
          is ringing. A steadily rotating one has no peak once the trend is removed.

        Rotation with a strong peak is neither: it is a rotor slipping poles.
    """

    from scipy.signal import lombscargle

    if len(t) < 32:
        return {"verdict": "too few samples", "n": len(t)}
    d = np.diff(phase)
    tot = np.abs(d).sum()
    coherence = float(abs(d.sum()) / tot) if tot else 0.0

    fs = 1.0 / float(np.median(np.diff(t)))
    trend = np.polyval(np.polyfit(t, phase, 1), t)
    grid = np.linspace(0.3, 0.45 * fs, 3000)
    power = lombscargle(t, phase - trend, 2.0 * np.pi * grid, normalize=True)
    peak_hz, peak = float(grid[np.argmax(power)]), float(power.max())

    if coherence > 0.5:
        verdict = "ROTATING"
    elif peak > 0.25:
        verdict = "OSCILLATING"
    else:
        verdict = "no coherent motion"
    # How far the rotor actually moves, in physical degrees. A few degrees of rocking
    # is invisible in coherence and in a normalised periodogram, but it is the
    # difference between "held still" and "being shaken but not carried round".
    amp_deg = float(np.std(phase - trend)) * 180.0 / (np.pi * BLADES)
    out = {"verdict": verdict, "n": len(t), "fs_hz": fs, "nyquist_hz": fs / 2,
           "amp_deg": amp_deg,
           "coherence": coherence, "peak_hz": peak_hz, "peak_power": peak,
           "spin_hz": float(d.sum() / (2 * np.pi * BLADES * (t[-1] - t[0])))}
    if field_hz:
        out["field_hz"] = field_hz
    return out


def _synthetic(angle_deg=0.0, size=241, blades=BLADES, r_out=100.0):
    """A disk with `blades` bright spokes, rotated by `angle_deg`. For the self-check."""

    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - c, yy - c
    rr = np.hypot(dx, dy)
    th = np.arctan2(dy, dx) - math.radians(angle_deg)
    img = np.where(rr <= r_out, 40, 0).astype(np.uint8)
    spoke = (np.cos(blades * th) > 0.5) & (rr <= 0.85 * r_out)
    img[spoke] = 220
    return img


def _self_check():
    ell = ((120.0, 120.0), (200.0, 200.0), 0.0)

    # 1. The blades are there, and they are the fourth harmonic.
    _, strength = blade_phase(_synthetic(0.0), ell)
    assert strength > 0.2, strength

    # 2. A known rotation comes back. Harmonic phase turns BLADES times as fast as the
    #    robot, so 10 deg of rotor is 40 deg of phase.
    p0, _ = blade_phase(_synthetic(0.0), ell)
    p1, _ = blade_phase(_synthetic(10.0), ell)
    got = math.degrees(math.remainder(p1 - p0, 2.0 * math.pi)) / BLADES
    assert abs(abs(got) - 10.0) < 1.0, got

    # 3. Nothing to see: report weakness, not a confident angle.
    _, blank = blade_phase(np.full((241, 241), 30, np.uint8), ell)
    assert blank < MIN_STRENGTH, blank

    # 4. A rotor held still reads "not turning"; one turning steadily reads TURNING.
    # Held still, and the field is slow enough that a strobe cannot be the cause.
    still = SpinWitness(fps=25.0)
    for i in range(10):
        still.update(i / 25.0, _synthetic(0.0), ell, field_hz=2.0)
    assert still.turning is False, still.summary()
    assert abs(still.drift_rev) < 0.01, still.drift_rev

    # The same still image at a field above fps/8 must NOT read as stopped: a rotor
    # spinning at a multiple of fps/4 looks exactly like this. This is the wagon-wheel
    # trap, and claiming "stopped" here was a real wrong call on real footage.
    strobe = SpinWitness(fps=25.0)
    for i in range(10):
        strobe.update(i / 25.0, _synthetic(0.0), ell, field_hz=13.0)
    assert strobe.turning is None, strobe.summary()

    # No field frequency given: stillness is never conclusive.
    blind = SpinWitness(fps=25.0)
    for i in range(10):
        blind.update(i / 25.0, _synthetic(0.0), ell)
    assert blind.turning is None, blind.summary()

    # 5 deg/frame at 25 fps is 0.35 Hz -- comfortably above the threshold, which sits
    # at 2.1 deg/frame. Sign of the harmonic phase follows the image convention, so
    # the magnitude is what is asserted.
    spun = SpinWitness()
    for i in range(10):
        spun.update(i / 25.0, _synthetic(i * 5.0), ell)
    assert spun.turning is True, spun.summary()
    assert abs(abs(spun.drift_rev) - 9 * 5.0 / 360.0) < 0.01, spun.drift_rev

    # `classify` is the general form of the rotating-vs-rocking question, and it must
    # separate the two cases the runner's capture trigger exists for. Phase is in blade
    # angle, so one revolution is 2*pi*BLADES.
    t = np.arange(256) / 60.0
    turn = classify(t, 2 * np.pi * BLADES * 3.0 * t)                    # 3 rev/s, steady
    rock = classify(t, 0.35 * np.sin(2 * np.pi * 9.0 * t))              # 9 Hz, nets to 0
    assert turn["verdict"] == "ROTATING", turn
    assert turn["coherence"] > 0.99, turn["coherence"]
    assert abs(turn["spin_hz"] - 3.0) < 0.05, turn["spin_hz"]
    assert rock["verdict"] == "OSCILLATING", rock
    assert rock["coherence"] < 0.5, rock["coherence"]
    assert abs(rock["peak_hz"] - 9.0) < 0.5, rock["peak_hz"]
    # Too short to judge is its own answer, never a default to one of the two above.
    assert classify(t[:8], np.zeros(8))["verdict"] == "too few samples"

    print("spin: 4th harmonic found, rotation recovered to <1 deg, blank rejected")
    print(f"  classify -> turning {turn['coherence']:.3f} coherence, "
          f"rocking {rock['peak_hz']:.1f} Hz peak")
    print(f"  still -> {still.summary()}")
    print(f"  spun  -> {spun.summary()}")
    print("\nall checks passed")


if __name__ == "__main__":
    _self_check()
