"""
The two condition tiers the estimator is measured against.

**Core** is what the ±1° / ±0.5 mm target is judged on.  **Edge** is measured and
reported alongside it, every iteration, but does not gate.  That split exists
because a single specification cannot honestly cover both: near-face-on under a
grazing light the rim goes unlit and no estimator recovers a pose, so folding
those frames into the gate would mean either failing forever or loosening the
spec until it stopped meaning anything.

Separate tiers instead of one weighted pool, deliberately.  A weight is an
arbitrary number that hides *which* regime moved when the total changes; two
numbers say it outright.

What differs between them:

|            | core                   | edge                     |
|------------|------------------------|--------------------------|
| ambient    | 0.20 – 0.60            | 0.15 – 0.28              |
| lights     | dome, lateral, or two opposed | single grazing lateral |
| backdrop   | peak ≤ 90 grey         | peak ≤ 120 grey          |
| opacity    | 0.8 – 1.0              | 0.7 – 1.0                |
| read noise | as the exposure implies | up to 1.5× that         |

Two things are deliberately *not* varied between tiers: the pose distribution and
the sensor's exposure/noise trade.  Poses stay identical so that a difference
between tiers is a difference in *conditions* and not in what was asked of the
estimator; and exposure stays physically coupled to noise in both, because a
camera cannot have a short exposure and a quiet read at the same time.

Every backdrop stays below the segmenter's threshold in both tiers -- see
`backgrounds.py` for why that is a design decision rather than an oversight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import backgrounds as bgmod
import render as rendermod


@dataclass
class Condition:
    """
    Everything that varies between frames except the pose itself.
    """

    background: np.ndarray
    light: rendermod.LightRig
    exposure: rendermod.Exposure
    alpha: float
    tier: str
    bg_name: str
    ambient: float

    @property
    def label(self):
        return f"{self.tier}/{self.bg_name}/amb{self.ambient:.2f}"


def _exposure(rng, index, subframes=5, noise_scale=1.0):
    """
    Sensor conditions for a high-frame-rate camera.

        Exposure has to fit inside the frame period, so a 420 fps camera is capped
        near 2.4 ms and in practice runs shorter.  Read noise is drawn
        **anti-correlated** with exposure because that is the real trade: a shorter
        exposure collects fewer photons and reads out noisier.  Sampling them
        independently would let the estimator see a bright, short, quiet frame that
        no sensor can produce.
    """

    exp_s = float(rng.uniform(1 / 4000, 1 / 1000))
    sigma = float(np.clip(4.5 * (1 / 1000) / exp_s, 3.0, 25.0)) * noise_scale
    return rendermod.Exposure(
        exposure_s=exp_s,
        subframes=subframes,
        spin_hz=float(rng.uniform(310.0, 350.0)),
        velocity_mm_s=tuple(rng.uniform(-40.0, 40.0, 3)),
        tilt_rate_deg_s=float(rng.uniform(-40.0, 40.0)),
        sigma=float(np.clip(sigma, 3.0, 40.0)),
        seed=int(index),
    )


def _core_light(rng):
    """
    Dome, single lateral, or two opposed laterals.

        The two-source case is the one that is new, and it is the realistic hard
        one: two lights at opposing bearings leave the rim lit on both sides and
        dark on the two between, which breaks the ring into arcs in a way a single
        source does not.  `segment.silhouette_hull` is built to survive exactly
        that, so it deserves to be tested on it.
    """

    amb = float(rng.uniform(0.20, 0.60))
    roll = rng.random()
    if roll < 0.40:
        rig = rendermod.LightRig(
            dome=((rng.uniform(30, 80), rng.uniform(0, 360)),),
            ambient=amb,
            intensity=rng.uniform(6, 20),
        )
    elif roll < 0.75:
        rig = rendermod.LightRig(
            lateral_deg=(rng.uniform(0, 360),),
            ambient=amb,
            intensity=rng.uniform(6, 20),
        )
    else:
        a = float(rng.uniform(0, 360))
        spread = float(rng.uniform(140.0, 220.0))
        rig = rendermod.LightRig(
            lateral_deg=(a, (a + spread) % 360.0),
            ambient=amb,
            intensity=rng.uniform(5, 14),
        )
    return rig, amb


def _edge_light(rng):
    """
    Low ambient with a grazing source -- degraded, but not hopeless.

        The band was chosen by measurement, not by taste.  At ambient 0.12 with no
        light down the optical axis, the robot itself stops clearing the segmenter's
        threshold: detection falls to a third and the frames that *do* detect are a
        sliver of rim back-projected to hundreds of millimetres away.  That is total
        failure, not difficulty, and an edge tier made of it measures nothing except
        how often segmentation dies.

        So this sits just above that cliff -- ambient 0.15-0.28, and the
        lens-mounted key light kept on, which is how the rig would actually be lit
        (see `render.LightRig`).  Hard enough that the rim breaks into arcs and the
        hull has to work; not so hard that there is nothing to measure.

        For reference, the monocular sweep's own characterisation of the cliff:
        at ambient 0.05 face-on, only 21% of the true silhouette clears the
        threshold and position error reaches 63 mm.
    """

    amb = float(rng.uniform(0.15, 0.28))
    if rng.random() < 0.75:
        rig = rendermod.LightRig(
            lateral_deg=(rng.uniform(0, 360),),
            ambient=amb,
            intensity=rng.uniform(6, 18),
        )
    else:
        rig = rendermod.LightRig(
            dome=((rng.uniform(15, 40), rng.uniform(0, 360)),),
            ambient=amb,
            intensity=rng.uniform(6, 18),
        )
    return rig, amb


def core(rng, n, shape, subframes=5):
    """
    ``n`` core-tier conditions. This tier gates the target.
    """

    out = []
    for i in range(n):
        field, name = bgmod.sample(
            shape, rng, peak=bgmod.CORE_PEAK, weights=bgmod.CORE_WEIGHTS
        )
        light, amb = _core_light(rng)
        out.append(
            Condition(
                background=field,
                light=light,
                exposure=_exposure(rng, i, subframes),
                alpha=float(rng.choice([0.8, 0.9, 1.0])),
                tier="core",
                bg_name=name,
                ambient=amb,
            )
        )
    return out


def edge(rng, n, shape, subframes=5):
    """
    ``n`` edge-tier conditions. Measured and reported, never gating.
    """

    out = []
    for i in range(n):
        field, name = bgmod.sample(
            shape, rng, peak=bgmod.EDGE_PEAK, weights=bgmod.EDGE_WEIGHTS
        )
        light, amb = _edge_light(rng)
        out.append(
            Condition(
                background=field,
                light=light,
                exposure=_exposure(
                    rng, i, subframes, noise_scale=rng.uniform(1.0, 1.5)
                ),
                alpha=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                tier="edge",
                bg_name=name,
                ambient=amb,
            )
        )
    return out


def sample_poses(rng, n, tilt_max=25.0, z_range=(-40.0, 40.0), lateral_mm=22.0):
    """
    Random world poses, identical between tiers.

        Tilt is drawn uniformly in the **cosine** rather than the angle, so samples
        spread evenly over the sphere of orientations instead of piling up near
        face-on where the geometry is degenerate anyway.

        The cap is near the platform's real attitude envelope -- 1.1° RMS and 5.2°
        peak-to-peak per the dynamics notes -- so 25° is already generous. The tilt
        that actually stresses the flat-circle model is set by where the *cameras*
        are, not by this.
    """

    cos_lo = math.cos(math.radians(tilt_max))
    tilt = np.degrees(np.arccos(rng.uniform(cos_lo, 1.0, n)))
    az = rng.uniform(0.0, 360.0, n)
    x = rng.uniform(-lateral_mm, lateral_mm, n)
    y = rng.uniform(-lateral_mm, lateral_mm, n)
    z = rng.uniform(*z_range, n)
    return tilt, az, np.column_stack([x, y, z])
