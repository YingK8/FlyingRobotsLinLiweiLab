"""Render the overlay tiles the visualisation page shows.

Two galleries, because they answer different questions:

**Conditions** -- a grid across tilt, noise and lighting, so you can see what the
estimator is looking at and how the fit tracks as things get harder.  This is the
"is it working" gallery.

**Failures** -- the poses with the worst residuals, reproduced deliberately.  This
is the one that earns its keep: the face-on, dimly-lit collapse is a paragraph of
explanation as a number, and obvious in one glance as a picture, because you can
see the fitted ellipse sitting on the blade cross instead of the rim.

Tiles are cropped around the detection and returned as PNG bytes, ready to be
base64'd into a self-contained page.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

import overlay as overlaymod  # noqa: E402
import render as rendermod  # noqa: E402
from estimator import PoseEstimator  # noqa: E402

TILE = 340


@dataclass
class Tile:
    """One gallery cell: the picture plus everything needed to caption it."""

    png: bytes
    title: str
    tilt_deg: float
    sigma: float
    ambient: float
    bg_level: float
    pos_err_mm: float = float("nan")
    dz_mm: float = float("nan")
    normal_err_deg: float = float("nan")
    ambiguity_deg: float = float("nan")
    fit_rms_px: float = float("nan")
    detected: bool = True
    note: str = ""
    layers: dict = field(default_factory=dict)


# JPEG, not PNG. These tiles are dominated by sensor noise, which is
# incompressible for a lossless codec -- a 340 px PNG tile runs 204 KB and the
# page would ship at 7 MB. At q92 the same tile is ~55 KB with no visible
# change to the overlay strokes, and the numbers all live in the CSVs anyway.
JPEG_QUALITY = 92


def _encode(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _measure(pose, sample):
    import math

    if pose is None:
        return {}
    d = np.asarray(pose.xyz_mm) - sample.center_mm
    cands = pose.extra.get("candidates") or []
    errs = [
        math.degrees(math.acos(float(np.clip(abs(c.normal @ sample.normal), -1.0, 1.0))))
        for c in cands
    ]
    return {
        "pos_err_mm": float(np.linalg.norm(d)),
        "dz_mm": float(d[2]),
        "normal_err_deg": min(errs) if errs else float("nan"),
        "ambiguity_deg": pose.ambiguity_margin_deg,
        "fit_rms_px": pose.fit_rms_px,
    }


def _tile(r, est, spec, gain, legend=False, layers=overlaymod.ALL_LAYERS, axial=None):
    """One overlay tile.

    ``axial=None`` inherits `segment.AXIAL_DEFAULT`, so the condition, failure and
    layer galleries show the build that ships. Only `build_weighting` passes it
    explicitly, and it pins *both* arms rather than letting either inherit --
    which is the whole point of an A/B. This defaulted to ``True`` until journal
    Iteration 12, so every gallery on the published page depicted the weighted
    fit while the shipped default was unweighted.
    """
    tilt, az, ctr, light, alpha, bg, sigma, title, note = spec
    exp = rendermod.Exposure(sigma=sigma) if sigma > 0 else None
    s = r.render(tilt, az, ctr, alpha=alpha, light=light, bg_level=bg, exposure=exp)
    est.reset()
    pose = est.update(s.image, axial=axial)

    if pose is None:
        img = cv2.cvtColor(s.image, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        c = min(h, w) // 3
        img = img[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c]
        img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
        overlaymod._text(img, (12, 28), "NO DETECTION", (0, 0, 255), 0.6, 2)
        return Tile(png=_encode(img), title=title, tilt_deg=tilt, sigma=sigma,
                    ambient=light.ambient, bg_level=bg, detected=False, note=note)

    img = overlaymod.draw(s.image, pose, s, K=r.K, radius_mm=est.radius_mm,
                          gain=gain, crop=2.2, size=TILE, legend=legend, layers=layers)
    return Tile(png=_encode(img), title=title, tilt_deg=tilt, sigma=sigma,
                ambient=light.ambient, bg_level=bg, note=note, **_measure(pose, s))


def condition_specs():
    """A grid that walks the two axes that actually matter: tilt and lighting.

    Opacity and background are deliberately *not* varied here -- both measured at
    ~1.1x spread across their whole range, so a row of them would be four
    identical pictures. Noise rides along with lighting because that is how a
    real camera couples them: less light, shorter exposure, more gain.
    """
    dome = lambda amb, i=12.0: rendermod.LightRig(  # noqa: E731
        dome=((60.0, 25.0),), ambient=amb, intensity=i)
    side = lambda amb: rendermod.LightRig(  # noqa: E731
        lateral_deg=(0.0,), ambient=amb, intensity=20.0, key_from_camera=False)

    out = []
    for tilt in (8.0, 22.0, 38.0, 55.0):
        out.append((tilt, 40.0, np.array([4.0, -3.0, 215.0]), dome(0.45), 0.9,
                    0.05, 8.0, f"tilt {tilt:.0f}°", "well lit, dome"))
    for amb, sig, label in ((0.50, 4.0, "bright"), (0.35, 10.0, "normal"),
                            (0.22, 20.0, "dim"), (0.12, 30.0, "very dim")):
        out.append((26.0, 40.0, np.array([4.0, -3.0, 215.0]), dome(amb), 0.9,
                    0.05, sig, f"ambient {amb:.2f}", f"{label}, σ={sig:.0f}"))
    for bg in (0.0, 0.2, 0.35, 0.5):
        out.append((26.0, 40.0, np.array([4.0, -3.0, 215.0]), dome(0.45), 0.9,
                    bg, 8.0, f"background {bg:.2f}", "grey level of the ground"))
    for a in (1.0, 0.85, 0.7):
        out.append((26.0, 40.0, np.array([4.0, -3.0, 215.0]), dome(0.45), a,
                    0.05, 8.0, f"opacity {a:.2f}", "material alpha"))
    return out


def failure_specs():
    """The known breakdowns, reproduced on purpose rather than hunted for.

    Each is a condition the sweep identified as a failure mode, so the gallery
    shows the *cause* next to the picture instead of leaving the reader to infer
    it from a scatter point.
    """
    dome = lambda amb, i=12.0: rendermod.LightRig(  # noqa: E731
        dome=((60.0, 25.0),), ambient=amb, intensity=i)
    side = lambda amb: rendermod.LightRig(  # noqa: E731
        lateral_deg=(0.0,), ambient=amb, intensity=20.0, key_from_camera=False)
    ctr = np.array([0.0, 0.0, 220.0])

    return [
        (2.0, 0.0, ctr, side(0.10), 1.0, 0.0, 6.0, "face-on + hard side light",
         "the rim is unlit; the hull collapses onto the blade cross"),
        (6.0, 0.0, ctr, dome(0.14), 1.0, 0.0, 25.0, "face-on + dim + noisy",
         "same collapse, with read noise inflating the hull"),
        (70.0, 0.0, ctr, dome(0.45), 0.9, 0.05, 8.0, "tilt 70°",
         "mast and magnet dominate the short axis; flat-circle model breaks"),
        (26.0, 40.0, ctr, dome(0.45), 0.9, 0.5, 8.0, "background at threshold",
         "grey 128 equals the threshold, so the whole frame is foreground"),
        (1.0, 0.0, ctr, dome(0.45), 0.9, 0.05, 8.0, "near head-on",
         "tilt is ill-conditioned: dθ/d(ratio) = -1/sinθ blows up"),
        (34.0, 210.0, np.array([0.0, 0.0, 350.0]), dome(0.40), 0.9, 0.05, 14.0,
         "long range", "fewer pixels across the rim, so depth degrades"),
    ]


def weighting_specs():
    """Poses that show what axial weighting changed, at a glance.

    Rendered twice each -- once with the weighting disabled, once with it on --
    so the fitted ellipse can be seen moving onto the rim instead of being
    dragged outward by the rod.
    """
    dome = lambda amb, i=12.0: rendermod.LightRig(  # noqa: E731
        dome=((60.0, 25.0),), ambient=amb, intensity=i)
    return [
        (70.0, 0.0, np.array([0.0, 0.0, 220.0]), dome(0.45), 0.9, 0.05, 8.0,
         "tilt 70°", "the rod pushes the silhouette outward along the short axis"),
        (60.0, 120.0, np.array([0.0, 0.0, 220.0]), dome(0.45), 0.9, 0.05, 8.0,
         "tilt 60°", ""),
        (50.0, 210.0, np.array([6.0, -4.0, 240.0]), dome(0.45), 0.9, 0.05, 8.0,
         "tilt 50°", ""),
    ]


def build_weighting(width=1024, height=768, gain=50.0):
    """Before/after tiles for the axial-weighting change.

    Kept in its own entry point because it must construct estimators with the
    weighting forced off, which the normal galleries never do.
    """
    out = []
    with rendermod.Renderer(width, height) as r:
        est = PoseEstimator(camera_matrix=r.K, dist_coeffs=None)
        for spec in weighting_specs():
            pair = {
                label: _tile(r, est, spec, gain, legend=(label == "unweighted"), axial=axial)
                for label, axial in (("unweighted", False), ("axial weighted", True))
            }
            out.append({"title": spec[7], "note": spec[8], "pair": pair})
    return out


def build(width=1024, height=768, gain=50.0):
    """Render both galleries. One Renderer for the process (pyglet allows one)."""
    with rendermod.Renderer(width, height) as r:
        est = PoseEstimator(camera_matrix=r.K, dist_coeffs=None)
        conditions = [_tile(r, est, s, gain, legend=(i == 0))
                      for i, s in enumerate(condition_specs())]
        failures = [_tile(r, est, s, gain) for s in failure_specs()]

        # A layer-by-layer breakdown of one clean frame, so the page can explain
        # what each overlay element means by showing it in isolation.
        spec = (26.0, 40.0, np.array([4.0, -3.0, 215.0]),
                rendermod.LightRig(dome=((60.0, 25.0),), ambient=0.45, intensity=12.0),
                0.9, 0.05, 8.0, "layers", "")
        layered = {
            "raw frame": _tile(r, est, spec, gain, layers=()),
            "fitted vs true ellipse": _tile(r, est, spec, gain, layers=("fit", "truth", "text")),
            "orientation vectors": _tile(r, est, spec, gain,
                                         layers=("fit", "truth", "normal", "text")),
            "both ambiguity branches": _tile(r, est, spec, gain,
                                             layers=("fit", "normal", "branches", "text")),
            "residual, amplified": _tile(r, est, spec, gain, layers=overlaymod.ALL_LAYERS,
                                         legend=True),
        }
    return {"conditions": conditions, "failures": failures, "layers": layered}
