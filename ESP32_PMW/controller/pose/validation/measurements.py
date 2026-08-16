"""The measurements behind the design decisions, recomputed at report time.

Every table here is produced from data generated in the same run that renders the
report -- the same poses, the same lighting draw, the same estimator.  Nothing is
transcribed from an earlier session.  That is the point: a diagnostics page whose
numbers were pasted in is a page that silently goes stale, and the interesting
numbers here are exactly the ones that would move if someone changed a threshold.

Four questions get answered, in the order they had to be answered:

1. **Where does the silhouette leave the rim?**  (`deviation_by_tilt`)
2. **Which part of the robot causes it?**  (`mesh_attribution` -- renders sliced
   copies of the mesh, which is the only way to attribute this rather than infer
   it from crossover arithmetic.)
3. **Which measurement channels survive?**  (`channel_reliability`)
4. **What actually fixes it?**  (`calibration_models`, `weighting_schemes`,
   `major_channel_ab`)
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import conic  # noqa: E402
import render as rendermod  # noqa: E402
import scene3d  # noqa: E402
import segment as segmod  # noqa: E402
import stereo as stereomod  # noqa: E402
import tune_weighting  # noqa: E402
from calibration import TiltCalibration  # noqa: E402

RESULTS = HERE.parents[2] / "results" / "pose_validation"

# Tilts for the mesh-attribution sweep. Chosen to straddle the mast crossover at
# arctan(R/h) = 52.5 deg, which is the thing being tested.
ATTRIB_TILTS = (25.0, 40.0, 55.0, 65.0, 70.0)
ATTRIB_LIGHT = dict(ambient=0.4, intensity=12.0)


# --------------------------------------------------------------------------
# 1 & 2: where the silhouette departs, and what causes it


def _mesh_variants(base):
    """The full mesh and three ablations, by dropping faces outside a z band.

    Faces rather than `trimesh.slice_plane`, which needs shapely and caps the
    cut; an uncapped shell is fine here because the renderer draws double-sided
    and only the silhouette is being measured.
    """
    V = np.asarray(base.vertices)
    F = np.asarray(base.faces)

    def band(lo, hi):
        ok = (V[:, 2] >= lo) & (V[:, 2] <= hi)
        return trimesh.Trimesh(vertices=V, faces=F[ok[F].all(axis=1)], process=False)

    return [
        ("full mesh", base),
        ("mast removed", band(-99.0, 2.0)),
        ("body removed", band(-2.0, 99.0)),
        ("rim ring only", band(-2.0, 2.0)),
    ]


def _centred_view(renderer):
    """A `render.View` whose intrinsics fit the renderer's frame.

    `Renderer.K` is the calibration's, whose principal point is near (498, 356).
    In a 1024x768 frame that is the centre; in the 500x375 frame this report uses
    it is the bottom-right corner, so a robot rendered on the optical axis falls
    out of frame and every measurement taken from it is nonsense. Scaling K to
    the frame -- through the identity `View`, which exists for exactly this -- puts
    it back in the middle.
    """
    scale = renderer.width / 1024.0
    K = renderer.K.copy()
    K[:2, :] *= scale
    return rendermod.View(K=K), K


def mesh_attribution(renderer, tilts=ATTRIB_TILTS):
    """Which part of the robot contaminates the silhouette, by deleting it.

    Renders the same poses with the mast, the magnet body, and both removed, and
    reports the peak outward deviation on each side of the minor axis. The
    attribution is then a fact rather than an inference: if deleting a part makes
    the contamination vanish, it was that part.

    Mutates and restores ``renderer.mesh``. One GL context per process, so the
    caller's renderer is borrowed rather than a second one created.
    """
    original = renderer.mesh
    view, K = _centred_view(renderer)
    light = rendermod.LightRig(dome=((60.0, 0.0),), **ATTRIB_LIGHT)
    rows = []
    try:
        for name, mesh in _mesh_variants(original):
            cells = []
            for tilt in tilts:
                renderer.mesh = mesh
                s = renderer.render(float(tilt), 0.0, [0.0, 0.0, 220.0], light=light,
                                    view=view)
                seg = segmod.segment(s.image)
                if seg is None:
                    cells.append(None)
                    continue
                truth = conic.normalise_ellipse(conic.project_circle(
                    s.center_mm, s.normal, rendermod.RIM_RADIUS_MM, K))
                t, dev = scene3d._deviation_profile(seg.contour, truth)
                plus = dev[(t > 45) & (t < 135)]
                minus = dev[(t < -45) & (t > -135)]
                cells.append((float(plus.max()) if len(plus) else 0.0,
                              float(minus.max()) if len(minus) else 0.0))
            rows.append({"name": name, "cells": cells})
    finally:
        renderer.mesh = original
    return {"tilts": list(tilts), "rows": rows}


def wall_model_fit(renderer, tilts=(15, 25, 35, 45, 55, 65, 70)):
    """Test the cylinder-wall hypothesis on the rim ring alone.

    If the rim's own height is what widens the silhouette, then the excess short
    half-extent is ``h sin(theta)`` and ``excess / sin(theta)`` must be constant
    and equal to ``h``.  Measured on the ring in isolation so the mast cannot
    contribute.
    """
    original = renderer.mesh
    V = np.asarray(original.vertices)
    F = np.asarray(original.faces)
    ok = (V[:, 2] >= -2.0) & (V[:, 2] <= 2.0)
    ring = trimesh.Trimesh(vertices=V, faces=F[ok[F].all(axis=1)], process=False)
    light = rendermod.LightRig(dome=((60.0, 0.0),), **ATTRIB_LIGHT)
    out = []
    view, K = _centred_view(renderer)
    try:
        renderer.mesh = ring
        mm_per_px = 220.0 / K[0, 0]
        for tilt in tilts:
            s = renderer.render(float(tilt), 0.0, [0.0, 0.0, 220.0], light=light,
                                view=view)
            seg = segmod.segment(s.image)
            if seg is None:
                continue
            truth = conic.normalise_ellipse(conic.project_circle(
                s.center_mm, s.normal, rendermod.RIM_RADIUS_MM, K))
            got = seg.ellipse[1][1] / 2 * mm_per_px
            want = truth[1][1] / 2 * mm_per_px
            excess = got - want
            out.append({"tilt": float(tilt), "excess_mm": excess,
                        "implied_h_mm": excess / math.sin(math.radians(tilt))})
    finally:
        renderer.mesh = original
    return out


# --------------------------------------------------------------------------
# 3: which channels survive


def _mad(v):
    """Median absolute deviation, scaled to be comparable to a standard deviation."""
    v = np.asarray(v, dtype=np.float64)
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def channel_reliability(batch, rig, radius_mm, bands=((20, 40), (40, 55), (55, 75))):
    """Major length, major direction and minor length, against analytic truth.

    Binned by the tilt each *camera* sees, not by the robot's lean, because the
    silhouette only knows about the former.
    """
    acc = {b: {"maj": [], "psi": [], "mnr": []} for b in bands}
    for item in batch:
        pose, sample = item["pose"], item["sample"]
        for cam, view, seg in zip(rig.cameras, sample.views, pose.per_view):
            if seg is None:
                continue
            seen = math.degrees(math.acos(min(1.0, abs(float(view.normal[2])))))
            band = next((b for b in bands if b[0] <= seen < b[1]), None)
            if band is None:
                continue
            try:
                truth = conic.normalise_ellipse(conic.project_circle(
                    view.center_mm, view.normal, radius_mm, cam.K))
            except Exception:
                continue
            got = conic.normalise_ellipse(seg.ellipse)
            acc[band]["maj"].append(got[1][0] / truth[1][0] - 1.0)
            acc[band]["mnr"].append(got[1][1] / truth[1][1] - 1.0)
            acc[band]["psi"].append(abs((got[2] - truth[2] + 90) % 180 - 90))

    rows = []
    for b in bands:
        d = acc[b]
        if not d["maj"]:
            continue
        rows.append({
            "band": f"{b[0]}–{b[1]}°",
            "n": len(d["maj"]),
            # MAD, not standard deviation: at n ~ 35 a single segmentation
            # failure moves the std by more than the effect being measured.
            "major_pct": 100 * float(np.median(d["maj"])),
            "major_sd": 100 * _mad(d["maj"]),
            "psi_deg": float(np.median(d["psi"])),
            "minor_pct": 100 * float(np.median(d["mnr"])),
            "minor_sd": 100 * _mad(d["mnr"]),
        })
    return rows


# --------------------------------------------------------------------------
# 4: what fixes it


def calibration_models(dataset=None, bands=((10, 20), (20, 40), (40, 55), (55, 71))):
    """Held-out comparison of the tilt models, from the tuning dataset.

    Uses the train/test split already in `dataset.npz`: every model is fitted on
    train and scored on test, so none of them is being graded on its own
    homework.
    """
    path = Path(dataset or RESULTS / "dataset.npz")
    if not path.exists():
        return None
    d = np.load(path)
    ratio = d["minor"] / d["major"]
    tilt = d["tilt_deg"]
    tr, te = d["split"] == 0, d["split"] == 1
    raw = np.degrees(np.arccos(np.clip(ratio, 0, 1)))

    models = [("none", None), ("quadratic", TiltCalibration.fit(raw[tr], tilt[tr]))]
    for lo, hi in ((5, 71), (20, 50)):
        try:
            models.append((f"cylinder, fit {lo}–{hi}°",
                           TiltCalibration.fit_cylinder(ratio[tr], tilt[tr],
                                                        tilt_range=(lo, hi))))
        except ValueError:
            pass

    rows = []
    for name, cal in models:
        cells = []
        for lo, hi in bands:
            m = te & (tilt >= lo) & (tilt < hi)
            if m.sum() < 5:
                cells.append(None)
                continue
            est = raw[m] if cal is None else np.array([cal.tilt(x) for x in raw[m]])
            cells.append((float(np.median(np.abs(est - tilt[m]))),
                          float(np.median(est - tilt[m]))))
        rows.append({"name": name, "cells": cells,
                     "k": None if cal is None or cal.model != "cylinder" else cal.k,
                     "floor": None if cal is None or cal.model != "cylinder"
                     else cal.resolution_floor_deg})
    return {"bands": [f"{a}–{b}°" for a, b in bands], "rows": rows,
            "n_test": int(te.sum())}


def weighting_schemes(batch, rig, radius_mm, bands=((20, 40), (40, 55), (55, 75))):
    """The three hull-fitting schemes, on the hulls this report already rendered.

    Cheap because the rendering is already paid for: this is ellipse fitting on
    existing contours. Fewer samples than `tune_weighting.py`'s dedicated sweep,
    but drawn from exactly the frames shown above, so the tables and the pictures
    describe the same data.
    """
    cal = TiltCalibration.load()
    acc = {name: {b: [] for b in bands} for name in tune_weighting.SCHEMES}
    for item in batch:
        pose, sample = item["pose"], item["sample"]
        for cam, view, seg in zip(rig.cameras, sample.views, pose.per_view):
            if seg is None:
                continue
            seen = math.degrees(math.acos(min(1.0, abs(float(view.normal[2])))))
            band = next((b for b in bands if b[0] <= seen < b[1]), None)
            if band is None:
                continue
            for name, fn in tune_weighting.SCHEMES.items():
                ell = fn(seg.contour)
                if ell is None:
                    continue
                r = min(1.0, max(0.0, ell[1][1] / max(ell[1][0], 1e-9)))
                acc[name][band].append(
                    cal.tilt(math.degrees(math.acos(r))) - seen)

    rows = []
    for name in tune_weighting.SCHEMES:
        cells = []
        for b in bands:
            v = np.array(acc[name][b])
            cells.append((float(np.median(np.abs(v))), float(np.median(v)))
                         if len(v) >= 5 else None)
        rows.append({"name": name, "cells": cells})
    return {"bands": [f"{a}–{b}°" for a, b in bands], "rows": rows}


def major_channel_ab(batch, rig, bands=((20, 40), (40, 55), (55, 75))):
    """Normal error with and without the major-axis channel, same frames.

    Re-solves from the stored per-view ellipses rather than re-rendering, so both
    arms see byte-identical inputs and the comparison isolates the channel.
    """
    rows = {False: {b: [] for b in bands}, True: {b: [] for b in bands}}
    for item in batch:
        pose, sample = item["pose"], item["sample"]
        if any(s is None for s in pose.per_view):
            continue
        seen = min(rig.tilt_seen_deg(sample.normal_world))
        band = next((b for b in bands if b[0] <= seen < b[1]), None)
        if band is None:
            continue
        base = pose.normal
        rows[False][band].append(
            stereomod.line_angle_deg(base, sample.normal_world))

        got = stereomod.solve_from_major(
            [s.ellipse for s in pose.per_view], rig, pose.xyz_mm)
        if got is None:
            rows[True][band].append(rows[False][band][-1])
            continue
        n_major, sigma = got
        blended, _ = stereomod.blend_normals(
            base, stereomod.ratio_sigma_deg(seen), n_major, sigma)
        rows[True][band].append(
            stereomod.line_angle_deg(blended, sample.normal_world))

    out = []
    for label, key in (("ratio channel only (shipped)", False),
                       ("+ major-axis channel", True)):
        cells = []
        for b in bands:
            v = np.array(rows[key][b])
            cells.append(float(np.median(v)) if len(v) >= 5 else None)
        out.append({"name": label, "cells": cells})
    return {"bands": [f"{a}–{b}°" for a, b in bands], "rows": out}


# --------------------------------------------------------------------------
# A chart


def wall_model_chart(fit, size=(6.4, 2.8), dpi=150):
    """``excess / sin(theta)`` against tilt -- flat if the wall model holds."""
    fig, ax = plt.subplots(figsize=size, dpi=dpi, facecolor=scene3d.PANEL_BG)
    ax.set_facecolor(scene3d.PANEL_BG)
    t = [f["tilt"] for f in fit]
    h = [f["implied_h_mm"] for f in fit]
    ax.plot(t, h, "o-", color=scene3d.COL_CAM[0], lw=1.2, ms=4,
            label="measured  excess / sin(tilt)")
    if h:
        ax.axhline(float(np.mean(h)), color=scene3d.COL_TRUTH, lw=1.0, ls="--",
                   label=f"mean = {np.mean(h):.3f} mm")
    ax.set_xlabel("tilt (deg)", color=scene3d.COL_INK_3, fontsize=8)
    ax.set_ylabel("implied wall\nhalf-height (mm)", color=scene3d.COL_INK_3, fontsize=8)
    ax.set_ylim(0, max(h + [1.0]) * 1.4)
    ax.tick_params(colors=scene3d.COL_INK_3, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(scene3d.COL_LINE)
    ax.grid(True, color=scene3d.COL_LINE, lw=0.5, alpha=0.6)
    leg = ax.legend(fontsize=7.5, frameon=True, framealpha=1.0,
                    edgecolor=scene3d.COL_LINE, facecolor=scene3d.PANEL_BG)
    for txt in leg.get_texts():
        txt.set_color(scene3d.COL_INK)
    buf = io.BytesIO()
    fig.tight_layout(pad=0.4)
    fig.savefig(buf, format="png", facecolor=scene3d.PANEL_BG, dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def centre_bias(renderer, tilts=(10, 25, 40, 55, 65, 70)):
    """Is the fitted ellipse's *centre* displaced, and does re-weighting fix it?

    Easy to overlook, because the obvious symptom of out-of-plane structure is a
    fattened short axis and that is what everything else here measures.  But the
    convex hull grows on **one** side, so the centre moves too -- and lateral
    position is read straight off the centre, which makes this a position error
    rather than an orientation one.

    Decomposed along the true ellipse's own axes, because the displacement is
    not isotropic: it is almost entirely along the short axis, which is the
    direction the rim wall and the rod protrude in.

    Scheme C is included because it is the natural thing to reach for and it
    mostly does not work here: its weights are computed from the current
    ellipse, which has already absorbed the shift, so an IRLS started from a
    biased fit converges to a nearby biased fit.  The rod's contribution is a
    coherent arc, not a sprinkling of outliers.
    """
    original = renderer.mesh
    view, K = _centred_view(renderer)
    light = rendermod.LightRig(dome=((60.0, 0.0),), **ATTRIB_LIGHT)
    mm_per_px = 220.0 / K[0, 0]
    rows = []
    try:
        for tilt in tilts:
            s = renderer.render(float(tilt), 0.0, [0.0, 0.0, 220.0], light=light,
                                view=view)
            seg = segmod.segment(s.image)
            if seg is None:
                continue
            truth = conic.normalise_ellipse(conic.project_circle(
                s.center_mm, s.normal, rendermod.RIM_RADIUS_MM, K))
            (tx, ty), _, tang = truth
            th = math.radians(tang)
            c, sn = math.cos(th), math.sin(th)

            def decomp(ellipse):
                (ex, ey), _, _ = conic.normalise_ellipse(ellipse)
                dx, dy = ex - tx, ey - ty
                return c * dx + sn * dy, -sn * dx + c * dy

            d_maj, d_min = decomp(seg.ellipse)
            fixed = tune_weighting.fit_one_sided(seg.contour)
            f_maj, f_min = decomp(fixed) if fixed else (float("nan"),) * 2
            rows.append({
                "tilt": float(tilt),
                "d_major_px": d_maj, "d_minor_px": d_min,
                "plain_mm": math.hypot(d_maj, d_min) * mm_per_px,
                "one_sided_mm": math.hypot(f_maj, f_min) * mm_per_px,
            })
    finally:
        renderer.mesh = original
    return rows
