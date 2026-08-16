"""The stereo geometry, drawn in 3-D: cameras, back-projection cones, pose.

Everything else in this package looks at the problem through a camera, in
pixels.  This is the one view from outside it -- where the cameras actually are,
what their back-projection cones look like, and where the estimated disk sits
against the true one.

**Ground truth is black, the estimate is amber.**  Amber (`#C8930A`) rather than
pure yellow because the panel is white: pure yellow on white is close to
unreadable, and the point of a diagnostics figure is to be read.  The canvas is
fixed white in both page themes, which needs no special handling -- an opaque PNG
is unaffected by the CSS around it.

**At true scale the two poses coincide.**  The estimate sits ~0.2 mm from truth
on a 20.4 mm disk, so on the context panel that is 0.04% of the frame and on the
zoom panel about 0.6%.  Nothing here is amplified: the disks really do draw on
top of each other, that really is what 0.2 mm looks like, and the numbers in the
caption carry the quantitative story.  A figure that spread them apart for
legibility would be claiming an error the estimator does not make.

**The rays are the measurement, not a decoration.**  Each one leaves a camera
centre through a pixel on that view's *fitted* ellipse and is cut at the plane of
the estimated disk.  So the ring of ray endpoints **is** the observed rim,
back-projected -- and how well that ring lands on the drawn circle is the
reprojection residual, in millimetres, in space.  Drawing rays to the disk
instead would have shown a tidier picture of nothing.

matplotlib's 3-D axes sort whole artists rather than fragments, so a ray crossing
the disk can paint over it.  Rays are therefore drawn first and faint, the disks
last and opaque; with 16 rays a camera the ordering reads correctly.
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

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

COL_TRUTH = "#000000"
COL_EST = "#C8930A"
# Camera colours are page.py's S1 and S3 -- blue and violet. S2 is orange, which
# sits too close to the estimate's amber to be told apart at 0.6 pt line width.
COL_CAM = ("#2A8FC4", "#8465E4")

PANEL_BG = "#FFFFFF"
COL_LINE = "#D6DCE4"  # page.py's --line, so the panel edges match the page
COL_INK = "#10161C"
COL_INK_3 = "#7A8896"

AXIS_LEN_MM = 14.0
N_RAYS = 16
N_DISK = 96
IMAGE_PLANE_MM = 45.0  # where to draw each camera's image quad along its axis


def _circle_points(center, normal, radius, n=N_DISK):
    """``n`` points on the circle of given centre, normal and radius."""
    n_hat = np.asarray(normal, dtype=np.float64)
    n_hat = n_hat / np.linalg.norm(n_hat)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_hat, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return (np.asarray(center, dtype=np.float64)[None, :]
            + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v))


def _ellipse_pixels(ellipse, n=N_RAYS):
    """``n`` evenly spaced points around an OpenCV ellipse, in pixels."""
    (cx, cy), (major, minor), ang = ellipse
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    a, b, th = major / 2.0, minor / 2.0, math.radians(ang)
    return np.column_stack([
        cx + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th),
        cy + a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th),
    ])


def _rays_to_plane(cam, pixels, plane_point, plane_normal, max_mm=1e4):
    """Where each pixel's viewing ray meets a plane, in world coordinates.

    ``K^-1 [u, v, 1]`` is the ray direction in camera coordinates (a point at
    unit depth), rotated into the world by the camera's ``R``.  The ray is then
    cut at the plane, which is what makes the endpoints meaningful: they are the
    observed rim back-projected onto the estimated disk's own plane, so their
    distance from the drawn circle is the reprojection residual in millimetres.

    Rays running nearly parallel to the plane are dropped rather than drawn to
    infinity; that only happens when the fit has already failed.
    """
    eye = cam.position
    dirs = np.hstack([np.asarray(pixels, dtype=np.float64),
                      np.ones((len(pixels), 1))]) @ cam.K_inv.T
    dirs = dirs @ cam.R.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    denom = dirs @ n
    num = float(n @ (np.asarray(plane_point, dtype=np.float64) - eye))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(denom) > 1e-6, num / denom, np.nan)
    ok = np.isfinite(t) & (t > 0) & (t < max_mm)
    return eye, eye[None, :] + t[:, None] * dirs, ok


def _image_quad(cam, width, height, depth_mm=IMAGE_PLANE_MM):
    """The camera's image rectangle, placed in world space at ``depth_mm``."""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height], [0, 0]],
                       dtype=np.float64)
    d = np.hstack([corners, np.ones((5, 1))]) @ cam.K_inv.T
    return (d * depth_mm) @ cam.R.T + cam.position


def _pixels_to_world(cam, pixels, depth_mm=IMAGE_PLANE_MM):
    """Pixels placed on the camera's image quad, in world space."""
    d = np.hstack([np.asarray(pixels, dtype=np.float64),
                   np.ones((len(pixels), 1))]) @ cam.K_inv.T
    return (d * depth_mm) @ cam.R.T + cam.position


def _view_for(normal, off_axis_deg=48.0, swing_deg=28.0):
    """A viewpoint that never looks down the disk's own plane.

    A fixed camera angle is fine until a pose happens to align with it, at which
    point the disk projects to a line and the panel shows nothing at all.  This
    places the viewer a fixed angle off the rotor axis instead, so the disk
    always presents as a reasonably open ellipse whatever the robot is doing.

    `matplotlib`'s ``view_init(elev, azim)`` positions the viewer on a sphere, so
    the viewing direction is that spherical point; taking the normal's own
    elevation and stepping ``off_axis_deg`` away from it gives the wanted angle
    between viewer and disk plane.  ``swing_deg`` rotates around the axis so the
    result is not accidentally aligned with a coordinate plane either.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    elev_n = math.degrees(math.asin(float(np.clip(n[2], -1.0, 1.0))))
    azim_n = math.degrees(math.atan2(float(n[1]), float(n[0])))
    return float(np.clip(elev_n - off_axis_deg, -78.0, 78.0)), azim_n + swing_deg


def _pose_artists(ax, center, normal, radius_mm, colour, label, lw, zorder):
    """One disk plus its rotation axis."""
    pts = _circle_points(center, normal, radius_mm)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colour, lw=lw,
            zorder=zorder, label=label, solid_capstyle="round")

    n_hat = np.asarray(normal, dtype=np.float64)
    n_hat = n_hat / np.linalg.norm(n_hat)
    c = np.asarray(center, dtype=np.float64)
    axis = np.array([c - AXIS_LEN_MM * n_hat, c + AXIS_LEN_MM * n_hat])
    ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], color=colour, lw=lw * 0.8,
            zorder=zorder, solid_capstyle="round")
    # A dot on the +n end, so the axis reads as directed rather than as a bar.
    tip = c + AXIS_LEN_MM * n_hat
    ax.scatter([tip[0]], [tip[1]], [tip[2]], color=colour, s=14, zorder=zorder,
               depthshade=False)


def _equalise(ax, pts, margin=0.52):
    """Isotropic axes around ``pts``.

    Without this the box is stretched to whatever range each axis happens to
    span, and a perfectly circular disk renders as an ellipse -- which in a
    figure about pose error is not a cosmetic problem, it is a wrong answer.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid = 0.5 * (lo + hi)
    span = float(np.max(hi - lo)) * margin or 1.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _style(ax, title):
    ax.set_facecolor(PANEL_BG)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 1.0))
        pane._axinfo["grid"].update(color=COL_LINE, linewidth=0.6)
        pane.line.set_color(COL_LINE)
    ax.tick_params(colors=COL_INK_3, labelsize=6.5, pad=0)
    # labelpad 0: mplot3d places axis labels outside the tick labels already,
    # and pulling them in further collides with the ticks on a square box.
    ax.set_xlabel("x (mm)", color=COL_INK_3, fontsize=8, labelpad=0)
    ax.set_ylabel("y (mm)", color=COL_INK_3, fontsize=8, labelpad=0)
    ax.set_zlabel("z (mm)", color=COL_INK_3, fontsize=8, labelpad=0)
    if title:
        ax.set_title(title, color=COL_INK, fontsize=10, pad=2)


def render(rig, truth_center, truth_normal, est_center, est_normal, radius_mm,
           ellipses=None, image_size=None, zoom=False, view=None,
           size=(6.4, 5.2), dpi=150, title=None):
    """Draw one case and return PNG bytes.

    ``ellipses`` are the per-view fitted ellipses in ideal pinhole pixels, one
    per camera; omit them to draw the poses and cameras without the cones.
    ``image_size`` is ``(width, height)``, needed only for the image quads.

    ``zoom`` swaps the framing: the context view spans both cameras and their
    cones, the zoom view spans the disk alone at true scale, which is the only
    place a 0.2 mm difference has any chance of being visible.

    ``view`` defaults to a fixed angle for the context panel -- which is about
    the rig, and the rig does not move -- and to `_view_for` on the zoom panel,
    which follows the pose so the disk is never caught edge-on.
    """
    fig = plt.figure(figsize=size, dpi=dpi, facecolor=PANEL_BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=PANEL_BG)

    truth_center = np.asarray(truth_center, dtype=np.float64)
    est_center = np.asarray(est_center, dtype=np.float64)
    extent = [truth_center, est_center]

    # --- rays first, faint, so the opaque disks paint over them -------------
    if ellipses is not None:
        for i, (cam, ellipse) in enumerate(zip(rig.cameras, ellipses)):
            if ellipse is None:
                continue
            colour = COL_CAM[i % len(COL_CAM)]
            pix = _ellipse_pixels(ellipse)
            eye, hits, ok = _rays_to_plane(cam, pix, est_center, est_normal)

            # The endpoints are kept on the zoom panel even though the rays
            # themselves are not: where the observed rim lands on the estimated
            # disk's plane *is* the reprojection residual, and at this scale it
            # is the one thing you can actually read off the figure.
            if ok.any():
                ax.scatter(hits[ok, 0], hits[ok, 1], hits[ok, 2], color=colour,
                           s=9 if zoom else 4, alpha=0.9, zorder=8 if zoom else 2,
                           depthshade=False,
                           label=f"rim seen by {cam.name or i}" if zoom else None)
            if zoom:
                continue

            for hit in hits[ok]:
                ax.plot([eye[0], hit[0]], [eye[1], hit[1]], [eye[2], hit[2]],
                        color=colour, lw=0.6, alpha=0.35, zorder=1)

            if image_size is not None:
                quad = _image_quad(cam, *image_size)
                ax.plot(quad[:, 0], quad[:, 1], quad[:, 2], color=colour, lw=0.8,
                        alpha=0.7, zorder=2)
                on_plane = _pixels_to_world(cam, _ellipse_pixels(ellipse, N_DISK))
                ax.plot(on_plane[:, 0], on_plane[:, 1], on_plane[:, 2],
                        color=colour, lw=1.2, zorder=3)
                extent.append(quad)

            ax.scatter([eye[0]], [eye[1]], [eye[2]], color=colour, s=34,
                       zorder=4, depthshade=False,
                       label=f"camera {cam.name or i}")
            ax.text(eye[0], eye[1], eye[2], f"  cam {cam.name or i}",
                    color=colour, fontsize=8, zorder=5)
            extent.append(eye[None, :])

    # --- the two poses, last and opaque -------------------------------------
    _pose_artists(ax, truth_center, truth_normal, radius_mm, COL_TRUTH,
                  "ground truth", 2.0, 6)
    _pose_artists(ax, est_center, est_normal, radius_mm, COL_EST,
                  "estimate", 1.6, 7)

    if zoom:
        # Framed on the axis, which is the widest thing here, with a little air.
        pad = AXIS_LEN_MM + 2.0
        _equalise(ax, np.array([truth_center - pad, truth_center + pad]), margin=0.5)
    else:
        _equalise(ax, np.vstack([np.atleast_2d(e) for e in extent]))

    if view is None:
        view = _view_for(est_normal) if zoom else (20.0, 40.0)
    ax.view_init(elev=view[0], azim=view[1])
    _style(ax, title)

    leg = ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=1.0,
                    edgecolor=COL_LINE, facecolor=PANEL_BG)
    for text in leg.get_texts():
        text.set_color(COL_INK)

    buf = io.BytesIO()
    fig.tight_layout(pad=0.4)
    fig.savefig(buf, format="png", facecolor=PANEL_BG, dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------
# The diagnostic that settles the shape question


def _deviation_profile(hull, ellipse):
    """Signed radial deviation of hull points from an ellipse, and where.

    Returns ``(t_deg, dev_px)``.  ``t`` is the ellipse parameter measured from
    the **major-axis tip**, so 0 and 180 are the ends of the long axis and +-90
    are the ends of the short one.  ``dev`` is positive outside the ellipse.

    That parameterisation is the whole point: it turns "the fit is bad at high
    tilt" into a statement about *where* on the silhouette the model fails, and
    the answer is not uniform. Measured against the true rim ellipse, the
    deviation is positive almost everywhere -- which it must be, since the rim's
    projection is convex and contained in the silhouette's convex hull, so the
    hull can only ever lie outside it.
    """
    (cx, cy), (major, minor), ang = ellipse
    a, b, th = major / 2.0, minor / 2.0, math.radians(ang)
    c, s = math.cos(th), math.sin(th)
    x = np.asarray(hull, dtype=np.float64)[:, 0] - cx
    y = np.asarray(hull, dtype=np.float64)[:, 1] - cy
    u = c * x + s * y      # along the major axis
    v = -s * x + c * y     # along the minor axis
    t = np.degrees(np.arctan2(v / max(b, 1e-9), u / max(a, 1e-9)))
    k = np.sqrt((u / max(a, 1e-9)) ** 2 + (v / max(b, 1e-9)) ** 2)
    r = np.hypot(a * np.cos(np.radians(t)), b * np.sin(np.radians(t)))
    return t, (k - 1.0) * r


def deviation_panel(views, size=(6.4, 3.0), dpi=150, title=None):
    """Where the silhouette departs from the rim ellipse, per view.

    ``views`` is a list of ``(name, hull, true_ellipse)``.

    Read it as follows. Contamination concentrated near ``t = +-90`` is
    out-of-plane structure -- the rim wall's own height, and past ~57 degrees of
    tilt the mast -- pushing the short direction outward. Contamination near
    ``t = 0, 180`` would be something wrong with the *long* direction, which is
    the axis depth is measured from; measured, there is essentially none, and
    that is why position survives tilts that destroy orientation.
    """
    fig, ax = plt.subplots(figsize=size, dpi=dpi, facecolor=PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    for i, (name, hull, ellipse) in enumerate(views):
        if hull is None or ellipse is None:
            continue
        t, dev = _deviation_profile(hull, ellipse)
        order = np.argsort(t)
        colour = COL_CAM[i % len(COL_CAM)]
        ax.plot(t[order], dev[order], "o-", color=colour, ms=2.6, lw=1.0,
                alpha=0.85, label=f"camera {name}")

    ax.axhline(0.0, color=COL_TRUTH, lw=1.0, zorder=1)
    for x in (-90, 90):
        ax.axvline(x, color=COL_LINE, lw=1.0, ls="--", zorder=0)
    ax.text(90, ax.get_ylim()[1], " minor axis\n (short direction)", fontsize=7,
            color=COL_INK_3, va="top")
    ax.text(0, ax.get_ylim()[1], " major axis", fontsize=7, color=COL_INK_3, va="top")

    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_xlabel("position around the rim, degrees from the major-axis tip",
                  color=COL_INK_3, fontsize=8)
    ax.set_ylabel("outward deviation (px)", color=COL_INK_3, fontsize=8)
    ax.tick_params(colors=COL_INK_3, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(COL_LINE)
    ax.grid(True, color=COL_LINE, lw=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=COL_INK, fontsize=10, pad=4)
    leg = ax.legend(fontsize=7.5, frameon=True, framealpha=1.0,
                    edgecolor=COL_LINE, facecolor=PANEL_BG, loc="lower center", ncol=2)
    for txt in leg.get_texts():
        txt.set_color(COL_INK)

    buf = io.BytesIO()
    fig.tight_layout(pad=0.4)
    fig.savefig(buf, format="png", facecolor=PANEL_BG, dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
