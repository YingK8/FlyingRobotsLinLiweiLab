#!/usr/bin/env python3
"""
What the solve is worth, and what gets written if it is worth anything.

`theory.md` section 14.5. The gate is not a formality: a bad extrinsic produces poses that
are smooth, plausible, self-consistent and wrong, so a failed run writes nothing.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from controller.calib.calibrate import OUT_DIR, RIG_PATH, SQUARE_MM, pose_matrix
from controller.calib.rig import Camera, StereoRig

# Two kinds of gate, and only one of them is a matter of taste.
#
# *Precision* gates are reprojection RMS. They convert: at a working distance $z$ with rays
# $\theta$ apart, a residual of $\sigma$ px is $z\sigma/(f\sin\theta)$ of 3D uncertainty.
# On this bench (119 mm, f = 2748, 83 deg) that is 22 um at 0.5 px and 44 um at 1.0. The
# robot is 10 mm across and the hover loop runs at 0.8 Hz, so anything under ~100 um is
# free. 0.5 px was a round number, not a requirement; 1.0 px is 44 um and still 200x
# smaller than the thing being measured.
MAX_RMS_PX = 1.0
MIN_PAIRS = 6              # the extrinsic is 6 DOF and every pair gives a whole estimate of
                           # it, so what matters is that they *agree* -- which is its own
                           # gate below. `MIN_VIEWS` = 8 is the intrinsics' number, where
                           # one view constrains a fraction of many more parameters.
#
# *Correctness* gates are the rest, and none of them move. A bad extrinsic is smooth,
# plausible and wrong: nothing downstream will notice it, so the closed-form seed, the
# residual structure and the pair-to-pair agreement are the only things standing in the
# way. Scale is worse still -- no residual here can see it at all (section 14.4).
MAX_RADIAL_RATIO = 1.5


def stereo_residuals(pairs, K_a, dist_a, K_b, dist_b, T_ba, rvecs, tvecs):
    """Residuals from the **joint** fit: one board pose per pair, explaining both views.

    Re-solving each camera independently would measure only that camera's intrinsics, so a
    completely wrong ``T_ba`` would still produce a clean residual and sail through the
    gate. Here camera B's residual carries the extrinsic error (section 14.5). The poses
    come from `refine_extrinsic`'s own bundle, not a second solver layered on top.
    """
    out = {"A": {"rad": [], "res": []}, "B": {"rad": [], "res": []}, "per_pair": []}
    for p, rvec, tvec in zip(pairs, rvecs, tvecs):
        obj = p["obj"].reshape(-1, 3).astype(np.float64)
        ia, ib = p["img_a"].reshape(-1, 2), p["img_b"].reshape(-1, 2)

        T_b = T_ba @ pose_matrix(rvec, tvec)
        rb, _ = cv2.Rodrigues(T_b[:3, :3])
        pa, _ = cv2.projectPoints(obj, rvec, tvec, K_a, dist_a)
        pb, _ = cv2.projectPoints(obj, rb, T_b[:3, 3], K_b, dist_b)
        da, db = pa.reshape(-1, 2) - ia, pb.reshape(-1, 2) - ib

        out["A"]["res"].append(da)
        out["B"]["res"].append(db)
        out["A"]["rad"].append(np.linalg.norm(ia - [K_a[0, 2], K_a[1, 2]], axis=1))
        out["B"]["rad"].append(np.linalg.norm(ib - [K_b[0, 2], K_b[1, 2]], axis=1))
        out["per_pair"].append([float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
                                for d in (da, db)])

    for tag in "AB":
        out[tag]["res"] = np.concatenate(out[tag]["res"])
        out[tag]["rad"] = np.concatenate(out[tag]["rad"])
    out["per_pair"] = np.asarray(out["per_pair"])
    out["rms_px"] = float(np.sqrt(np.mean(np.sum(
        np.vstack([out["A"]["res"], out["B"]["res"]]) ** 2, axis=1))))
    print(f"joint stereo residual: {out['rms_px']:.4f} px RMS over both views "
          f"(A {np.sqrt(np.mean(np.sum(out['A']['res'] ** 2, axis=1))):.4f}, "
          f"B {np.sqrt(np.mean(np.sum(out['B']['res'] ** 2, axis=1))):.4f})")
    return out


def structure_report(radii, res, n_bins=4):
    """Radial trend and isotropy of one camera's residuals."""
    edges = np.quantile(radii, np.linspace(0, 1, n_bins + 1))
    binned = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (radii >= lo) & (radii <= hi)
        binned.append(float(np.sqrt(np.mean(np.sum(res[m] ** 2, axis=1)))) if m.any()
                      else float("nan"))
    ratio = binned[-1] / binned[0] if binned[0] > 0 else float("inf")
    return {"rms_px": float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))),
            "radial_bins_px": binned, "radial_ratio": float(ratio),
            "anisotropy": float(np.std(res[:, 0]) / max(np.std(res[:, 1]), 1e-12)),
            "n_points": int(len(res))}


def uncertainty_um(rms_px, z_mm, f_px, sep_deg):
    """What a reprojection residual is worth as 3D noise, in microns."""

    return 1e3 * z_mm * rms_px / (f_px * max(math.sin(math.radians(sep_deg)), 1e-6))


def acceptance(stereo_info, resid, intr_a, intr_b, struct_a, struct_b, spread, um=None):
    """Print the gate and return True only if everything passes."""
    px = f"< {MAX_RMS_PX:g} px"
    checks = [
        (f"joint stereo RMS {px}", resid["rms_px"] < MAX_RMS_PX,
         f"{resid['rms_px']:.4f} px" + (f" = {um(resid['rms_px']):.0f} um" if um else "")),
        (f"bundle RMS {px}", stereo_info["rms_px"] < MAX_RMS_PX,
         f"{stereo_info['rms_px']:.4f} px"),
        ("bundle agrees with closed-form seed", stereo_info["seed_gap_deg"] < 1.0,
         f"{stereo_info['seed_gap_deg']:.3f} deg, {stereo_info['seed_gap_mm']:.3f} mm"),
        (f"camera A intrinsics RMS {px}", intr_a["rms_px"] < MAX_RMS_PX,
         f"{intr_a['rms_px']:.4f} px"),
        (f"camera B intrinsics RMS {px}", intr_b["rms_px"] < MAX_RMS_PX,
         f"{intr_b['rms_px']:.4f} px"),
        ("A residuals structureless", struct_a["radial_ratio"] < MAX_RADIAL_RATIO,
         f"outer/inner {struct_a['radial_ratio']:.2f}"),
        ("B residuals structureless", struct_b["radial_ratio"] < MAX_RADIAL_RATIO,
         f"outer/inner {struct_b['radial_ratio']:.2f}"),
        ("pairs agree on rotation", spread["rot_deg_max"] < 2.0,
         f"worst {spread['rot_deg_max']:.2f} deg"),
        ("enough pairs", stereo_info["n_pairs"] >= MIN_PAIRS,
         f"{stereo_info['n_pairs']} pairs"),
    ]
    # Advisory, not a gate. Directional residuals name a noise source with a preferred
    # direction -- a board that is not flat, a drift during the pair -- and are worth
    # chasing. They cannot mean a *wrong* extrinsic while the closed-form seed still agrees
    # and the radial trend is flat, and at 30 um of total 3D uncertainty against a 10 mm
    # robot, doubling one axis changes nothing downstream (section 14.5a).
    warnings = [(f"{tag} residuals isotropic", 0.67 < st["anisotropy"] < 1.5,
                 f"sx/sy {st['anisotropy']:.2f}")
                for tag, st in (("A", struct_a), ("B", struct_b))]

    print("acceptance (theory.md section 14.5)")
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:36s} {detail}")
    for label, ok, detail in warnings:
        print(f"  [{'ok  ' if ok else 'WARN'}] {label:36s} {detail}")
    passed = all(ok for _, ok, _ in checks)
    print(f"\n{'PASS' if passed else 'FAIL'} overall")
    if passed and not all(ok for _, ok, _ in warnings):
        print("Passed, with directional residuals. Look for a board that is not flat, or "
              "movement\nduring the pair, before trusting this rig below ~50 um.")
    if not passed:
        print("A failed gate is not a formality: a bad extrinsic produces poses that are\n"
              "smooth, plausible, self-consistent and wrong. Fix the capture, not the limit.")
    return passed


def build_rig(K_a, dist_a, K_b, dist_b, T_ba, spec, image_size,
              intr_a, intr_b, stereo_info, spread, ids=()):
    """StereoRig in the camera-A world frame."""
    return StereoRig(
        cameras=(
            Camera(K=K_a, dist=dist_a, T_world_cam=np.eye(4), name="A"),
            Camera(K=K_b, dist=dist_b, T_world_cam=np.linalg.inv(T_ba), name="B"),
        ),
        meta={
            "source": "calib/calibrate.py",
            # Which two devices shot this bag. A set, not an assignment: no macOS
            # listing enumerates in OpenCV's order, so nothing can say which of
            # these was A. Used only to catch a *different* camera in the pair.
            "elp_ids": sorted(ids),
            "world_frame": "camera_A",
            "world_frame_note": (
                "T_world_camA = I. +z is camera A's optical axis, NOT up. baseline_mm and "
                "axis_separation_deg are frame-independent and valid; tilt_seen_deg and any "
                "elevation reading are not, until the rig is rebased onto the drone disk "
                "frame at integration with the pose estimator."),
            "square_mm_measured": spec.square_mm != SQUARE_MM,
            **spec.summary(),
            "image_size": list(image_size),
            "n_pairs": stereo_info["n_pairs"],
            "rms_px": stereo_info["rms_px"],
            "rms_px_intrinsics": {"A": intr_a["rms_px"], "B": intr_b["rms_px"]},
            "pair_spread": spread,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


def save_intrinsics(path, K, dist, spec, info):
    np.savez(path, camera_matrix=K, dist_coeffs=dist,
             image_size=np.array(info["image_size"]), rms=info["rms_px"],
             board=spec.name, board_squares=np.array([spec.cols, spec.rows]),
             square_len=spec.square_mm, marker_len=spec.marker_mm)
    return path


def write_results(cal, spec, rig_path=RIG_PATH, out_dir=OUT_DIR, meta=None):
    """The rig in the camera-A world frame, plus each camera's intrinsics."""

    rig = build_rig(cal["K_a"], cal["dist_a"], cal["K_b"], cal["dist_b"], cal["T_ba"],
                    spec, cal["image_size"], cal["intr_a"], cal["intr_b"],
                    cal["stereo_info"], cal["spread"],
                    ids=cal.get("elp_ids", ()))
    rig.meta.update(meta or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"wrote {rig.save(rig_path)}")
    for tag in "AB":
        k = tag.lower()
        print("wrote", save_intrinsics(out_dir / f"camera_intrinsics_{tag}.npz",
                                       cal[f"K_{k}"], cal[f"dist_{k}"], spec,
                                       cal[f"intr_{k}"]))

    print("\nrig geometry (frame-independent quantities only):")
    print(f"  baseline_mm          {rig.baseline_mm():.3f}")
    print(f"  axis_separation_deg  {rig.axis_separation_deg():.3f}")
    print(f"  bisector_incidence   {rig.bisector_incidence_deg():.3f}  "
          f"(best a flat board can show both cameras)")
    if rig.axis_separation_deg() < 20.0:
        print("  WARNING: under 20 deg between the optical axes. Fusion needs the cameras "
              "to disagree about which direction is depth; below ~20 deg they barely do.")
    print("\nStill owed, and no code here can do it: triangulate a caliper-measured "
          "distance in the working volume and compare.")
    return rig
