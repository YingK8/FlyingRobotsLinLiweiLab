"""
The compiled pose core, wrapped as a drop-in `stereo.StereoPoseEstimator`.

    `pmw_pose` (controller/native, C++) ports everything `update` does per frame for the
    live configuration -- segmentation, back-projection, match, fuse, the image-mode
    refine -- and hands back plain numbers. This module turns those into the same
    `StereoPose` every consumer already reads, so nothing downstream knows which core
    produced it. The Python estimator stays as the reference; `native_parity.py` holds
    the two to each other. See `theory.md` 21.

    Built lazily on the first frame, because the pixel scale is only known then
    (`_match_scale`), and rebuilt if the scale changes. Every tuning constant the C++
    side needs is read from the Python module constants in `native_config` -- there is
    no second copy of a number anywhere in `controller/native`.
"""

from __future__ import annotations

import copy
import math
import time

import cv2
import numpy as np

from controller.calib.shape import TiltCalibration
from controller.pose import conic
from controller.pose import segment as segmod
from controller.pose import stereo
from controller.pose.estimator import _angles_from_normal
from controller.pose.filter import ACCEL_MM_S2
from controller.pose.stereo import Match, StereoPose, StereoPoseEstimator

try:
    import pmw_pose
except ImportError:  # `uv sync --extra native` builds it
    pmw_pose = None


def available():
    return pmw_pose is not None


def native_config(est):
    """The C++ `Config`, from the Python module constants and this estimator's state.

    ``est`` must have seen its first frame (`_match_scale`), so `_ksize` and
    `min_area` are already at the frame's scale.
    """

    return dict(
        ring_ksize=int(segmod.RING_KSIZE if est._ksize is None else est._ksize),
        ring_blur_sigma=float(segmod.RING_BLUR_SIGMA),
        ring_plate_weight=float(segmod.RING_PLATE_WEIGHT),
        ring_samples=int(segmod.RING_SAMPLES),
        ring_coverage_floor=float(segmod.RING_COVERAGE_FLOOR),
        roi_margin=float(segmod.ROI_MARGIN),
        bg_diff_thresh=int(segmod.BG_DIFF_THRESH),
        blob_keep_fraction=float(segmod._BLOB_KEEP_FRACTION),
        ring_max_spread=float(segmod.RING_MAX_SPREAD),
        shape_tol=float(segmod.SHAPE_TOL),
        ring_band=float(segmod.RING_BAND),
        ring_regrow_iters=int(segmod.RING_REGROW_ITERS),
        max_anchors=int(segmod._MAX_ANCHORS),
        min_contour_pts=int(segmod._MIN_CONTOUR_PTS),
        axial=bool(segmod.AXIAL_DEFAULT),
        axial_weight_iters=int(segmod.AXIAL_WEIGHT_ITERS),
        axial_weight_power=float(segmod.AXIAL_WEIGHT_POWER),
        axial_weight_floor=float(segmod.AXIAL_WEIGHT_FLOOR),
        axial_skip_ratio=float(segmod.AXIAL_SKIP_RATIO),
        one_sided_weight=float(segmod.ONE_SIDED_WEIGHT),
        one_sided_iters=int(segmod.ONE_SIDED_ITERS),
        trim_fraction=float(segmod.TRIM_FRACTION),
        trim_min_coverage=float(segmod.TRIM_MIN_COVERAGE),
        trim_iters=int(segmod.TRIM_ITERS),
        trim_coverage_bins=int(segmod.TRIM_COVERAGE_BINS),
        ring_seed_fraction=float(segmod.RING_SEED_FRACTION),
        ring_seed_percentile=float(segmod.RING_SEED_PERCENTILE),
        undistort_samples=180,
        refine_tol=float(stereo.REFINE_TOL_ANALYTIC),
        max_refine_iter=int(stereo.MAX_REFINE_ITER),
        ref_percentile=float(stereo.REF_PERCENTILE),
        jac_rel_step=float(stereo._JAC_REL_STEP),
        jac_distort_step_px=float(stereo._JAC_DISTORT_STEP_PX),
        jac_grad_step_px=float(stereo._JAC_GRAD_STEP_PX),
        jac_min_resid_frac=float(stereo._JAC_MIN_RESID_FRAC),
        far_px=float(stereo._FAR_PX),
        cal_disp_tol_mm=float(stereo.CAL_DISP_TOL_MM),
        cal_disp_tol_n=float(stereo.CAL_DISP_TOL_N),
        max_discrepancy_mm=est.max_discrepancy_mm,
        suspect_mm=float(est._suspect_mm),
        min_ridge=float(est.min_ridge),
        max_fit_rms_rel=est.max_fit_rms_rel,
        dropout_s=float(est.dropout_s),
        window_frames=int(stereo.WINDOW_FRAMES),
        min_window_support=int(stereo.MIN_WINDOW_SUPPORT),
        prior_sigma_deg=float(stereo.PRIOR_SIGMA_DEG),
        sigma_normal_rad=float(stereo.SIGMA_NORMAL_RAD),
        accel_mm_s2=float(ACCEL_MM_S2),
        sigma_lat_mm=float(est.sigma_lat_mm),
        sigma_depth_mm=float(est.sigma_depth_mm),
        radius_mm=float(est.radius_mm),
        level=int(segmod.THRESH if est.thresh is None else est.thresh),
        min_area=float(est.min_area),
        verify_tol=est.verify_tol,
        require_stereo=bool(est.require_stereo),
    )


def camera_dict(cam):
    return {
        "name": cam.name,
        "K": np.ascontiguousarray(cam.K, dtype=np.float64),
        "dist": None if cam.dist is None else np.ascontiguousarray(cam.dist, dtype=np.float64),
        "T_world_cam": np.ascontiguousarray(cam.T_world_cam, dtype=np.float64),
    }


def centre_cal_dict(cc):
    return {
        "tilt_knots_deg": [float(x) for x in cc.tilt_knots_deg],
        "offset_over_major": [float(x) for x in cc.offset_over_major],
    }


def _gray(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return np.ascontiguousarray(g, dtype=np.uint8)


def _circle_poses(view):
    return [conic.CirclePose(np.asarray(c), np.asarray(n)) for c, n in view]


def _segmentation(d):
    if d is None:
        return None
    return segmod.Segmentation(
        mask=d["mask"], contour=d["contour"], ellipse=d["ellipse"], area_px=d["area_px"],
        n_points=d["n_points"], fit_rms_px=d["fit_rms_px"], threshold=d["threshold"],
        t_ms=0.0, valid=d["valid"], valid_from=d["valid_from"],
    )


class NativeStereoPoseEstimator(StereoPoseEstimator):
    """`StereoPoseEstimator` with `update` running in C++.

    Only the live configuration is ported. Anything outside it raises here rather than
    silently running the Python path: a caller that asked for it should know.
    """

    def __init__(self, rig, **kwargs):
        if pmw_pose is None:
            raise ImportError("pmw_pose is not built; run `uv sync --extra native`")
        super().__init__(rig, **kwargs)
        self._native = None
        self._native_scale = None
        self._native_thresh = None
        self._check_supported()

    @classmethod
    def from_python(cls, est):
        """The native twin of an existing Python estimator, same configuration."""

        if pmw_pose is None:
            raise ImportError("pmw_pose is not built; run `uv sync --extra native`")
        obj = copy.copy(est)
        obj.__class__ = cls
        obj._native = None
        obj._native_scale = None
        obj._native_thresh = None
        obj.reset()
        obj._check_supported()
        return obj

    def _check_supported(self):
        bad = []
        if not self.direct:
            bad.append("direct=False")
        if not self.do_refine:
            bad.append("do_refine=False")
        if not self.undistort:
            bad.append("undistort=False")
        if self.use_major_channel:
            bad.append("use_major_channel=True")
        if self.max_jump_deg_s is not None:
            bad.append("max_jump_deg_s")
        if self.apply_centre_to_measurement:
            bad.append("apply_centre_to_measurement=True")
        if self.tilt_cal is not None and not self.tilt_cal.is_identity:
            bad.append("a non-identity tilt_cal")
        if segmod.APPEARANCE != "bright":
            bad.append(f"appearance={segmod.APPEARANCE!r}")
        if bad:
            raise NotImplementedError(
                "the native estimator ports the live configuration only; not " + ", ".join(bad))

    def _ensure_native(self):
        if self._native is None or self._native_scale != self._px_scale:
            self._native = pmw_pose.Estimator(
                [camera_dict(c) for c in self.rig.cameras], native_config(self),
                centre_cal_dict(self.centre_cal),
                np.ascontiguousarray(self.reference, dtype=np.float64))
            self._native_scale = self._px_scale
            # Rebuilt from `native_config`, so the core's level is whatever `thresh` was
            # at that moment; `update` pushes any later change.
            self._native_thresh = None if self.thresh is None else int(self.thresh)

    def reset(self):
        super().reset()
        if getattr(self, "_native", None) is not None:
            self._native.reset()

    def update(self, frames, t=None, frame_index=None, stamps=None, motion=None):
        now = time.monotonic() if t is None else float(t)
        stamps = None if stamps is None else [float(x) for x in stamps]
        vel = None if motion is None else np.ascontiguousarray(motion.rate, dtype=np.float64)
        vel_cov = (None if motion is None
                   else np.ascontiguousarray(motion.rate_cov, dtype=np.float64))
        self.frame_index = (
            self.frame_index + 1 if frame_index is None else int(frame_index))
        self._match_scale(frames[0])
        self._ensure_native()
        # `thresh` reaches the core as `Config.level`, read once at construction, and
        # `_ensure_native` only rebuilds on a scale change -- so the viser slider
        # (`live_viz` does `est.thresh = viz.thresh` every iteration) had done nothing
        # since the native core became the default. Pushed through each frame instead;
        # it is one int and the core reads it inside `segment`.
        if self.thresh is not None and self.thresh != self._native_thresh:
            self._native.set_thresh(int(self.thresh))
            self._native_thresh = int(self.thresh)

        grays = [_gray(f) for f in frames]
        plates, versions = [], []
        for cam, g in zip(self.rig.cameras, grays):
            plate = self.backgrounds.get(cam.name)
            # The plate-response cache key, as `_view_candidates` computes it: a running
            # plate's frame counter before this update; a saved plate has none (-1).
            versions.append(getattr(plate, "n", 0) // segmod.PLATE_REFRESH_FRAMES
                            if hasattr(plate, "update") else -1)
            if hasattr(plate, "update"):
                plate = plate.update(g)
            plates.append(None if plate is None else np.ascontiguousarray(plate, dtype=np.uint8))

        r = self._native.update(grays[0], grays[1], plates[0], plates[1], versions[0], versions[1],
                                now, stamps, vel, vel_cov)
        nat = self._native
        self.n_lost, self.n_detected = nat.n_lost, nat.n_detected
        self.n_rejected, self.n_rejected_fit = nat.n_rejected, nat.n_rejected_fit
        self.n_rejected_mono, self.n_rearbitrated = nat.n_rejected_mono, nat.n_rearbitrated
        for cam, i in zip(self.rig.cameras, range(len(self.rig.cameras))):
            self._prev_ellipse[cam.name] = nat.prev_ellipse(i)
        if r is None:
            return None
        return self._gate_predicted(stereo_pose(self, r, now, self.frame_index, t))


def stereo_pose(est, r, now, frame_index, t=None):
    """
    The C++ `update` dict, rebuilt as the `StereoPose` every consumer downstream reads.

        Split out of `NativeStereoPoseEstimator.update` because `pose/tracker.py` needs
        exactly the same conversion off exactly the same dict, and a second copy of it
        is a second place for the zeroing, the psi convention or a field name to drift.

        Mutates `est._prev_normal` / `_prev_t`, which is the jump gate's memory: the
        caller owns the estimator, so the state stays where the gates can find it.
    """

    segs = [_segmentation(d) for d in r["segs"]]
    cands = [_circle_poses(v) for v in r["candidates"]]
    usable = list(r["views_used"])
    m = Match(poses=tuple(_circle_poses(r["match_poses"])), indices=tuple(r["match_indices"]),
              discrepancy_mm=float(r["discrepancy_mm"]), margin=float(r["margin"]))
    centre, normal = r["center"], r["normal"]
    alive = None if r["alive"] is None else r["alive"].astype(bool)
    est._prev_normal = normal
    est._prev_t = t

    xyz, n_zeroed = est.zero.apply(centre, normal)
    theta, phi = _angles_from_normal(n_zeroed)
    ref = segs[usable[0]]
    pose = StereoPose(
        t=now,
        frame_index=frame_index,
        xyz_mm=xyz,
        normal=n_zeroed,
        theta_deg=theta,
        phi_deg=phi,
        n_views=len(usable),
        discrepancy_mm=m.discrepancy_mm,
        skew_ms=float(r["skew_ms"]),
        margin=m.margin,
        refine_rms_px=float(r["refine_rms_px"]),
        refine_iters=int(r["refine_iters"]),
        jump_deg=float(r["jump_deg"]),
        t_seg_ms=float(r["t_seg_ms"]),
        t_est_ms=float(r["t_est_ms"]),
        psi_deg=est.zero.apply_psi(ref.ellipse[2]),
        ellipse=ref.ellipse,
        area_px=ref.area_px,
        fit_rms_px=ref.fit_rms_px,
        ambiguity_margin_deg=float(r["ambiguity_margin_deg"]),
        n_solutions=int(r["n_solutions"]),
        union_coverage=float(r["union_coverage"]),
        per_view=tuple(segs),
        extra={"match": m, "candidates": cands, "views_used": usable,
               "world": (centre, normal), "alive": alive},
    )
    return pose


def _self_check():
    """Every constant the core needs reaches it, and the wrapper round-trips.

    The C++ constructor throws on a missing config key, so building one estimator is
    the whole test of `native_config`. A blank frame pair must come back as ``None``,
    not as an exception, since a lost frame is the normal case in flight.
    """

    from controller.calib import rig as rigmod
    from controller.pose.estimator import RADIUS_BENCH_MM

    assert available(), "pmw_pose is not built; run `uv sync --extra native`"
    assert rigmod.DEFAULT_PATH.exists(), f"needs the measured rig at {rigmod.DEFAULT_PATH}"
    rig = rigmod.StereoRig.load()
    est = NativeStereoPoseEstimator(rig, radius_mm=RADIUS_BENCH_MM, backgrounds={}, error_model=False)
    w, h = (rig.meta.get("image_size") or (640, 400))
    blank = [np.zeros((int(h), int(w)), np.uint8) for _ in rig.cameras]
    assert est.update(blank, t=0.0) is None
    assert est.n_lost == 1 and est.frame_index == 0
    twin = NativeStereoPoseEstimator.from_python(StereoPoseEstimator(
        rig, radius_mm=RADIUS_BENCH_MM, backgrounds={}, error_model=False))
    assert twin.update(blank, t=0.0) is None
    for bad in (dict(direct=False), dict(do_refine=False), dict(use_major_channel=True)):
        try:
            NativeStereoPoseEstimator(rig, radius_mm=RADIUS_BENCH_MM, error_model=False, **bad)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{bad} should have been refused")
    print("stereo_native: config reaches the core, blank frames are None, unsupported kwargs refuse")


if __name__ == "__main__":
    _self_check()
