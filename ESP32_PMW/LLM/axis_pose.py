"""Offline, explicitly experimental silhouette-axis diagnostics for the notebook.

Reuses the Python axis_pose core without changing its constants. These estimates
are separate from the physical-point pose/takeoff pipeline: a silhouette has no
observed directed endpoint and its centroid is not a measured material point.
"""
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd

from controller.pose import axis_pose as core
from LLM.record.takes import load_flight, _capture_dir
from LLM.record.io import write_json, sha256, snapshot
from LLM.analyse.pose import _validated_rig


def _roi(value, width, height):
    a = np.asarray(value)
    if (a.shape != (4,) or a.dtype.kind not in 'ifu' or not np.isfinite(a).all()
            or not np.equal(a, np.floor(a)).all()):
        raise ValueError('Each robot ROI must be [x, y, width, height] in saved-image pixels')
    x, y, w, h = map(int, a)
    if min(x, y) < 0 or min(w, h) < 3 or x+w > width or y+h > height:
        raise ValueError('Robot ROI is outside the saved image or too small')
    return x, y, w, h


def _view(gray, plate, roi, threshold):
    x, y, w, h = roi
    mask = core.mask(gray[y:y+h, x:x+w],
                     None if plate is None else plate[y:y+h, x:x+w], thresh=threshold)
    if mask is None:
        return None, mask, 'no_mask'
    # A clipped silhouette changes its moments; do not fit the visible fragment.
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return None, mask, 'mask_touches_roi_border'
    m = core.moments(mask)
    if m is None:
        return None, mask, 'too_small_or_round'
    n, cx, cy, ux, uy = m
    return (n, cx+x, cy+y, ux, uy), mask, 'ok'


def _overlay(frame, mask, m, roi, label):
    out = frame.copy()
    x, y, w, h = roi
    if mask is not None:
        area = out[y:y+h, x:x+w]
        area[mask > 0] = (.6*area[mask > 0] + .4*np.array([40, 210, 40])).astype(np.uint8)
    cv2.rectangle(out, (x, y), (x+w-1, y+h-1), (220, 180, 40), 1)
    if m is not None:
        _, cx, cy, ux, uy = m
        c = np.array([cx, cy]); u = np.array([ux, uy])*25
        cv2.line(out, tuple(np.rint(c-u).astype(int)), tuple(np.rint(c+u).astype(int)), (0, 220, 255), 2)
        cv2.circle(out, tuple(np.rint(c).astype(int)), 3, (0, 0, 255), -1)
    out = cv2.resize(out, (640, round(out.shape[0]*640/out.shape[1])))
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 200, 255), 1)
    return out


def run_axis_pose(run_dir, calibration_path, *, robot_rois, background_paths=None,
                  background_mode='difference', background_confirmed=False,
                  preprocessing_confirmed=False, reference_axis_camera=(0., -1., 0.),
                  threshold=core.THRESH, max_skew_s=.01, max_gap_s=.05,
                  max_step_deg=20., max_frames=None, progress=None, show=None, show_every=10):
    """Write a new trial containing CSV, curves, overlays and input snapshots.

    Works with measured or sync_check runs, or an imported FlightWriter take.
    No hardware, no world-frame claim, no takeoff, and no promotion of experimental
    axes to orientation_valid. Reference is an explicit camera-A direction, not
    gravity. After rejection/gaps continuity resets to that reference.
    ``show(panel)``: every ``show_every``-th frame's A|B segmentation overlay (the
    same drawing as overlays.jpg), for a live view while it runs.
    """
    run, calibration_path = Path(run_dir).resolve(), Path(calibration_path).resolve()
    limits = np.asarray([threshold, max_skew_s, max_gap_s, max_step_deg], float)
    if not np.isfinite(limits).all() or np.any(limits <= 0) or threshold > 255 or max_step_deg >= 90:
        raise ValueError('Invalid threshold, timing or angular-step limits')
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('max_frames must be a positive integer or None')
    ref_a = np.asarray(reference_axis_camera, float)
    if ref_a.shape != (3,) or not np.isfinite(ref_a).all() or np.linalg.norm(ref_a) < 1e-10:
        raise ValueError('reference_axis_camera must be a finite nonzero vector')
    ref_a /= np.linalg.norm(ref_a)
    if background_mode not in ('difference', 'threshold'):
        raise ValueError('background_mode must be difference or threshold')
    if background_mode == 'difference' and (not background_confirmed or set(background_paths or {}) != set('AB')):
        raise ValueError('Provide empty-scene A/B backgrounds and set background_confirmed=True after checking them')
    if background_mode == 'threshold' and background_paths:
        raise ValueError('threshold mode does not consume background_paths')
    flight = load_flight(run)
    rig, _ = _validated_rig(calibration_path, flight, preprocessing_confirmed)
    for cam in rig.cameras:
        if (cam.dist is not None and len(cam.dist) != 5) or not np.allclose(cam.K, [[cam.K[0, 0], 0, cam.K[0, 2]], [0, cam.K[1, 1], cam.K[1, 2]], [0, 0, 1]]):
            raise ValueError('axis_pose requires zero-skew intrinsics and the 5-term OpenCV distortion model')
    if set(robot_rois or {}) != set('AB'):
        raise ValueError('Specify robot_rois for both A and B; exclude holder markers and LED')
    rois, plates = {}, {}
    sources = {'rig.json': calibration_path}
    for tag in 'AB':
        w, h = flight['metadata']['decoded_image_sizes'][tag]
        rois[tag] = _roi(robot_rois[tag], w, h)
        if background_mode == 'difference':
            p = Path(background_paths[tag]).resolve()
            plate = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if plate is None or plate.shape != (h, w):
                raise ValueError(f'Background {tag} has wrong dimensions or cannot be read')
            plates[tag] = plate; sources[f'background_{tag}.png'] = p
    capture_root = _capture_dir(run)
    raw_paths = [capture_root/p for p in ('frames.csv', 'meta.json', 'A/A.mp4', 'B/B.mp4')]
    all_inputs = [*raw_paths, *sources.values(), Path(__file__), Path(core.__file__)]
    hashes = {str(p): sha256(p) for p in all_inputs}
    n = len(flight['frame_indices']) if max_frames is None else min(max_frames, len(flight['frame_indices']))
    ref_world = rig.a.R @ ref_a
    last_axis = last_t = None
    output, panels, decoders = [], [], []
    preview_indices = set(np.linspace(0, n-1, min(n, 6), dtype=int))
    try:
        for tag in 'AB':
            decoder = cv2.VideoCapture(flight[f'video_{tag.lower()}'])
            decoders.append(decoder)
            if not decoder.isOpened():
                raise ValueError(f'Cannot reopen camera {tag} video')
        for k in range(n):
            ta, tb = float(flight['times_a'][k]), float(flight['times_b'][k])
            t = (ta+tb)/2
            row = dict(frame_index=int(flight['frame_indices'][k]), encoded_frame_index=k,
                       t_source_a_s=ta, t_source_b_s=tb, elapsed_capture_s=t-(flight['times_a'][0]+flight['times_b'][0])/2,
                       stereo_skew_s=abs(ta-tb), axis_estimate_valid=False, position_estimate_valid=False,
                       orientation_valid=False, position_valid=False, physical_accuracy_validated=False,
                       coordinate_frame='camera_A', method='experimental_axis_pose_moments',
                       reason='not_detected', axis_sign_observed=False)
            row.update({name: np.nan for name in ('x_est_mm', 'y_est_mm', 'z_est_mm', 'nx_est', 'ny_est', 'nz_est', 'angle_to_reference_deg', 'ray_gap_mm')})
            frames, masks, views = [], [], []
            for tag, decoder in zip('AB', decoders):
                ok, frame = decoder.read()
                if not ok:
                    raise ValueError(f'Camera {tag} ended early at encoded frame {k}')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                m, mask, reason = _view(gray, plates.get(tag), rois[tag], threshold)
                frames.append(frame); masks.append(mask); views.append(m)
                row[f'{tag.lower()}_reason'] = reason
                row[f'{tag.lower()}_area_px'] = int(mask.sum()) if mask is not None else 0
            continuity = last_axis is not None and t-last_t <= max_gap_s
            row['reference_reset'] = not continuity
            ref = last_axis if continuity else ref_world
            if abs(ta-tb) > max_skew_s:
                row['reason'] = 'stereo_skew_exceeds_limit'
            elif any(v is None for v in views):
                row['reason'] = 'missing_or_degenerate_view'
            else:
                pose = core.pose(rig.cameras, views, ref)
                row['reason'] = 'geometry_rejected'
                if pose is not None:
                    p, a, gap = pose
                    step = np.degrees(np.arccos(np.clip(a @ ref, -1, 1)))
                    row['ray_gap_mm'] = float(gap)
                    if not np.isfinite(np.r_[p, a, gap]).all():
                        row['reason'] = 'nonfinite_estimate'
                    elif any((c.R.T @ (p-c.position))[2] <= 0 for c in rig.cameras):
                        row['reason'] = 'behind_camera'
                    elif continuity and step > max_step_deg:
                        row['reason'] = 'axis_jump'
                    else:
                        pc, ac = rig.a.R.T @ (p-rig.a.position), rig.a.R.T @ a
                        row.update(zip(('x_est_mm', 'y_est_mm', 'z_est_mm', 'nx_est', 'ny_est', 'nz_est'), map(float, np.r_[pc, ac])))
                        row.update(axis_estimate_valid=True, position_estimate_valid=True,
                                   angle_to_reference_deg=float(np.degrees(np.arccos(np.clip(ac @ ref_a, -1, 1)))),
                                   reason='experimental_estimate_only')
                        last_axis, last_t = a, t
            if not row['axis_estimate_valid']:
                last_axis = last_t = None
            output.append(row)
            if k in preview_indices or (show is not None and k % show_every == 0):
                panel = np.hstack([_overlay(f, mask, m, rois[tag],
                    f'{tag} frame={row["frame_index"]} t={row["elapsed_capture_s"]:.3f}s {row["reason"]}')
                    for tag, f, mask, m in zip('AB', frames, masks, views)])
                if k in preview_indices:
                    panels.append(panel)
                if show is not None and k % show_every == 0:
                    show(panel)
            if progress is not None and (k % 250 == 0 or k+1 == n):
                progress(k+1, n)
        if max_frames is None and any(d.read()[0] for d in decoders):
            raise ValueError('Video grew after input validation')
    finally:
        for d in decoders:
            d.release()
    if any(sha256(p) != h for p, h in hashes.items()):
        raise ValueError('Inputs changed during axis_pose; rerun on a finalized take')
    folder = run/'processed'/'axis_pose_trials'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        snapshot(source, folder/'inputs'/name)
    df = pd.DataFrame(output)
    df.to_csv(folder/'axis_pose.csv', index=False)
    metadata = dict(status='experimental_only', method='controller.pose.axis_pose (Python)',
                    frames=n, accepted=int(df.axis_estimate_valid.sum()),
                    reasons={str(k): int(v) for k, v in df.reason.value_counts().items()},
                    physical_accuracy_validated=False, orientation_valid=False, position_valid=False,
                    coordinate_frame='camera_A', time_basis='mean host capture time minus first saved pair; NOT LED zero',
                    background_mode=background_mode, background_confirmed=bool(background_confirmed),
                    preprocessing_confirmed=bool(preprocessing_confirmed), reference_axis_camera=ref_a.tolist(),
                    robot_rois={k: list(v) for k, v in rois.items()}, threshold=float(threshold),
                    max_skew_s=float(max_skew_s), max_gap_s=float(max_gap_s), max_step_deg=float(max_step_deg),
                    max_frames=max_frames, core_config=core.CFG, core_alpha=core.ALPHA, input_sha256=hashes,
                    limitation='Silhouette centroid proxy and unsigned axis, branch/sign selected by reference. No roll, gravity frame, takeoff or physical accuracy claim; inspect hub/blade contamination.')
    _plot(df, folder)
    if not cv2.imwrite(str(folder/'overlays.jpg'), np.vstack(panels)):
        raise ValueError('Could not write axis_pose overlays')
    write_json(folder/'metadata.json', metadata)
    return metadata | {'directory': str(folder), 'csv': str(folder/'axis_pose.csv'),
                       'curves': str(folder/'curves.png'), 'overlays': str(folder/'overlays.jpg')}


def _plot(df, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    t = df.elapsed_capture_s
    axs[0].plot(t, df.angle_to_reference_deg, '.-', ms=2)
    axs[0].set_ylabel('Axis / reference (deg)')
    for name in ('x', 'y', 'z'):
        axs[1].plot(t, df[f'{name}_est_mm'], '.-', ms=2, label=f'{name} camera A')
    axs[1].set_ylabel('Centroid proxy (mm)'); axs[1].legend()
    axs[2].plot(t, df.ray_gap_mm, '.', ms=2, label='Ray gap')
    failed = ~df.axis_estimate_valid
    axs[2].scatter(t[failed], np.zeros(failed.sum()), color='red', marker='x', label='Rejected')
    axs[2].set_ylabel('Ray gap (mm)'); axs[2].legend()
    axs[2].set_xlabel('Time since first saved pair (s); host timestamps, NOT LED zero')
    for ax in axs:
        ax.grid(alpha=.25)
    fig.suptitle(f'EXPERIMENTAL axis_pose | {int(df.axis_estimate_valid.sum())}/{len(df)} accepted | no physical validation')
    fig.savefig(folder/'curves.png', dpi=140)
    plt.close(fig)
