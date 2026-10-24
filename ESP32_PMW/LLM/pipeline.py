"""Offline orchestration: observed LED events -> common time -> evidence package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from LLM.capture import load_flight
from LLM.io import inventory, invalidate_report, read_json, sha256, write_json
from LLM.pattern import sample_pattern
from LLM.sync import detect_led, fit_clock, map_times
from LLM.takeoff import detect_takeoff
from LLM.telemetry import parse_log
from LLM.world import transform_pose_dataframe


def synchronize(run, roi_a, roi_b, pulse_times=(0.0,)):
    run = Path(run)
    invalidate_report(run, "Alignment changed or failed; rerun process after a successful sync.")
    flight = load_flight(run)
    cfg = read_json(run / "config.json")["led"]
    # A failed new attempt must not leave an old clock silently usable.
    (run / "processed/sync.json").unlink(missing_ok=True)
    result = {"schema_version": 1, "clocks": {},
              "association": "operator identifies these LED pulses as the listed experiment events",
              "hardware_sync_validated": False, "led": {}}
    result["input_sha256"] = {name: sha256(run / name) for name in
                              ("raw/camera/frames.csv", "raw/camera/A/A.mp4", "raw/camera/B/B.mp4")}
    result["led_config"] = cfg
    for ch, roi in (("A", roi_a), ("B", roi_b)):
        detected = detect_led(flight[f"video_{ch.lower()}"], flight[f"times_{ch.lower()}"], roi, cfg)
        pd.DataFrame(detected.pop("trace")).to_csv(run / f"processed/led_{ch}.csv", index=False)
        detected["video_path"] = f"raw/camera/{ch}/{ch}.mp4"
        result["led"][ch] = detected
        if detected["status"] != "detected" or len(detected["edges"]) != len(pulse_times):
            write_json(run / "processed/sync_attempt.json", result)
            raise ValueError(f"camera {ch}: detected {len(detected['edges'])} LED rises, expected {len(pulse_times)}; inspect ROI/pulse association")
        result["clocks"][f"camera_{ch}"] = fit_clock(
            [e["source_time_s"] for e in detected["edges"]], pulse_times, f"camera_{ch}")
    write_json(run / "processed/sync.json", result)
    return result


def add_clock(run, name, source_anchors, experiment_anchors, evidence):
    if name != "esp":
        raise ValueError("manual clock currently supports esp only; use sync for cameras")
    if not evidence.strip():
        raise ValueError("clock anchors require an evidence description")
    path = Path(run) / "processed/sync.json"
    result = read_json(path) if path.exists() else {"schema_version": 1, "clocks": {}}
    clock = fit_clock(source_anchors, experiment_anchors, name)
    invalidate_report(run, "ESP clock anchors changed; rerun process.")
    clock["evidence"] = evidence
    result["clocks"][name] = clock
    write_json(path, result)
    return clock


def _strict_times(df, column):
    values = df[column].to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"{column} must be nonempty, finite and strictly increasing")
    return values


def _plots(run, pose, plan, current, takeoff, window, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    fig.suptitle(f"{run.name} | {source.upper()} | LED-aligned evidence, timing not hardware-validated")
    for c in "xyz":
        axes[0].plot(pose.t_s, pose[f"{c}_mm"], label=c)
    axes[0].set_ylabel("Position (mm)")
    for col in ("alpha_x_deg", "alpha_y_deg"):
        axes[1].plot(pose.t_s, pose[col], label=col)
    axes[1].set_ylabel("Lean (deg)")
    if not pose.orientation_valid.any():
        axes[1].text(.02, .5, "Orientation unavailable: directed axis not observed", transform=axes[1].transAxes)
    if plan is not None:
        axes[2].plot(plan.t_s, plan.frequency_hz, label="planned frequency")
        for ch in "ABCD":
            axes[3].step(plan.t_s, plan[f"duty_{ch}_pct"], where="post", label=ch)
            axes[4].step(plan.t_s, plan[f"phase_{ch}_deg"], where="post", label=ch)
    else:
        axes[2].text(.02, .5, "No planned pattern supplied", transform=axes[2].transAxes)
    axes[2].set_ylabel("Frequency (Hz)")
    axes[3].set_ylabel("Planned duty (%)")
    axes[4].set_ylabel("Planned phase (deg)")
    if current is not None and "t_s" in current:
        for ch in "ABCD":
            axes[5].plot(current.t_s, current[f"current_{ch}_A"], ".-", label=ch)
    else:
        axes[5].text(.02, .5, "Current missing or unsynchronized; no aligned curve shown", transform=axes[5].transAxes)
    axes[5].set_ylabel("Filtered CS (A)")
    axes[5].set_xlabel("Experiment time (s), LED START = 0")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(0, color="gray", linestyle=":")
        if takeoff["status"] == "detected":
            ax.axvline(takeoff["onset_s"], color="black", linestyle="--")
        ax.axvspan(*window, color="green", alpha=.07)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", ncol=4, fontsize=8)
    fig.savefig(run / "export/overview.png", dpi=140)
    plt.close(fig)


def export_clips(run, flight, clocks, window, fps=60.0, max_error_s=.03):
    """Make two previews on one common grid. Originals and source-frame mapping remain authoritative."""
    import cv2

    timeline = np.arange(window[0], window[1] + 1e-9, 1/fps)
    if not len(timeline):
        raise ValueError("empty clip window")
    for ch in "AB":
        key = ch.lower()
        times = map_times(flight[f"times_{key}"], clocks[f"camera_{ch}"])
        right = np.minimum(np.searchsorted(times, timeline), len(times)-1)
        left = np.maximum(right-1, 0)
        indices = np.where(abs(times[left]-timeline) <= abs(times[right]-timeline), left, right)
        cap = cv2.VideoCapture(str(flight[f"video_{key}"]))
        writer, decoded, frame = None, -1, None
        mapping = []
        try:
            for out_index, (t, index) in enumerate(zip(timeline, indices)):
                while decoded < index:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError("clip decoder ended before validated frame count")
                    decoded += 1
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(run / f"export/camera_{ch}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise ValueError("cannot open preview video encoder")
                delta = float(times[index] - t)
                valid = abs(delta) <= max_error_s
                view = frame.copy() if valid else np.zeros_like(frame)
                cv2.putText(view, f"{ch} t={t:.3f}s PREVIEW {'MISSING' if not valid else ''}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                writer.write(view)
                mapping.append({"output_frame": out_index, "playback_s": out_index/fps, "t_s": float(t),
                                "decoded_source_frame": int(index), "capture_frame_index": int(flight["frame_indices"][index]),
                                "source_t_s": float(times[index]), "delta_s": delta, "image_valid": valid})
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        pd.DataFrame(mapping).to_csv(run / f"export/camera_{ch}_frame_map.csv", index=False)


def process_run(run, clips=True):
    run = Path(run).resolve()
    invalidate_report(run, "Processing in progress or failed; no current complete report.")
    manifest = read_json(run / "manifest.json")
    config = read_json(run / "config.json")
    sync = read_json(run / "processed/sync.json")
    if not sync.get("input_sha256") or sync.get("led_config") != config["led"]:
        raise ValueError("sync provenance missing or LED config changed; rerun sync")
    for name, digest in sync["input_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError("capture changed after LED alignment; rerun sync")
    clocks = sync["clocks"]
    flight = load_flight(run)
    camera_pose = pd.read_csv(run / "processed/pose_camera.csv")
    pose_meta = read_json(run / "processed/pose_camera.meta.json")
    if not pose_meta.get("input_sha256"):
        raise ValueError("pose provenance missing; rerun pose/auto-pose")
    for name, digest in pose_meta["input_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"pose input changed: {name}; rerun pose/auto-pose")
    if len(camera_pose) != len(flight["frame_indices"]) or not np.array_equal(camera_pose.frame_index, flight["frame_indices"]):
        raise ValueError("pose rows must match every encoded stereo pair, including invalid observations")
    for ch in "AB":
        col = f"t_source_{ch.lower()}_s"
        if not np.allclose(camera_pose[col], flight[f"times_{ch.lower()}"], atol=1e-7, rtol=0):
            raise ValueError("pose timestamps differ from preserved raw capture")
    pose = transform_pose_dataframe(camera_pose, read_json(run / "calibration/world_frame.json"))
    ta = map_times(pose.t_source_a_s, clocks["camera_A"])
    tb = map_times(pose.t_source_b_s, clocks["camera_B"])
    pose["t_s"] = (ta + tb) / 2
    pose["aligned_stereo_skew_s"] = abs(ta-tb)
    _strict_times(pose, "t_s")
    # Original triangulation gates source skew; optical alignment cannot retroactively
    # make two moving-robot images simultaneous. Gate corrected skew as well.
    skew_limit = float(config.get("max_stereo_skew_s", .01))
    if not np.isfinite(skew_limit) or skew_limit <= 0:
        raise ValueError("max_stereo_skew_s must be finite and positive")
    pose["stereo_skew_s"] = abs(flight["times_a"] - flight["times_b"])
    bad = (pose.aligned_stereo_skew_s > skew_limit) | (pose.stereo_skew_s > skew_limit)
    pose.loc[bad, ["position_valid", "orientation_valid"]] = False
    pose.loc[bad, ["x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "alpha_x_deg", "alpha_y_deg"]] = np.nan
    # Invalidate the previous report before writing derived outputs. Clear only
    # files owned by this pipeline; raw evidence and user notes are untouched.
    (run / "export/INCOMPLETE.txt").write_text("Processing in progress or failed; no current complete report.\n")
    for name in ("report.md", "package.json", "quality.json", "overview.png", "pose.csv", "current.csv",
                 "command_planned.csv", "command_applied.csv", "camera_A.mp4", "camera_B.mp4",
                 "camera_A_frame_map.csv", "camera_B_frame_map.csv"):
        (run / "export" / name).unlink(missing_ok=True)
    for name in ("command_planned.csv", "command_applied.csv", "current.csv", "current.meta.json", "events.json"):
        (run / "processed" / name).unlink(missing_ok=True)
    pose.to_csv(run / "processed/pose.csv", index=False)
    takeoff = detect_takeoff(pose, config["takeoff"])
    common = [max(float(ta[0]), float(tb[0])), min(float(ta[-1]), float(tb[-1]))]
    if common[1] <= common[0]:
        raise ValueError("camera recordings have no common experiment interval")
    window = list(common)
    if takeoff["status"] == "detected":
        pre, post = (float(config["export"][k]) for k in ("pre_roll_s", "post_takeoff_s"))
        if not np.isfinite([pre, post]).all() or pre < 0 or post <= 0:
            raise ValueError("export requires finite pre_roll_s>=0, post_takeoff_s>0")
        window = [max(common[0], takeoff["onset_s"] - pre), min(common[1], takeoff["onset_s"] + post)]
    events = {"schema_version": 1, "takeoff": takeoff, "crop_window_s": window,
              "crop_reason": "detected takeoff with context" if takeoff["status"] == "detected" else "full overlap; no confirmed takeoff",
              "start_event": "LED START; actual current rise may occur later"}
    write_json(run / "processed/events.json", events)
    plan = sample_pattern(read_json(run / "pattern.json")) if (run / "pattern.json").exists() else None
    if plan is not None:
        plan.to_csv(run / "processed/command_planned.csv", index=False)
    current, current_status = None, "missing"
    if (run / "raw/telemetry.jsonl").exists():
        current = parse_log(run / "raw/telemetry.jsonl")
        current_status = "no_parseable_samples" if current.empty else "unsynchronized"
        if "esp" in clocks and not current.empty:
            current["t_s"] = map_times(current.t_device_s, clocks["esp"])
            current_status = "aligned_by_explicit_esp_anchors"
        current.to_csv(run / "processed/current.csv", index=False)
        write_json(run / "processed/current.meta.json", current.attrs)
    applied_status = "missing"
    applied_path = run / "raw/applied_commands.csv"
    if applied_path.exists():
        applied = pd.read_csv(applied_path)
        _strict_times(applied, "t_device_s")
        if "provenance" not in applied or not applied.provenance.eq("device_applied").all():
            raise ValueError("applied command log requires provenance=device_applied; host send time is not applied time")
        value_columns = ["frequency_hz"] + [f"{kind}_{ch}_{unit}" for kind, unit in (("duty", "pct"), ("phase", "deg")) for ch in "ABCD"]
        if not set(value_columns).issubset(applied) or not np.isfinite(applied[value_columns].to_numpy(float)).all():
            raise ValueError("applied log requires finite frequency_hz, four duty_*_pct and four phase_*_deg snapshots")
        duties = applied[[f"duty_{ch}_pct" for ch in "ABCD"]].to_numpy(float)
        if np.any((duties < 0) | (duties > 100)) or np.any(applied.frequency_hz < 0):
            raise ValueError("applied log duty/frequency outside valid range")
        applied_status = "unsynchronized"
        if "esp" in clocks:
            applied["t_s"] = map_times(applied.t_device_s, clocks["esp"])
            applied_status = "aligned_by_explicit_esp_anchors"
        applied.to_csv(run / "processed/command_applied.csv", index=False)
    for name, df in (("pose", pose), ("command_planned", plan), ("current", current),
                     ("command_applied", applied if applied_path.exists() else None)):
        if df is not None:
            selected = df[(df.t_s >= window[0]) & (df.t_s <= window[1])].copy() if "t_s" in df else df.copy()
            if name == "command_applied" and "t_s" in df:
                # Keep the last observed snapshot as context, at its true time;
                # this is not a fabricated application event at the crop boundary.
                selected = pd.concat([df[df.t_s < window[0]].tail(1), selected]).copy()
                selected["context_before_window"] = selected.t_s < window[0]
            if "t_s" in selected and takeoff["status"] == "detected":
                selected["t_from_takeoff_s"] = selected.t_s - takeoff["onset_s"]
            selected.to_csv(run / f"export/{name}.csv", index=False)
    _plots(run, pose, plan, current, takeoff, window, manifest["source"])
    if clips:
        export_clips(run, flight, clocks, window)
    quality = {"source": manifest["source"], "n_frames": len(pose), "valid_position_frames": int(pose.position_valid.sum()),
               "valid_orientation_frames": int(pose.orientation_valid.sum()), "current_status": current_status,
               "applied_command_status": applied_status, "takeoff_status": takeoff["status"],
               "hardware_timing_validated": False, "videos_are_resampled_previews": bool(clips),
               "pose_methods": sorted(pose.pose_method.dropna().unique().tolist()),
               "metric_scale_measured": pose_meta.get("metric_scale_measured"),
               "warnings": ["LED observation timestamps are not exposure timestamps.",
                            "Current sense is a low-rate filtered estimate, not signed waveform/RMS/current phase.",
                            "Planned pattern is not evidence of exact firmware execution.",
                            "Center-height takeoff is a proxy; verify against actual bearing clearance."]}
    if "experimental_ellipse_center_proxy" in quality["pose_methods"]:
        quality["warnings"].append("Automatic ellipse centers are unvalidated silhouette proxies; geometric acceptance is not physical tracking accuracy.")
    if quality["metric_scale_measured"] is not True:
        quality["warnings"].append("Calibration board metric scale is not verified; mm values inherit that uncertainty.")
    write_json(run / "export/quality.json", quality)
    code = list(Path(__file__).parent.glob("*.py")) + [Path(__file__).parents[1] / "controller/control/ramp.py"]
    write_json(run / "export/package.json", {"schema_version": 1, "run_id": manifest["run_id"],
        "source": manifest["source"], "inputs_and_processing": inventory(run),
        "code_sha256": {str(p.relative_to(Path(__file__).parents[1])): sha256(p) for p in code},
        "crop_window_s": window, "quality": quality})
    report = f"""# {run.name} — {manifest['source'].upper()}

This package is an offline experimental evidence bundle. Hardware timing has not been validated.

![Overview](overview.png)

- Position: {quality['valid_position_frames']} / {len(pose)} frames valid.
- Directed orientation: {quality['valid_orientation_frames']} / {len(pose)} frames valid.
- Pose source: {', '.join(quality['pose_methods'])}. Metric scale verified: {quality['metric_scale_measured']}.
- Takeoff: **{takeoff['status']}**; onset {takeoff.get('onset_s')} s; confirmation {takeoff.get('confirmed_s')} s.
- Common crop: {window[0]:.6f} to {window[1]:.6f} s, preserving LED START time zero.
- Current: {current_status}. Actual applied commands: {applied_status}.

The green region is the exported interval; the dashed line is inferred takeoff. `pose.csv` retains
invalid samples and independent position/orientation flags. `t_from_takeoff_s` is an additional
coordinate, never a replacement for experiment time. `camera_A.mp4` and `camera_B.mp4`, when present,
are synchronized viewing previews; consult their frame maps and original raw videos for analysis.

`command_planned.csv` contains intended frequency, carrier duty and commanded phase. Duty is not
current magnitude. Filtered CS values do not measure current phase. A hardware-verified ESP START
timestamp is required before real electrical data can share the LED time axis reliably.

See `quality.json`, `package.json`, `../processed/events.json`, and `../processed/sync.json` for
provenance, hashes, thresholds and LED observation brackets. Share the **whole run folder**, including
raw videos, calibration and metadata. Do not share only this plot.
"""
    (run / "export/report.md").write_text(report, encoding="utf-8")
    (run / "export/INCOMPLETE.txt").unlink()
    return quality
