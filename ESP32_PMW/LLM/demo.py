"""Synthetic example verifies plumbing/geometry, never real tracking or flight accuracy."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from LLM.io import init_run, sha256, write_json
from LLM.pattern import example_pattern
from LLM.pipeline import add_clock, process_run, synchronize
from LLM.pose import triangulate_annotations
from LLM.world import build_world_frame
from controller.control import ramp


def make_demo(path):
    run = init_run(path, source="synthetic")
    root = run / "raw/camera"
    width, height, fps, count = 640, 400, 60, 240
    K = np.array([[500., 0, 320], [0, 500, 200], [0, 0, 1]])
    transforms = [np.eye(4), np.eye(4)]
    transforms[1][0, 3] = 60
    cameras = [{"name": ch, "K": K.tolist(), "dist": [0, 0, 0, 0, 0], "T_world_cam": t.tolist()}
               for ch, t in zip("AB", transforms)]
    rig_path = run / "calibration/stereo_rig.json"
    write_json(rig_path, {"cameras": cameras, "world_frame": "camera_A", "image_size": [width, height],
                          "rotate180": False, "square_mm_measured": False, "source": "synthetic_known_geometry"})
    world = build_world_frame([0, 0, 500], [0, -1, 0])
    world["evidence"] = "synthetic known gravity/origin; not a plumb-line measurement"
    write_json(run / "calibration/world_frame.json", world)
    writers = []
    for ch in "AB":
        (root / ch).mkdir(parents=True)
        w = cv2.VideoWriter(str(root / ch / f"{ch}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not w.isOpened():
            raise RuntimeError("demo video encoder unavailable")
        writers.append(w)
    stamps, annotations = [], []
    try:
        for i in range(count):
            t = i/fps - .5
            rise = max(0, t-1) * 15
            center = np.array([2*np.sin(max(t-1, 0)*2), -rise, 500.])
            axis = np.array([.1*np.sin(t*2), -1, .05*np.cos(t*2)])
            axis /= np.linalg.norm(axis)
            tip = center + axis * 15
            row = {"frame_index": i}
            for ch, T, writer in zip("AB", transforms, writers):
                pixels = []
                for point in (center, tip):
                    p = K @ (point - T[:3, 3]); pixels.append(p[:2]/p[2])
                c, tip_px = pixels
                key = ch.lower()
                row.update({f"{key}x_px": c[0], f"{key}y_px": c[1], f"{key}_tip_x_px": tip_px[0], f"{key}_tip_y_px": tip_px[1]})
                frame = np.full((height, width, 3), 25, dtype=np.uint8)
                led = 240 if 30 <= i < 42 or 210 <= i < 222 else 25
                frame[20:44, 20:44] = led
                cv2.line(frame, tuple(c.astype(int)), tuple(tip_px.astype(int)), (200, 200, 200), 2)
                cv2.ellipse(frame, tuple(c.astype(int)), (18, 4), 0, 0, 360, (220, 220, 220), 2)
                cv2.putText(frame, "SYNTHETIC DEMO", (12, height-18), cv2.FONT_HERSHEY_SIMPLEX, .6, (120, 180, 220), 1)
                writer.write(frame)
            annotations.append(row)
            stamps.append({"index": i, "t_capture": 100+i/fps+.0005, "skew_s": .001,
                           "t_a": 100+i/fps, "t_b": 100+i/fps+.001, "written": 1})
    finally:
        for writer in writers:
            writer.release()
    pd.DataFrame(stamps).to_csv(root / "frames.csv", index=False)
    write_json(root / "meta.json", {"mode": [width, height], "fps": fps, "n_frames": count,
                                     "dropped": 0, "encoder_errors": 0, "rotate180": False,
                                     "timestamp_semantics": "synthetic_host_receipt"})
    annotation_path = run / "raw/annotations.csv"
    pd.DataFrame(annotations).to_csv(annotation_path, index=False)
    pose = triangulate_annotations(run, annotation_path, rig_path)
    pose["pose_method"] = "synthetic_ground_truth_pixel_correspondences"
    pose.to_csv(run / "processed/pose_camera.csv", index=False)
    sources = ["calibration/stereo_rig.json", "raw/camera/frames.csv", "raw/camera/meta.json",
               "raw/camera/A/A.mp4", "raw/camera/B/B.mp4", "raw/annotations.csv"]
    write_json(run / "processed/pose_camera.meta.json", pose.attrs | {
        "input_sha256": {name: sha256(run / name) for name in sources}})
    write_json(run / "pattern.json", example_pattern())
    with (run / "raw/telemetry.jsonl").open("w") as stream:
        for i, t in enumerate(np.arange(0, 3.5, .5)):
            vals = [(.3+.1*np.sin(t))*v for v in (1, .95, 1.03, .97)]
            frequency = ramp.frequency_at(example_pattern()["frequency_segments"], float(t))
            line = f"t={10000+int(t*1000)} freq={frequency:.1f} | I[A]: " + " ".join(f"{c}={v:.3f}" for c, v in zip("ABCD", vals)) + " | duty[%]: A=20 B=20 C=20 D=20 | trip=0"
            stream.write(json.dumps({"host_receive_s": 100.5+t+.02, "line": line})+"\n")
    synchronize(run, [20, 20, 24, 24], [20, 20, 24, 24], [0, 3])
    add_clock(run, "esp", [10, 13], [0, 3], "synthetic known START/end events; no hardware timing measurement")
    quality = process_run(run)
    assert quality["valid_position_frames"] == count and quality["valid_orientation_frames"] == count
    assert quality["takeoff_status"] == "detected"
    return {"run": str(run), "report": str(run / "export/report.md"), "source": "synthetic", **quality}
