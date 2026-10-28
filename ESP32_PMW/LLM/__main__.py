"""Small explicit CLI. No command in this CLI energizes the coils."""
from __future__ import annotations

import argparse
from pathlib import Path

from LLM.io import init_run, invalidate_report, read_json, sha256, snapshot, write_json


def _pose(run, annotations, calibration, confirmed=False, automatic=False):
    from LLM.pose import triangulate_annotations
    run = Path(run)
    invalidate_report(run, "Pose changed or failed; rerun pose then process.")
    snapshot(calibration, run / "calibration/stereo_rig.json")
    if automatic:
        from LLM.auto_pose import generate_annotations
        data = generate_annotations(run, run / "calibration/stereo_rig.json", preprocessing_confirmed=confirmed,
                                    **read_json(run / "config.json").get("auto_pose", {}))
        annotations = run / "processed/auto_annotations.csv"
        data.to_csv(annotations, index=False)
        write_json(run / "processed/auto_annotations.meta.json", data.attrs)
    else:
        snapshot(annotations, run / "raw/annotations.csv")
        annotations = run / "raw/annotations.csv"
    data = triangulate_annotations(run, annotations, run / "calibration/stereo_rig.json", preprocessing_confirmed=confirmed,
                                   max_skew_s=read_json(run / "config.json").get("max_stereo_skew_s", .01))
    if automatic:
        data["pose_method"] = "experimental_ellipse_center_proxy"
    data.to_csv(run / "processed/pose_camera.csv", index=False)
    sources = ["calibration/stereo_rig.json", "raw/camera/frames.csv", "raw/camera/meta.json",
               "raw/camera/A/A.mp4", "raw/camera/B/B.mp4", str(Path(annotations).relative_to(run))]
    write_json(run / "processed/pose_camera.meta.json", data.attrs | {"automatic_detector": automatic,
               "input_sha256": {name: sha256(run / name) for name in sources}})
    return {"position_valid": int(data.position_valid.sum()), "orientation_valid": int(data.orientation_valid.sum()), "frames": len(data)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM experiment capture and offline evidence pipeline")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("init", help="create an empty run; no hardware access")
    p.add_argument("run")
    p = subs.add_parser("demo", help="generate synthetic stereo videos, then run the full pipeline")
    p.add_argument("run")
    p = subs.add_parser("example-pattern", help="write a schema example, not a validated flight recipe")
    p.add_argument("path")
    p = subs.add_parser("import-flight", help="copy and validate a completed recorder take")
    p.add_argument("run"); p.add_argument("source")
    p = subs.add_parser("world", help="define origin/gravity in calibrated camera A coordinates")
    p.add_argument("run")
    p.add_argument("--origin", nargs=3, type=float, required=True)
    p.add_argument("--up", nargs=3, type=float, required=True)
    p.add_argument("--x-hint", nargs=3, type=float, default=[1, 0, 0])
    p.add_argument("--evidence", required=True, help="reference frame/point IDs used to measure origin and plumb up")
    p = subs.add_parser("world-from-points", help="triangulate bearing and plumb endpoints from a static stereo pair")
    p.add_argument("run"); p.add_argument("--reference", required=True); p.add_argument("--calibration", required=True)
    p.add_argument("--confirm-preprocessing", action="store_true")
    p = subs.add_parser("sync", help="associate observed LED rises with experiment event times")
    p.add_argument("run")
    p.add_argument("--roi-a", nargs=4, type=int, required=True)
    p.add_argument("--roi-b", nargs=4, type=int, required=True)
    p.add_argument("--pulse-times", nargs="+", type=float, default=[0.0])
    p = subs.add_parser("sync-start", help="match one optical START per camera to the recorded flight firmware event")
    p.add_argument("run")
    p.add_argument("--roi-a", nargs=4, type=int, required=True)
    p.add_argument("--roi-b", nargs=4, type=int, required=True)
    p = subs.add_parser("clock", help="explicit measured ESP clock anchors; never host-send guesses")
    p.add_argument("run")
    p.add_argument("--source-times", nargs="+", type=float, required=True)
    p.add_argument("--experiment-times", nargs="+", type=float, required=True)
    p.add_argument("--evidence", required=True)
    for name in ("pose", "auto-pose"):
        p = subs.add_parser(name, help="triangulate physical-point annotations" if name == "pose" else "experimental rim detector; no validated axis estimate")
        p.add_argument("run")
        p.add_argument("--calibration", required=True)
        if name == "pose":
            p.add_argument("--annotations", required=True)
        p.add_argument("--confirm-preprocessing", action="store_true")
    p = subs.add_parser("process", help="world/time conversion, takeoff, plots and synchronized preview clips")
    p.add_argument("run"); p.add_argument("--no-clips", action="store_true")
    p = subs.add_parser("record", help="open existing dual-camera recorder; SPACE starts/stops, q quits")
    p.add_argument("out")
    p.add_argument("--indices", nargs=2, type=int)
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=400)
    p.add_argument("--fps", type=float, default=120)
    p.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "avfoundation", "v4l2"])
    p.add_argument("--codec", default="auto", choices=["auto", "avc1", "mp4v"])
    p.add_argument("--seconds", type=float)
    p.add_argument("--start", action="store_true", help="record immediately; useful for timed notebook/CLI runs")
    p.add_argument("--no-flip", action="store_true")
    p = subs.add_parser("record-telemetry", help="passive serial receive; port must not be owned by a running controller")
    p.add_argument("path"); p.add_argument("--port", required=True); p.add_argument("--seconds", type=float, required=True)
    subs.add_parser("self-check", help="offline tests; no camera, serial, or coil drive")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = str(init_run(args.run))
        elif args.command == "demo":
            from LLM.demo import make_demo
            result = make_demo(args.run)
        elif args.command == "example-pattern":
            from LLM.pattern import example_pattern
            if Path(args.path).exists():
                raise ValueError("pattern file already exists")
            write_json(args.path, example_pattern()); result = args.path
        elif args.command == "import-flight":
            from LLM.capture import import_flight
            result = import_flight(args.source, args.run)["validation"]
        elif args.command == "world":
            from LLM.world import build_world_frame
            path = Path(args.run) / "calibration/world_frame.json"
            if path.exists():
                raise ValueError("world frame snapshot exists; use a new run for a different frame")
            result = build_world_frame(args.origin, args.up, args.x_hint)
            result["evidence"] = args.evidence
            write_json(path, result)
        elif args.command == "world-from-points":
            from LLM.reference import build_world_from_reference
            run = Path(args.run)
            destination = run / "calibration/world_frame.json"
            if destination.exists():
                raise ValueError("world frame snapshot already exists; use a new run for a different frame")
            snapshot(args.calibration, run / "calibration/stereo_rig.json")
            snapshot(args.reference, run / "raw/world_reference.json")
            result = build_world_from_reference(run, run / "raw/world_reference.json", run / "calibration/stereo_rig.json",
                args.confirm_preprocessing, max_skew_s=read_json(run / "config.json").get("max_stereo_skew_s", .01))
            write_json(destination, result)
        elif args.command == "sync":
            from LLM.pipeline import synchronize
            result = synchronize(args.run, args.roi_a, args.roi_b, args.pulse_times)["clocks"]
        elif args.command == "sync-start":
            from LLM.start import synchronize_start
            result = synchronize_start(args.run, args.roi_a, args.roi_b)["clocks"]
        elif args.command == "clock":
            from LLM.pipeline import add_clock
            result = add_clock(args.run, "esp", args.source_times, args.experiment_times, args.evidence)
        elif args.command in ("pose", "auto-pose"):
            result = _pose(args.run, getattr(args, "annotations", None), args.calibration,
                           args.confirm_preprocessing, args.command == "auto-pose")
        elif args.command == "process":
            from LLM.pipeline import process_run
            result = process_run(args.run, clips=not args.no_clips)
        elif args.command == "record":
            from LLM.capture import record_flight
            result = record_flight(args.out, indices=args.indices, width=args.width, height=args.height,
                                   fps=args.fps, rotate180=not args.no_flip, backend=args.backend,
                                   codec=args.codec, seconds=args.seconds, start=args.start)
        elif args.command == "record-telemetry":
            from LLM.telemetry import record_telemetry
            result = record_telemetry(args.path, args.port, args.seconds)
        else:
            from LLM.tests.run import run
            run(); result = "All offline checks passed"
        print(result)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.exit(2, f"LLM: {error}\n")


if __name__ == "__main__":
    main()
