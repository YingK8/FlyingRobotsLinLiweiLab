"""Run directories and portable, strict JSON metadata."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from LLM import __version__


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_run(path, source="real"):
    path = Path(path).resolve()
    if source not in ("real", "synthetic"):
        raise ValueError("source must be real or synthetic")
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty run: {path}")
    for name in ("raw", "calibration", "processed", "export"):
        (path / name).mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True)
    write_json(path / "manifest.json", {
        "schema_version": 1, "run_id": path.name, "source": source,
        "created_utc": datetime.now(timezone.utc).isoformat(), "pipeline_version": __version__,
        "git_commit": revision.stdout.strip() or None, "tracked_worktree_dirty": bool(dirty.stdout.strip()),
        "initial_conditions": {"robot_id": None, "coil_setup_id": None, "bearing_id": None, "notes": "Fill before a real experiment"},
        "time_zero": "operator-associated START LED pulse; not actual current-rise time",
        "hardware_validated": False,
    })
    write_json(path / "config.json", {
        "schema_version": 1,
        "max_stereo_skew_s": 0.01,
        "led": {"threshold": 160, "low_threshold": 80, "min_on_frames": 2},
        "takeoff": {"baseline_start_s": 0.0, "baseline_end_s": 0.2,
                    "min_height_mm": 1.0, "noise_sigma_factor": 5.0,
                    "min_upward_speed_mm_s": 1.0, "sustain_s": 0.05,
                    "max_gap_s": 0.05, "min_baseline_samples": 5},
        "export": {"pre_roll_s": 0.2, "post_takeoff_s": 2.0},
        "defaults_status": "engineering starting values; validate on stationary/spinning bearing recordings",
    })
    return path


def snapshot(source, destination):
    """Copy an input once. Never silently replace calibration or raw evidence."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        return
    if destination.exists():
        if sha256(source) == sha256(destination):
            return
        raise ValueError(f"snapshot exists with different content: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def invalidate_report(run, reason):
    """A changed/failed processing attempt must not advertise an old completed report."""
    folder = Path(run) / "export"
    if folder.is_dir():
        (folder / "INCOMPLETE.txt").write_text(reason + "\n", encoding="utf-8")
        for name in ("report.md", "package.json"):
            (folder / name).unlink(missing_ok=True)


def inventory(run):
    run = Path(run)
    return {str(p.relative_to(run)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(run.rglob("*")) if p.is_file() and
            "export" not in p.relative_to(run).parts and not p.name.endswith(".tmp")}
