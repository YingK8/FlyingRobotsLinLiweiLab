"""Import the existing camera recorder without changing its timing or driving coils.

The source timestamps are host read-completion times, not exposure timestamps.
``written`` maps an encoded frame back to its original capture row; it must never
be replaced by the video's nominal frame rate. See controller/camera/theory.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _capture_dir(path):
    path = Path(path).resolve()
    return path / "raw" / "camera" if (path / "raw" / "camera").is_dir() else path


def _decode_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    n, size = 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current = (frame.shape[1], frame.shape[0])
            if size is not None and current != size:
                raise ValueError(f"Frame dimensions change within {path}")
            size = current
            n += 1
    finally:
        cap.release()
    if not n:
        raise ValueError(f"Video has no decodable frames: {path}")
    return n, size


def load_flight(run_or_capture_dir):
    """Validate a paired recording and return encoded-order timestamps/indexes.

    Decodes both complete videos so a missing encoded frame cannot silently move
    all subsequent timestamps. Dropped capture rows stay in the original CSV.
    Legacy indexes without ``written`` are accepted only when metadata explicitly
    reports zero drops and the decoded counts match.
    """
    root = _capture_dir(run_or_capture_dir)
    paths = {tag: root / tag / f"{tag}.mp4" for tag in "AB"}
    required = [*paths.values(), root / "frames.csv", root / "meta.json"]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing recording file: {path}")
    metadata = json.loads((root / "meta.json").read_text())
    if metadata.get("encoder_errors", 0):
        raise ValueError("Recording reports encoder errors; frame/time mapping is unsafe")
    rows = pd.read_csv(root / "frames.csv", comment="#")
    if rows.empty:
        raise ValueError("Recording has no timestamp rows")
    for name in ("index", "t_a", "t_b"):
        if name not in rows:
            raise ValueError(f"frames.csv requires {name}; do not infer time from FPS")
        values = pd.to_numeric(rows[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError(f"frames.csv {name} must be finite and strictly increasing")
        if name == "index" and (np.any(values != np.floor(values)) or np.any(values < 0)):
            raise ValueError("Capture indices must be nonnegative integers")
    if "n_frames" in metadata and metadata["n_frames"] != len(rows):
        raise ValueError("meta.json n_frames disagrees with frames.csv")
    legacy = "written" not in rows
    if legacy:
        if metadata.get("dropped") != 0:
            raise ValueError("Legacy index has no written flag and zero drops are not established")
        written = np.ones(len(rows), dtype=bool)
    else:
        flags = pd.to_numeric(rows["written"], errors="raise").to_numpy()
        if not np.isin(flags, (0, 1)).all():
            raise ValueError("written must contain only 0 or 1")
        written = flags == 1
        if "dropped" in metadata and metadata["dropped"] != int((~written).sum()):
            raise ValueError("meta.json dropped disagrees with written flags")
    kept = rows.loc[written]
    if kept.empty:
        raise ValueError("No encoded frames are indexed")
    sizes = {}
    for tag, path in paths.items():
        count, size = _decode_info(path)
        if count != len(kept):
            raise ValueError(f"{tag}: {count} decoded frames but {len(kept)} written rows")
        sizes[tag] = list(size)
    if "mode" in metadata and metadata["mode"] is not None:
        if any(size != list(metadata["mode"]) for size in sizes.values()):
            raise ValueError("Video dimensions disagree with metadata mode")
    return {
        "video_a": str(paths["A"]), "video_b": str(paths["B"]),
        "times_a": kept["t_a"].to_numpy(dtype=float),
        "times_b": kept["t_b"].to_numpy(dtype=float),
        "frame_indices": kept["index"].to_numpy(dtype=np.int64),
        "metadata": {**metadata, "validated_written_frames": len(kept),
                     "dropped_capture_rows": int((~written).sum()),
                     "legacy_index": legacy, "decoded_image_sizes": sizes,
                     "timestamp_semantics": metadata.get(
                         "timestamp_semantics", "host_read_completion_not_exposure")},
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_flight(source_dir, run_dir):
    """Copy a validated capture into a run; refuse to replace existing raw data."""
    source = _capture_dir(source_dir)
    target = Path(run_dir).resolve() / "raw" / "camera"
    if target.exists():
        raise FileExistsError(f"Raw recording already exists: {target}")
    load_flight(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".camera-import-", dir=target.parent))
    files = ("A/A.mp4", "B/B.mp4", "frames.csv", "meta.json")
    try:
        for relative in files:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
        copied = load_flight(staging)
        metadata = {
            "source_directory": str(source),
            "imported_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": {relative: _sha256(staging / relative) for relative in files},
            "validation": copied["metadata"],
        }
        (staging / "import.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if target.exists():
            raise FileExistsError(f"Raw recording already exists: {target}")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return metadata


def record_flight(out_dir, *, indices=None, width=1280, height=800, fps=120.0,
                  rotate180=True, max_skew_s=0.01, preview=True, start=False, note=None):
    """Use the established interactive camera recorder; never open serial or drive.

    This creates dated takes beneath ``out_dir``. Import a completed take using
    ``import_flight``. Do not run concurrently with another owner of the cameras.
    """
    from controller.camera.record import record

    return record(out_dir=out_dir, indices=indices, width=width, height=height,
                  fps=fps, rotate180=rotate180, max_skew_s=max_skew_s,
                  preview=preview, start=start, note=note)


if __name__ == "__main__":
    from LLM.tests.test_capture_pose import run_tests
    run_tests()
