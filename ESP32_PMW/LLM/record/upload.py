"""Copy run folders to the lab Google Drive, filed FLAT as `<YYYY-MM-DD>/<run name>/<file>`.

Operator 2026-11-15: no subfolders on Drive. `raw/camera/A/A.mp4` -> `A.mp4`, every other
file -> its path joined with `_` (`raw/telemetry.jsonl` -> `raw_telemetry.jsonl`), and
`raw/captures/` is not sent: `import.json` records it byte-identical to `raw/camera/`.
SOURCE.md lists every Drive name against its local path. The local run is never changed.

    .venv/bin/python -m LLM.record.upload LLM/runs/experiment_20261115_142201_ab12cd
    .venv/bin/python -m LLM.record.upload LLM/runs          # every dated run in it
    .venv/bin/python -m LLM.record.upload --self-check

One-time: `brew install rclone`, then `rclone config` -> new remote `gdrive`, storage
`drive`, defaults, browser login. `rclone copy` only: it never deletes, locally or on
Drive, and a re-run skips unchanged files and adds outputs written since.
"""
import json
import os
import re
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

REMOTE = "gdrive:"                                 # the remote name `rclone config` was given
FOLDER_ID = "14STwkriICMQNyhygB8EQEcvuXoE2YKNV"    # the operator's lab folder link, 2026-11-15
REPO = "https://github.com/YingK8/FlyingRobotsLinLiweiLab"   # origin, 2026-11-15
DATED = re.compile(r"(\d{4})(\d{2})(\d{2})_\d{6}")
SKIP = ("raw/captures/",)
# OpenCV's AVFoundation writer stamps H.264 level 3.2, which allows 216000 macroblocks/s;
# 1280x800 at 120 fps is 480000 (640x400 at 210: 210000). 4.2 allows 522240. Rewriting the
# level field (and moving moov first) changes no coded frame. Whether it is what crops
# Drive's player (operator, 2026-11-15) is UNVERIFIED: it is the only anomaly ffprobe found.
H264_LEVEL = "4.2"


def flat_name(rel):
    """`raw/camera/A/A.mp4` -> `A.mp4` (file named after its folder); else parts joined by `_`."""
    parts = Path(rel).parts
    if len(parts) >= 2 and Path(parts[-1]).stem == parts[-2]:
        return parts[-1]
    return "_".join(parts)


def flat_files(run):
    """``{drive name: local path}``; two files mapping to one name is refused, not overwritten."""
    out = {}
    for path in sorted(Path(run).rglob("*")):
        rel = path.relative_to(run).as_posix()
        if path.is_file() and not rel.startswith(SKIP):
            name = flat_name(rel)
            if name in out:
                raise ValueError(f"{rel} and {out[name]} both flatten to {name}")
            out[name] = path
    return out


def dest(run):
    """`YYYY-MM-DD/<name>` from the date in the run's own name; no date, no guess."""
    name = Path(run).name
    m = DATED.search(name)
    if m is None:
        raise ValueError(f"{name!r} has no YYYYMMDD_HHMMSS in its name; not filing it under a guessed date")
    return f"{m[1]}-{m[2]}-{m[3]}/{name}"


def runs_in(path):
    """A run folder is itself; any other folder (e.g. LLM/runs) is its dated children."""
    path = Path(path)
    if DATED.search(path.name):
        return [path]
    return sorted(p for p in path.iterdir() if p.is_dir() and DATED.search(p.name))


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()


def source_md(run):
    """SOURCE.md: the branch and the commit the run itself recorded (raw/run_parameters.json),
    which is the code that made it -- not the code checked out at upload time."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    params = Path(run) / "raw/run_parameters.json"
    rec = (json.loads(params.read_text()).get("software") or {}).get("git") if params.exists() else None
    lines = [f"# {Path(run).name}", "", f"- Branch: [{branch}]({REPO}/tree/{branch})"]
    manifest = Path(run) / "manifest.json"
    m = json.loads(manifest.read_text()) if manifest.exists() else {}
    note = (m.get("initial_conditions") or {}).get("notes")
    if note:
        lines.append(f"- Note: {note.strip()}")
    if m.get("labels"):
        lines.append("- Labels: " + "; ".join(f"{cat}: {', '.join(picks)}" for cat, picks in m["labels"].items()))
    if rec and rec.get("commit"):
        c = rec["commit"]
        lines.append(f"- Recorded at commit: [{c[:7]}]({REPO}/tree/{c})")
        if rec.get("dirty_files"):
            lines.append(f"- **{rec['dirty_files']} uncommitted file(s) at record time**: "
                         "the exact code is not on GitHub")
    else:
        lines.append("- Recorded commit: not in raw/run_parameters.json")
    return "\n".join(lines) + "\n"


def upload(run, on_staged=None):
    """Stage, then copy to Drive. ``on_staged()`` is called once the run folder is no longer
    read: the stage holds remuxed videos and HARD links (not symlinks), so the run can be
    moved to the Trash while rclone is still sending."""
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone not found: `brew install rclone`, then `rclone config` (remote name gdrive)")
    if not (Path(run) / "manifest.json").is_file():
        raise FileNotFoundError(f"{run}: no manifest.json -- moved (discarded?) or not a run")
    target = REMOTE + dest(run)
    files = flat_files(run)
    ffmpeg = shutil.which("ffmpeg")
    md = source_md(run) + "\n## Files (Drive name <- local path in the run)\n\n" + "".join(
        f"- `{name}` <- `{path.relative_to(run).as_posix()}`\n" for name, path in files.items())
    md += ("\n`*.mp4`: H.264 level field set to " + H264_LEVEL + " and moov moved first "
           "(`ffmpeg -c copy`), so their SHA-256 differs from `raw_camera_import.json`; "
           "the frames are bit-identical.\n" if ffmpeg else "")
    # Staged beside, never written into, the run folder: that folder is evidence.
    with tempfile.TemporaryDirectory() as stage:
        for name, path in files.items():
            if path.suffix == ".mp4" and ffmpeg:
                subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-c", "copy", "-bsf:v",
                                f"h264_metadata=level={H264_LEVEL}", "-movflags", "+faststart",
                                os.path.join(stage, name)], check=True)
            else:
                try:   # same volume: free and instant; the inode outlives a move of the run
                    os.link(path, os.path.join(stage, name))
                except OSError:
                    shutil.copy2(path, os.path.join(stage, name))
        Path(stage, "SOURCE.md").write_text(md)
        if on_staged is not None:
            on_staged()
        # --checksum: a re-staged video has a new mtime but the same bytes, so it is skipped
        subprocess.run(["rclone", "copy", "--checksum", stage, target,
                        "--drive-root-folder-id", FOLDER_ID, "--transfers", "4",
                        "--stats-one-line", "--stats", "10s"], check=True)
    return target


def _self_check():
    assert dest("LLM/runs/experiment_20261115_142201_ab12cd") == "2026-11-15/experiment_20261115_142201_ab12cd"
    assert dest("sync_check_20261106_191129_0fd047") == "2026-11-06/sync_check_20261106_191129_0fd047"
    try:
        dest("sessions")
    except ValueError:
        pass
    else:
        raise AssertionError("a dateless name must be refused")
    md = source_md("LLM/runs/sessions")
    assert md.startswith("# sessions") and f"{REPO}/tree/" in md and "not in raw/run_parameters.json" in md
    assert flat_name("raw/camera/A/A.mp4") == "A.mp4"
    assert flat_name("raw/telemetry.jsonl") == "raw_telemetry.jsonl"
    assert flat_name("manifest.json") == "manifest.json"
    with tempfile.TemporaryDirectory() as tmp:
        for rel in ("raw/camera/A/A.mp4", "raw/captures/x/A/A.mp4", "raw/camera/frames.csv", "ramp.json"):
            Path(tmp, rel).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp, rel).write_text(rel)
        assert sorted(flat_files(tmp)) == ["A.mp4", "ramp.json", "raw_camera_frames.csv"], "captures skipped"
        Path(tmp, "raw_camera_frames.csv").write_text("clash")
        try:
            flat_files(tmp)
        except ValueError:
            pass
        else:
            raise AssertionError("a flattened-name collision must be refused")
    with tempfile.TemporaryDirectory() as tmp:     # the stage survives the run being moved
        run = Path(tmp, "sync_check_20261106_191129_0fd047")
        (run / "raw").mkdir(parents=True)
        (run / "manifest.json").write_text("{}")
        (run / "raw/telemetry.jsonl").write_text("t")
        real = subprocess.run
        def fake(cmd, **kw):
            if cmd[0] != "rclone":
                return real(cmd, **kw)
            stage = Path(cmd[3])
            assert not run.exists() and (stage / "raw_telemetry.jsonl").read_text() == "t", "staged by hard link"
            return real(["true"])
        subprocess.run = fake
        try:
            upload(run, on_staged=lambda: shutil.move(str(run), str(Path(tmp, "trash"))))
        finally:
            subprocess.run = real
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("sessions", "discarded", "sync_check_20261106_191129_0fd047", "experiment_20261115_142201_ab12cd"):
            (Path(tmp) / name).mkdir()
        assert [p.name for p in runs_in(tmp)] == ["experiment_20261115_142201_ab12cd", "sync_check_20261106_191129_0fd047"]
        assert runs_in(Path(tmp) / "sync_check_20261106_191129_0fd047")[0].name.startswith("sync_check_")
    print("upload self-check passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        _self_check()
    elif not sys.argv[1:]:
        sys.exit(__doc__)
    else:
        for arg in sys.argv[1:]:
            for run in runs_in(arg):
                print(f"{run} -> {upload(run)}", flush=True)
