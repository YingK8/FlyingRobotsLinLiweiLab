"""Opening the ELP, and the one thing `pose/sources.py` does not already do.

`sources.CameraSource` is the capture code -- threaded grabber, drop-oldest slot,
dropped-frame counting -- and nothing here reimplements any of it. What is added
is small and specific:

* a **device profile** (`elp_camera.json`) so the sensor's modes are data rather
  than a list retyped into each script;
* **strict mode checking**, because `cv2.VideoCapture.set` returns success while
  quietly giving you something else, and a silent fallback from 640x400 to
  640x480 is discovered as bad poses rather than as an error;
* **one call that opens either one camera or two** (`open_group`), returning a
  `CameraSource` or a `StereoSource`, plus `as_frames` to flatten the difference.
  That is what lets `modes.py`, `capture.py` and the notebook be written once,
  branch-free, and it means the stereo path is exercised today by pairing the ELP
  with the built-in FaceTime camera rather than waiting for a second ELP.

macOS only in one respect: AVFoundation gives OpenCV no device *names*, so
cameras can be addressed by index alone. `probe_indices` exists because of that
and for no better reason.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. camera is stage 1 of 4.
sys.path[:0] = [str(HERE)]

import sources  # noqa: E402

DEFAULT_PATH = HERE / "elp_camera.json"

Mode = namedtuple("Mode", "width height fps")

_MODE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*(?:@\s*(\d+(?:\.\d+)?))?\s*$")


def _is_macos():
    return sys.platform == "darwin"


def default_backend():
    return cv2.CAP_AVFOUNDATION if _is_macos() else cv2.CAP_V4L2


def load_profile(path=DEFAULT_PATH):
    """The device profile, or ``{}`` if it has not been written yet."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def modes(profile=None):
    """The sensor's modes, as `Mode` tuples, in the profile's order."""
    profile = load_profile() if profile is None else profile
    return [Mode(m["width"], m["height"], m["fps"]) for m in profile.get("modes", [])]


def parse_mode(spec):
    """``"1280x800@120"`` -> `Mode`. The fps part is optional."""
    if isinstance(spec, Mode):
        return spec
    m = _MODE_RE.match(str(spec))
    if not m:
        raise ValueError(f"cannot parse mode {spec!r}; expected WxH or WxH@FPS")
    w, h, f = m.groups()
    return Mode(int(w), int(h), float(f) if f else None)


def probe_indices(max_index=4, backend=None):
    """Which camera indices actually open and deliver a frame.

    Opening is not enough -- a device can open and then fail to read -- so this
    takes a frame before believing it. Costs a second or two, and is only ever
    used interactively.
    """
    backend = default_backend() if backend is None else backend
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append((i, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        finally:
            cap.release()
    return found


def open_elp(index=0, mode=None, grayscale=True, strict=True, fourcc="MJPG", **kw):
    """A `sources.CameraSource` on the ELP, with the mode checked rather than hoped.

    ``strict`` raises when the driver granted something other than what was asked.
    On by default because the failure it catches is silent: a mode that quietly
    falls back changes the intrinsics' scale, and every distance downstream is
    then wrong by a factor nothing else can detect.

    Note the frame rate is deliberately *not* checked -- `CAP_PROP_FPS` reports
    what was requested, not what arrives, so it cannot verify anything. Use
    `modes.py`, which times real reads.
    """
    m = parse_mode(mode) if mode is not None else None
    cam = sources.CameraSource(
        index=index,
        width=m.width if m else kw.pop("width", None),
        height=m.height if m else kw.pop("height", None),
        fps=(m.fps if m else None) or kw.pop("fps", None),
        fourcc=fourcc,
        grayscale=grayscale,
        backend=kw.pop("backend", default_backend()),
    )
    if strict and m is not None:
        got = cam.actual
        if (got["width"], got["height"]) != (m.width, m.height):
            cam.close()
            raise OSError(
                f"camera {index}: asked {m.width}x{m.height}, got "
                f"{got['width']}x{got['height']}. The driver substituted a mode "
                f"rather than refusing; pass strict=False to accept it.")
    return cam


def open_group(indices, mode=None, max_skew_s=None, **kw):
    """One index gives a `CameraSource`, several give a `StereoSource`.

    Returns ``(source, cameras)`` -- the second is always the flat list of
    `CameraSource`, because per-camera counters (`n_grabbed`, `n_dropped`,
    `actual`) are not reachable through `StereoSource` and are exactly what a
    throughput report needs.
    """
    idx = [indices] if isinstance(indices, int) else list(indices)
    cams = [open_elp(index=i, mode=mode, **kw) for i in idx]
    if len(cams) == 1:
        return cams[0], cams
    return sources.StereoSource(cams, max_skew_s=max_skew_s), cams


def as_frames(item):
    """Normalise a read to ``(t, [frame, ...])`` whether mono or stereo."""
    if item is None:
        return None
    t, payload = item
    return t, (list(payload) if isinstance(payload, (list, tuple)) else [payload])


def native_modes_ffmpeg(index=0):
    """The modes AVFoundation says the device *actually* has, or ``None``.

    OpenCV cannot report this: it will happily accept a resolution the sensor
    does not have and let the driver synthesise it, which is the leading
    explanation for a mode that measures slower than its neighbours. ffmpeg lists
    the real ones on stderr. Optional -- returns ``None`` when ffmpeg is absent
    rather than making it a dependency.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "-hide_banner", "-f", "avfoundation", "-list_formats", "all",
         "-i", str(index)],
        capture_output=True, text=True, timeout=30)
    text = (proc.stderr or "") + (proc.stdout or "")
    out = []
    for line in text.splitlines():
        m = re.search(r"(\d+)x(\d+)@\[([\d.]+)\s+([\d.]+)\]fps", line)
        if m:
            out.append({"width": int(m.group(1)), "height": int(m.group(2)),
                        "fps_min": float(m.group(3)), "fps_max": float(m.group(4))})
    return out or None
