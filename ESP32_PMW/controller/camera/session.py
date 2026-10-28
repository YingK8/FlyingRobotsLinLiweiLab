"""Camera-only notebook helpers. No serial imports or actuator commands."""
from __future__ import annotations

import platform
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from controller.camera import sources
from controller.camera.record import FlightWriter


def check_environment(codec="auto"):
    """Encode AND decode two synthetic clips before opening any real device."""
    with tempfile.TemporaryDirectory(prefix="pmw-codec-check-") as tmp:
        writer = FlightWriter(tmp, fps=30, codec=codec, meta={"source": "synthetic_codec_check"})
        try:
            for i in range(12):
                frame = np.full((64, 80), 20 + i * 15, dtype=np.uint8)
                writer.add(i / 30, [frame, frame], (i / 30, i / 30 + .001), .001)
        finally:
            writer.close()
        counts = {}
        for tag in "AB":
            cap = cv2.VideoCapture(str(writer.dir / tag / f"{tag}.mp4"))
            try:
                n = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (64, 80):
                        raise RuntimeError("codec changed image dimensions")
                    n += 1
                counts[tag] = n
            finally:
                cap.release()
        if writer.errors or writer.dropped or counts != {"A": 12, "B": 12}:
            raise RuntimeError(f"codec round-trip failed: {counts}, errors={writer.errors}, drops={writer.dropped}")
        return {"python": sys.version.split()[0], "python_executable": sys.executable,
                "system": platform.system(), "opencv": cv2.__version__,
                "camera_backends": [cv2.videoio_registry.getBackendName(b)
                                    for b in cv2.videoio_registry.getCameraBackends()],
                "default_backend": cv2.videoio_registry.getBackendName(sources.default_backend()),
                "codec": writer.codec, "two_clip_encode_decode": "PASS (12 frames each)",
                "hardware_tested": False}


def probe(max_index=6, *, width=1280, height=800, fps=120, backend="auto", seconds=1.0):
    """Probe sequentially and release each camera. Results are candidates, not IDs.

    Re-probe after changing backend: DirectShow and MSMF need not enumerate identically.
    Two independent successes do not prove simultaneous capture; the next notebook cell
    opens the chosen pair together.
    """
    if max_index < 1 or seconds <= 0:
        raise ValueError("positive max_index and seconds required")
    rows = []
    for index in range(max_index):
        cam = None
        row = {"index": index}
        try:
            cam = sources.MonoCamera(index=index, width=width, height=height, fps=fps,
                                     backend=backend, grayscale=True)
            row.update(cam.actual)
            stamps = []
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                item = cam.read()
                if item is None:
                    break
                t, frame = item
                stamps.append(t)
                row["delivered_size"] = f"{frame.shape[1]}x{frame.shape[0]}"
            row["read_frames"] = len(stamps)
            row["read_fps"] = ((len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else None)
            row["status"] = ("candidate; confirm A/B visually" if stamps and
                             row["delivered_size"] == f"{width}x{height}" else "no matching frames")
        except (OSError, cv2.error) as exc:
            row["status"] = str(exc)
        finally:
            if cam is not None:
                cam.close()
        rows.append(row)
    return rows
