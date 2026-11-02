"""No hardware: Bench step order/state hand-off, and find_led on a synthetic START pulse."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


def _pulse_video(path, fps=120.0, n=240, start_s=1.0, on_s=0.03, gap_s=0.03):
    """Dark scene; an LED at (200,50) goes on-off-on at start_s; a bright block oscillates."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    times = np.arange(n) / fps
    for i, t in enumerate(times):
        frame = np.full((240, 320), 20, np.uint8)
        led_on = (start_s <= t < start_s + on_s) or t >= start_s + on_s + gap_s
        if led_on:
            cv2.rectangle(frame, (196, 46), (204, 54), 255, -1)
        x = 60 + (20 if (i // 3) % 2 else 0)                    # the "robot": always moving
        cv2.rectangle(frame, (x, 150), (x + 30, 180), 230, -1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return times


def check_find_led():
    from LLM.analyse.start import find_led
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "A.mp4"
        times = _pulse_video(video)
        x, y, w, h = find_led(video, times, start_host_s=1.1, on_s=0.03, gap_s=0.03)
        assert x <= 200 < x + w and y <= 50 < y + h, (x, y, w, h)
        assert w < 30 and h < 30, "box should hug the LED, not the moving block"
        try:   # no pulse in the window -> refuse, never guess
            find_led(video, times, start_host_s=0.5, on_s=0.03, gap_s=0.03, before_s=0.3, after_s=0.3)
        except ValueError:
            pass
        else:
            raise AssertionError("find_led must refuse a window with no START pair")


def check_bench_flow():
    import LLM.record.experiment as experiment
    import LLM.record.panel as panel
    import controller.calib.capture as capture
    import LLM.analyse.rig as geometry
    from LLM.bench import Bench
    calls = []

    def fake_panel(session, *, output_root, run=None, **kw):
        calls.append(("panel", Path(output_root).name, run is not None, kw.get("roi_a")))
        return {"take": str(output_root) + "/take", "reference_pair": str(output_root) + "/ref"}

    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(panel, "run_panel", fake_panel), \
            patch.object(capture, "capture", lambda bag, **kw: calls.append(("capture", Path(bag).parent.name[:6]))), \
            patch.object(geometry, "verify_bag", lambda *a, **k: {"passed": True}), \
            patch.object(experiment, "audit_take", lambda take, s: {"integrity": "PASS", "take": take}), \
            patch.object(experiment, "prepare_sync_run", lambda root, s, trial, ref, **k: (calls.append(("sync_run", trial["take"][-4:])), Path(tmp) / "run")[1]), \
            patch.object(experiment, "finish_take", lambda take, s, run: {"imported": str(run)}), \
            patch.object(experiment, "postprocess", lambda run, a, b, **k: {"run": run, "rois": (a, b)}):
        (Path(tmp) / "run").mkdir()
        bench = Bench((1, 0), "fake-port", runs=Path(tmp))
        for step in ("run", "analyse", "verify_calibration"):
            try:
                getattr(bench, step)("x", note="n") if step == "run" else getattr(bench, step)()
            except ValueError:
                pass
            else:
                raise AssertionError(f"{step} must refuse before its prerequisites")
        bench.use_calibration(Path(tmp) / "rig.json")
        assert bench.verify_calibration() == {"passed": True}
        bench.photograph_reference()
        bench.record()
        report = bench.run("alignment_50Hz", note="n")
        assert report == {"imported": str(Path(tmp) / "run")}
        assert (Path(tmp) / "run/ramp.json").exists()
        out = bench.analyse()
        assert out["rois"] == (None, None), "no LED ROI given -> found during analyse"
    assert [c[0] for c in calls] == ["capture", "panel", "panel", "sync_run", "panel"], calls
    assert calls[-1][2] is True and calls[-1][3] is None, "the run panel opens serial, ROI optional"


def run():
    check_find_led()
    check_bench_flow()
    print("bench: OK")


if __name__ == "__main__":
    run()
