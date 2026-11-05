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
        calls.append(("panel", Path(output_root).name, run is not None, kw.get("roi_a"), kw.get("auto_start")))
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
        report = bench.run("alignment_50Hz", note="n", auto_start=True)
        assert report == {"imported": str(Path(tmp) / "run")}
        assert (Path(tmp) / "run/ramp.json").exists()
        out = bench.analyse()
        assert out["rois"] == (None, None), "no LED ROI given -> found during analyse"
    assert [c[0] for c in calls] == ["capture", "panel", "panel", "sync_run", "panel"], calls
    assert calls[-1][2] is True and calls[-1][3] is None, "the run panel opens serial, ROI optional"
    assert calls[-1][4] is True, "Block 8 automatic start must reach the recording panel"


def check_exposure_modes():
    from controller.camera import uvc
    from LLM.bench import Bench
    from LLM.record.io import read_json
    with patch.object(uvc, "apply", return_value=[{"mode": "manual"}]) as manual, \
            patch.object(uvc, "set_auto", return_value=[{"exposure_mode": "auto"}]) as auto:
        assert uvc.configure(None, 100) is None  # legacy None leaves device controls alone
        manual.assert_not_called(); auto.assert_not_called()
        assert uvc.configure(None, 100, auto_exposure=True)[0]["exposure_mode"] == "auto"
        auto.assert_called_once(); manual.assert_not_called()
        uvc.configure(.004, 100)
        manual.assert_called_once_with(.004, 100)
        try:
            uvc.configure(.004, 100, auto_exposure=True)
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting auto/manual settings must be refused")
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(uvc, "meter") as preview, \
            patch("controller.calib.capture.capture") as capture:
        bench = Bench(width=1280, height=800, fps=120, exposure_ms=None, runs=Path(tmp))
        bench.preview_cameras()
        bench.collect_charuco_frames()
        for call in (preview.call_args, capture.call_args):
            assert call.kwargs["auto_exposure"] is True and call.kwargs["fps"] == 120
        assert preview.call_args.args[0] is None and capture.call_args.kwargs["exposure_s"] is None
        camera = read_json(bench.folder / "session.json")["camera"]
        assert camera["auto_exposure"] is True and camera["exposure_s"] is None
        assert camera["rotate180"] is True


def run():
    check_find_led()
    check_bench_flow()
    check_exposure_modes()
    print("bench: OK")


if __name__ == "__main__":
    run()
