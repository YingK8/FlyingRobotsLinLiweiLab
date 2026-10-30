"""Recorded START: fake serial owner, real encoded videos, no physical hardware."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from LLM.io import init_run, read_json, sha256, write_json
from LLM.start import read_start, synchronize_start
from controller.control.link import SerialComm
from controller.control.hover_controller_runner import CommandLink, RunConfig, fly

EVENT = ("SYNC_START v=2 led0=10000000 led1=10000010 drive0=10000012 drive1=10000100 "
         "on_us=100000 gap_us=100000")
STOP = "SYNC_STOP v=2 t0=10680000 t1=10680010 on_us=100000 gap_us=100000"
CURRENT = "t=10500 freq=20.0 | I[A]: A=1 B=2 C=3 D=4 | duty[%]: A=10 B=20 C=30 D=40"


def reject(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("invalid START input accepted")


def log(path, lines):
    path.write_text("".join(json.dumps({"host_receive_s": 100 + i * .1, "line": line}) + "\n"
                            for i, line in enumerate(lines)))


class Port:
    def __init__(self):
        self.chunks, self.sent = [], []
        self.opened = self.closed = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def reset_input_buffer(self):
        self.chunks = []

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, _n):
        return self.chunks.pop(0)

    def write(self, value):
        self.sent.append(value)


def run():
    with tempfile.TemporaryDirectory(prefix="llm-start-") as directory:
        root = Path(directory)
        path = root / "raw.jsonl"
        port = Port()
        with patch("controller.control.link.serial.Serial", return_value=port):
            link = SerialComm(port="fake", raw_log_path=path)
            # Long split line, batched lines and a response received DURING a send.
            port.chunks = [b"SYNC_READY v=1\r\n", EVENT[:30].encode(),
                           (EVENT[30:] + "\r\n" + CURRENT + "\n").encode()]
            assert link.handle_serial_comm() == "SYNC_READY v=1"
            assert link.handle_serial_comm() is None
            assert link.handle_serial_comm("phase=0:90:180:270") == EVENT
            assert link.handle_serial_comm() == CURRENT
            link.close()
            assert port.closed and len(path.read_text().splitlines()) == 3
            assert read_start(path)["start"]["drive1_us"] == 10000100
            assert read_start(path)["stop"] is None
            reject(lambda: SerialComm(port="fake", raw_log_path=path))
        # File-close failure must not prevent thermal bookkeeping (mocked, no real stamp).
        class BrokenLog:
            def close(self):
                raise OSError("test disk failure")
        from types import SimpleNamespace
        fake = SerialComm.__new__(SerialComm)
        fake.ser, fake._raw_log = Port(), BrokenLog()
        fake._drive_since, fake._drive_s = None, 1.0
        stamped = []
        thermal = SimpleNamespace(add_energised=lambda seconds: stamped.append(seconds) or 25)
        with patch.dict("sys.modules", {"ai.thermal": SimpleNamespace(coil_thermal=thermal)}):
            fake.close()
        assert stamped == [1.0]
        for lines in (["seq go: 1 segment(s)"], [EVENT],
                      ["SYNC_READY v=1", EVENT, EVENT],
                      ["SYNC_READY v=1", EVENT, "SYNC_READY v=1"],
                      ["SYNC_READY v=1", EVENT.replace("led1=10000010", "led1=9999999")],
                      ["SYNC_READY v=1", EVENT + " garbage"],
                      ["SYNC_READY v=1", EVENT, CURRENT.replace("t=10500", "t=500")],
                      ["SYNC_READY v=1", EVENT.replace("v=2", "v=1").split(" on_us")[0]],
                      ["SYNC_READY v=1", EVENT.replace("on_us=100000", "on_us=0")],
                      ["SYNC_READY v=1", STOP, EVENT],
                      ["SYNC_READY v=1", EVENT, STOP, STOP],
                      ["SYNC_READY v=1", EVENT, STOP.replace("t0=10680000", "t0=9999999")],
                      ["SYNC_READY v=1", EVENT, STOP.replace("on_us=100000", "on_us=10000")]):
            log(path, lines)
            reject(lambda: read_start(path))
        log(path, ["SYNC_READY v=1", EVENT, STOP, CURRENT])
        parsed = read_start(path)
        assert parsed["stop"]["t0_us"] == 10680000 and parsed["start"]["on_us"] == 100000

        # Existing sole reader must not drop a response returned by send().
        command = CommandLink(None, True, root / "commands.log")
        class Response:
            def handle_serial_comm(self, outgoing):
                return "state=1 freq=2.0"
        command.comm = Response()
        command.send("phase=0:90:180:270")
        assert command.state == 1 and command.freq == 2
        command.comm = None
        # A dry run has no transport: stop must report unacknowledged, not wait a second.
        assert command.stop() is False
        command.sync_start, command.sync_ready = True, True
        reject(command.send_takeoff)  # no recorded pre-roll
        command.note_capture(50)
        command.note_capture(50.1)
        reject(command.send_takeoff)
        command.note_capture(50.3)
        sent = []
        command.send = sent.append
        command.arm()  # must not reset/re-open a port
        assert sent[-1] == "seq=go" and command.sync_sent
        reject(command.arm)
        command.close()
        reject(lambda: fly(RunConfig(sync_start=True, dry_run=True)))

        # Actual CommandLink construction/reset/READY path against a fake transport.
        for ready_at in (None, 1.0, 3.0):
            port = Port()
            elapsed = [0.0]
            emitted = [False]
            def boot_wait(seconds):
                elapsed[0] += seconds
                if ready_at is not None and elapsed[0] >= ready_at and not emitted[0]:
                    port.chunks.append(b"SYNC_READY v=1\n")
                    emitted[0] = True
            with patch("controller.control.link.serial.Serial", return_value=port), \
                    patch("controller.control.hover_controller_runner.time.sleep", side_effect=boot_wait), \
                    patch("controller.control.hover_controller_runner.time.monotonic", side_effect=lambda:elapsed[0]):
                def construct():
                    return CommandLink("fake", False, root / f"ready-{ready_at}.log",
                                       raw_log_path=root / f"ready-{ready_at}.jsonl", sync_start=True)
                if ready_at is None:
                    try: construct()
                    except ValueError as exc: assert 'no serial bytes received' in str(exc)
                    else: raise AssertionError('Missing READY accepted')
                    assert port.closed and not port.sent
                    assert elapsed[0] >= 5
                else:
                    owner = construct()
                    assert owner.sync_ready and not port.sent
                    reject(owner.send_takeoff)
                    owner.note_capture(1)
                    owner.note_capture(1.3)
                    owner.arm()
                    assert port.sent[-1] == b"seq=go\n"
                    reject(owner.arm)
                    owner.send("stop")
                    owner.close()

        def write_take(directory, count, fps, lit):
            """A stereo take whose ROI follows `lit(frame)`."""
            camera = directory / "raw/camera"
            for tag in "AB":
                (camera / tag).mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(camera / tag / f"{tag}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
                assert writer.isOpened()
                try:
                    for i in range(count):
                        frame = np.zeros((64, 64, 3), np.uint8)
                        if lit(i):
                            frame[16:40, 16:40] = 255
                        writer.write(frame)
                finally:
                    writer.release()
            stamps = 100 + np.arange(count) / fps
            pd.DataFrame({"index": np.arange(count), "t_capture": stamps,
                          "t_a": stamps, "t_b": stamps + .004,
                          "skew_s": .004, "written": 1}).to_csv(camera / "frames.csv", index=False)
            write_json(camera / "meta.json", {"n_frames": count, "dropped": 0,
                                               "encoder_errors": 0, "mode": [64, 64]})
            return stamps

        count, fps = 40, 50
        def start_only(i):
            return 6 <= i <= 10 or i >= 16
        run_dir = init_run(root / "run", source="synthetic")
        write_take(run_dir, count, fps, start_only)
        raw = run_dir / "raw/telemetry.jsonl"
        log(raw, ["SYNC_READY v=1", EVENT, CURRENT])
        result = synchronize_start(run_dir, [16, 16, 24, 24], [16, 16, 24, 24])
        assert set(result["clocks"]) == {"camera_A", "camera_B", "esp"}
        assert abs(result["clocks"]["esp"]["offset_s"] + 10.000005) < 1e-9
        assert result["device_start"]["drive_apply_interval_t_s"][0] > 0
        assert result["input_sha256"]["raw/telemetry.jsonl"] == sha256(raw)
        assert result["hardware_sync_validated"] is False
        assert [m["kind"] for m in result["markers"]["A"]["markers"]] == ["start"]
        assert [m["kind"] for m in result["markers"]["B"]["markers"]] == ["start"]
        assert result["clocks"]["camera_A"]["drift_measured"] is False
        assert "device_stop" not in result
        assert abs(result["cross_camera"]["start_skew_s"]) < result["cross_camera"]["bound_s"]
        assert read_json(run_dir / "processed/sync.json") == result
        # Inputs-only processing must succeed with no rig/world/pose and decode its plot.
        from LLM.experiment import postprocess
        manifest=read_json(run_dir/'manifest.json')
        write_json(run_dir/'manifest.json',manifest | {'experiment_mode':'sync_check'})
        write_json(run_dir/'pattern.json', {'schema_version':1,'trigger':'time',
            'frequency_segments':[[2,20,1,0,1]],'channel_updates':[],
            'channel_settings_status':'firmware_defaults_unrecorded'})
        checked=postprocess(run_dir,[16,16,24,24],[16,16,24,24])
        assert checked['status']=='sync_check_inputs_only'
        assert checked['hardware_sync_validated'] is False
        assert cv2.imread(checked['input_curves']) is not None
        assert not (run_dir/'processed/pose.csv').exists()
        run2 = init_run(root / "run_stop", source="synthetic")
        def start_stop(i):
            if i <= 5: return False
            if i <= 10: return True
            if i <= 15: return False
            if i <= 39: return True
            if i <= 44: return False
            if i <= 49: return True
            return False
        stamps2 = write_take(run2, 60, fps, start_stop)
        stop_us = int(round((10.000005 + (stamps2[40] - stamps2[6])) * 1e6))
        stop_line = f"SYNC_STOP v=2 t0={stop_us} t1={stop_us + 10} on_us=100000 gap_us=100000"
        raw2 = run2 / "raw/telemetry.jsonl"
        log(raw2, ["SYNC_READY v=1", EVENT, stop_line, CURRENT])
        two = synchronize_start(run2, [16, 16, 24, 24], [16, 16, 24, 24])
        assert [m["kind"] for m in two["markers"]["A"]["markers"]] == ["start", "stop"]
        assert two["clocks"]["camera_A"]["drift_measured"] is True
        assert abs(two["clocks"]["camera_A"]["scale"] - 1) < .01
        assert two["device_stop"]["stop_reference_device_t_s"] > 0
        write_take(run2, 60, fps, lambda i: i >= 16 or 6 <= i <= 10)
        reject(lambda: synchronize_start(run2, [16, 16, 24, 24], [16, 16, 24, 24]))

        # A failed re-association must remove the old working clocks.
        log(raw, ["SYNC_READY v=1"])
        reject(lambda: synchronize_start(run_dir, [16, 16, 24, 24], [16, 16, 24, 24]))
        assert not (run_dir / "processed/sync.json").exists()
    print("LLM START: raw RX tee, one-start/pre-roll gates, device-clock validation and optical association passed")


if __name__ == "__main__":
    run()
