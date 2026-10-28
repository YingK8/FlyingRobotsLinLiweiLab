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

EVENT = "SYNC_START v=1 led0=10000000 led1=10000010 drive0=10000012 drive1=10000100"
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
            assert read_start(path)["drive1_us"] == 10000100
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
                      ["SYNC_READY v=1", EVENT, CURRENT.replace("t=10500", "t=500")]):
            log(path, lines)
            reject(lambda: read_start(path))

        # Existing sole reader must not drop a response returned by send().
        command = CommandLink(None, True, root / "commands.log")
        class Response:
            def handle_serial_comm(self, outgoing):
                return "state=1 freq=2.0"
        command.comm = Response()
        command.send("phase=0:90:180:270")
        assert command.state == 1 and command.freq == 2
        command.comm = None
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

        # Real video decoder/edge detection + device event association.
        run_dir = init_run(root / "run", source="synthetic")
        camera = run_dir / "raw/camera"
        count, fps = 20, 20
        for tag in "AB":
            (camera / tag).mkdir(parents=True)
            writer = cv2.VideoWriter(str(camera / tag / f"{tag}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
            assert writer.isOpened()
            try:
                for i in range(count):
                    frame = np.zeros((64, 64, 3), np.uint8)
                    if i >= 6:
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
        raw = run_dir / "raw/telemetry.jsonl"
        log(raw, ["SYNC_READY v=1", EVENT, CURRENT])
        result = synchronize_start(run_dir, [16, 16, 24, 24], [16, 16, 24, 24])
        assert set(result["clocks"]) == {"camera_A", "camera_B", "esp"}
        assert abs(result["clocks"]["esp"]["offset_s"] + 10.000005) < 1e-9
        assert result["device_start"]["drive_apply_interval_t_s"][0] > 0
        assert result["input_sha256"]["raw/telemetry.jsonl"] == sha256(raw)
        assert result["hardware_sync_validated"] is False
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
        # A failed re-association must remove the old working clocks.
        log(raw, ["SYNC_READY v=1"])
        reject(lambda: synchronize_start(run_dir, [16, 16, 24, 24], [16, 16, 24, 24]))
        assert not (run_dir / "processed/sync.json").exists()
    print("LLM START: raw RX tee, one-start/pre-roll gates, device-clock validation and optical association passed")


if __name__ == "__main__":
    run()
