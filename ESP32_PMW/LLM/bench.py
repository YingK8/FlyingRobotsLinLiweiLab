"""The bench session as plain-English steps; the order and the "why" are in README.md.

Each method only orders existing functions (controller/calib, LLM/experiment*.py) and keeps
the state between steps. Only `run` and `drive_without_cameras` energise the coils.
"""
from pathlib import Path
from uuid import uuid4

from controller.calib.calibrate import BOARDS

PROJECT = Path(__file__).resolve().parents[1]
PATTERNS = PROJECT / "LLM/patterns"
FIXTURE_LAYOUT = PROJECT / "LLM/analyse/fixture_layout.json"
#: The printed 9x6 letter board at its caliper-measured pitch, 16.52 mm (marker scales to
#: 12.39 mm). The value setup_002 was calibrated with; re-measure if you reprint.
BOARD = BOARDS["9x6_letter"].with_square_mm(16.52)


def pattern_path(pattern):
    """A pattern name from LLM/patterns, or a path to a pattern JSON."""
    path = Path(pattern)
    return path if path.suffix == ".json" else PATTERNS / f"{pattern}.json"


class Bench:
    def __init__(self, cameras=(1, 0), serial_port=None, *, width=1280, height=800, fps=120,
                 upside_down=True, backend="auto", codec="auto", board=BOARD,
                 runs=PROJECT / "LLM/runs"):
        from LLM.record.experiment import new_session
        self.camera = dict(indices=tuple(cameras), width=width, height=height, fps=fps,
                           rotate180=upside_down, backend=backend, codec=codec)
        self.serial_port, self.board, self.runs = serial_port, board, Path(runs)
        self.session = new_session(self.runs / "sessions", self.camera)
        self.camera = self.session["camera"]
        self.folder = Path(self.session["directory"])
        self.calibration = self.verification = self.world = None
        self.reference_pair = self.trial = self.last_run = None
        self.leds = (None, None)
        self._calibration_bag = self._views = self._intrinsics = None
        print("Session:", self.folder)

    # ---- cameras ----------------------------------------------------------------------
    def preview_cameras(self):
        """Live view: cover one lens to confirm A/B, set focus. Don't touch lenses after."""
        from LLM.record.panel import run_panel
        return run_panel(self.session, output_root=self.folder / "preview_takes", max_seconds=10)

    # ---- calibration ------------------------------------------------------------------
    def calibrate(self):
        """Full ChArUco calibration: intrinsics of A and B, then the A->B extrinsics."""
        self.collect_charuco_frames()
        self.calibrate_intrinsics()
        return self.calibrate_extrinsics()

    def collect_charuco_frames(self, folder=None):
        """Auto-shutter capture of still board poses seen by both cameras (q to finish)."""
        if (self.camera["width"], self.camera["height"]) != (1280, 800):
            raise ValueError("Full calibration captures at 1280x800; build the bench in that mode")
        bag = Path(folder) if folder else self.folder / f"calibration_{uuid4().hex[:8]}" / "images"
        self._capture(bag)
        self._calibration_bag = bag
        return bag

    def calibrate_intrinsics(self, bag=None):
        """K and distortion for each camera, from the collected frames."""
        from controller.calib.calibrate import calibrate_intrinsics, load_views
        bag = Path(bag) if bag else self._calibration_bag
        if bag is None:
            raise ValueError("collect_charuco_frames() first, or pass an existing bag")
        self._calibration_bag = bag
        views_a, views_b, size = load_views(self.board, bag)
        self._views = (views_a, views_b, size)
        self._intrinsics = {tag: calibrate_intrinsics(self.board, views, size, tag)
                            for tag, views in (("A", views_a), ("B", views_b))}
        return {tag: info for tag, (_, _, info) in self._intrinsics.items()}

    def calibrate_extrinsics(self):
        """A->B pose with intrinsics held fixed; judged, then saved as stereo_rig.json."""
        from controller.calib.calibrate import (calibrate_extrinsics, judge_calibration,
                                                save_calibration)
        from controller.calib.capture import read_meta
        if self._intrinsics is None:
            raise ValueError("calibrate_intrinsics() first")
        (K_a, dist_a, intr_a), (K_b, dist_b, intr_b) = self._intrinsics["A"], self._intrinsics["B"]
        views_a, views_b, size = self._views
        T_ba, pairs, seed_T, spread, info = calibrate_extrinsics(
            self.board, views_a, views_b, K_a, dist_a, K_b, dist_b, size)
        result = judge_calibration(pairs, size, K_a, dist_a, intr_a, K_b, dist_b, intr_b, T_ba,
                                   seed_T, spread, info,
                                   read_meta(self._calibration_bag).get("elp_ids", []))
        folder = self._calibration_bag.parent
        rig = folder / "stereo_rig.json"
        meta = {"camera_indices": list(self.camera["indices"]), "rotate180": self.camera["rotate180"]}
        if not save_calibration(result, self.board, self._calibration_bag, rig, folder / "diagnostics", meta):
            raise RuntimeError(f"Calibration FAILED; read {folder / 'diagnostics'}. Fix the capture, not the limit.")
        self.use_calibration(rig)
        return rig

    def use_calibration(self, rig):
        """Adopt an existing stereo_rig.json instead of calibrating again."""
        self.calibration, self.verification, self.world = Path(rig), None, None
        print("Calibration to verify:", self.calibration)

    def verify_calibration(self):
        """Fresh still board poses at the final rig position; only the board pose is fitted."""
        from LLM.analyse.rig import verify_bag
        if self.calibration is None:
            raise ValueError("calibrate() or use_calibration(path) first")
        self.verification = self.world = None
        bag = self.folder / f"verify_{uuid4().hex[:8]}" / "images"
        self._capture(bag)
        report = verify_bag(bag, self.calibration, self.camera, session_id=self.session["session_id"])
        if not report["passed"]:
            raise RuntimeError("Verification FAILED; this calibration cannot be used for a measured run")
        self.verification = report
        return report

    def _capture(self, bag):
        from controller.calib.capture import capture
        c = self.camera
        capture(bag, indices=c["indices"], spec=self.board, width=c["width"], height=c["height"],
                rotate180=c["rotate180"], backend=c["backend"])

    # ---- scene ------------------------------------------------------------------------
    def photograph_reference(self):
        """Save one still pair with the board removed: LED and plumb line in both views."""
        from LLM.record.panel import run_panel
        self.world = None
        result = run_panel(self.session, output_root=self.folder / "reference_takes", max_seconds=10)
        if result["reference_pair"] is None:
            raise ValueError("Click 'Save static reference pair' before closing the preview")
        self.reference_pair = Path(result["reference_pair"])
        return self.reference_pair

    def locate_fixture(self, plumb, layout=FIXTURE_LAYOUT):
        """Optional: AprilTags + plumb line -> world origin and gravity (needs a verified rig)."""
        from LLM.record.io import write_json
        from LLM.analyse.rig import localize_fixture
        if self.reference_pair is None:
            raise ValueError("photograph_reference() first")
        self.world = localize_fixture(self.reference_pair, self.calibration, layout, plumb,
                                      self.camera, session_id=self.session["session_id"])
        write_json(self.folder / "world_frame.json", self.world)
        return self.world

    # ---- recording --------------------------------------------------------------------
    def record(self, seconds=10):
        """Camera-only trial recording, then a frame-by-frame audit of the files."""
        from LLM.record.experiment import audit_take
        from LLM.record.panel import run_panel
        self.trial = None
        result = run_panel(self.session, output_root=self.folder / "trial_takes", max_seconds=seconds)
        if result["take"] is None:
            raise ValueError("Nothing recorded; click Record")
        self.trial = audit_take(result["take"], self.session)
        return self.trial

    def show_pattern(self, pattern):
        """Plot a pattern's planned frequency before flying it."""
        from LLM.record.io import read_json
        from LLM.record.pattern import sample_pattern
        path = pattern_path(pattern)
        return sample_pattern(read_json(path)).plot(x="t_s", y="frequency_hz", grid=True, title=path.stem)

    def check_serial(self):
        """Reset the ESP and wait for SYNC_READY; sends no drive. True on PASS."""
        from LLM.record.serial_check import check
        return check(self._port(), self.runs / "serial_checks") == 0

    def drive_without_cameras(self, pattern):
        """Drive a pattern with no video (power/current debugging). ENERGISES THE COILS."""
        from LLM.record.drive_only import run_frequency_pattern
        return run_frequency_pattern(pattern_path(pattern), self._port(), output_root=self.runs / "drive_only")

    def run(self, pattern, *, note, warnings_reviewed="", led_a=None, led_b=None, extra_seconds=30):
        """Record -> START (LED + drive) -> automatic stop -> import. ENERGISES THE COILS.

        Measured run when the calibration is verified and the fixture located; otherwise
        a sync check (video + CS + LED, no geometry). LED ROIs [x,y,w,h] are optional:
        without them they are found from the START pulse during analyse().
        """
        from LLM.record.experiment import finish_take, prepare_run, prepare_sync_run
        from LLM.record.panel import run_panel
        from LLM.record.io import read_json, snapshot
        from controller.control import ramp
        port = self._port()
        if self.trial is None:
            raise ValueError("record() a trial first")
        if self.verification and self.world:
            run = prepare_run(self.runs, self.session, self.calibration, self.verification, self.world,
                              self.trial, note=note, warning_review=warnings_reviewed)
        else:
            if self.reference_pair is None:
                raise ValueError("photograph_reference() first")
            print("No verified calibration + fixture: recording a sync check (no geometry).")
            run = prepare_sync_run(self.runs, self.session, self.trial, self.reference_pair,
                                   note=note, warning_review=warnings_reviewed)
        self.last_run, self.leds = run, (led_a, led_b)
        snapshot(pattern_path(pattern), run / "pattern.json")
        segments = read_json(run / "pattern.json")["frequency_segments"]
        result = run_panel(self.session, output_root=run / "raw/captures", run=run,
                           pattern_path=run / "pattern.json", serial_port=port, roi_a=led_a, roi_b=led_b,
                           max_seconds=ramp.duration_s(segments) + extra_seconds)
        if result["take"] is None:
            raise ValueError("Preview only, nothing recorded; call run() again for a new run")
        report = finish_take(result["take"], self.session, run)
        print("Run:", run)
        return report

    def analyse(self, annotations=None):
        """LED START/STOP alignment -> export/input_curves.png; pose/takeoff with annotations."""
        from LLM.record.experiment import postprocess
        if self.last_run is None:
            raise ValueError("run() first")
        return postprocess(self.last_run, *self.leds, annotations=annotations)

    def _port(self):
        if not self.serial_port:
            raise ValueError("Give Bench(serial_port=...) the ESP port (list: serial.tools.list_ports)")
        return self.serial_port
