"""The bench session as plain-English steps; the order and the "why" are in README.md.

Each method is one bench step; record_experiment.ipynb calls them in order. Only `run`
and `drive_without_cameras` energise the coils.
"""
from pathlib import Path
from uuid import uuid4

from controller.calib.calibrate import BOARDS

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE_LAYOUT = PROJECT / "LLM/analyse/fixture_layout.json"   # old paper AprilTags
HOLDER_LAYOUT = PROJECT / "LLM/analyse/holder_layout.json"     # v3 CAD checker marks (default)
#: The printed 9x6 letter board at its caliper-measured pitch, 16.52 mm (marker scales to
#: 12.39 mm). The value setup_002 was calibrated with; re-measure if you reprint.
BOARD = BOARDS["9x6_letter"].with_square_mm(16.52)


class Bench:
    def __init__(self, cameras=(1, 0), serial_port=None, *, width=640, height=400, fps=210,
                 exposure_ms=2.0, gain=100, upside_down=True, backend="auto", codec="auto",
                 board=BOARD, runs=PROJECT / "LLM/runs", resume_session=None):
        """One set of camera settings for every step that opens the cameras: mode, fps,
        exposure and gain. A numeric exposure selects manual UVC controls;
        exposure_ms=None explicitly restores camera auto exposure at every capture
        entry point (operator request, 2026-11-04). upside_down: one bool or (A, B)."""
        if exposure_ms is not None and not 0 < exposure_ms < 1e3 / fps:
            raise ValueError(f"exposure {exposure_ms} ms must be under one frame "
                             f"({1e3 / fps:.2f} ms at {fps} fps)")
        self.serial_port, self.board, self.runs = serial_port, board, Path(runs)
        camera = dict(indices=tuple(cameras), width=width, height=height, fps=fps,
                               rotate180=upside_down, backend=backend, codec=codec,
                               exposure_s=None if exposure_ms is None else exposure_ms / 1e3,
                               auto_exposure=exposure_ms is None, gain=gain)
        if resume_session is None:
            self._new_session(camera)
        else:
            from LLM.analyse.rig import camera_key, same_rotation
            from LLM.record.io import read_json
            folder = (self.runs / "sessions" / resume_session).resolve()
            session = read_json(folder / "session.json")
            if folder != Path(session["directory"]) or folder.name != session["session_id"]:
                raise ValueError("Saved session ID or directory mismatch")
            expected = camera_key(camera)
            if same_rotation(expected["rotate180"], session["camera"]["rotate180"]):
                expected["rotate180"] = session["camera"]["rotate180"]
            if expected != session["camera"]:
                raise ValueError("Camera settings differ from the saved session; choose a new session")
            self._load_session(session)

    def _new_session(self, camera):
        """Sessions are immutable: a new camera setting is a new session, and everything
        verified under the old one is void."""
        from LLM.record.experiment import new_session
        self._load_session(new_session(self.runs / "sessions", camera))

    def _load_session(self, session):
        self.session = session
        self.camera = self.session["camera"]
        self.folder = Path(self.session["directory"])
        self.calibration = self.verification = self.world = None
        self.reference_pair = self.trial = self.last_run = None
        self.leds = (None, None)
        self._calibration_bag = self._views = self._intrinsics = None
        print("Session:", self.folder)

    # ---- cameras ----------------------------------------------------------------------
    def preview_cameras(self, rotate180=None, scale=1.0):
        """Live A | B inline in the notebook at the session's own mode, fps, exposure and
        gain, nothing recorded: confirm A/B, aim, focus, light. Shows each camera's grab
        rate (average, and best 0.5 s as max) and brightness.

        rotate180: (A, B) 180-degree flips, or one bool; None keeps the session's. A change
            starts a NEW session (sessions are immutable), so calibrate in the new
            orientation or load a calibration shot in it.
        scale: display size only. Interrupt the kernel to stop. Rotation in software is
        180 deg only; any other angle is the camera itself."""
        from controller.camera import uvc
        from controller.camera.sources import rotations
        if rotate180 is not None and rotations(rotate180, 2) != rotations(self.camera["rotate180"], 2):
            self._new_session(dict(self.camera, rotate180=rotations(rotate180, 2)))
            print(f"rotation (A, B) = {tuple(rotations(rotate180, 2))}: new session, "
                  f"calibration cleared")
        c = self.camera
        uvc.meter(c["exposure_s"], c["gain"], width=c["width"], height=c["height"],
                  fps=c["fps"], rotate180=c["rotate180"], indices=c["indices"],
                  backend=c["backend"], scale=scale, auto_exposure=c.get("auto_exposure", False))

    # ---- calibration ------------------------------------------------------------------
    def collect_charuco_frames(self, folder=None):
        """Auto-shutter capture of still board poses seen by both cameras (q to finish)."""
        # Calibrated in the bench's own mode, the one it records in: intrinsics are never
        # rescaled, and fixed_rig refuses a calibration from another mode.
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
            raise ValueError("calibrate_extrinsics() or use_calibration(path) first")
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
                rotate180=c["rotate180"], backend=c["backend"], fps=c["fps"],
                exposure_s=c["exposure_s"], gain=c["gain"], auto_exposure=c.get("auto_exposure", False))

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

    def find_holder_marks(self):
        """Detect the checker centres in the reference pair (no hardware).

        First call writes an empty `holder_regions.json` next to the photos; fill in one
        [x, y, w, h] per visible CAD name and call again to name them. See analyse/HOLDER_MARKERS.md."""
        from LLM.analyse.holder import inspect_holder, make_holder_regions
        if self.reference_pair is None:
            raise ValueError("photograph_reference() first")
        regions = self.reference_pair / "holder_regions.json"
        if not regions.exists():
            make_holder_regions(self.reference_pair, regions)
        report = inspect_holder(self.reference_pair, regions_path=regions)
        print("Edit, then call again:", regions)
        return {tag: report["cameras"][tag]["landmarks"] for tag in "AB"}

    def locate_fixture(self, plumb, layout=None):
        """Optional: named holder marks (or AprilTags) + plumb line -> world origin and gravity.

        `layout` defaults to a per-session copy of holder_layout.json: measure the bearing,
        fill `bearing_center_fixture_mm`, then set `survey_confirmed=true`. Pass
        FIXTURE_LAYOUT for the old AprilTag sheets."""
        import shutil
        from LLM.record.io import read_json, write_json
        from LLM.analyse.rig import localize_fixture
        from LLM.analyse.holder_geometry import localize_holder
        if self.reference_pair is None:
            raise ValueError("photograph_reference() first")
        if layout is None:
            layout = self.folder / "holder_layout.json"
            if not layout.exists():
                shutil.copyfile(HOLDER_LAYOUT, layout)
        args = dict(session_id=self.session["session_id"])
        if read_json(layout).get("pattern_type") == "checker_quadrants":
            self.world = localize_holder(self.reference_pair, self.calibration, layout,
                                         self.reference_pair / "holder_regions.json", plumb, self.camera, **args)
        else:
            self.world = localize_fixture(self.reference_pair, self.calibration, layout, plumb, self.camera, **args)
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

    def show_ramp(self, ramp):
        """Plot a ramp's planned frequency before driving it."""
        from LLM.record.ramps import load_ramp, planned_frequency, ramp_path
        import matplotlib.pyplot as plt
        path = ramp_path(ramp)
        ax = planned_frequency(load_ramp(path)).plot(x="t_s", y="frequency_hz", grid=True, title=path.stem)
        plt.close(ax.figure)
        return ax.figure

    def check_serial(self):
        """Reset the ESP and wait for SYNC_READY; sends no drive. True on PASS."""
        from LLM.record.serial_check import check
        return check(self._port(), self.runs / "serial_checks") == 0

    def drive_without_cameras(self, ramp, *, confirm_start=False):
        """ENERGISES THE COILS, without video. confirm_start waits for START after reset."""
        from LLM.record.drive_only import drive_ramp
        from LLM.record.ramps import ramp_path
        return drive_ramp(ramp_path(ramp), self._port(), output_root=self.runs / "drive_only",
                          confirm_start=confirm_start)

    def run(self, ramp, *, note, warnings_reviewed="", led_a=None, led_b=None, extra_seconds=30,
            auto_start=False):
        """Record -> START (LED + drive) -> automatic stop -> import. ENERGISES THE COILS.

        `ramp` is a name in LLM/ramps (or a path). Call analyse() afterwards.

        Measured run when the calibration is verified and the fixture located; otherwise
        a sync check (video + CS + LED, no geometry). LED ROIs [x,y,w,h] are optional:
        without them they are found from the START pulse during analyse().
        """
        from LLM.record.experiment import finish_take, prepare_run, prepare_sync_run
        from LLM.record.panel import run_panel
        from LLM.record.io import snapshot
        from LLM.record.ramps import load_ramp, ramp_path
        from controller.control.ramp import duration_s, label
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
        snapshot(ramp_path(ramp), run / "ramp.json")
        segments = load_ramp(run / "ramp.json")
        print("Ramp:", label(segments))
        result = run_panel(self.session, output_root=run / "raw/captures", run=run,
                           ramp_path=run / "ramp.json", serial_port=port, roi_a=led_a, roi_b=led_b,
                           max_seconds=duration_s(segments) + extra_seconds, auto_start=auto_start)
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
