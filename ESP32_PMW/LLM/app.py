"""The LLM bench in one viser page: a button per step, greyed until its inputs exist.

Every step is an existing function; this file only orders them and draws. Buttons only
set events and the main thread runs the step, as in `record/panel.py`: signals and
serial cleanup need the main thread. The camera stays open across steps and is lent to
`capture` (calibration) and `run_panel` (the run). During a step the full camera
overlay replaces the 3D scene (`scene.set_background_image`); tick "3D view" to see
the scene. The platform and z are fitted ONCE, before a run, and never updated while
the coils are driven (open loop).

    .venv/bin/python -m LLM.app            # or LLM/viser_app.ipynb
    .venv/bin/python -m LLM.app --self-check
"""
import json
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from LLM.bench import HOLDER_LAYOUT, PROJECT, Bench
from LLM.record.io import read_json, sha256, write_json

STILL_S = 10.0       # operator request: 10 s of still footage for the platform fit
HOLDOUT_EVERY = 5    # every 5th calibration pair is held out for verify_bag (needs >= 5)
# The drone sits on the platform's bearing, on the mounting-boss axis -- NOT on the rope,
# which is a separate plumb line (2026-11-05 recordings: the moving robot is mid-array).
# Its ROI is this cylinder about that axis, projected: guesses, not measurements.
ROBOT_RADIUS_MM = 40.0          # robot (~20 mm, render.load_mesh) plus a wobble margin
ROBOT_Z_MM = (0.0, 150.0)       # CAD z span: bearing unsurveyed, plus the flight volume above
ROPE_DRAW_MM = (-100.0, 500.0)   # drawn extent of the rope along world z; display only
FRUSTUM_MM = 60.0    # drawn frustum depth; display only
VIEW_FROM_MM = (450.0, -450.0, 350.0)   # where the viewer starts, looking at the origin
STEP_MESH = PROJECT / "LLM/analyse/coil_base.step"
LABEL_PRESETS = PROJECT / "LLM/labels.json"   # the label bubbles; edit the file, restart the app
TICK = "✓ "                                     # marks a picked bubble
# Operator 2026-11-15: three blocks and a collapsed review block, as few buttons as possible.
STEPS = ("connect", "calibrate", "setup", "run", "discard", "keep", "upload", "playback", "pose")
LABELS = {"connect": "Connect cameras",
          "calibrate": "Calibrate (start recording the board)",
          "setup": "Detect setup (LED, then platform if calibrated)",
          "run": "Record + run", "discard": "Reset",
          "keep": "Save local (keep run, ready next)",
          "upload": "Save to Drive (note + upload; does not keep)",
          "playback": "Playback / pause", "pose": "Preprocess drone pose (segmentation)"}
CAL_STOP_LABEL = "Calibrate: stop recording + solve"
ON_CLICK = ("keep", "upload")   # run on the click itself (files only), never greyed by a long step
NO_CALIBRATION = "(none)"
# B saw only 5 of its targets at OpenCV's default 500 Shi-Tomasi corners (a bright room
# outranks them); 3000 gave the same pose to 0.1 mm, +1 target, ~3 s (2026-11-05).
MAX_CORNERS = 3000
# `led=code` is a start bit + 7 slots + a dark slot at scene.CODE_BIT_S (0.72 s); the
# rest is serial + camera latency margin. A guess, not measured.
LED_TEST_S = 1.0


def _led_roi(found, shape, pad_px=4):
    """`scene.locate_led`'s (cx, cy, r) as the [x, y, w, h] the run panel and
    `synchronize_start` take, clipped to the frame; None stays None."""
    if found is None:
        return None
    cx, cy, r = found
    h, w = shape[:2]
    x0, y0 = max(0, int(cx - r - pad_px)), max(0, int(cy - r - pad_px))
    x1, y1 = min(w, int(np.ceil(cx + r + pad_px))), min(h, int(np.ceil(cy + r + pad_px)))
    return [x0, y0, x1 - x0, y1 - y0]


def _rgb(frames, width):
    """A|B side by side, each `width` px wide, as RGB for viser."""
    out = [cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1]))) for f in frames]
    out = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in out]
    return np.hstack(out)


def _wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (w, x, y, z)


def _z_to(n):
    """A rotation taking +z onto unit `n` (roll is unobserved, so arbitrary)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    x = np.cross(n, [1.0, 0, 0] if abs(n[0]) < .9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(n, x), n]


class ViserSink:
    """`sources.Sink` stand-in for `capture`: its A|B view split into the two camera images
    in the side panel; the Calibrate button's second press = q."""
    inline = False

    def __init__(self, app, hz=10.0):
        self.app, self.period, self._last = app, 1.0 / hz, 0.0

    def open(self):
        return self

    def due(self):
        return time.monotonic() - self._last >= self.period

    def show(self, view):
        if view is None or not self.due():
            return -1
        self._last = time.monotonic()
        half = view.shape[1] // 2              # capture._overlay: two equal panels, A then B
        self.app.show_pair([view[:, :half], view[:, half:]])
        return ord("q") if self.app.events["done"].is_set() else -1

    def close(self):
        self.app.events["done"].clear()


class App:
    def __init__(self, port=None, cameras=(1, 0), mode=(1280, 800, 120), *, exposure_ms=None,
                 gain=100, upside_down=(False, False), calibration_exposure_ms=10.0,
                 session=None, calibration="latest", duty_dist=None, viser_port=8090,
                 runs=PROJECT / "LLM/runs"):
        """calibration: "latest" (newest saved rig at this resolution), a stereo_rig.json
        path, or None. A loaded rig is unverified, so runs are sync checks until
        Calibrate verifies a new one on held-out frames."""
        import viser
        w, h, fps = mode
        self.bench = Bench(cameras, port, width=w, height=h, fps=fps, exposure_ms=exposure_ms,
                           gain=gain, upside_down=upside_down, runs=runs, resume_session=session)
        self.src, self.run_done, self.player = None, False, None
        # Calibration's own exposure (operator 2026-11-16: the board scene was too dark at 3 ms
        # and calibration needs no frame rate). Over one frame period, so the ELP slows its
        # frame rate while it is set; the session exposure is restored after the recording.
        self.calibration_exposure_s = None if calibration_exposure_ms is None else calibration_exposure_ms / 1e3
        # After a take: audit + import + LED sync on a worker thread (files and decoding
        # only, never serial or cameras), so the save buttons / Reset / the note are usable at once.
        # Everything that moves or reads the run folder waits for it first.
        self.analysis, self.analysis_done, self.keep_when_analysed = None, True, False
        self._analysis_lock = threading.Lock()
        self.fixture = self.robot_roi = self.rope = self.annot = None
        self.frusta = []
        self.events = {k: threading.Event() for k in (*STEPS, "done", "close")}
        s = self.server = viser.ViserServer(host="127.0.0.1", port=viser_port, label="LLM bench")
        s.gui.configure_theme(control_width="large")
        # Millimetres: viser's default camera (3, 3, 3) would sit inside the platform.
        s.scene.set_up_direction("+z")
        s.initial_camera.position, s.initial_camera.look_at = VIEW_FROM_MM, (0.0, 0.0, 40.0)
        # near < 1: viser draws the background image (every camera overlay, `big`) on a
        # plane 1 scene unit in front of the camera (client App.tsx BackgroundImage); at
        # near=5 that plane was clipped and Calibrate's preview never appeared.
        s.initial_camera.near, s.initial_camera.far = 0.5, 5000.0
        self.bench.duty_dist.update(duty_dist or {})   # per-run coil duty randomisation
        # Ends a run early by the panel's Stop path (firmware `stop`, wait OFF, save the
        # take). E-STOP removed (operator 2026-11-15): when idle the kills are
        # controller/control/safe_off.py, the GPIO14 button and the supply.
        self.finish = threading.Event()
        self.calibrating = False                # Calibrate's second press stops the recording
        # record/upload.py on a worker; set once that upload no longer reads the run folder
        self.upload_staged = threading.Event()
        self.upload_staged.set()
        # Every status message, timestamped and tagged with the run, so it can be read back
        # (operator 2026-11-16). The 5 Hz run-panel line is not logged: it would flood it.
        self.log_path = self.bench.folder / "app_log.jsonl"
        self._log_lock = threading.Lock()
        self._cal_pick = None                   # dropdown choice, loaded on the main thread
        self.status = s.gui.add_markdown("**Connect the cameras.**")
        self.buttons = {}

        def button(key, on_click=None):
            self.buttons[key] = b = s.gui.add_button(LABELS[key], disabled=True)
            b.on_click(on_click or (lambda _: self.events[key].set()))

        with s.gui.add_folder("1 · Setup"):
            button("connect")
            self.cal_choice = s.gui.add_dropdown("Calibration", (NO_CALIBRATION,), initial_value=NO_CALIBRATION)
            self.cal_choice.on_update(lambda _: setattr(self, "_cal_pick", self.cal_choice.value))
            button("calibrate", lambda _: self.events["done" if self.calibrating else "calibrate"].set())
            button("setup")
        # Camera views live in the side panel only, one per row at full panel width, under
        # Setup (operator 2026-11-15); the main display is always the 3D scene.
        self.cams = [s.gui.add_image(np.zeros((400, 640, 3), np.uint8), label=f"Camera {t}", format="jpeg")
                     for t in "AB"]
        with s.gui.add_folder("2 · Run"):
            self.note = s.gui.add_text("Run note (saved by Save local / Save to Drive)", "", multiline=True)
            self._labels_gui(s)
            button("run")
            self.stop_rec = s.gui.add_button("Stop recording early (coils off, saves)", disabled=True, color="red")
            self.stop_rec.on_click(lambda _: self.finish.set())
            button("discard")
            button("keep", lambda _: self.do_keep())
            button("upload", lambda _: self._start_upload())
            self.curves = s.gui.add_image(np.zeros((10, 10, 3), np.uint8), label="Input curves", visible=False)
        with s.gui.add_folder("3 · Run parameters"):
            self._design_gui(s)
        with s.gui.add_folder("4 · Review last run", expand_by_default=False):
            button("playback", lambda _: self.do_play() if self.player is not None else self.events["playback"].set())
            button("pose")
            self.slider = s.gui.add_slider("Playback frame", 0, 1, 1, 0, disabled=True)
            self.playback_img = s.gui.add_image(np.zeros((10, 10, 3), np.uint8), label="Playback A | B",
                                                format="jpeg", visible=False)
        print(f"Open http://127.0.0.1:{s.get_port()}")
        self._refresh_calibrations(calibration)
        self._draw_scene()

    def _design_gui(self, s):
        """Run parameters: per parameter a mode (fixed / random / step), value, step and
        range; `record/design.py` turns them into each run's ramp and duty. The defaults
        are all `fixed` (deterministic); Load parameters brings a randomisation setup."""
        from LLM.record.design import MODES, PARAMS
        self.design = {}
        load = s.gui.add_upload_button("Load parameters (.json: design or a run's run_parameters.json)",
                                       mime_type="application/json")
        load.on_upload(lambda _: self._load_params(load.value.content, load.value.name))
        # Frequency-scheduled per-coil trim (LLM/analyse/coil_trim.py): off unless chosen.
        from LLM.analyse.coil_trim import TRIMS
        trims = sorted((p.name for p in TRIMS.glob("coil_trim_*.json")), reverse=True)
        self.trim = s.gui.add_dropdown("Coil trim (per-frequency duty; needs duty 100 %)", ("off", *trims),
                                       initial_value="off")
        self.duty_scale = s.gui.add_number("Duty random: mean drop from 100 (%)",
                                           self.bench.duty_dist["scale_pct"], min=1.0, max=70.0, step=1.0)
        for name, p in PARAMS.items():
            with s.gui.add_folder(p["label"], expand_by_default=False):
                self.design[name] = {
                    "mode": s.gui.add_dropdown("mode", MODES, initial_value=p["mode"]),
                    "value": s.gui.add_number("value (fixed / step start)", p["value"], step=0.1),
                    "step": s.gui.add_number("step after kept run (+/-)", p["step"], step=0.1),
                    "lo": s.gui.add_number("min (random / bound)", p["lo"], step=0.1),
                    "hi": s.gui.add_number("max (random / bound)", p["hi"], step=0.1)}

    def _load_params(self, content, name="parameters"):
        """A design file ({"parameters": {name: {mode, value, step, lo, hi}}, "duty_scale_pct"})
        or a run's raw/run_parameters.json (its "design") into the widgets -- all or nothing:
        an unknown name, a bad mode or a non-number refuses the file and changes nothing."""
        from LLM.record.design import MODES
        try:
            d = json.loads(content)
            d = d.get("design") or d
            params, new = d["parameters"], {}
            for pname, spec in params.items():
                if pname not in self.design:
                    raise ValueError(f"unknown parameter {pname!r}")
                if "mode" in spec and spec["mode"] not in MODES:
                    raise ValueError(f"{pname}: mode {spec['mode']!r} not in {MODES}")
                new[pname] = {k: (spec[k] if k == "mode" else float(spec[k]))
                              for k in ("mode", "value", "step", "lo", "hi") if k in spec}
            scale = d.get("duty_scale_pct")
            scale = None if scale is None else float(scale)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self.say(f"**Load parameters refused** (`{name}`): {exc}. Nothing changed.")
            return
        for pname, spec in new.items():
            for k, v in spec.items():
                self.design[pname][k].value = v
        if scale is not None:
            self.duty_scale.value = scale
        varied = [f"{n} ({sp['mode'].value})" for n, sp in self.design.items() if sp["mode"].value != "fixed"]
        self.say(f"**Loaded parameters** from `{name}`: {len(new)} set. "
                 + ("Varied each run: " + ", ".join(varied) + "." if varied else "All fixed: deterministic."))

    def _specs(self):
        return {name: {k: (w.value if k == "mode" else float(w.value)) for k, w in widgets.items()}
                for name, widgets in self.design.items()}

    # ---- plumbing ---------------------------------------------------------------------
    def say(self, text):
        self.status.content = text
        print(text.replace("**", ""), flush=True)
        self.log("status", text)

    def log(self, kind, text):
        run = self.bench.last_run
        line = json.dumps({"t": datetime.now().isoformat(timespec="milliseconds"), "kind": kind,
                           "run": Path(run).name if run is not None else None, "text": text})
        with self._log_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_exc(self):
        traceback.print_exc()
        self.log("error", traceback.format_exc())

    def _labels_gui(self, s):
        """One row of bubbles per category of LLM/labels.json; a click toggles the label."""
        cats = read_json(LABEL_PRESETS)["categories"]
        self.labels = {cat: [] for cat in cats}
        self.label_groups = {}
        with s.gui.add_folder("Labels (click to toggle; saved with the note)"):
            for cat, opts in cats.items():
                group = s.gui.add_button_group(cat, tuple(opts))
                group.on_click(lambda e, cat=cat: self.toggle_label(cat, e.target.value))
                self.label_groups[cat] = (group, tuple(opts))

    def toggle_label(self, cat, clicked):
        name, picks = clicked.removeprefix(TICK), self.labels[cat]
        if name in picks:
            picks.remove(name)
        else:
            picks.append(name)
        group, opts = self.label_groups[cat]
        group.options = tuple(TICK + o if o in picks else o for o in opts)

    def clear_labels(self):
        for cat, (group, opts) in self.label_groups.items():
            self.labels[cat].clear()
            group.options = opts

    def show_pair(self, views, width=640):
        """Camera A and B into their side-panel images (BGR or grey, any size)."""
        for handle, view in zip(self.cams, views):
            handle.image = _rgb([view], width)

    def states(self):
        """Which steps can run, from in-memory state only (it is polled at 5 Hz)."""
        b, cams, done = self.bench, self.src is not None, self.run_done
        ran = b.last_run is not None and (Path(b.last_run) / "raw/camera/meta.json").exists()
        uploading = not self.upload_staged.is_set()     # only while staging: transfers overlap
        return {"connect": not cams, "calibrate": cams, "setup": cams and not done,
                "run": cams and bool(b.serial_port) and not done,
                "discard": done, "keep": done, "upload": b.last_run is not None and not uploading,
                "playback": ran or self.player is not None,
                "pose": ran and self.player is None and self.robot_roi is not None}

    def gate(self, busy=False):
        on = self.states()
        for key in STEPS:
            # Save local stays live through a long step; Save to Drive does not, since a
            # run in progress would copy a half-written folder.
            self.buttons[key].disabled = (busy and key not in ("keep",)) or not on[key]

    def frame(self):
        item = self.src.read()
        if item is None:
            raise OSError("Camera stream ended")
        return item[0], list(item[1])

    def tick(self):
        ui = time.monotonic() - getattr(self, "_last_ui", 0) >= .2
        if self.player is not None:
            self._advance()
        if self.src is not None:
            self.preview()
        elif self.player is None:
            time.sleep(.05)
        if ui:
            self._last_ui = time.monotonic()
            self.gate()

    def preview(self, frames=None, light=False):
        """One live-preview update (at most 5 Hz); reads a pair if none is given. Used by
        `tick` and as the pump while a step blocks, so the preview never pauses. Never
        touches the buttons: a step in progress keeps them greyed.

        ``light``: plain frames at 1 Hz, no overlay and no frustum images -- for use while
        a take records. The full preview cost the encoder enough CPU that its 64-pair
        queue filled after 4.7 s and the take was refused (2026-11-05, 1 drop at 120 fps).
        """
        if frames is None:
            _, frames = self.frame()
        if time.monotonic() - getattr(self, "_shown", 0) < (1.0 if light else .2):
            return
        self._shown = time.monotonic()
        if light:
            self.show_pair(frames, 320)
            return
        marked = self.annot is not None or self.bench.leds[0] is not None
        self.show_pair(self._annotate(frames) if marked else frames)
        for frustum, f in zip(self.frusta, frames):
            frustum.image = _rgb([f], 160)

    def loop(self):
        try:
            while not self.events["close"].is_set():
                if self._cal_pick is not None:
                    pick, self._cal_pick = self._cal_pick, None
                    try:
                        self._load_calibration(pick)
                    except Exception as exc:
                        self.log_exc()
                        self.say(f"**Loading calibration failed:** {exc}")
                for key in STEPS:
                    if self.events[key].is_set():
                        self.events[key].clear()
                        if not self.states()[key]:
                            continue
                        self.gate(busy=True)
                        try:
                            getattr(self, "do_" + key)()
                        except Exception as exc:
                            self.log_exc()
                            self.say(f"**{LABELS[key]} failed:** {exc}")
                try:
                    self.tick()
                except OSError as exc:      # a camera dropped out: keep the app, drop the source
                    if self.src is not None:
                        self.src.close()
                    self.src, self.frusta = None, []
                    self.say(f"**Camera lost** ({exc}). Check the USB, then Connect cameras again.")
        except KeyboardInterrupt:
            pass
        finally:
            self._wait_for_analysis()
            if self.player is not None:
                for d in self.player["video"]:
                    d.release()
            if self.src is not None:
                self.src.close()
            self.server.stop()

    def _saved_rigs(self):
        size, out = [self.bench.camera["width"], self.bench.camera["height"]], []
        for p in sorted(self.bench.runs.glob("sessions/*/calibration_*/**/stereo_rig.json"), reverse=True):
            try:
                if read_json(p).get("image_size") == size:
                    out.append(str(p))
            except (OSError, ValueError):   # a half-written or foreign file is skipped, not fatal
                pass
        return out

    def _rig(self):
        from LLM.analyse.rig import fixed_rig
        return fixed_rig(self.bench.calibration, self.bench.camera)[0]

    def _world(self, p, vector=False):
        """camera A -> world (mm); identity until the platform is fitted."""
        w = self.bench.world
        if w is None:
            return np.asarray(p, float)
        R = np.asarray(w["R_camera_from_world"])
        return R.T @ (np.asarray(p, float) - (0 if vector else np.asarray(w["origin_camera_mm"])))

    # ---- steps ------------------------------------------------------------------------
    def do_connect(self):
        from controller.camera import sources, uvc
        c = self.bench.camera
        self.src = sources.open_stereo([f"camera:{i}" for i in c["indices"]], grayscale=True, max_skew_s=None,
                                       **{k: c[k] for k in ("width", "height", "fps", "rotate180", "backend")})
        if c.get("exposure_s") or c.get("auto_exposure"):
            uvc.configure(c.get("exposure_s"), c.get("gain", 100), auto_exposure=c.get("auto_exposure", False))
        _, frames = self.frame()
        self.say("**Cameras connected.** Frame mean/max " + ", ".join(f"{f.mean():.0f}/{f.max()}" for f in frames)
                 + " (healthy ~59/82, max 235). "
                 + ("Detect setup next." if self.bench.calibration else
                    "No calibration loaded: Calibrate, or pick one under Calibration."))

    def _rig_label(self, path):
        try:
            return str(Path(path).resolve().relative_to((self.bench.runs / "sessions").resolve()))
        except ValueError:
            return str(path)

    def _refresh_calibrations(self, select=None):
        """The Calibration dropdown: saved rigs at this resolution, newest first. ``select``:
        "latest" (autoload the newest, if any), a stereo_rig.json path, or None."""
        rigs = self._saved_rigs()
        if select not in (None, "latest") and str(select) not in rigs:
            rigs.insert(0, str(select))
        self._rig_paths = {self._rig_label(r): r for r in rigs}
        self.cal_choice.options = (NO_CALIBRATION, *self._rig_paths)
        pick = (next(iter(self._rig_paths), None) if select == "latest" else
                None if select is None else self._rig_label(select))
        if pick is not None:
            self._load_calibration(pick)

    def _load_calibration(self, label):
        """Adopt the rig ``label`` names ((none) = none). Anything fitted with the previous rig
        -- platform, rope, robot ROI, overlay -- is dropped with it; the scene is redrawn."""
        b = self.bench
        path = None if label == NO_CALIBRATION else self._rig_paths.get(label, label)
        if (path is None and b.calibration is None) or (
                path is not None and b.calibration is not None and Path(path).resolve() == Path(b.calibration).resolve()):
            if self.cal_choice.value != label:  # e.g. Calibrate's new rig: show it, load nothing
                self.cal_choice.value = label
            return                              # already in force (this also absorbs the dropdown echo)
        if path is None:
            b.calibration = b.verification = b.world = None
        else:
            b.use_calibration(path)
            try:
                self._rig()                     # refuses a file from another camera mode
            except Exception:
                b.calibration = None
                raise
        self.fixture = self.robot_roi = self.rope = self.annot = None
        self.server.scene.reset()
        self._draw_scene()
        if self.cal_choice.value != label:
            self.cal_choice.value = label
        self.say(f"**Calibration:** `{label}`" + (" loaded, **unverified**: runs are sync checks until a "
                                                    "Calibrate verifies one." if path else " -- none.")
                 + (" Detect setup next." if path and self.src is not None else ""))

    def do_calibrate(self):
        """One button: record board footage (nothing judged live, theory.md 16.5f); its second
        press stops the recording, then the still frames are selected and solved here."""
        from LLM.analyse.rig import verify_bag
        b = self.bench
        root = b.folder / f"calibration_{uuid4().hex[:8]}"
        self.say("**Recording calibration footage.** Move the board around both cameras and pause "
                 "~1 s at each pose (only still frames are kept). Press **Calibrate: stop recording + "
                 "solve** to finish.")
        self.calibrating = True
        self.buttons["calibrate"].label, self.buttons["calibrate"].disabled = CAL_STOP_LABEL, False
        from controller.camera import uvc
        c, exposure_s = b.camera, self.calibration_exposure_s
        try:
            if exposure_s is not None:
                uvc.apply(exposure_s, c.get("gain", 100))
            n = b.record_calibration(root / "raw", self.src, ViserSink(self), exposure_s=exposure_s)
        finally:
            if exposure_s is not None:   # back to the session's own setting before any run
                uvc.configure(c.get("exposure_s"), c.get("gain", 100), auto_exposure=c.get("auto_exposure", False))
            self.calibrating = False
            self.buttons["calibrate"].label = LABELS["calibrate"]
        self.say(f"**Recorded** {n} burst(s) in `{root / 'raw'}` (kept even if the solve fails). "
                 "**Selecting** still frames...")
        n, m = b.select_calibration(root / "raw", root / "capture", exposure_s=exposure_s)
        self.say(f"**Selected** {n} pair(s) + {m[0]}/{m[1]} solo (A/B). **Calibrating...**")
        # The raw recording and the selected bag stay untouched; fit and held-out are copies.
        pairs = sorted((root / "capture/A").glob("pair_*.png"))
        held = {p.name for p in pairs[HOLDOUT_EVERY - 1::HOLDOUT_EVERY]}
        for name, keep in (("images", lambda f: f.name not in held), ("holdout", lambda f: f.name in held)):
            for tag in "AB":
                (root / name / tag).mkdir(parents=True)
                for f in (root / "capture" / tag).glob("*.png"):
                    if keep(f) and (name == "images" or f.name.startswith("pair_")):
                        shutil.copy2(f, root / name / tag / f.name)
            for f in ("meta.json", "pairs.csv"):
                shutil.copy2(root / "capture" / f, root / name / f)
        b.calibrate_intrinsics(root / "images")
        b.calibrate_extrinsics()
        rms = read_json(b.calibration)["rms_px"]
        if len(held) < 5:
            self._refresh_calibrations(b.calibration)
            self._draw_scene()
            self.say(f"**Calibrated** (stereo rms {rms:.2f} px) but **not verified**: {len(held)} held-out "
                     f"pairs, verification needs 5 (capture >= {5 * HOLDOUT_EVERY} pairs).")
            return
        report = verify_bag(root / "holdout", b.calibration, b.camera, session_id=b.session["session_id"])
        b.verification = report if report["passed"] else None
        self._refresh_calibrations(b.calibration)   # the new rig shows as the dropdown's choice
        self._draw_scene()
        self.say(f"**Calibrated**, stereo rms {rms:.2f} px. Held-out check on {len(held)} pairs: "
                 f"**{'PASS' if report['passed'] else 'FAIL'}**. Held-out frames prove the fit, "
                 "not that the rig stays put afterwards: re-run if a camera or focus moves.")

    def _annotate(self, frames):
        """The last fit drawn on full-resolution frames -- the still means or the live
        preview: grey every checker candidate, yellow the named ones, orange the rope,
        green the robot ROI. Strokes are sized to survive the thumbnail."""
        a, views = self.annot or {}, []
        for k, (tag, f) in enumerate(zip("AB", frames)):
            view = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f.copy()
            for u, v in a.get("cands", [[], []])[k]:
                cv2.circle(view, (int(u), int(v)), 10, (170, 170, 170), 2)
            named = (a.get("obs") or {}).get(tag, {})
            for label, (u, v) in named.items():
                cv2.drawMarker(view, (int(u), int(v)), (0, 230, 255), cv2.MARKER_CROSS, 22, 3)
                cv2.putText(view, label, (int(u) + 10, int(v) - 10), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 230, 255), 2)
            if a.get("rope") is not None:
                cv2.line(view, *map(tuple, a["rope"][k]["ends"]), (30, 140, 255), 4)
            if self.robot_roi and tag in self.robot_roi:
                x, y, w, h = self.robot_roi[tag]
                cv2.rectangle(view, (x, y), (x + w, y + h), (40, 210, 40), 3)
            if self.bench.leds[k] is not None:     # the LED ROI the run and LED sync use
                x, y, w, h = self.bench.leds[k]
                cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.putText(view, "LED", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 255), 2)
            if "cands" in a:
                for colour, width in (((0, 0, 0), 6), ((255, 255, 255), 2)):
                    cv2.putText(view, f"{tag}: {len(a['cands'][k])} candidates, {len(named)} named", (10, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, width)
            views.append(view)
        return views

    def _overlay(self, mean, cands, rope=None, obs=None):
        """Kept as `annot`, so the live preview carries it too; drawn before AND after
        the fit, so a failed fit still shows what was seen."""
        self.annot = dict(cands=cands, rope=rope, obs=obs)
        self.show_pair(self._annotate(mean))

    def do_setup(self):
        """LED first (it needs no calibration), then the platform fit only with a rig loaded.
        The LED word ends inside LED_TEST_S, before the still window starts, so it is never
        in the averaged image."""
        b = self.bench
        try:
            led = self._find_led() if b.serial_port else "No ESP port given: LED not searched."
        except Exception as exc:               # the platform fit does not need the LED
            self.log_exc()
            led = f"**LED check failed:** {exc}. Detect setup retries."
        if b.calibration is None:
            self.say(led + "\n\n**No calibration loaded:** LED only. Calibrate, or pick one under "
                           "Calibration, for the platform fit.")
            return
        self.say(led + "\n\n" + self._fit_platform() + "\n\n" + led)

    def _fit_platform(self):
        """10 s still -> rope x holder fit -> world frame; returns the status text."""
        from LLM.analyse.holder_detection import detect_checker_centers
        from LLM.analyse.holder_geometry import find_rope_and_holder, fit_holder
        from LLM.analyse.world import build_world_frame
        b = self.bench
        self.say(f"**Keep the bench still:** {STILL_S:.0f} s of footage is averaged into one image per camera.")
        rig = self._rig()
        acc, n, t0 = None, 0, time.monotonic()
        while time.monotonic() - t0 < STILL_S:
            _, frames = self.frame()
            acc = [f.astype(np.float64) for f in frames] if acc is None else [a + f for a, f in zip(acc, frames)]
            n += 1
            self.preview(frames)
            if n % 30 == 0:
                self.say(f"**Averaging** {time.monotonic() - t0:.0f}/{STILL_S:.0f} s, {n} pairs. Keep still.")
        mean = [np.rint(a / n).astype(np.uint8) for a in acc]
        ref = b.folder / f"reference_{uuid4().hex[:8]}"
        ref.mkdir()
        for tag, img in zip("AB", mean):
            cv2.imwrite(str(ref / f"{tag}.png"), img)
        write_json(ref / "meta.json", dict(session_id=b.session["session_id"], camera=b.camera,
                                           times=list(self.src.last_stamps), averaged_pairs=n, window_s=STILL_S))
        b.reference_pair = ref
        layout = read_json(HOLDER_LAYOUT)
        cands = [[c["center_px"] for c in detect_checker_centers(img, max_candidates=MAX_CORNERS)] for img in mean]
        self._overlay(mean, cands)
        self.say(f"**Searching** rope x holder ({len(cands[0])}/{len(cands[1])} checker candidates)…")
        from LLM.record.panel import run_while
        up, rope, obs, info = run_while(lambda: find_rope_and_holder(rig, mean, cands, layout), self.preview)
        self._overlay(mean, cands, rope, obs)
        if up is None:
            raise ValueError(f"no thin vertical line in both views (per view: {info['lines']}); is the rope lit "
                             f"and in frame?")
        T, report = fit_holder(rig, layout, obs)        # raises with its reason; the overlay shows why
        tilt = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ up, -1, 1))))
        origin = layout.get("bearing_center_fixture_mm") or layout["mounting_boss"]["axis_xy_mm"] + [0.0]
        world = build_world_frame(T[:3, :3] @ origin + T[:3, 3], up, T[:3, :3] @ layout["x_hint_fixture"])
        world["evidence"] = dict(method="viser_app_rope_x_holder", session_id=b.session["session_id"],
                                 camera=b.camera, rig_sha256=sha256(b.calibration), fixture_fit=report,
                                 match=info, T_camera_A_from_fixture=T.tolist(), holder_tilt_from_rope_deg=tilt,
                                 rope_image_ends={t: r["ends"] for t, r in zip("AB", rope)},
                                 origin="surveyed bearing" if layout.get("bearing_center_fixture_mm") else
                                 "mounting-boss axis, UNSURVEYED", physical_accuracy_validated=False,
                                 input_sha256={str(p.resolve()): sha256(p) for p in
                                               (ref / "A.png", ref / "B.png", ref / "meta.json", HOLDER_LAYOUT)})
        b.world, self.fixture = world, T
        self.robot_roi = self._pose_rois()
        self._overlay(mean, cands, rope, obs)
        write_json(b.folder / "world_frame.json", world)
        self.rope = self._rope_point([r["line"] for r in rope], rig, world["origin_camera_mm"])
        self._draw_scene()
        for client in self.server.get_clients().values():
            client.camera.position, client.camera.look_at = VIEW_FROM_MM, (0.0, 0.0, 40.0)
        fitted = (f"**Platform fitted:** {sum(map(len, obs.values()))} keypoints, rms {report['rms_px']:.2f} px; "
                 f"holder tilt from rope {tilt:.1f} deg; origin {world['evidence']['origin']}"
                 + (". **Yaw ambiguous** (ID dots not separated): x may be off by 90 deg" if info["ambiguous"] else "")
                 + ". Camera images: grey candidates, yellow named, orange rope, green robot ROI."
                 " Frozen from here: nothing is re-fitted during a run.")
        return fitted

    @staticmethod
    def _rope_point(lines, rig, near):
        """The rope's 3D point nearest `near` (camera A, mm).

        Each view's rope line and its lens centre span a plane (`disc_pose.mast_plane`,
        the same planes `mast_direction` crosses for the direction); the rope is their
        intersection, and the minimum-norm step from `near` onto both planes lands on it.
        """
        from controller.pose.disc_pose import mast_plane
        N = np.array([mast_plane(line, cam) for line, cam in zip(lines, rig.cameras)])
        d = np.array([n @ cam.position for n, cam in zip(N, rig.cameras)])
        near = np.asarray(near, float)
        return near - N.T @ np.linalg.solve(N @ N.T, N @ near - d)

    def _draw_scene(self):
        import trimesh
        sc, T, w = self.server.scene, self.fixture, self.bench.world
        sc.add_grid("/grid", width=400.0, height=400.0, cell_size=10.0, section_size=50.0)
        sc.add_frame("/world", axes_length=60.0, axes_radius=1.0)
        sc.add_label("/world/z", "z = rope" if w else "camera A frame", position=(0, 0, 65))
        if self.bench.calibration is not None:
            c, R_w = self.bench.camera, (np.asarray(w["R_camera_from_world"]).T if w else np.eye(3))
            self.frusta = []
            for cam in self._rig().cameras:   # OpenCV axes, as viser's frustum expects
                self.frusta.append(sc.add_camera_frustum(
                    f"/cameras/{cam.name}", 2 * np.arctan(c["height"] / (2 * cam.K[1, 1])), c["width"] / c["height"],
                    scale=FRUSTUM_MM, thickness=1.0, color=(40, 120, 255), wxyz=_wxyz(R_w @ cam.R),
                    position=self._world(cam.position)))
                sc.add_label(f"/cameras/{cam.name}/name", f"camera {cam.name}")
        if self.rope is not None and w is not None:
            x, y, _ = self._world(self.rope)       # world z IS the rope direction
            rope = trimesh.creation.cylinder(radius=1.0, height=ROPE_DRAW_MM[1] - ROPE_DRAW_MM[0])
            rope.visual.face_colors = (230, 120, 30, 255)
            sc.add_mesh_trimesh("/rope", rope, position=(x, y, sum(ROPE_DRAW_MM) / 2))
        if T is not None:
            if not hasattr(self, "_platform"):
                mesh = trimesh.load(STEP_MESH, force="mesh")
                mesh.apply_scale(1000.0)        # cascadio reads the STEP in metres; CAD is mm
                self._platform = mesh
            R = np.asarray(self.bench.world["R_camera_from_world"]).T @ T[:3, :3]
            sc.add_mesh_trimesh("/platform", self._platform, wxyz=_wxyz(R), position=self._world(T[:3, 3]))
            layout = read_json(HOLDER_LAYOUT)
            pts = np.array([self._world(T[:3, :3] @ p + T[:3, 3]) for p in layout["points"].values()])
            sc.add_point_cloud("/keypoints", pts, (255, 220, 0), point_size=3.0, point_shape="circle")

    def _wake(self):
        """`serial_check.wake` on a worker (the preview stays live); one status line."""
        from LLM.record.panel import run_while
        from LLM.record.serial_check import wake
        w = run_while(lambda: wake(self.bench.serial_port), self.preview)
        return (f"wake: drained {w['drained']} B, ROM boot {'ok' if w['rom_boot'] else 'NOT seen'}, "
                f"READY {'ok' if w['ready'] else 'NOT seen'}")

    def _find_led(self):
        """`led=code` (IDLE only, energises nothing) and `scene.locate_led` on both
        cameras: the ROIs found are kept for the run (START dark check) and LED sync.
        The decode is the identity test, so a found ROI is our LED, not a reflection."""
        from controller.control.hover_controller_runner import CommandLink
        from controller.pose.scene import locate_led
        from LLM.record.panel import run_while
        b = self.bench
        folder = b.folder / f"led_test_{datetime.now():%Y%m%d_%H%M%S}"
        folder.mkdir(parents=True)
        woke = self._wake()
        self.say(f"**Testing LED** ({woke}): blinking the ID word; coil power can stay off.")
        link = run_while(lambda: CommandLink(b.serial_port, False, str(folder / "host_commands.log"),
                                             takeoff_cmd=[], raw_log_path=folder / "telemetry.jsonl",
                                             require_ready=True), self.preview)
        times, frames = [], []
        try:
            link.send("led=code")
            end = time.monotonic() + LED_TEST_S
            while time.monotonic() < end:
                t, pair = self.frame()
                times.append(t)
                frames.append(pair)
                link.drain()
                self.preview(pair, light=True)
        finally:
            link.close()
        acked = "SYNC_CODE" in (folder / "telemetry.jsonl").read_text(errors="replace")
        rois, report = [], []
        for k, tag in enumerate("AB"):
            views = [p[k] for p in frames]
            roi = _led_roi(locate_led(views, np.asarray(times)), views[0].shape)
            rois.append(roi)
            if roi is None:
                report.append(f"{tag}: **not found**")
            else:
                x, y, w, h = roi
                level = [float(v[y:y + h, x:x + w].mean()) for v in views]
                report.append(f"{tag}: {roi}, ROI mean {min(level):.0f}..{max(level):.0f}")
        for k, tag in enumerate("AB"):          # where anything blinked: max - min over the window
            views = np.stack([p[k] for p in frames])
            cv2.imwrite(str(folder / f"range_{tag}.png"), views.max(axis=0) - views.min(axis=0))
        if any(r is None for r in rois):         # the evidence to diagnose a miss offline
            np.savez_compressed(folder / "frames.npz", times=np.asarray(times),
                                A=np.stack([p[0] for p in frames]), B=np.stack([p[1] for p in frames]))
        write_json(folder / "result.json", {"firmware_acked": acked, "rois_xywh": rois,
                                            "pairs": len(frames), "window_s": LED_TEST_S})
        self.show_pair(self._annotate(frames[len(frames) // 2]))
        if all(r is not None for r in rois):
            b.leds = tuple(rois)
            return ("**LED found in both cameras**: " + "; ".join(report)
                    + ". Runs check it is dark before START and sync on it.")
        why = "" if acked else " The firmware never answered SYNC_CODE: check it is `flight` and IDLE."
        return "**LED not found**: " + "; ".join(report) + f".{why} Is the LED in both views? Detect setup retries."

    def do_run(self):
        # No PSU-off prompt (operator request 2026-11-05): the app cannot see the supply,
        # and a reset can still float the carrier pins until forceAllGatesLow() runs.
        from LLM.record.design import duty_values, plan, write_ramp
        b = self.bench
        # Planned and validated BEFORE the reset: a bad design or a ramp over the 55 s
        # cap is refused with nothing energised.
        design = plan(self._specs(), duty_scale_pct=float(self.duty_scale.value))
        ramp = write_ramp(b.folder / "ramps" / f"{datetime.now():%Y%m%d_%H%M%S}.json", design)
        v = {k: p["value"] for k, p in design["parameters"].items()}
        from LLM.analyse.coil_trim import TRIMS
        trim = None if self.trim.value == "off" else TRIMS / self.trim.value
        if trim is not None and any(d != 100.0 for d in duty_values(design)):
            raise ValueError("Coil trim needs every coil's duty fixed at 100 % (it only lowers duty); "
                             "set the duty parameters to fixed 100 or the trim to off")
        woke = self._wake()                  # the recorder's own link follows; see serial_check.wake
        self.say(f"**Running** 1 -> {v['f_top_hz']:g} Hz at {v['rate_hz_s']:g} Hz/s, hold {v['hold_s']:g} s; duty "
                 + (f"100 % until {min(v['duty_switch_hz'], v['f_top_hz']):g} Hz, then " if v['duty_switch_hz'] > 1 else "from START ")
                 + "/".join(f"{d:g}" for d in duty_values(design))
                 + (f" %, coil trim `{trim.name}`" if trim else " %") + f" ({woke}). The ESP32 resets, the ramp "
                 "starts after 5 s of pre-roll and stops on its own. Stop recording early ends it and saves. "
                 "After the stop the take is checked and LED-synced in the background; the save buttons work at once.")
        # LED sync (operator 2026-11-06, replacing the host-clock default of 2026-11-05).
        # No LED word before START: send_takeoff sleeps through it and the recording
        # drops those frames. The ROIs come from Test LED, or else find_led after the run.
        if self.player is not None:            # its decoders and the drone mesh belong to the last run
            self._close_player()
        self.finish.clear()
        self.stop_rec.disabled = False
        try:
            take = b.run(ramp, finish=False, note=self.note.value, led_code=False, led_a=b.leds[0], led_b=b.leds[1],
                  duty=duty_values(design), design=design, duty_switch_hz=v["duty_switch_hz"], trim=trim,
                  finish_event=self.finish, auto_start=True, server=self.server, src=self.src,
                  preview=lambda frames: self.preview(frames, light=True),
                  say=lambda text: setattr(self.status, "content", text))   # 5 Hz: not printed
        finally:
            self.stop_rec.disabled = True
        self.run_done, self.analysis_done = True, False
        self.analysis = threading.Thread(target=self._check_and_analyse, args=(take, b.last_run, b.leds), daemon=True)
        self.analysis.start()
        self.say(f"**Take recorded** `{b.last_run}`. Checking + LED sync in the background; "
                 "type the note, then Save local, Save to Drive or Reset (they finish once the check is done).")

    def _check_and_analyse(self, take, run, leds):
        from LLM.record.experiment import finish_take
        b, summary = self.bench, ""
        try:
            finish_take(take, b.session, run)
            out = b.analyse(sync="led", run=run, leds=leds)
            curves = Path(run) / "export/input_curves.png"
            if curves.exists():
                self.curves.image, self.curves.visible = cv2.cvtColor(cv2.imread(str(curves)), cv2.COLOR_BGR2RGB), True
            duty = read_json(Path(run) / "raw/run_parameters.json")["duty"]["percent"]
            pkg = (f" Duty {duty} %." if duty else "") + (f" Package: `{out['package']}`." if out.get("package") else
                   f" Package failed: {out['package_error']}." if out.get("package_error") else "")
            dead = out.get("unpowered_coils")
            warn = (f" **Coils {', '.join(dead)} drew NO current: not powered? Their effective duty is 0.**"
                    if dead else "")
            summary = f"**Run saved** `{run}`: {out.get('status')}.{warn}{pkg}"
        except Exception as exc:
            self.log_exc()
            summary = f"**Run saved** `{run}`, but the check / analysis failed: {exc}."
        finally:
            with self._analysis_lock:
                self.analysis_done, keep = True, self.keep_when_analysed
                self.keep_when_analysed = False
            if keep:
                self.do_keep(also=summary)
            else:
                self.say(summary + " Edit the note, then Save local, Save to Drive, or Reset.")

    def _wait_for_analysis(self):
        if self.analysis is not None and self.analysis.is_alive():
            self.say("**Waiting** for the background check + LED sync to finish...")
            self.analysis.join()

    def _start_upload(self):
        run = self.bench.last_run
        if run is None or not self.upload_staged.is_set():
            return
        self.buttons["upload"].disabled = True
        # One event per upload: an older transfer finishing must not release a newer stage.
        self.upload_staged = staged = threading.Event()
        threading.Thread(target=self._upload, args=(Path(run), staged), daemon=True).start()

    def _upload(self, run, staged):
        from LLM.record.upload import upload
        try:
            self._wait_for_analysis()           # it writes into the folder being copied
            # The note lives only in the text box until saved; upload what is typed now.
            # Save local saves it again later, so re-upload if the note changes after this.
            if self.run_done and self.bench.last_run is not None and Path(self.bench.last_run) == run:
                self._save_note(run)
            self.say(f"**Uploading** `{run.name}` to Drive in the background; Reset and the next run need not wait.")
            target = upload(run, on_staged=staged.set)
            self.say(f"**Uploaded** `{run.name}` -> `{target}`.")
        except Exception as exc:
            self.log_exc()
            self.say(f"**Upload failed:** {exc}")
        finally:
            staged.set()

    def _save_note(self, run):
        """The note as typed after the run; the manifest is metadata, not raw evidence."""
        path = Path(run) / "manifest.json"
        manifest = read_json(path)
        manifest["initial_conditions"]["notes"] = self.note.value
        manifest["labels"] = {cat: list(picks) for cat, picks in self.labels.items() if picks}
        manifest["note_saved_after_run"] = datetime.now().isoformat(timespec="seconds")
        write_json(path, manifest)

    def do_discard(self):
        self._wait_for_analysis()               # it writes into the folder about to move
        if not self.upload_staged.is_set():
            self.say("**Waiting** for the Drive upload to stage the run (seconds)...")
            # Only until staged: the stage is remuxed copies + hard links, so the transfer
            # carries on after the folder moves to the Trash (upload.upload).
            self.upload_staged.wait()
        run = Path(self.bench.last_run)
        if self.player is not None:             # its decoders hold this run's videos open
            self._close_player()
        self._save_note(run)
        # Operator 2026-11-15: to the macOS Trash (Finder "Put Back" works), not discarded/.
        # Still a move, never a delete: raw video is evidence until the Trash is emptied.
        if shutil.which("trash"):
            subprocess.run(["trash", str(run)], check=True)
            where = "the Trash (Finder: Put Back to restore)"
        else:
            target = self.bench.runs / "discarded" / run.name
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(run), str(target))
            where = f"`{target}`"
        self.bench.last_run, self.run_done = None, False
        self.curves.visible = False
        self.say(f"**Reset:** `{run.name}` moved to {where}. Ready for a new run.")

    def do_keep(self, also=""):
        if not self.run_done:
            return
        with self._analysis_lock:               # runs on viser's thread: never join here
            if not self.analysis_done:
                self.keep_when_analysed = True
                self.say("**Will keep** this run once the background check finishes, with the note as it reads then.")
                return
        from LLM.record.design import advance
        self._save_note(self.bench.last_run)
        self.run_done = False
        specs, notes = advance(self._specs())      # step modes move only after a KEPT run
        for name, spec in specs.items():
            self.design[name]["value"].value = spec["value"]
        stepped = [f"{n} -> {sp['value']:g}" for n, sp in specs.items() if sp["mode"] == "step"]
        self.say((also + " " if also else "") + f"**Kept** `{self.bench.last_run}` with its note. Ready for the next run."
                 + (" Stepped: " + ", ".join(stepped) + "." if stepped else "")
                 + (" **" + "; ".join(notes) + "**" if notes else ""))
        self.note.value = ""
        self.clear_labels()

    def _pose_rois(self):
        """axis_pose ROIs: the ROBOT_RADIUS_MM x ROBOT_Z_MM cylinder about the mounting-boss
        axis, projected into each view through this session's platform fit; None before one."""
        from LLM.analyse.holder_geometry import _project
        w, c = self.bench.world, self.bench.camera
        T = (w or {}).get("evidence", {}).get("T_camera_A_from_fixture")
        if T is None or self.bench.calibration is None:
            return None
        T = np.asarray(T)
        cx, cy = read_json(HOLDER_LAYOUT)["mounting_boss"]["axis_xy_mm"]
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        pts = np.array([[cx + ROBOT_RADIUS_MM * np.cos(t), cy + ROBOT_RADIUS_MM * np.sin(t), z]
                        for z in np.linspace(*ROBOT_Z_MM, 6) for t in a]) @ T[:3, :3].T + T[:3, 3]
        rois = {}
        for tag, cam in zip("AB", self._rig().cameras):
            uv = _project(cam, pts)
            x0, y0 = np.clip(np.floor(uv.min(0)).astype(int), 0, None)
            x1, y1 = np.minimum(np.ceil(uv.max(0)).astype(int), [c["width"], c["height"]])
            rois[tag] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
        return rois

    @staticmethod
    def _segmented(frame, plate, roi, label):
        """One BGR frame with axis_pose's robot mask (green), axis (yellow) and ROI drawn,
        exactly as `LLM.axis_pose` segments it against `plate`."""
        from LLM.axis_pose import _overlay, _view
        from controller.pose import axis_pose as core
        m, mask, reason = _view(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), plate, roi, core.THRESH)
        return _overlay(frame, mask, m, roi, f"{label} {reason}")

    def _trials(self, run):
        return sorted((Path(run) / "processed/axis_pose_trials").glob("*/axis_pose.csv"))

    def _plates(self, run):
        """Per-camera background plates of the take itself (`disc_pose.build_plate`: a low
        percentile over 300 frames), so the bright coil cans and cables cancel and what
        moves -- the drone in flight -- is what segments. Threshold mode segmented the
        cans (2026-11-05, 0/2400 posed). A robot that never leaves its rest pose is in
        its own plate and does not segment. Saved under processed/, never in raw/."""
        from controller.pose.disc_pose import build_plate
        from LLM.record.takes import load_flight
        folder = Path(run) / "processed/pose_plates"
        folder.mkdir(parents=True, exist_ok=True)
        flight, paths = load_flight(run), {}
        for t in "AB":
            paths[t] = folder / f"percentile_plate_{t}.png"
            if not paths[t].exists():
                cv2.imwrite(str(paths[t]), build_plate(flight[f"video_{t.lower()}"]))
        return paths

    def do_pose(self):
        """axis_pose over the whole take, segmentation shown live; saved inside the run."""
        self._wait_for_analysis()
        from LLM.axis_pose import run_axis_pose
        from LLM.record.panel import run_while
        run, rois, live = Path(self.bench.last_run), self.robot_roi, {}
        rig = run / "calibration/stereo_rig.json"
        rig = rig if rig.exists() else self.bench.calibration

        def pump():                            # the camera images show the segmentation, not live
            if self.src is not None:
                self.frame()                   # keep the stream drained
            else:
                time.sleep(.05)
            if "panel" in live:
                panel = live.pop("panel")
                half = panel.shape[1] // 2
                self.show_pair([panel[:, :half], panel[:, half:]])
            if "done" in live:
                k, n = live.pop("done")
                self.say(f"**Drone pose:** frame {k}/{n}. Green = segmented robot, yellow = axis, box = ROI.")
        self.say("**Drone pose:** building the take's background plates…")
        plates = run_while(lambda: self._plates(run), pump)
        self.say("**Drone pose:** segmenting every frame against them (green = robot mask, yellow = axis)…")
        # background_confirmed: the plate is computed from the take, not an empty-scene
        # photo -- its file name says so, and axis_pose snapshots it into the trial
        out = run_while(lambda: run_axis_pose(run, rig, robot_rois=rois, background_mode="difference",
                                              background_paths=plates, background_confirmed=True,
                                              show=lambda panel: live.__setitem__("panel", panel),
                                              progress=lambda k, n: live.__setitem__("done", (k, n))), pump)
        self.playback_img.image = cv2.cvtColor(cv2.imread(out["overlays"]), cv2.COLOR_BGR2RGB)
        self.playback_img.visible = True
        self.say(f"**Drone pose saved** with the recording: `{out['directory']}` -- {out['accepted']}/{out['frames']} "
                 f"frames posed. Reasons: {out['reasons']}. Experimental silhouette axis + centroid; ROI {rois}.")

    def do_playback(self):
        self._wait_for_analysis()
        from LLM.record.takes import load_flight
        from controller.pose import render
        import pandas as pd
        run = Path(self.bench.last_run)
        if not self._trials(run):            # solve once, shown live, saved with the run
            if self.robot_roi is None:
                raise ValueError("no saved drone pose for this run and no platform fit to place its ROI: Fit setup first")
            self.do_pose()
        trial = self._trials(run)[-1]
        pose = pd.read_csv(trial)
        rois = read_json(trial.parent / "metadata.json")["robot_rois"]
        plates = {t: cv2.imread(str(trial.parent / "inputs" / f"background_{t}.png"), cv2.IMREAD_GRAYSCALE)
                  for t in "AB" if (trial.parent / "inputs" / f"background_{t}.png").exists()}
        flight = load_flight(run)
        drone = self.server.scene.add_mesh_trimesh("/drone", render.load_mesh(), visible=False)
        self.player = dict(pose=pose, video=[cv2.VideoCapture(flight[f"video_{t}"]) for t in "ab"], drone=drone,
                           rois=rois, plates=plates, k=-1, playing=False, t0=0.0, k0=0,
                           times=np.asarray(flight["times_a"], float))
        self.slider.max, self.slider.value, self.slider.disabled = len(pose) - 1, 0, False
        self._draw_scene()
        self.say(f"**Playback** `{run.name}`: {int(pose.axis_estimate_valid.sum())}/{len(pose)} frames posed "
                 "(silhouette axis + centroid proxy, roll unobserved). The platform is static. "
                 "Playback / pause toggles; scrub with the slider.")

    def _close_player(self):
        for d in self.player["video"]:
            d.release()
        self.player["drone"].remove()
        self.player, self.playback_img.visible = None, False
        self.slider.disabled = True

    def do_play(self):
        p = self.player
        p["playing"] = not p["playing"]
        p["t0"], p["k0"] = time.monotonic(), int(self.slider.value)

    def _advance(self):
        p = self.player
        if p["playing"]:
            target = int(np.searchsorted(p["times"], p["times"][p["k0"]] + time.monotonic() - p["t0"]))
            if target >= len(p["pose"]):
                p["playing"], target = False, len(p["pose"]) - 1
            self.slider.value = target
        k = int(self.slider.value)
        if k == p["k"]:
            if self.src is None:            # with cameras on, their read paces the loop
                time.sleep(.02)
            return
        if k != p["k"] + 1:            # a jump: seek; consecutive frames just decode on
            for d in p["video"]:
                d.set(cv2.CAP_PROP_POS_FRAMES, k)
        frames = [d.read()[1] for d in p["video"]]
        p["k"] = k
        if all(f is not None for f in frames):
            panel = np.hstack([self._segmented(f, p["plates"].get(t), p["rois"][t], f"{t} k={k}")
                               for t, f in zip("AB", frames)])   # the mask + axis axis_pose saved
            self.playback_img.image = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            self.playback_img.visible = True
        row, drone = p["pose"].iloc[k], p["drone"]
        drone.visible = bool(row.axis_estimate_valid)
        if drone.visible:
            drone.position = self._world([row.x_est_mm, row.y_est_mm, row.z_est_mm])
            drone.wxyz = _wxyz(_z_to(self._world([row.nx_est, row.ny_est, row.nz_est], vector=True)))


def main(port=None, cameras=(1, 0), mode=(1280, 800, 120), **kw):
    App(port, cameras, mode, **kw).loop()


def _self_check():
    """Gating with a fake camera and no serial: no hardware is touched."""
    import tempfile

    class FakeStereo:
        last_stamps, max_skew_s = (0.0, 0.0), None

        def read(self, **_):
            return time.monotonic(), [np.full((800, 1280), 60, np.uint8)] * 2

        def close(self):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        app = App(None, runs=Path(tmp), viser_port=0)
        try:
            on = lambda: {k for k in STEPS if not app.buttons[k].disabled}
            app.gate()
            assert on() == {"connect"}, on()
            assert app.bench.calibration is None and app.cal_choice.value == NO_CALIBRATION, "nothing saved, nothing loaded"
            app.src = FakeStereo()
            app.gate()
            assert on() == {"calibrate", "setup"}, f"no port: {on()}"
            app.bench.serial_port = "fake"
            app.gate()
            assert on() == {"calibrate", "setup", "run"}, on()
            app.run_done = True
            app.gate()
            assert on() == {"calibrate", "discard", "keep"}, f"no take folder yet: {on()}"
            app.gate(busy=True)
            assert on() == {"keep"}, "a long step greys everything but Save local"
            from LLM.record.experiment import prepare_sync_run
            app.bench.last_run = prepare_sync_run(Path(tmp), app.bench.session, note="typed before")
            app.gate()
            assert {"discard", "keep", "upload"} <= on(), on()
            app.note.value = "typed after the run"
            cat = next(iter(app.labels))
            first, second = app.label_groups[cat][1][:2]
            app.toggle_label(cat, first)
            app.toggle_label(cat, second)
            app.toggle_label(cat, TICK + second)          # a second click un-picks it
            assert app.labels[cat] == [first] and app.label_groups[cat][0].options[0] == TICK + first
            app.do_keep()
            manifest = read_json(app.bench.last_run / "manifest.json")
            assert manifest["initial_conditions"]["notes"] == "typed after the run" and not app.run_done
            assert manifest["labels"] == {cat: [first]}, manifest.get("labels")
            assert not any(app.labels.values()) and TICK not in "".join(app.label_groups[cat][0].options)
            from LLM.record.upload import source_md
            assert f"- Labels: {cat}: {first}" in source_md(app.bench.last_run)
            logged = [json.loads(x) for x in app.log_path.read_text().splitlines()]
            assert logged[-1]["kind"] == "status" and logged[-1]["run"] == app.bench.last_run.name
            assert "Kept" in logged[-1]["text"], logged[-1]
            app.gate()
            assert "upload" in on() and "keep" not in on(), "Save to Drive stays available after Save local"
            # Deterministic by default; Load parameters is all-or-nothing.
            from LLM.record.design import PARAMS, plan
            assert all(p["mode"] == "fixed" for p in PARAMS.values()), "defaults must be deterministic"
            assert all(w["mode"].value == "fixed" for w in app.design.values())
            setup = {"parameters": {"duty_A_pct": {"mode": "random", "lo": 40.0, "hi": 100.0}},
                     "duty_scale_pct": 20.0}
            app._load_params(json.dumps({"design": setup}).encode(), "run_parameters.json")
            assert app.design["duty_A_pct"]["mode"].value == "random" and app.design["duty_A_pct"]["lo"].value == 40.0
            assert app.duty_scale.value == 20.0 and "Loaded" in app.status.content
            app._load_params(json.dumps({"parameters": {"f_top_hz": {"value": 200.0},
                                                        "hold_s": {"mode": "sometimes"}}}).encode(), "bad.json")
            assert app.design["f_top_hz"]["value"].value == PARAMS["f_top_hz"]["value"], "a refused file changes nothing"
            assert "refused" in app.status.content
            app._load_params(json.dumps({"parameters": {"duty_A_pct": {"mode": "fixed"}}}).encode())
            assert app._specs()["f_top_hz"]["value"] == PARAMS["f_top_hz"]["value"]
            app.design["f_top_hz"]["mode"].value = "step"
            plan(app._specs(), seed=1)       # the GUI's specs are a valid design
            app.run_done = True
            app.do_keep()                    # a kept run steps the value shown in the GUI
            assert app.design["f_top_hz"]["value"].value == PARAMS["f_top_hz"]["value"] + PARAMS["f_top_hz"]["step"]
            app.run_done, app.analysis_done = True, False
            app.do_keep()                    # pressed while the background check runs
            assert app.run_done and app.keep_when_analysed, "Save local during the check is deferred, not lost"
            import contextlib, io
            with contextlib.redirect_stderr(io.StringIO()):
                app._check_and_analyse(Path(tmp) / "no_take", app.bench.last_run, (None, None))
            assert not app.run_done and "Kept" in app.status.content and "failed" in app.status.content, app.status.content
            # Camera views: side panel only, one image per camera, each from its own half.
            ViserSink(app).show(np.hstack([np.full((50, 80), 10, np.uint8), np.full((50, 80), 200, np.uint8)]))
            a, b = (np.asarray(h.image) for h in app.cams)
            assert a.mean() < 50 < 150 < b.mean(), (a.mean(), b.mean())
            app.tick()                      # one preview frame through the fake camera
            app.bench.leds = ([10, 10, 20, 20], [30, 30, 20, 20])
            app._shown = 0
            app.tick()                      # ...and with the LED ROIs drawn
            assert not any(hasattr(app, k) for k in ("estop", "show3d", "seg", "overlay", "thumbs")), "removed widgets"
            assert len(app.buttons) == len(STEPS) == 9, "nine step buttons (+ Stop early + Load parameters = 11)"
        finally:
            app.server.stop()
    # The rope: a known vertical line through the synthetic holder rig, drawn into both
    # views, must come back as the same 3D line; then the scene draws cameras + rope.
    from LLM.tests.test_holder_geometry import scene as holder_scene, project
    from LLM.analyse.world import build_world_frame
    rig, _, _, truth = holder_scene()
    up = truth[:3, :3] @ [0, 0, 1]
    base = truth[:3, :3] @ [30.0, 5.0, 0.0] + truth[:3, 3]
    lines = []
    for cam in rig.cameras:              # zero distortion: image points are undistorted
        a, b = project(cam, [base - 80 * up, base + 120 * up])
        lines.append(((a + b) / 2, (b - a) / np.linalg.norm(b - a)))
    near = truth[:3, :3] @ [49.5, 10.5, 0.0] + truth[:3, 3]
    p = App._rope_point(lines, rig, near)
    off = (p - base) - ((p - base) @ up) * up
    assert np.linalg.norm(off) < 0.5, f"rope point {np.linalg.norm(off):.2f} mm off the true line"
    with tempfile.TemporaryDirectory() as tmp:
        app = App(None, runs=Path(tmp), viser_port=0, upside_down=True)
        try:
            rig.save(Path(tmp) / "rig.json")
            app.bench.calibration, app.rope, app.fixture = Path(tmp) / "rig.json", p, truth
            app.bench.world = build_world_frame(near, up, truth[:3, :3] @ [1, 0, 0])
            app._draw_scene()
            assert len(app.frusta) == 2
            assert np.allclose(app._world(rig.a.position), app._world([0, 0, 0]))
        finally:
            app.server.stop()
    # Autoload: the newest saved rig at this resolution is in force at startup, unverified;
    # (none) drops it and everything fitted with it.
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "sessions/s1/calibration_x/stereo_rig.json"
        saved.parent.mkdir(parents=True)
        rig.save(saved)
        if read_json(saved).get("image_size") == [1280, 800]:
            app = App(None, runs=Path(tmp), viser_port=0, upside_down=True)
            try:
                assert Path(app.bench.calibration).resolve() == saved.resolve(), app.bench.calibration
                assert app.bench.verification is None and app.cal_choice.value == "s1/calibration_x/stereo_rig.json"
                app.robot_roi = {"A": [0, 0, 1, 1]}
                app._load_calibration(NO_CALIBRATION)
                assert app.bench.calibration is None and app.robot_roi is None
            finally:
                app.server.stop()
        else:
            print("  autoload not checked: the synthetic rig is", read_json(saved).get("image_size"))
    from LLM.record.panel import run_while
    pumped = []
    assert run_while(lambda: (time.sleep(.3), 7)[1], lambda: (pumped.append(1), time.sleep(.01))) == 7
    assert len(pumped) > 5, "the preview must be pumped while the worker blocks"
    try:
        run_while(lambda: 1 / 0, lambda: time.sleep(.01))
        raise AssertionError("a worker error must reach the caller")
    except ZeroDivisionError:
        pass
    assert _led_roi(None, (400, 640)) is None
    assert _led_roi((5.0, 300.0, 6.0), (400, 640)) == [0, 290, 15, 20]
    assert _led_roi((630.0, 395.0, 6.0), (400, 640)) == [620, 385, 20, 15]
    R = _z_to([0, 0.6, 0.8])
    assert np.allclose(R @ [0, 0, 1], [0, .6, .8]) and np.allclose(R.T @ R, np.eye(3))
    print("LLM app: button gating, rope line, cameras + platform scene ok (no cameras, serial or coils)")


if __name__ == "__main__":
    import sys
    _self_check() if "--self-check" in sys.argv else main()
