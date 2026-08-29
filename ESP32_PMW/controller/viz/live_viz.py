#!/usr/bin/env python3
"""
Live 3-D view of a flight: pose, rig, trajectory and the controller's own signals.

`hover_controller_runner.py` reduces a full 5-DOF pose to two scalars and logs only the
commands it sent, so a bad run gives you no way to tell a bad controller from lying
vision. This puts both on one screen while the loop is running: the robot mesh at the
estimated pose, the measured rim circle, the cameras that measured it, a time-coloured
trace of where it has been, and four stacked plots of position, angle, command and
estimator health.

The control loop calls `push()` and nothing else. `push()` is one `deque.append` -- O(1),
bounded, atomic under the GIL -- and a daemon thread does every expensive thing at its own
rate. Nothing here can block the loop, and `make_viz()` returns a no-op `NullViz` if viser
is missing, the port is taken, or construction fails for any reason: the coils have no
firmware watchdog, so a visualiser must never be able to end a flight.

Usage:
  live_viz.py --demo                 # synthetic helix, no hardware
  live_viz.py --replay poses.csv     # a recorder.py CSV
  live_viz.py --camera camera:0      # live vision, no controller
"""

from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parent / "pose"),
    str(HERE.parent / "calib"),
    str(HERE.parent / "camera"),
]

# Palette lifted from ai/validation/scene3d.py so the live view and the offline
# validation renders name the same thing with the same colour.
C_ESTIMATE = (0xC8, 0x93, 0x0A)
C_REF = (0x2E, 0xC4, 0x8F)
C_CAM = ((0x2A, 0x8F, 0xC4), (0x84, 0x65, 0xE4))
C_NORMAL = (0x22, 0xC8, 0xD8)
# Five visually separable strokes for the four command channels plus the adaptive
# hover-frequency estimate, which shares their plot because it is only readable
# against the `freq` the loop is actually commanding.
C_CMD = ("#e0533d", "#2e9bd6", "#e8b33c", "#7c5ce0", "#3fbf7f")

TRACE_CMAP = "viridis"
NORMAL_MM = 25.0
RIM_SEGMENTS = 64


@dataclass
class Sample:
    """One tick as the control loop saw it. `pose` is a `Pose` or a `StereoPose`."""

    t: float
    pose: object = None
    ref: tuple | None = None  # (lateral, vertical) metres, matching cfg.axes
    u: tuple | None = None  # (mag, az_deg, f_field_hz, throttle, f_hat_hz)
    frames: list | None = None  # one BGR/grey frame per rig camera
    stats: dict = field(default_factory=dict)


@dataclass
class Tick:
    """One frame's worth of the live stereo pipeline, for whoever is consuming it."""

    t: float                    # capture time, the estimator's clock
    xyz_mm: "np.ndarray | None" # datum frame, +z up; None when the filter has nothing
    pose: object | None         # full 5-DOF, for the viz and its overlays
    frames: list | None
    lost: int                   # cumulative lost-frame count
    viz: object                 # the LiveViz (or NullViz) to push to


class NullViz:
    """
    Every method a no-op, so a broken visualiser cannot become a broken flight.

        `make_viz` returns one of these instead of raising. The runner's call sites stay
        unconditional -- no `if viz is not None` scattered through the control loop, which
        is exactly the kind of branch that gets it wrong at 3am.
    """

    enabled = False
    thresh = None

    # The control knobs, answered as plain class attributes for the same reason
    # `thresh` is: a control loop polls them unconditionally, once per iteration, and
    # a dead viewer must have an answer rather than an AttributeError. The defaults
    # are the *inert* ones, not the useful ones -- `armed` False and `mag_max` 0.0
    # mean a session that lost its viser server flies nothing and energises no coil.
    # The coils have no firmware watchdog, so that is the only safe direction to fail.
    armed = False
    land = False
    mag_max = 0.0
    gain_scale = (1.0, 1.0)
    setpoint = (0.0, 0.0, 60.0)

    def push(self, *a, **kw):
        pass

    def set_zero(self, *a, **kw):
        pass

    def close(self):
        pass


def make_viz(enabled=True, **kw):
    """Build a `LiveViz`, or a `NullViz` if anything at all goes wrong."""

    if not enabled:
        return NullViz()
    try:
        return LiveViz(**kw)
    except Exception as e:  # noqa: BLE001 -- see the class docstring
        print(f"live_viz: disabled ({type(e).__name__}: {e})", file=sys.stderr)
        return NullViz()


def circle_points(center, normal, radius, n=RIM_SEGMENTS):
    """`n` points on the circle with the given centre, normal and radius."""

    n_hat = np.asarray(normal, dtype=np.float64)
    n_hat = n_hat / np.linalg.norm(n_hat)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_hat, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.asarray(center, dtype=np.float64)[None, :] + radius * (
        np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    )


def closed_loop_segments(pts):
    """(N,3) ring -> (N,2,3) line segments, wrapping the last point to the first."""

    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)


def wxyz_from_matrix(r):
    """3x3 rotation -> viser's (w, x, y, z) quaternion."""

    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(r, dtype=np.float64)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def trace_colors(n, cmap=TRACE_CMAP):
    """
    Per-segment RGB for an n-point trace, oldest to newest along the colormap.

        Returns (n-1, 2, 3) uint8 to match `add_line_segments`' per-endpoint colour
        array.  The n-1 is the whole point: one segment fewer than points, and getting
        it wrong drops the gradient silently rather than raising.
    """

    from matplotlib import colormaps

    if n < 2:
        return np.zeros((0, 2, 3), dtype=np.uint8)
    rgb = (colormaps[cmap](np.linspace(0.0, 1.0, n))[:, :3] * 255).astype(np.uint8)
    return np.stack([rgb[:-1], rgb[1:]], axis=1)


def load_rig(path=None):
    """
    The rig to draw, falling back to a one-camera rig at the origin.

        `StereoRig.load` refuses to guess an extrinsic, and rightly so -- but that is a
        rule for the estimator, not for a picture.  Drawing one frustum at the origin
        while the stereo calibration is still un-solved is honest; refusing to draw
        anything is not.
    """

    import rig as rigmod

    p = Path(path) if path else rigmod.DEFAULT_PATH
    if p.exists():
        return rigmod.StereoRig.load(p)
    print(f"live_viz: no rig at {p}, drawing a monocular rig at the origin")
    return rigmod.StereoRig.monocular()


def nominal_rig(elev_deg=(45.0, 45.0), azim_deg=(0.0, 90.0), range_mm=None):
    """
    The stereo rig as *planned* rather than as measured -- for laying one out.

        `StereoRig.from_spherical` defaults to the configuration the plan settles on:
        both cameras 45 degrees up, 90 degrees apart in bearing.  Drawing it needs no
        calibration, so you can see the geometry, the occlusion angles and how much of
        the flight volume each camera covers before building anything.

        `elev_deg=(45, -45)` is the mixed-hemisphere variant: optically identical, but
        it splits the occlusion risk between the takeoff rod below and the coils above.
    """

    import rig as rigmod

    kw = {} if range_mm is None else {"range_mm": float(range_mm)}
    return rigmod.StereoRig.from_spherical(
        elev_deg=tuple(elev_deg), azim_deg=tuple(azim_deg), **kw
    )


def axis_map(axes):
    """
    `("x", "-y")` -> ([0, 1], [+1, -1]): which world axis each controlled DOF is.

        The runner regulates two scalars it calls lateral and vertical, and
        `CameraSource._component` is the only place that says which world axes those
        are.  A reference plotted or drawn on the wrong axis is worse than none at
        all -- it looks like a tracking error -- so read the same spec rather than
        assuming the vertical DOF is world z.
    """

    idx = [{"x": 0, "y": 1, "z": 2}[a[-1]] for a in axes]
    sign = [-1.0 if a.startswith("-") else 1.0 for a in axes]
    return idx, sign


def ref_to_world(ref, idx, sign, measured=None):
    """
    Controlled reference (metres) -> a world point in mm.

        The third axis is uncontrolled, so it is held at the measurement: the marker
        then sits on the plane the loop actually regulates instead of floating at an
        arbitrary depth.  NaN there when there is no measurement, which draws nothing.
    """

    w = np.full(3, np.nan)
    if measured is not None:
        w[:] = np.asarray(measured, dtype=float)
    for k in range(len(idx)):
        w[idx[k]] = sign[k] * 1e3 * float(ref[k])
    return w


def up_direction(rig, zero=None):
    """
    Which world axis viser should treat as up.  Three frames get confused here.

        A `zeroing.Zero` datum wins outright: `Zero.apply` puts the reference normal on
        +z, so in a zeroed frame +z *is* the rotor axis, i.e. up.

        Otherwise it depends on where the rig came from, and the two producers disagree.
        `calibrate_stereo.py` and `StereoRig.monocular()` use camera A's optical frame,
        where +y is image-down and so up is **-y**.  `StereoRig.from_spherical()` builds
        a lab frame instead -- `spherical_position` measures elevation above the
        horizontal and bearing about world +z -- where up is **+z**.

        Guessing wrong lays the robot on its side, so read `meta["world_frame"]` (the key
        `rig.py` documents for exactly this) and fall back to geometry: camera A sitting
        at the identity extrinsic means the world *is* its optical frame.
    """

    if zero is not None and not zero.is_identity:
        # `_rotate_zero` leaves this when a `rotate` has turned the datum's frame; a
        # bare datum has not moved +z and does not set it.
        return zero.meta.get("up", "+z")
    frame = rig.meta.get("world_frame")
    if frame is None:
        frame = "camera_A" if np.allclose(rig.a.T_world_cam, np.eye(4)) else "lab"
    return "-y" if frame == "camera_A" else "+z"


def camera_in_datum(cam, zero):
    """
    A camera's `T_world_cam` expressed in the datum frame, mirroring `Zero.apply`.

        Without this the robot is drawn in the datum frame and the cameras in the
        camera frame -- two different origins in one scene, which looks plausible and
        is completely wrong.
    """

    T = np.asarray(cam.T_world_cam, dtype=np.float64).copy()
    T[:3, :3] = zero.R.T @ T[:3, :3]
    T[:3, 3] = zero.R.T @ (T[:3, 3] - zero.t)
    return T


#: The sidebar is a few hundred pixels wide. A frustum image was scaled down by the
#: renderer before it ever reached the browser; a GUI image is sent as authored, so
#: pushing 1280x800 twice at `image_hz` is 20 full-resolution JPEGs a second down the
#: websocket for a panel that shows none of it.
SIDEBAR_W = 480


def _fit_width(img, width=SIDEBAR_W):
    """Downscale to ``width`` if wider, keeping the aspect ratio. Never upscales."""

    if img.shape[1] <= width:
        return img
    h = max(1, int(round(img.shape[0] * width / img.shape[1])))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def smoothed(pose, state, last=None):
    """The pose to *draw*: filtered position and normal, everything else untouched.

    The estimator's answer is what gets logged; this is what gets rendered, and they are
    not the same job. Two things go wrong when the raw pose is drawn directly. Depth
    carries 2.4 mm of frame-to-frame jitter against 1.4 mm laterally (`pose/theory.md`
    S13), and depth is the axis a perspective view turns into apparent *size*, so the
    robot pulses. And a lost frame writes NaN into every series, which blanks the plot
    lines for as long as the gap lasts and then brings them back -- a flicker that says
    "tracking is broken" when one frame in three is simply hard.

    `PoseFilter` already coasts through those gaps and was already being computed and
    discarded. ``state`` is its ``(xyz, velocity, normal)``; ``last`` is the most recent
    real pose, which supplies the fields the filter has no opinion about when the current
    frame was lost. Returns ``None`` only when nothing is tracked at all.
    """

    import dataclasses

    from estimator import _angles_from_normal

    if state is None:
        return pose
    xyz, _vel, n = state
    src = pose if pose is not None else last
    if src is None:
        return None
    theta, phi = _angles_from_normal(np.asarray(n, dtype=float))
    fields = dict(xyz_mm=np.asarray(xyz, dtype=float), normal=np.asarray(n, dtype=float),
                  theta_deg=theta, phi_deg=phi)
    if pose is None:
        # Coasting. Keep the geometry, drop what belongs to a frame that never solved,
        # so the overlay says "no detection" instead of redrawing a stale ellipse.
        fields.update(per_view=(), extra={}, n_views=0,
                      discrepancy_mm=float("nan"), fit_rms_px=float("nan"))
    return dataclasses.replace(src, **fields)


def normal_segment_px(pose, cam, zero=None, length_mm=NORMAL_MM, centre_px=None):
    """
    The rotor axis as ``((x0, y0), (x1, y1))`` in one camera's pixels, or ``None``.

        **Three frames, and getting any of them wrong points the arrow somewhere
        plausible and false.** Poses are reported in the datum frame, so this undoes the
        datum to reach the world, then `Camera.to_camera` to reach this camera, and only
        then projects. Multiplying by `K` alone -- which is what this did -- is correct
        only while the world frame *is* that camera's optical frame and no datum is set:
        true once, and false since `prime_zero` began installing one by default. It also
        meant the arrow could only ever be drawn for camera A.

        ``centre_px`` pins the tail to a point already measured in the image, normally
        the fitted ellipse's centre. Worth doing even though the projected centre should
        land there: the ellipse is in *distorted* pixels and this projection is an ideal
        pinhole, and any residual error in the datum or the extrinsic shows up as the
        arrow floating off the robot, which reads as a direction error rather than the
        position error it is. Pinned, the arrow shows only what it can actually say --
        which way the axis points.
    """

    c = np.asarray(pose.xyz_mm, dtype=float).reshape(3)
    n = np.asarray(pose.normal, dtype=float).reshape(3)
    if zero is not None and not zero.is_identity:
        # `Zero.apply` reports R' (c - t); this is its inverse.
        c = zero.R @ c + zero.t
        n = zero.R @ n
    c_cam, n_cam = cam.to_camera(c, n)

    p0 = np.asarray(c_cam, dtype=float).reshape(3)
    p1 = p0 + length_mm * np.asarray(n_cam, dtype=float).reshape(3)
    if p0[2] <= 1e-6 or p1[2] <= 1e-6:
        return None
    uv = (cam.K @ np.stack([p0, p1]).T).T
    uv = uv[:, :2] / uv[:, 2:3]
    if centre_px is not None:
        uv = uv - uv[0] + np.asarray(centre_px, dtype=float).reshape(2)
    return tuple(map(tuple, uv))


class LiveViz:
    """
    A viser scene and GUI fed from a control loop, one bounded queue in between.

        Construct, `push()` per tick, `close()` at the end.  Everything costly -- mesh
        transforms, colormapping, JPEG encoding, websocket traffic -- happens on the
        render thread at `hz`, decoupled from however fast the loop runs, the same split
        `online_camera.ipynb` uses for its preview.
    """

    enabled = True

    def __init__(
        self,
        rig=None,
        port=8080,
        trace_len=2000,
        hz=30.0,
        image_hz=10.0,
        plot_len=900,
        radius_mm=None,
        axes=("x", "-y"),
        zero=None,
        label="flight",
        backgrounds=None,
        estimator=None,
    ):
        import viser
        from zeroing import Zero

        # The datum the estimator reports in. `Zero.load` returns identity when there
        # is no file, which is the un-zeroed camera-frame case, so this is safe to
        # call unconditionally.
        self.zero = Zero.load() if zero is None else zero
        # The estimator's own plates, keyed by camera name. Only the mask overlay
        # wants them, and only on frames that produced no pose -- see `_update_images`.
        self.backgrounds = dict(backgrounds or {})
        # Only for the threshold slider's starting value. The loop, not the render
        # thread, is what actually writes `estimator.thresh` -- see `thresh`.
        self.estimator = estimator
        self.axes = tuple(axes)
        self._ref_idx, self._ref_sign = axis_map(self.axes)
        self._plots_spec = plots_spec(self.axes)
        self.trace_len = int(trace_len)
        self.plot_len = int(plot_len)
        self._period = 1.0 / float(hz)
        self._image_period = 1.0 / float(image_hz) if image_hz else None

        self.rig = rig if rig is not None else load_rig()
        self._q = deque(maxlen=4096)
        self._stop = threading.Event()
        self._errors = 0
        self._t0 = None
        self._stats = {}

        # History lives here and is touched only by the render thread.
        self._trace = deque(maxlen=self.trace_len)
        self._hist = {k: deque(maxlen=self.plot_len) for k in _PLOT_KEYS}

        # The server binds its port in the constructor, so anything that fails after
        # this point has to hand the port back -- otherwise a single bad argument
        # leaves 8080 occupied and every later attempt degrades to NullViz.
        self.server = viser.ViserServer(port=int(port), label=label)
        self.up = up_direction(self.rig, self.zero)
        try:
            self.server.scene.set_up_direction(self.up)
            self._build_scene(radius_mm)
            self._build_gui()
        except Exception:
            self.server.stop()
            raise

        self._thread = threading.Thread(target=self._run, name="live_viz", daemon=True)
        self._thread.start()
        print(f"live_viz: http://localhost:{self.server.get_port()}")

    # ---- the only method the control loop touches -------------------------

    def push(self, pose=None, ref=None, u=None, frames=None, t=None, **stats):
        """
        Hand one tick to the renderer.  Never blocks, never raises.

            A bounded `deque.append` is a single atomic bytecode under the GIL, so this
            costs about a microsecond against a 33 ms control period.  If the renderer
            ever falls behind the deque drops its oldest entries rather than growing --
            a stalled browser must cost frames of trace, not memory.
        """

        self._q.append(
            Sample(
                t=time.monotonic() if t is None else t,
                pose=pose,
                ref=ref,
                u=u,
                frames=frames,
                stats=stats,
            )
        )

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.server.stop()
        except Exception:  # noqa: BLE001 -- shutdown must not raise into a landing
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- static scene -----------------------------------------------------

    def _build_scene(self, radius_mm):
        import render

        scene = self.server.scene
        mesh = render.load_mesh()
        self.radius_mm = float(radius_mm or render.RIM_RADIUS_MM)

        self._grid = scene.add_grid(
            "/grid",
            width=400.0,
            height=400.0,
            cell_size=10.0,
            section_size=50.0,
            plane="xy" if self.up == "+z" else "xz",
            position=(0.0, 0.0, 0.0),
        )
        scene.add_frame("/world", axes_length=40.0, axes_radius=0.8)

        self._drone = scene.add_mesh_trimesh("/drone", mesh)
        self._rim = scene.add_line_segments(
            "/rim",
            points=closed_loop_segments(circle_points((0, 0, 0), (0, 0, 1), self.radius_mm)),
            colors=C_ESTIMATE,
            thickness=2.0,
            thickness_units="screen",
        )
        self._normal = scene.add_line_segments(
            "/normal",
            points=np.zeros((1, 2, 3)),
            colors=C_NORMAL,
            thickness=3.0,
            thickness_units="screen",
        )
        self._trace_node = scene.add_line_segments(
            "/trace",
            points=np.zeros((1, 2, 3)),
            colors=np.zeros((1, 2, 3), dtype=np.uint8),
            thickness=2.5,
            thickness_units="screen",
        )
        self._ref_node = scene.add_icosphere(
            "/ref", radius=3.0, color=C_REF, opacity=0.65, visible=False
        )

        self._frustums = []
        for i, cam in enumerate(self.rig.cameras):
            h = 2.0 * cam.K[1, 2]
            w = 2.0 * cam.K[0, 2]
            # Poses arrive in the datum frame, so the cameras have to be drawn there
            # too -- otherwise the two halves of the scene use different origins.
            T = camera_in_datum(cam, self.zero)
            self._frustums.append(
                scene.add_camera_frustum(
                    f"/cam/{cam.name or i}",
                    fov=2.0 * math.atan(h / (2.0 * cam.K[1, 1])),
                    aspect=w / h,
                    scale=30.0,
                    color=C_CAM[i % len(C_CAM)],
                    wxyz=wxyz_from_matrix(T[:3, :3]),
                    position=T[:3, 3],
                )
            )

    def _datum_text(self):
        if self.zero is None or self.zero.is_identity:
            return f"none -- up is {self.up} (rig frame)"
        return f"set, up +z (spread {self.zero.meta.get('spread_deg', '?')} deg)"

    def set_zero(self, zero):
        """
        Install a datum on a scene that is already running.

            The live path cannot prime a datum before the viewer exists -- the robot is
            usually not in shot when the session starts, and waiting for it blocks
            startup entirely (see `prime_zero`). So the viewer comes up in whatever
            frame the rig provides and re-orients when the first stable poses arrive.

            Everything derived from the datum is redone: the up direction, the grid it
            implies, and the camera frustums, which `camera_in_datum` places in the
            datum frame so both halves of the scene share an origin.
        """

        self.zero = zero
        self.up = up_direction(self.rig, zero)
        try:
            self.server.scene.set_up_direction(self.up)
            self._grid.plane = "xy" if self.up == "+z" else "xz"
            for cam, frustum in zip(self.rig.cameras, self._frustums):
                T = camera_in_datum(cam, zero)
                frustum.wxyz = wxyz_from_matrix(T[:3, :3])
                frustum.position = T[:3, 3]
            if getattr(self, "_datum_label", None) is not None:
                self._datum_label.value = self._datum_text()
        except Exception as e:  # noqa: BLE001 -- see the class docstring
            print(f"live_viz: set_zero failed ({type(e).__name__}: {e})", file=sys.stderr)

    # ---- gui --------------------------------------------------------------

    def _build_gui(self):
        import viser.uplot as up

        gui = self.server.gui
        with gui.add_folder("Run"):
            # Which frame the scene is drawn in. Without this the two states are not
            # distinguishable by looking: un-zeroed, "up" is the rig's world frame,
            # which for a camera_A rig is that camera's optical frame and has nothing
            # to do with vertical, so the robot lies on its side and it reads as a
            # rotation bug rather than a datum that has not landed yet.
            self._datum_label = gui.add_text(
                "datum", self._datum_text(), disabled=True
            )
            self.paused = gui.add_checkbox("pause", False)
            self.follow = gui.add_checkbox("follow robot", False)
            self.trace_slider = gui.add_slider(
                "trace", min=100, max=5000, step=100, initial_value=self.trace_len
            )
            clear = gui.add_button("clear trace")

            @clear.on_click
            def _(_):
                self._trace.clear()

        with gui.add_folder("Segment"):
            # Polled by the loop, not pushed through a callback: every other control
            # here is read the same way, and a callback would be writing the
            # estimator's threshold from the render thread while the loop reads it.
            # The range brackets the default rather than being fixed at [100, 250].
            # It was fixed, and when `segment.THRESH` moved to 72 for the black
            # backdrop the slider refused its own initial value, `add_slider` raised,
            # and the whole GUI was disabled -- a constant in one module silently
            # turning off the viewer in another.
            lo, hi = self._thresh_range()
            self._thresh = gui.add_slider(
                "threshold", min=lo, max=hi, step=1,
                initial_value=self._default_thresh(),
            )
            self.show_mask = gui.add_checkbox("mask", True)

        with gui.add_folder("Control"):
            # Polled, like the threshold slider above and for the same reason: the
            # render thread owns the widget, the control loop reads it once per
            # iteration, and nothing here ever writes into the loop's state from
            # another thread. `land` is the single exception -- see below.
            self._armed = gui.add_checkbox("armed", False)
            self._sp_x = gui.add_slider(
                "x (mm)", min=-60.0, max=60.0, step=1.0, initial_value=0.0)
            self._sp_y = gui.add_slider(
                "y (mm)", min=-60.0, max=60.0, step=1.0, initial_value=0.0)
            self._sp_z = gui.add_slider(
                "z (mm)", min=0.0, max=200.0, step=1.0, initial_value=60.0)
            self._gain_lat = gui.add_slider(
                "gain lateral", min=0.0, max=3.0, step=0.05, initial_value=1.0)
            self._gain_vert = gui.add_slider(
                "gain vertical", min=0.0, max=3.0, step=0.05, initial_value=1.0)
            self._mag_max = gui.add_slider(
                "mag max", min=0.0, max=0.8, step=0.01, initial_value=0.0)
            # Set before the button exists: the click callback runs on the render
            # thread and may fire the instant the widget is created.
            self._land = False
            land = gui.add_button("land")

            @land.on_click
            def _(_):
                # The one callback here, because a button press is an event and
                # polling would miss it between two frames. It writes a bool and
                # nothing else; the getter clears it. See `land`.
                self._land = True

        with gui.add_folder("Now"):
            self.readout = gui.add_markdown("waiting for the first pose")

        # In the sidebar rather than on the frustums. Hung in the scene they face
        # wherever the camera faces, so reading them means orbiting to each one in turn
        # and losing sight of the robot -- and they are the one panel you want to keep
        # an eye on while the robot moves.
        self._views = {}
        with gui.add_folder("Cameras"):
            blank = np.zeros((90, 144, 3), np.uint8)
            for i, cam in enumerate(self.rig.cameras if self.rig else ()):
                self._views[i] = gui.add_image(blank, label=f"camera {cam.name or i}",
                                               format="jpeg", jpeg_quality=75)

        empty = (np.zeros(1),)
        self._plots = {}
        for key, title, series in self._plots_spec:
            self._plots[key] = gui.add_uplot(
                data=empty + tuple(np.zeros(1) for _ in series),
                series=(up.Series(label="t (s)"),)
                + tuple(
                    up.Series(label=lbl, stroke=col, width=1.6) for lbl, col in series
                ),
                title=title,
                legend=up.Legend(show=True),
                aspect=1.6,
            )

    # ---- render thread ----------------------------------------------------

    def _run(self):
        t_image = 0.0
        while not self._stop.is_set():
            t_tick = time.monotonic()
            try:
                batch = self._drain()
                if batch and not self.paused.value:
                    self._ingest(batch)
                    self._update_scene(batch[-1])
                    self._update_plots()
                    if self._image_period and t_tick - t_image >= self._image_period:
                        t_image = t_tick
                        self._update_images(batch[-1])
            except Exception:  # noqa: BLE001 -- a dead view must not end a flight
                self._errors += 1
                if self._errors == 1:
                    traceback.print_exc()
                    print("live_viz: continuing despite the error above", file=sys.stderr)
            self._stop.wait(max(0.0, self._period - (time.monotonic() - t_tick)))

    def _drain(self):
        """Take everything queued. `popleft` on an empty deque is the loop's exit."""

        out = []
        while True:
            try:
                out.append(self._q.popleft())
            except IndexError:
                return out

    def _ingest(self, batch):
        """Fold a batch of samples into the trace and the plot histories."""

        for s in batch:
            if self._t0 is None:
                self._t0 = s.t
            t = s.t - self._t0
            p, ref, u = s.pose, s.ref, s.u
            # Anything without `xyz_mm` is treated as a lost frame rather than an
            # error: a dropped detection arrives as None every few frames, and one
            # malformed sample must not cost the whole batch its trace.
            raw = getattr(p, "xyz_mm", None)
            xyz = np.asarray(raw, dtype=float) if raw is not None else None
            if xyz is not None:
                self._trace.append(xyz)

            # The reference arrives in metres on the runner's (lateral, vertical)
            # axes; everything here is world mm. Convert once, on the right axes.
            rw = (
                ref_to_world(ref, self._ref_idx, self._ref_sign)
                if ref is not None
                else np.full(3, np.nan)
            )

            self._hist["t"].append(t)
            for k, v in (
                ("x", xyz[0] if xyz is not None else np.nan),
                ("y", xyz[1] if xyz is not None else np.nan),
                ("z", xyz[2] if xyz is not None else np.nan),
                ("ref_x", rw[0]),
                ("ref_y", rw[1]),
                ("ref_z", rw[2]),
                ("theta", getattr(p, "theta_deg", np.nan) if p is not None else np.nan),
                ("phi", getattr(p, "phi_deg", np.nan) if p is not None else np.nan),
                ("psi", getattr(p, "psi_deg", np.nan) if p is not None else np.nan),
                ("mag", u[0] if u else np.nan),
                ("az", u[1] if u and len(u) > 1 else np.nan),
                ("freq", u[2] if u and len(u) > 2 else np.nan),
                ("throttle", u[3] if u and len(u) > 3 else np.nan),
                # Optional fifth slot. A loop with no adaptive estimate sends a
                # 4-tuple and gets a NaN line rather than an IndexError.
                ("f_hat", u[4] if u and len(u) > 4 else np.nan),
                ("rms", getattr(p, "fit_rms_px", np.nan) if p is not None else np.nan),
                (
                    "margin",
                    getattr(p, "ambiguity_margin_deg", np.nan) if p is not None else np.nan,
                ),
                (
                    "discrepancy",
                    getattr(p, "discrepancy_mm", np.nan) if p is not None else np.nan,
                ),
            ):
                self._hist[k].append(float(v))

        # Whatever the caller passed as keywords -- lost frames, dropped frames, dt --
        # goes to the readout verbatim, so a new counter needs no change here.
        self._stats = batch[-1].stats

        if self.trace_slider.value != self._trace.maxlen:
            self._trace = deque(self._trace, maxlen=int(self.trace_slider.value))

    def _update_scene(self, last):
        import render

        p = last.pose if getattr(last.pose, "xyz_mm", None) is not None else None
        if p is not None:
            xyz = np.asarray(p.xyz_mm, dtype=float)
            m = render.pose_matrix(p.theta_deg, p.phi_deg, xyz)
            q = wxyz_from_matrix(m[:3, :3])
            self._drone.wxyz, self._drone.position = q, xyz
            self._rim.wxyz, self._rim.position = q, xyz
            self._normal.points = np.stack(
                [xyz, xyz + NORMAL_MM * np.asarray(p.normal, dtype=float)]
            )[None]
            if self.follow.value:
                for client in self.server.get_clients().values():
                    client.camera.look_at = xyz

        if last.ref is not None:
            w = ref_to_world(
                last.ref,
                self._ref_idx,
                self._ref_sign,
                measured=p.xyz_mm if p is not None else np.zeros(3),
            )
            self._ref_node.position = w
            self._ref_node.visible = bool(np.isfinite(w).all())

        pts = np.asarray(self._trace, dtype=float)
        if len(pts) >= 2:
            self._trace_node.points = np.stack([pts[:-1], pts[1:]], axis=1)
            self._trace_node.colors = trace_colors(len(pts))

    def _update_plots(self):
        t = np.asarray(self._hist["t"], dtype=float)
        if len(t) < 2:
            return
        for key, _, series in self._plots_spec:
            self._plots[key].data = (t,) + tuple(
                np.asarray(self._hist[k], dtype=float) for k, _ in series
            )
        self.readout.content = self._readout()

    def _readout(self):
        def last(k):
            h = self._hist[k]
            return h[-1] if h else float("nan")

        return (
            f"**xyz** {last('x'):+8.1f} {last('y'):+8.1f} {last('z'):+8.1f} mm\n\n"
            f"**tilt** {last('theta'):5.1f}°  **az** {last('phi'):+6.1f}°  "
            f"**psi** {last('psi'):+6.1f}°\n\n"
            f"**cmd** mag {last('mag'):+.3f}  az {last('az'):.0f}°  "
            f"f {last('freq'):.2f} Hz\n\n"
            f"**fit** {last('rms'):.2f} px  **margin** {last('margin'):.1f}°\n\n"
            f"`{self._status}`"
        )

    @property
    def _status(self):
        h = self._hist["t"]
        hz = (len(h) - 1) / max(h[-1] - h[0], 1e-9) if len(h) > 1 else 0.0
        extra = "  ".join(f"{k} {v}" for k, v in sorted(self._stats.items()))
        return (
            f"{hz:5.1f} Hz   trace {len(self._trace)}   "
            f"viz errors {self._errors}   {extra}"
        )

    @staticmethod
    def _default_thresh():
        """The level the segmenter would use on its own, so the slider starts neutral."""

        import segment

        return int(segment.DARK_THRESH if segment.APPEARANCE == "dark"
                   else segment.THRESH)

    @classmethod
    def _thresh_range(cls):
        """Slider bounds that always contain the default, whatever it has been set to."""

        d = cls._default_thresh()
        return min(40, d), max(250, d)

    @property
    def thresh(self):
        """
        The slider's level, for the loop to hand the estimator each frame.

                A property rather than a bare attribute so `NullViz` can answer ``None``
                -- which is also what the estimator wants when there is no GUI, since
                ``thresh=None`` means "use the appearance's own default".
        """

        return int(self._thresh.value)

    @property
    def armed(self):
        """Whether the loop may command the coils at all."""

        return bool(self._armed.value)

    @property
    def setpoint(self):
        """``(x_mm, y_mm, z_mm)`` in the datum frame, where +z is the rotor axis."""

        return (float(self._sp_x.value), float(self._sp_y.value), float(self._sp_z.value))

    @property
    def gain_scale(self):
        """``(lateral, vertical)`` multipliers on the loop's gain matrix, 1.0 = as designed."""

        return (float(self._gain_lat.value), float(self._gain_vert.value))

    @property
    def mag_max(self):
        """Ceiling on the lateral field magnitude. **Zero by default, i.e. disabled.**"""

        return float(self._mag_max.value)

    @property
    def land(self):
        """True once per button press, then False again -- a one-shot latch."""

        pressed, self._land = self._land, False
        return pressed

    def _update_images(self, last):
        """Put each camera's annotated view in the sidebar."""

        import segment

        if not last.frames:
            return
        p = last.pose
        per_view = getattr(p, "per_view", ()) if p is not None else ()
        for i, cam in enumerate(self.rig.cameras):
            frame = last.frames[i] if i < len(last.frames) else None
            if frame is None:
                continue
            if per_view and i < len(per_view):
                seg = per_view[i]
            else:
                seg = p.extra.get("segmentation") if p is not None else None
            mask = self.show_mask.value
            if mask and seg is None:
                # The frames worth looking at are the ones with no pose, and those
                # carry no `Segmentation` to take a mask from. Re-thresholding costs
                # one pass over a frame the pipeline has already given up on.
                mask = segment.threshold_mask(
                    frame, thresh=self.thresh,
                    background=self.backgrounds.get(cam.name),
                )
            # Both views, not just camera A: the arrow is now projected through each
            # camera's own extrinsic, so it means something in either. Tail pinned to
            # that view's own ellipse centre.
            npx = None
            if p is not None:
                npx = normal_segment_px(
                    p, cam, self.zero,
                    centre_px=seg.ellipse[0] if seg is not None else None,
                )
            out = segment.draw(frame, seg, normal_px=npx, mask=mask)
            rgb = out[:, :, ::-1] if out.ndim == 3 else out
            view = self._views.get(i)
            if view is not None:
                view.image = _fit_width(rgb, SIDEBAR_W)


_XYZ = ("x", "y", "z")
_C_XYZ = ("#e0533d", "#2e9bd6", "#e8b33c")
_C_XYZ_REF = ("#a03027", "#1c6790", "#9a7420")


def plots_spec(axes):
    """
    Plot layout: [(key, title, [(history key, stroke)])], given the controlled axes.

        The reference is drawn on the same axis as the measurement it is steering, so
        `x` and `x_ref` are the same line when tracking is perfect.  Which axes those
        are is not fixed -- hence the argument -- and the uncontrolled one gets no
        reference series rather than a flat all-NaN one cluttering the legend.
    """

    idx, _ = axis_map(axes)
    return [
        (
            "pos",
            "position (mm)",
            [(a, _C_XYZ[i]) for i, a in enumerate(_XYZ)]
            + [(f"ref_{_XYZ[i]}", _C_XYZ_REF[i]) for i in idx],
        ),
        (
            "ang",
            "angle (deg)",
            [("theta", "#e0533d"), ("phi", "#2e9bd6"), ("psi", "#e8b33c")],
        ),
        # Distinct strokes so a saturating actuator is obvious at a glance. `f_hat`
        # shares the plot with `freq` deliberately: it is the frequency at which the
        # adaptive estimate says lift balances weight, and the only useful thing to
        # read off it is the gap to the frequency actually being commanded. On its
        # own axis that gap becomes an eyeball comparison between two plots.
        (
            "cmd",
            "command",
            [
                ("mag", C_CMD[0]),
                ("az", C_CMD[1]),
                ("freq", C_CMD[2]),
                ("throttle", C_CMD[3]),
                ("f_hat", C_CMD[4]),
            ],
        ),
        (
            "health",
            "estimator health",
            [("rms", "#e0533d"), ("margin", "#2e9bd6"), ("discrepancy", "#e8b33c")],
        ),
    ]


# The superset, so history buffers exist whatever `axes` turns out to be.
_PLOT_KEYS = ["t", "ref_x", "ref_y", "ref_z"] + [
    k for _, _, series in plots_spec(("x", "-y")) for k, _ in series
]


# ---- entry points ---------------------------------------------------------


def _fake_pose(t, hover_mm=60.0):
    """
    A wobbling hover near the origin -- enough shape to see every element move.

        Written in the **datum** frame, the one a zeroed estimator reports in: +z is the
        rotor axis, so a level robot has ``normal = +z`` and ``theta = 0``.  That is what
        makes it render upright.  Feeding these poses to a viewer told that up is -y (the
        raw optical frame) is what lays the robot on its side.
    """

    from estimator import Pose, _angles_from_normal

    xyz = np.array(
        [40.0 * math.cos(t), 40.0 * math.sin(t), hover_mm + 20.0 * math.sin(0.5 * t)]
    )
    tilt = math.radians(12.0 * math.sin(0.7 * t))
    n = np.array([math.sin(tilt) * math.cos(t), math.sin(tilt) * math.sin(t), math.cos(tilt)])
    theta, phi = _angles_from_normal(n)
    return Pose(
        t=t,
        frame_index=0,
        xyz_mm=xyz,
        normal=n,
        theta_deg=theta,
        phi_deg=phi,
        psi_deg=phi,
        ellipse=((0.0, 0.0), (1.0, 1.0), 0.0),
        area_px=0.0,
        fit_rms_px=0.4 + 0.1 * math.sin(3 * t),
        ambiguity_margin_deg=20.0,
        n_solutions=2,
        jump_deg=0.0,
        t_seg_ms=1.0,
        t_est_ms=1.0,
    )


def demo(port=8080, seconds=None, hz=30.0, rig=None):
    """
    Drive the viewer from a synthetic hover. No camera, no serial.

        Defaults to the nominal stereo rig, so this doubles as the rig-layout preview:
        two cameras where `StereoRig.from_spherical` puts them, aimed at a robot
        hovering over the origin.  Its world is the lab frame, +z up.
    """

    import zeroing

    viz = make_viz(
        port=port,
        rig=rig if rig is not None else nominal_rig(),
        axes=("x", "z"),  # lab frame: lateral is +x, height is +z
        zero=zeroing.Zero.identity(),
        label="live_viz demo",
    )
    t0 = time.monotonic()
    try:
        while seconds is None or time.monotonic() - t0 < seconds:
            t = time.monotonic() - t0
            viz.push(
                _fake_pose(t),
                ref=(0.04 * math.cos(t), 0.060),
                # Five channels, the last a fake adaptive hover-frequency estimate
                # converging on the commanded one, so --demo exercises that series.
                u=(0.3 * math.sin(t), 0.0, 140.0 + 5.0 * math.sin(0.4 * t), 80.0,
                   140.0 + 12.0 * math.exp(-0.2 * t)),
            )
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()


def replay(path, port=8080, speed=1.0):
    """Play a `recorder.py` CSV back through the viewer at `speed` x real time."""

    import pandas as pd
    from estimator import Pose

    df = pd.read_csv(path, comment="#")
    viz = make_viz(port=port, label=f"replay {Path(path).name}")
    # np.array, not to_numpy: pandas hands back a read-only view of its own block.
    t = np.array(df["t_capture"], dtype=float)
    t -= t[0]
    try:
        t_start = time.monotonic()
        for i, row in enumerate(df.itertuples(index=False)):
            wait = t[i] / speed - (time.monotonic() - t_start)
            if wait > 0:
                time.sleep(wait)
            viz.push(
                Pose(
                    t=t[i],
                    frame_index=int(row.frame),
                    xyz_mm=np.array([row.x_mm, row.y_mm, row.z_mm]),
                    normal=np.array([row.nx, row.ny, row.nz]),
                    theta_deg=row.theta_deg,
                    phi_deg=row.phi_deg,
                    psi_deg=row.psi_deg,
                    ellipse=((row.ellipse_cx, row.ellipse_cy), (row.major_px, row.minor_px), row.ellipse_deg),
                    area_px=row.area_px,
                    fit_rms_px=row.fit_rms_px,
                    ambiguity_margin_deg=row.ambiguity_margin_deg,
                    n_solutions=int(row.n_solutions),
                    jump_deg=row.jump_deg,
                    t_seg_ms=row.t_seg_ms,
                    t_est_ms=row.t_est_ms,
                ),
                t=t[i],
            )
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()


def from_camera(source="camera:0", width=1280, height=800, port=8080):
    """Live vision with no controller in the loop -- the pose pipeline on its own."""

    import sources
    from estimator import PoseEstimator, load_intrinsics

    K, dist = load_intrinsics()
    cam = sources.open_source(source, width=width, height=height)
    est = PoseEstimator(camera_matrix=K, dist_coeffs=dist)
    viz = make_viz(port=port, label=f"live {source}", estimator=est)
    lost = 0
    try:
        while True:
            item = cam.read()
            if item is None:
                break
            t_cap, frame = item
            if viz.enabled:
                est.thresh = viz.thresh
            pose = est.update(frame, t=t_cap)
            lost += pose is None
            viz.push(pose, frames=[frame], t=t_cap, lost=lost)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()
        cam.close()


#: Consecutive read timeouts before a live session gives up. Each one is `MonoCamera`'s
#: own 2 s wait, so this is ~10 s of silence -- long enough to ride out a USB stall,
#: short enough that an unplugged camera does not hang the notebook.
MAX_READ_MISSES = 5

#: Frames of nothing-at-all before the viewer explains itself. A live run that shows an
#: empty scene and says nothing is the worst failure mode here: the cameras are fine, the
#: loop is running, and the only signal is an absence.
DIAGNOSE_AFTER = 60


def _diagnose_silence(frames, backgrounds, tags):
    """Say why nothing is being detected, once, after `DIAGNOSE_AFTER` empty frames."""

    import segment

    print("\nno detections yet. What the segmenter can see:", file=sys.stderr)
    for tag, f in zip(tags, frames):
        g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        region = segment.valid_region(g, bg=backgrounds.get(tag))
        cover = "nothing" if region is None else f"{100 * region.mean() / 255:.0f}% of frame"
        inside = "" if region is None else (
            f", darkest pixel inside it {int(g[region > 0].min())}"
            if (region > 0).any() else "")
        print(f"  camera {tag}: valid region {cover}{inside}", file=sys.stderr)
    if not backgrounds:
        print("  With no plate the region is the bright backdrop only. A robot posed over "
              "a dark\n  gap is outside it. Either put the backdrop behind the robot, or "
              "take the robot\n  out of frame once and run background.capture_stereo(), "
              "then pass backgrounds='saved'.", file=sys.stderr)
    else:
        print("  A plate is in use, so the region is what differs from it. If that is most "
              "of the\n  frame the plate is stale -- the scene moved since it was taken.",
              file=sys.stderr)


def stereo_frames(specs="camera:0,camera:1", rig_path=None, width=1280, height=800,
                  port=8080, rotate180=True, backgrounds=None, zero="auto", flip=False,
                  rotate=None, axes=("x", "y", "z"), label=None):
    """Generator over the live stereo pipeline. Yields one `Tick` per frame."""

    import sources
    from filter import PoseFilter

    rig, est = _stereo_estimator(rig_path)
    cams = [x.strip() for x in specs.split(",")]
    try:
        src = sources.open_stereo(cams, max_skew_s=None, width=width, height=height,
                                  grayscale=True, rotate180=rotate180)
    except OSError as e:
        # "could not open camera index 1" does not say whether the camera is unplugged,
        # held by another process, or simply enumerated somewhere else today.
        from elp import probe_indices
        found = probe_indices()
        raise SystemExit(
            f"{e}\n\nCameras that open and deliver a frame right now: "
            + (", ".join(f"index {i} at {w}x{h}" for i, w, h in found) or "none")
            + f"\nAsked for {specs}. The ELP OV9281 reports 1280x800; a 1920x1080 device "
            f"is the built-in FaceTime camera.\nUSB cameras enumerate before it, so two "
            f"ELPs plugged in are normally 0 and 1.") from e
    tags = [c.name or str(i) for i, c in enumerate(rig.cameras)]
    if backgrounds is None or backgrounds == "saved":
        # Default to the saved plates when they exist. Having to name them was a trap:
        # `capture_stereo` writes them, the bare call ignored them, and the only symptom
        # was a viewer that detected nothing.
        from background import load_stereo
        required = backgrounds == "saved"
        backgrounds = load_stereo(tags)
        if backgrounds:
            print(f"using saved background plates for {', '.join(sorted(backgrounds))}")
        elif required:
            raise SystemExit(
                "no saved plates. Take the robot out of frame and run\n"
                "    python -c \"import sys; sys.path.insert(0, 'pose'); "
                "import background; background.capture_stereo()\"")
    elif backgrounds == "running":
        # No plate to shoot and nothing to hold still for: `RunningPlate` is a running
        # median that converges on whatever does not move, so it builds the plate out of
        # the live stream while the robot flies through it. Costs the first
        # `background.WARMUP_FRAMES` frames, which it reports as "no plate" so the
        # top-hat runs unaided until then.
        import background as bgmod
        backgrounds = {t: bgmod.RunningPlate() for t in tags}
        print("building background plates from the stream as it runs")
    elif backgrounds == "auto":
        import background as bgmod
        print("building background plates -- move the robot around for a few seconds")
        backgrounds = bgmod.from_stereo_stream(src, tags=tags)
        item = src.read()
        if item is not None and bgmod.plate_holds_still_subject(backgrounds, item[1], tags):
            raise SystemExit(
                "the plates came out with the robot still in them: a temporal median "
                "removes\nwhat moves, and nothing moved. Either move the robot while "
                "the plates build, or\ntake it out of frame once and use "
                "backgrounds='saved' (background.capture_stereo).")
    elif backgrounds is not None and not isinstance(backgrounds, dict):
        from background import load_for_flight
        backgrounds = load_for_flight(backgrounds)
    if not backgrounds:
        # Not `backdrop_mask`. A running median needs nothing shot in advance and beats
        # the backdrop fallback outright, so an absent plate is no longer a degraded mode.
        import background as bgmod
        backgrounds = {t: bgmod.RunningPlate() for t in tags}
        print("no saved plates: building them from the stream as it runs "
              "(background.capture_stereo() if you would rather shoot one)",
              file=sys.stderr)
    est.backgrounds = dict(backgrounds)

    filt = PoseFilter()

    # **Not primed before the viewer exists.** A live session normally starts before
    # the robot is in shot, and waiting for a stable pose there blocks `make_viz`
    # outright -- no viser URL, no output, nothing to interrupt. The datum is
    # collected from the running loop instead and installed when it arrives.
    auto_zero = zero == "auto"
    if not auto_zero and zero is not None:
        est.zero = zero
    pending = _DatumPrimer(flip=flip, rotate=rotate) if auto_zero else None

    viz = make_viz(port=port, rig=rig, radius_mm=est.radius_mm,
                   label=f"live stereo {specs}" if label is None else label,
                   axes=axes, backgrounds=est.backgrounds,
                   estimator=est, zero=est.zero)
    if pending is not None:
        print("live_viz: datum not set yet -- the scene re-orients once the robot "
              "holds still enough to fix one")
    lost, last_seen, misses = 0, None, 0
    try:
        while True:
            item = src.read()
            if item is None:
                # `MonoCamera.read` returns None after a 2 s timeout, and a live session
                # must not end on one. A USB stall, another process touching the device,
                # a dropped frame -- all transient, all indistinguishable here from an
                # unplugged camera, and treating the first as the second used to close
                # the viewer with nothing said but "(viser) Server stopped".
                misses += 1
                if misses < MAX_READ_MISSES:
                    print(f"camera read timed out ({misses}/{MAX_READ_MISSES}), retrying",
                          file=sys.stderr)
                    continue
                print(f"source stopped: {MAX_READ_MISSES} reads in a row timed out. The "
                      f"camera was unplugged, or another process has it.", file=sys.stderr)
                break
            misses = 0
            t_cap, frames = item
            # Same poll as the replay loop: the slider is only useful live if the
            # estimator reads it, and this is the loop `run.ipynb` actually calls.
            if viz.enabled:
                est.thresh = viz.thresh
            pose = est.update(frames, t=t_cap,
                              stamps=getattr(src, "last_stamps", None) or None,
                              motion=filt.pos if filt.pos.initialised else None)
            state = filt.update(pose, t=t_cap)
            lost += pose is None
            if pose is not None:
                last_seen = pose
                if pending is not None:
                    # Offered raw, before the datum exists -- once it is installed the
                    # estimator reports in it and the primer has already stopped.
                    z = pending.offer(pose.xyz_mm, pose.normal)
                    if z is not None:
                        est.zero = z
                        viz.set_zero(z)
                        filt.reset()
                        print(f"live_viz: datum set, spread "
                              f"{z.meta['spread_deg']} deg -- scene re-oriented")
            elif last_seen is None and lost == DIAGNOSE_AFTER:
                _diagnose_silence(frames, est.backgrounds, tags)
            # What the *view* gets, not what the log gets -- see `smoothed`. It is
            # also what carries `xyz_mm` through a coasted frame, so the controller
            # and the picture agree about where the robot is on a lost frame instead
            # of the loop seeing a hole the viewer has already filled in.
            drawn = smoothed(pose, state, last_seen)
            raw = getattr(drawn, "xyz_mm", None)
            yield Tick(
                t=t_cap,
                xyz_mm=None if raw is None else np.asarray(raw, dtype=float),
                pose=drawn,
                frames=list(frames),
                lost=lost,
                viz=viz,
            )
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()
        src.close()


def from_stereo(*a, **kw):
    """Live stereo tracking: two cameras, the measured rig, and the pose filter.

    Two differences from `from_camera`, both of them the point of having a rig.  A second
    view kills the monocular tilt ambiguity outright, and the fusion is information
    weighted, so each camera dominates the world directions it measures *laterally*
    (`pose/theory.md` section 12.6).

    The frames are not simultaneous and this does not pretend otherwise.  The per-camera
    capture times go to `stereo.fuse`, together with the filter's velocity and its
    covariance, which moves each view to a common instant and inflates its covariance to
    pay for the move.  At hover that correction is worth more than it sounds: the skew
    displacement is 1.5-2.2x the lateral noise floor (`pose/theory.md` section 17).

    No rejection sampling on skew here.  That is the calibration path's trick and it costs
    seven frames out of eight, which a control loop cannot pay.

    ``backgrounds`` defaults to whatever `background.capture_stereo` last wrote, and
    falls back to ``"running"`` when there is nothing saved. Otherwise: ``"running"`` for
    a `background.RunningPlate` per camera, built from the stream as it runs and needing
    nothing shot in advance; ``"auto"`` for a temporal median up front (move the robot
    while it builds); ``{camera name: plate}``; or a flight directory to take them from.
    A shot plate is only good while the cameras *and the scene* hold still -- one taken
    from a flight two days earlier differed from the live view on 44% of the frame,
    because the foam had been moved -- which is the argument for ``"running"``.

    Open loop: everything above is `stereo_frames`, and this is the whole difference
    between watching a flight and flying one -- a push with no reference and no command.
    Arguments are `stereo_frames`' arguments.
    """

    for tick in stereo_frames(*a, **kw):
        tick.viz.push(tick.pose, frames=tick.frames, t=tick.t, lost=tick.lost)


def _stereo_estimator(rig_path=None, backgrounds=None):
    """``(rig, estimator)`` for the measured rig, or a clear refusal.

    ``backgrounds`` is one empty-rig plate per camera, or ``"running"`` for a
    `background.RunningPlate` each. Without them the segmenter falls back to
    `segment.backdrop_mask`, which cannot separate the robot's dark rim from a dark gap
    in the scene behind it -- worth 24% of frames against 59% on this bench.

    The radius is `estimator.RADIUS_BENCH_MM`, **not** `estimator.RADIUS_MM`.
    That constant is `RADIUS_BY_APPEARANCE[APPEARANCE]`, and for `bright` it is 10.2446,
    which was fitted on *renders* -- `estimator.py` says in as many words to pass this
    explicitly for the bench rig. This path did not, so every viser session ran the rig
    half a percent large. A radius error is a systematic depth offset along each
    camera's own axis and the axes point different ways, so it lands straight on
    cross-view discrepancy: at 10.20 only 66% of strong-fit frames come under the 5 mm
    gate, against 100% at 9.95.
    """

    import rig as rigmod
    from estimator import RADIUS_BENCH_MM
    from shape import CentreCalibration, TiltCalibration
    from stereo import StereoPoseEstimator

    p = Path(rig_path) if rig_path else rigmod.DEFAULT_PATH
    if not p.exists():
        raise SystemExit(
            f"no rig at {p}. Stereo needs the measured extrinsic. Run\n"
            f"    python calib/calibrate.py\n"
            f"and let it pass its acceptance gate. A failed calibration writes nothing, "
            f"deliberately: a bad extrinsic gives poses that are smooth, plausible and "
            f"wrong.")
    rig = rigmod.StereoRig.load(p)
    if backgrounds == "running":
        import background as bgmod
        backgrounds = {c.name: bgmod.RunningPlate() for c in rig.cameras}
    return rig, StereoPoseEstimator(rig, tilt_cal=TiltCalibration.load(),
                                    centre_cal=CentreCalibration.load(),
                                    radius_mm=RADIUS_BENCH_MM,
                                    backgrounds=backgrounds)


# How many consecutive accepted poses must agree before one is taken as the datum, and
# how closely. The datum becomes a fixed bias on the whole run, so it is worth waiting
# for: 12 frames is a fifth of a second at 60 fps, and 3 degrees is comfortably above
# the per-view scatter (`pose/theory.md` 12.12 puts it at 2.6 deg at 40-50 deg tilt,
# which stereo fusion takes to ~0.3) while well below any real attitude change.
AUTO_ZERO_FRAMES = 12
AUTO_ZERO_TOL_DEG = 3.0

# How long `prime_zero` may draw on a source before giving up. 600 frames is ten
# seconds at 60 fps: long enough to cover a robot that is in shot but briefly lost,
# short enough that a session started before takeoff is not mistaken for a crash.
# The live path also stops at the first frame, and installs the datum later instead.
AUTO_ZERO_MAX_FRAMES = 600


def _rotation(spec):
    """
    ``(axis, degrees)`` or a 3x3 matrix -> a 3x3 rotation, or ``None``.

        Right-handed about the named axis, which is the only convention numpy and scipy
        offer and not the only one a viewer suggests: "counter-clockwise" depends on
        which side you look from, so the sign is worth checking against the scene rather
        than reasoning about. `rotate=("y", -90)` and `("y", 90)` differ.
    """

    if spec is None:
        return None
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        from scipy.spatial.transform import Rotation

        return Rotation.from_euler(spec[0], float(spec[1]), degrees=True).as_matrix()
    return np.asarray(spec, dtype=np.float64).reshape(3, 3)


def _rotate_zero(zero, spec):
    """
    Turn the datum's frame by ``spec``, leaving the origin where it is.

        `Zero.apply` reports ``R' (c - t)``, so to express poses in a frame rotated by
        ``S`` -- new coordinates ``S x`` -- the datum's rotation becomes ``R S'``. Doing
        it here rather than in the scene keeps the poses, the grid and the camera
        frustums in one frame; rotating only the view would separate them.
    """

    S = _rotation(spec)
    if S is None:
        return zero
    from zeroing import Zero

    # Record where the rotation *puts* up, not just that one was applied. The datum
    # alone always puts the rotor axis on +z, so `up_direction` used to answer "+z" for
    # any datum -- and then a `rotate` moved the poses out from under it, leaving viser
    # drawing an xy grid and calling z up while the robot's axis sat on x. A z motion
    # then reads as an x motion, which is exactly how it looked.
    up = S @ np.array([0.0, 0.0, 1.0])
    i = int(np.argmax(np.abs(up)))
    return Zero(R=zero.R @ S.T, t=zero.t, psi_ref_deg=zero.psi_ref_deg,
                meta={**zero.meta, "rotate": str(spec),
                      "up": f"{'+' if up[i] >= 0 else '-'}{'xyz'[i]}"})


class _DatumPrimer:
    """
    `prime_zero`'s rule, fed one pose at a time from a loop that is already running.

        Same test -- `AUTO_ZERO_FRAMES` poses agreeing to `AUTO_ZERO_TOL_DEG` -- but
        pull instead of push, because the live path cannot afford to own the read loop
        before the viewer exists. `offer()` returns a `Zero` the first time the run
        settles and ``None`` every other time, so the caller can install it once.
    """

    def __init__(self, n=AUTO_ZERO_FRAMES, tol_deg=AUTO_ZERO_TOL_DEG, flip=False,
                 rotate=None):
        self.n, self.tol_deg, self.flip = n, tol_deg, flip
        self.rotate = rotate
        self._poses = deque(maxlen=n)
        self.done = False

    def offer(self, center, normal):
        from zeroing import Zero

        if self.done:
            return None
        self._poses.append((np.asarray(center, float), _unit_np(normal)))
        if len(self._poses) < self.n:
            return None
        mean_n = _unit_np(np.mean([b for _, b in self._poses], axis=0))
        spread = max(
            math.degrees(math.acos(min(1.0, abs(float(b @ mean_n)))))
            for _, b in self._poses
        )
        if spread > self.tol_deg:
            return None
        self.done = True
        mean_c = np.mean([a for a, _ in self._poses], axis=0)
        return _rotate_zero(Zero.from_pose(
            mean_c, -mean_n if self.flip else mean_n,
            meta={"source": "_DatumPrimer", "n": self.n, "flip": bool(self.flip),
                  "spread_deg": round(spread, 2)},
        ), self.rotate)


def prime_zero(est, frames, n=AUTO_ZERO_FRAMES, tol_deg=AUTO_ZERO_TOL_DEG,
               flip=False, max_frames=AUTO_ZERO_MAX_FRAMES, rotate=None):
    """
    A `zeroing.Zero` datum from the first stable poses the estimator produces.

        The rig's world frame comes from the calibration board, which is not gravity
        aligned -- on the 2026-08-28 flight the robot's median tilt from world +z reads
        38 degrees, which is not the robot leaning but the frame being tipped. That
        makes "up" wrong in the scene and every angle read off it misleading.

        This puts the robot's own rotor axis on +z instead, so the viewer sees the
        hover attitude as vertical and later poses as departures from it.
        `up_direction` and `camera_in_datum` already follow the datum, so nothing else
        in the scene needs to know.

        **Averaged, and stable-gated.** The datum is subtracted from every later
        measurement, so noise in it is a fixed bias on the whole run rather than
        something that averages out (`calib/zeroing.py`). Poses are collected until `n`
        of them agree with their own running mean to within ``tol_deg``; a run that
        never settles returns ``None`` rather than a datum built from scatter.

        ``flip`` negates the datum's axis, turning the scene the other way up.

        It is a real choice and not a preference, because **the sign of the reported
        normal is asserted rather than measured**. A circle is unoriented, so the rim
        carries nothing about which face is which (`pose/theory.md` 16.13); `stereo.orient`
        picks the hemisphere agreeing with its reference, and on a rig whose world frame
        is camera A's optical frame that reference points *away from camera A*, not up.
        Whether that coincides with the rod's direction depends on which side of the
        robot camera A sits, which nothing in the pipeline knows.

        So the datum's +z is the rotor axis with an arbitrary sign, and `flip` is how you
        say which. Look at the scene: if the rod hangs down, flip it.

        ``frames`` is any iterable of stereo frame lists, and ``max_frames`` bounds how
        long it is drawn from. **The bound is not optional on a live source.** A camera
        generator does not end, and this returns only once it has seen `n` agreeing
        poses, so without a cap a session started before the robot is in shot never
        reaches `make_viz` -- no viser, no output, nothing to interrupt. The module
        docstring's promise that nothing here blocks the loop covers startup too.

        Returns ``(zero, report)``, with ``zero`` ``None`` if none was found.
    """

    from zeroing import Zero

    kept = []
    for i, fs in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            return None, (f"no stable pose in {max_frames} frames "
                          f"({len(kept)} detections) -- datum left as loaded")
        p = est.update(fs, t=i / 60.0, frame_index=i)
        if p is None:
            continue
        kept.append((np.asarray(p.xyz_mm, float), _unit_np(p.normal)))
        if len(kept) < n:
            continue
        window = kept[-n:]
        mean_n = _unit_np(np.mean([b for _, b in window], axis=0))
        spread = max(
            math.degrees(math.acos(min(1.0, abs(float(b @ mean_n))))) for _, b in window
        )
        if spread <= tol_deg:
            mean_c = np.mean([a for a, _ in window], axis=0)
            axis = -mean_n if flip else mean_n
            return _rotate_zero(Zero.from_pose(
                mean_c, axis,
                meta={"source": "prime_zero", "n": n, "flip": bool(flip),
                      "spread_deg": round(spread, 2)},
            ), rotate), (f"datum from {n} poses, spread {spread:.2f} deg"
                + (" (flipped)" if flip else "")
                + (f" rotated {rotate}" if rotate else ""))
    return None, f"no stable pose in {len(kept)} detections -- datum left as loaded"


def _unit_np(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(np.linalg.norm(v), 1e-12)


def from_recording(rec_dir, rig_path=None, port=8080, csv_out=None, speed=1.0,
                   rig=None, est=None, viz=None, loop=False, zero="auto", flip=False,
                   rotate=None):
    """Replay a `camera/record.py` recording through the pose pipeline.

    The offline twin of `from_stereo`, and the reason the recorder writes `frames.csv`:
    the per-camera capture times go into the estimate exactly as they would live, so an
    offline run and a live run of the same flight give the same answer. Assuming the two
    videos are simultaneous instead is worth 1.5-2.2x the lateral noise floor at hover
    (`pose/theory.md` section 17).

    ``csv_out`` also writes a `recorder.PoseRecorder` log, which `replay` can then show
    without decoding video again.
    """

    import sys as _sys
    from filter import PoseFilter
    _sys.path.insert(0, str(HERE.parent / "camera"))
    from record import open_recording

    if est is None:
        # A `RunningPlate`, not this recording's own median plate. Both are built from
        # this footage, so both match its exposure and focus, but the running one tracks
        # the scene as it drifts instead of freezing one estimate of it. Over all 1013
        # frames of `2026-08-28_135533`: discrepancy p90 4.94 mm against 5.94, and 90.0%
        # of frames under the 5 mm gate against 87.2%. It also makes this path identical
        # to the live one, which is worth more than the 1 mm.
        rig, est = _stereo_estimator(rig_path, backgrounds="running")
    caps, stamps = open_recording(rec_dir)
    if stamps is None:
        print(f"{rec_dir}: no frames.csv, so the two views are assumed simultaneous")

    if zero == "auto":
        # A priming pass, before the scene exists: the viz reads the datum once at
        # construction to place its cameras and its "up", so installing one afterwards
        # would leave both stale.
        def _stereo_frames():
            while True:
                got = [c.read() for c in caps]
                if not all(ok for ok, _ in got):
                    return
                yield [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                       for _, f in got]

        zero, note = prime_zero(est, _stereo_frames(), flip=flip, rotate=rotate)
        print(f"live_viz: {note}")
        for c in caps:
            c.release()
        caps, stamps = open_recording(rec_dir)
        est.reset()
    if zero is not None:
        est.zero = zero

    filt = PoseFilter()
    log = None
    if csv_out:
        from recorder import PoseRecorder
        log = PoseRecorder(csv_out, meta={"source": str(rec_dir)})
    own_viz = viz is None
    if own_viz:
        # The rim the scene draws is the one the estimator assumed, not the mesh's own:
        # they differ by 2% since the radius was refitted (`pose/fit_radius.py`), and a
        # ring that does not sit on the mesh reads as a pose error.
        viz = make_viz(port=port, rig=rig, radius_mm=est.radius_mm,
                       label=f"replay {Path(rec_dir).name}",
                       backgrounds=est.backgrounds, estimator=est, zero=est.zero)
    poses, lost, i, last_seen = [], 0, 0, None
    try:
        while True:
            got = [c.read() for c in caps]
            if not all(ok for ok, _ in got):
                if not loop:
                    break
                # Tuning against a recording means watching one pass, moving the
                # threshold, and watching the same pass again. Fifteen seconds of
                # video is not long enough to drag a slider, so start it over.
                for c in caps:
                    c.release()
                caps, stamps = open_recording(rec_dir)
                i, filt = 0, PoseFilter()
                # Dropped, not kept: a tuning session runs for hundreds of passes and
                # the only pass anyone reads is the one on screen.
                poses.clear()
                lost = 0
                est.reset()
                continue
            # The slider is the whole point of replaying. Guarded on `enabled` so a
            # headless run cannot overwrite a level its caller set deliberately.
            if viz.enabled:
                est.thresh = viz.thresh
            frames = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                      for _, f in got]
            row = stamps[i] if stamps is not None and i < len(stamps) else None
            t = float(np.mean(row)) if row is not None else i / 60.0
            pose = est.update(frames, t=t, frame_index=i,
                              stamps=None if row is None else list(row),
                              motion=filt.pos if filt.pos.initialised else None)
            state = filt.update(pose, t=t)
            lost += pose is None
            poses.append(pose)
            if log is not None and pose is not None:
                log.write(pose, t_capture=t, frame_index=i)
                last_seen = pose
            # The log gets the estimate; the view gets the filter. See `smoothed`.
            viz.push(smoothed(pose, state, last_seen), frames=frames, t=t, lost=lost)
            if speed and own_viz:
                time.sleep(max(0.0, (1.0 / 60.0) / speed))
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        for c in caps:
            c.release()
        if log is not None:
            log.close()
        if own_viz:
            viz.close()
    got = [p for p in poses if p is not None]
    print(f"{i} frames, {len(got)} solved, {lost} lost")
    return poses


def _self_check():
    """Assert the things that fail silently rather than loudly."""

    import render

    # The off-by-one that kills the gradient without raising.
    for n in (0, 1, 2, 37):
        assert trace_colors(n).shape == (max(n - 1, 0), 2, 3), n
    assert (trace_colors(9)[0, 0] != trace_colors(9)[-1, -1]).any(), "gradient is flat"

    # A closed ring must have as many segments as points, last wrapping to first.
    ring = circle_points((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 10.0, n=8)
    seg = closed_loop_segments(ring)
    assert seg.shape == (8, 2, 3)
    assert np.allclose(seg[-1, 1], ring[0])
    assert np.allclose(np.linalg.norm(ring - np.array([1.0, 2.0, 3.0]), axis=1), 10.0)

    # pose_matrix -> quaternion -> matrix must round-trip, or the mesh and the
    # normal disagree about where the robot is pointing.
    from scipy.spatial.transform import Rotation

    for tilt, azim in ((0.0, 0.0), (13.0, 47.0), (-30.0, 200.0)):
        m = render.pose_matrix(tilt, azim, (1.0, 2.0, 3.0))
        w, x, y, z = wxyz_from_matrix(m[:3, :3])
        back = Rotation.from_quat([x, y, z, w]).as_matrix()
        assert np.allclose(back, m[:3, :3], atol=1e-9), (tilt, azim)

    # Frame convention: the un-zeroed rig world is camera A's optical frame, +y down.
    import rig as rigmod

    # All four frames, because getting this wrong renders the robot on its side and
    # nothing else in the scene complains.
    import zeroing

    mono = rigmod.StereoRig.monocular()
    assert up_direction(mono) == "-y", "camera A optical: +y is image-down"
    assert up_direction(rigmod.StereoRig(cameras=mono.cameras,
                                         meta={"world_frame": "camera_A"})) == "-y"
    assert up_direction(nominal_rig()) == "+z", "from_spherical builds a lab frame"
    # A datum wins over the rig: Zero.apply puts the reference normal on +z.
    level = zeroing.Zero.from_pose((0.0, 0.0, 300.0), (0.0, 0.0, 1.0))
    tipped = zeroing.Zero.from_pose((0.0, 0.0, 300.0), (0.0, -1.0, 0.0))
    assert up_direction(mono, zeroing.Zero.identity()) == "-y"
    assert up_direction(mono, tipped) == "+z"
    assert np.allclose(tipped.apply((0.0, 0.0, 300.0), (0.0, -1.0, 0.0))[1], [0, 0, 1])
    # Cameras must move into the datum frame with the robot, or the scene has two
    # origins. At the datum's own reference pose the camera lands at -R'@t.
    T = camera_in_datum(mono.a, level)
    assert np.allclose(T[:3, 3], level.R.T @ -level.t), T[:3, 3]
    assert np.allclose(camera_in_datum(mono.a, zeroing.Zero.identity()),
                       mono.a.T_world_cam)

    # The runner's default axes put "vertical" on world -y, NOT on world z. Getting
    # this wrong draws the setpoint on the wrong axis, which reads as a tracking
    # error that is not there.
    idx, sign = axis_map(("x", "-y"))
    assert (idx, sign) == ([0, 1], [1.0, -1.0])
    w = ref_to_world((0.05, 0.30), idx, sign, measured=(1.0, 2.0, 333.0))
    assert np.allclose(w, [50.0, -300.0, 333.0]), w  # z is uncontrolled: held
    assert np.allclose(
        ref_to_world((0.05, 0.30), *axis_map(("x", "z"))), [50.0, np.nan, 300.0],
        equal_nan=True,
    )
    keys = {k for _, _, s in plots_spec(("x", "-y")) for k, _ in s}
    assert "ref_y" in keys and "ref_z" not in keys, keys
    assert set(_PLOT_KEYS) >= {k for a in (("x", "-y"), ("x", "z"), ("y", "z"))
                               for _, _, s in plots_spec(a) for k, _ in s}

    # The adaptive hover-frequency estimate rides in the command plot, and its history
    # buffer is derived from that spec rather than listed twice. If the derivation ever
    # stops working the series draws nothing and says nothing, which is the whole
    # reason this is asserted rather than eyeballed.
    cmd = next(s for k, _, s in plots_spec(("x", "-y")) if k == "cmd")
    assert "f_hat" in [k for k, _ in cmd], cmd
    assert "f_hat" in _PLOT_KEYS, _PLOT_KEYS
    assert len({c for _, c in cmd}) == len(cmd), "two command channels share a stroke"

    # The cross-agent contract. `stereo_frames` yields these and the controller reads
    # them by name, so a rename here is a runtime AttributeError over there.
    import dataclasses as _dc
    import inspect

    assert inspect.isgeneratorfunction(stereo_frames), "stereo_frames must be a generator"
    assert [f.name for f in _dc.fields(Tick)] == [
        "t", "xyz_mm", "pose", "frames", "lost", "viz"], _dc.fields(Tick)

    # A dead visualiser must be silently harmless at every call site.
    null = make_viz(enabled=False)
    null.push(None, ref=(0.0, 0.0), u=(1, 2, 3, 4), anything="goes")
    # ...including the control knobs, and *inert* there: a control loop polls these
    # unconditionally, and a viewer that failed to start must arm nothing.
    assert null.thresh is None
    assert null.armed is False and null.land is False
    assert null.mag_max == 0.0, "a dead viewer must not enable the lateral channel"
    assert null.gain_scale == (1.0, 1.0)
    assert null.setpoint == (0.0, 0.0, 60.0)
    null.close()
    # A bad argument must degrade to NullViz *and* hand its port back, or one failed
    # attempt would squat on 8080 and force every later one to degrade too.
    assert make_viz(enabled=True, rig=object(), port=0).enabled is False
    survivor = make_viz(enabled=True, port=0)
    assert survivor.enabled is True, "a failed build did not hand its port back"
    survivor.close()

    # The replay path, end to end on a recorder-format CSV. Cheap, and it catches
    # the column-name drift and read-only-view traps that only bite at run time.
    import tempfile

    import recorder

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write("# source, self-check\n" + ",".join(recorder.COLUMNS) + "\n")
        for i in range(3):
            fh.write(",".join(str(1.0 + i) for _ in recorder.COLUMNS) + "\n")
    replay(fh.name, port=0, speed=1000.0)
    Path(fh.name).unlink()

    # push() must be cheap enough to disappear inside a 33 ms control period.
    viz = LiveViz(port=0, label="self-check")
    try:
        # The real widgets must answer with the same defaults `NullViz` fakes, or a
        # headless run and a live one start from different places.
        assert viz.armed is False and viz.mag_max == 0.0
        assert viz.gain_scale == (1.0, 1.0), viz.gain_scale
        assert viz.setpoint == (0.0, 0.0, 60.0), viz.setpoint
        # One-shot: a press must be consumed exactly once, or a landing restarts on
        # every iteration for as long as the flag stays set.
        assert viz.land is False
        viz._land = True
        assert viz.land is True and viz.land is False

        p = _fake_pose(0.0)
        t0 = time.perf_counter()
        for _ in range(10_000):
            viz.push(p, ref=(0.0, 0.3), u=(0.1, 0.0, 140.0, 80.0))
        per_us = 1e6 * (time.perf_counter() - t0) / 10_000
        assert per_us < 20.0, f"push() cost {per_us:.1f} us"
        assert len(viz._q) <= 4096, "the queue must stay bounded"

        # A dropped detection is None every few frames and must be ordinary.
        viz.push(None)
        # A sample that genuinely raises inside the render path must cost one frame,
        # not the thread. (`xyz_mm` present but unconvertible, so the getattr guard
        # lets it through to np.asarray.)
        viz.push(type("Bad", (), {"xyz_mm": "not an array"})())
        time.sleep(0.4)
        assert viz._thread.is_alive(), "render thread died on a bad sample"
        assert viz._errors >= 1, "the bad sample did not actually exercise the guard"
        viz.push(p)
        time.sleep(0.2)
        assert viz._thread.is_alive()
    finally:
        viz.close()
    print(f"self-check ok (push {per_us:.1f} us/call)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--demo", action="store_true", help="synthetic helix, no hardware")
    ap.add_argument("--replay", metavar="CSV", help="play a recorder.py pose CSV")
    ap.add_argument("--camera", metavar="SPEC", help="live vision, e.g. camera:0")
    ap.add_argument("--stereo", metavar="A,B", nargs="?", const="camera:0,camera:1",
                    help="live stereo, e.g. --stereo camera:0,camera:1")
    ap.add_argument("--rig", metavar="JSON", help="rig for --stereo (default stereo_rig.json)")
    ap.add_argument("--recording", metavar="DIR", help="replay a camera/record.py recording")
    ap.add_argument("--csv", metavar="OUT", help="also write a pose CSV while replaying")
    ap.add_argument("--layout", metavar="eA,eB,aA,aB", nargs="?", const="45,45,0,90",
                    help="preview a nominal rig geometry (deg) instead of a measured "
                         "one; implies --demo. e.g. --layout 45,-45,0,90")
    ap.add_argument("--range", type=float, help="--layout camera distance, mm")
    ap.add_argument("--check", action="store_true", help="run the self-check and exit")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--speed", type=float, default=1.0, help="--replay rate multiplier")
    a = ap.parse_args()

    if a.check:
        _self_check()
    elif a.layout:
        eA, eB, aA, aB = (float(x) for x in a.layout.split(","))
        rig = nominal_rig((eA, eB), (aA, aB), range_mm=a.range)
        print(rig.summary() if hasattr(rig, "summary") else rig.meta)
        demo(port=a.port, rig=rig)
    elif a.replay:
        replay(a.replay, port=a.port, speed=a.speed)
    elif a.recording:
        from_recording(a.recording, rig_path=a.rig, port=a.port, csv_out=a.csv,
                       speed=a.speed)
    elif a.stereo:
        from_stereo(a.stereo, rig_path=a.rig, port=a.port)
    elif a.camera:
        from_camera(a.camera, port=a.port)
    elif a.demo:
        demo(port=a.port)
    else:
        ap.error("pick one of --demo, --layout, --replay, --recording, --camera, "
                 "--stereo, --check")
