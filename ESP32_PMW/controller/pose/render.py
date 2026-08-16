"""Render the robot mesh at known poses, so residuals have something to be
measured against.

There is no ground truth on the bench -- nothing observes the robot but the
camera we are trying to validate.  So we synthesise it: place the mesh at a pose
we chose, render it through the *real* calibrated intrinsics, run the estimator,
and compare.

Renderer choice, for the record.  Open3D's `OffscreenRenderer` cannot run on
this machine at all: the macOS wheel is built EGL-headless-only and dies with
"EGL Headless is not supported on this platform", and calling
`gui.Application.instance.initialize()` first does not help.  Its legacy
`Visualizer` does render, but `RenderOption` exposes no material opacity, which
makes the alpha sweep -- the whole point of this harness -- impossible.  pyrender
gives per-material alpha via `alphaMode="BLEND"`, takes calibrated intrinsics
directly through `IntrinsicsCamera`, and hands back a pixel-exact segmentation
mask for free.  It needs `pyglet<2`; pyrender 0.1.45 predates the pyglet 2 API
break.

Coordinate conventions are the trap here.  OpenCV looks down +z with +y down;
OpenGL, which pyrender follows, looks down -z with +y up.  Getting that wrong
produces images that look perfectly plausible and ground truth that is silently
mirrored.  `selftest.py` checks the rendered silhouette against the analytic
projection from `conic.project_circle` and will catch exactly that.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# pyrender picks its GL backend from PYOPENGL_PLATFORM, but only accepts "egl"
# or "osmesa" there -- both Linux-only. An unset variable means "use pyglet",
# which is the path that works on macOS, so clear any inherited value rather
# than set one. Must happen before the pyrender import.
if os.environ.get("PYOPENGL_PLATFORM") not in (None, "egl", "osmesa"):
    del os.environ["PYOPENGL_PLATFORM"]

import pyrender  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

MESH_PATH = (
    Path(__file__).resolve().parent / "assets" / "flyingrobot_rod2.STL"
)
INTRINSICS_PATH = (
    Path(__file__).resolve().parents[1] / "calib" / "assets" / "camera_intrinsics.npz"
)

# OpenCV camera axes -> OpenGL camera axes: flip y and z, keep x.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

#: Base-colour factor for a matt red body, for the red-on-white rig. Chosen so
#: the rendered chroma ``R - max(G, B)`` lands in the 48-135 band measured off
#: real red surfaces (`test/test_appearance.py`), rather than a saturated primary
#: that no paint produces.
RED_BODY = (0.70, 0.13, 0.12)

#: Base-colour factor for a matt black body, for the black-on-white rig. Not
#: zero: real black paint reflects, and a body rendered at 0 would make the
#: threshold look far better than the bench ever will.
BLACK_BODY = (0.09, 0.095, 0.10)

#: Bronze drive coils, the clutter beside the robot.
#:
#: **Unused, and kept only as a record of a superseded idea.** It was chosen so the
#: rendered chroma would land in a band measured off real coils, back when
#: `segment.py` separated them from the robot by colour. The rig's camera is a mono
#: OV9281 -- chroma is identically zero at every pixel -- so that gate is gone and
#: nothing reads this constant. The citation this line used to carry, to
#: `test/test_appearance.py`, was also wrong: that file has never contained a coil.
#: Clutter is now rejected by region and spread instead; see lecture notes section 15.
COIL_BODY = (0.16, 0.34, 0.56)


@dataclass
class LightRig:
    """A lighting configuration for one render.

    Two families, matching the two ways light actually reaches the robot on the
    bench:

    ``lateral_deg`` -- directional lights on a ring in the plane normal to the
    camera axis, i.e. lit from the side at the given bearings.  This is the
    harsh case: a low side light leaves half the rim in shadow and the threshold
    eats it, which is exactly the failure the sweep should surface.

    ``dome`` -- ``(elevation_deg, azimuth_deg)`` pairs on a hemisphere above the
    robot, the diffuse overhead case.

    ``ambient`` turns out to be the dominant knob, and not for a trivial reason.
    The duct is a thin ring, so face-on its outer wall is nearly parallel to the
    view and catches almost nothing from a directional source: measured over the
    ground-truth silhouette, raising ``intensity`` from 3 to 80 moves the
    fraction of lit pixels only 0.21 -> 0.39, while raising ``ambient`` from
    0.05 to 0.3 moves it to 0.89.  Directional light controls contrast; ambient
    controls whether the silhouette exists at all.  Both are swept.

    ``key_from_camera`` adds a light down the optical axis -- a lens-mounted
    ring light, which is how you would actually illuminate this rig.
    """

    lateral_deg: tuple = ()
    dome: tuple = ()
    intensity: float = 10.0
    ambient: float = 0.35
    key_from_camera: bool = True

    def directions(self):
        """Unit vectors the light travels *toward*, in OpenCV camera axes."""
        out = []
        for a in self.lateral_deg:
            r = math.radians(a)
            out.append(np.array([math.cos(r), math.sin(r), 0.0]))
        for elev, az in self.dome:
            e, a = math.radians(elev), math.radians(az)
            out.append(
                np.array([math.cos(e) * math.cos(a), -math.sin(e), math.cos(e) * math.sin(a)])
            )
        if self.key_from_camera or not out:
            out.append(np.array([0.0, 0.0, 1.0]))
        return out

    def label(self):
        lat = "-".join(f"{a:g}" for a in self.lateral_deg) or "none"
        dome = "-".join(f"{e:g}@{a:g}" for e, a in self.dome) or "none"
        return f"lat[{lat}]_dome[{dome}]_i{self.intensity:g}_amb{self.ambient:g}"


@dataclass
class Exposure:
    """Sensor realism: finite exposure time and read noise.

    Renders are otherwise instantaneous and noise-free, which flatters the
    estimator.  Two effects are worth simulating because they act in opposite
    directions:

    **Motion blur** is accumulated properly, by rendering ``subframes`` across
    the exposure window and averaging, rather than convolving with a kernel.
    A kernel would be wrong here: the dominant motion is the rotor *spinning*,
    not translating, and a rotation is not a shift-invariant blur.  At 330 Hz
    and a 1/500 s exposure the rotor turns 238 degrees within one frame, which
    no linear kernel represents.

    **Gaussian read noise** is added after accumulation, since it is a per-frame
    sensor effect and does not average down with exposure the way scene motion
    does.  ``sigma`` is in grey levels out of 255.

    High frame rates make these fight each other: a 420 fps camera must use a
    short exposure, which reduces blur but collects fewer photons and so raises
    noise.  ``exposure_s`` and ``sigma`` should be moved together to model a real
    camera rather than independently.
    """

    exposure_s: float = 0.0
    subframes: int = 1
    spin_hz: float = 0.0
    velocity_mm_s: tuple = (0.0, 0.0, 0.0)
    tilt_rate_deg_s: float = 0.0
    sigma: float = 0.0
    seed: int = 0

    @property
    def blurs(self):
        return self.exposure_s > 0 and self.subframes > 1 and (
            self.spin_hz != 0
            or any(self.velocity_mm_s)
            or self.tilt_rate_deg_s != 0
        )

    def label(self):
        return f"exp{self.exposure_s*1e3:g}ms_spin{self.spin_hz:g}_sig{self.sigma:g}"


@dataclass
class View:
    """A camera other than the reference one, simulated by moving the world.

    `Renderer` holds its pyrender camera at the origin because pyglet's Cocoa
    backend allows exactly one GL context per process -- a second `Renderer`
    raises, and so would a second camera node with a different pose if we wanted
    a *different image* out of the same pass.  So the second view is produced by
    left-multiplying every object pose by ``T_this_ref``, which is
    algebraically identical to moving the camera and costs one 4x4 product.

    ``T_this_ref`` maps **reference-camera coordinates into this camera's
    coordinates**.  For a `rig.StereoRig` that is ``rig.relative(ref, this)``.
    Identity reproduces the reference view exactly, which is what
    `selftest_stereo.py` asserts first.

    ``occluders`` are ``(trimesh, 4x4 pose in reference-camera coords, rgb)``.
    They are rendered but kept out of the ground-truth mask, so the mask stays
    the robot's true silhouette and occlusion becomes something to measure
    rather than something baked into the answer.
    """

    T_this_ref: np.ndarray = field(default_factory=lambda: np.eye(4))
    K: np.ndarray = field(default=None)
    # Which background generator produced this frame's backdrop, or None for the
    # flat `bg_level`. Carried so a failure can be traced to a condition.
    bg_name: str = field(default=None)
    occluders: tuple = ()
    name: str = ""

    def __post_init__(self):
        self.T_this_ref = np.asarray(self.T_this_ref, dtype=np.float64).reshape(4, 4)
        if self.K is not None:
            self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)


@dataclass
class Sample:
    """One rendered frame plus the ground truth that produced it."""

    image: np.ndarray  # uint8 grayscale
    mask: np.ndarray  # bool, True where the mesh is
    center_mm: np.ndarray  # circle centre, OpenCV camera coords
    normal: np.ndarray  # unit rotor normal, oriented toward the camera
    tilt_deg: float
    azimuth_deg: float
    alpha: float
    bg_level: float
    light: LightRig = field(default=None)
    exposure: "Exposure" = field(default=None)
    # Intrinsics this sample was rendered through.  Carried explicitly because a
    # stereo pair's two views need not share a camera; ``None`` means the
    # default calibration, which is what every monocular caller gets.
    K: np.ndarray = field(default=None)
    # Which background generator produced this frame's backdrop, or None for the
    # flat `bg_level`. Carried so a failure can be traced to a condition.
    bg_name: str = field(default=None)

    @property
    def ellipse_gt(self):
        """Analytic image ellipse of the rim at this pose.

        Independent of the render -- computed straight from the geometry -- so
        comparing it against the fitted ellipse separates renderer error from
        estimator error.
        """
        import conic

        K = load_K() if self.K is None else self.K
        return conic.project_circle(self.center_mm, self.normal, RIM_RADIUS_MM, K)


_MESH_CACHE = {}
RIM_RADIUS_MM = 10.204  # overwritten by `load_mesh` with the value measured here


def load_K(path=INTRINSICS_PATH):
    d = np.load(Path(path))
    return np.asarray(d["camera_matrix"], dtype=np.float64)


def load_mesh(path=MESH_PATH):
    """Load the robot, recentred and axis-aligned so pose means something.

    The STL sits in Fusion's assembly frame with the rotor plane in local y-z.
    We rotate the rotor axis onto +z and translate the rim centroid to the
    origin, so "pose" is the pose of the duct circle itself -- the very feature
    `conic.py` recovers -- rather than of some arbitrary CAD datum.

    Also measures the rim radius from the mesh instead of hardcoding it, so
    ground truth and the estimator's radius come from one source.
    """
    global RIM_RADIUS_MM
    key = str(path)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]

    mesh = trimesh.load(str(path), process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64)

    # Rotor axis = direction of least extent (the disc is thin along its axis).
    centroid = v.mean(axis=0)
    x = v - centroid
    _, vecs = np.linalg.eigh(np.cov(x.T))
    axis = vecs[:, 0]  # smallest variance
    e1, e2 = vecs[:, 2], vecs[:, 1]
    rot = np.column_stack([e1, e2, axis])
    if np.linalg.det(rot) < 0:
        rot[:, 1] *= -1
    aligned = x @ rot

    radial = np.hypot(aligned[:, 0], aligned[:, 1])
    rim = aligned[radial > 0.93 * radial.max()]
    # Outer radius, not mean radius: the segmenter hulls the silhouette, and a
    # hull rides the outermost surface. A high percentile rather than the raw
    # max so one stray vertex cannot set the scale.
    RIM_RADIUS_MM = float(np.percentile(radial, 99.9))

    # Put the origin at the rim's centre, in the rim's own plane.
    aligned[:, 2] -= rim[:, 2].mean()
    aligned[:, 0] -= rim[:, 0].mean()
    aligned[:, 1] -= rim[:, 1].mean()

    mesh.vertices = aligned
    _MESH_CACHE[key] = mesh
    return mesh


def pose_matrix(tilt_deg, azimuth_deg, center_mm, spin_deg=0.0):
    """4x4 model->camera transform (OpenCV axes) for a tilt/azimuth/position.

    Tilt swings the rotor axis away from the camera axis; azimuth spins that
    tilt around.  At tilt 0 the rotor faces the camera dead on.

    ``spin_deg`` rotates the robot about its own rotor axis -- blade phase.  It
    does not change the pose being estimated (the rotor axis is unmoved), only
    which way the blades happen to point, so it is the right way to model a
    robot turning at 310-350 Hz.
    """
    t, a = math.radians(tilt_deg), math.radians(azimuth_deg)
    # Rotate +z by `tilt` about a unit axis in the x-y plane set by `azimuth`.
    # `k` is already normalised, so `t * k` is the rotation vector directly and
    # scipy does the Rodrigues expansion -- previously written out here by hand
    # along with its skew-symmetric matrix.
    axis = np.array([-math.sin(a), math.cos(a), 0.0])
    r = Rotation.from_rotvec(t * axis)
    if spin_deg:
        r = r * Rotation.from_euler("z", spin_deg, degrees=True)

    m = np.eye(4)
    m[:3, :3] = r.as_matrix()
    m[:3, 3] = np.asarray(center_mm, dtype=np.float64)
    return m


def _add_noise(image, sigma, seed=0):
    """Additive Gaussian read noise, in grey levels.

    Deliberately applied after any exposure averaging: read noise is a per-frame
    sensor effect, so averaging sub-frames must not attenuate it the way it
    attenuates scene motion.
    """
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0.0, float(sigma), image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def normal_from_pose(model_to_cam):
    """The rotor normal a pose implies, oriented toward the camera.

    Matches how `conic.backproject` orients its normals, so ground truth and
    estimate are directly comparable without a sign convention in between.
    """
    n = model_to_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    return -n if n @ model_to_cam[:3, 3] > 0 else n


class Renderer:
    """Offscreen renderer with a fixed camera and a swappable material/lighting.

    Held as a class because `pyrender.OffscreenRenderer` owns a GL context that
    costs far more to create than to reuse -- rebuilding it per frame would
    dominate a sweep of a few hundred renders.

    One per process, and only one.  pyglet's Cocoa backend cannot build a second
    NSOpenGL pixel format after the first window exists, so constructing another
    `Renderer` -- even after `close()` -- raises deep inside pyglet
    (``ObjCInstance PygletDelegate has no attribute initWithAttributes_``).  To
    compare resolutions, render once and resize the image, or use a subprocess.
    """

    def __init__(self, width=1024, height=768, mesh_path=MESH_PATH, intrinsics=INTRINSICS_PATH):
        self.width, self.height = width, height
        self.mesh = load_mesh(mesh_path)
        self.K = load_K(intrinsics)
        self._renderer = pyrender.OffscreenRenderer(width, height)

    def close(self):
        self._renderer.delete()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def render(self, tilt_deg, azimuth_deg, center_mm, alpha=1.0, light=None, bg_level=0.0,
               exposure=None, spin_deg=0.0, view=None, background=None,
               body_colour=None):
        """Render one frame and return it with its ground truth.

        ``exposure`` adds motion blur and read noise; see `Exposure`.  Ground
        truth is taken at the **middle** of the exposure window, which is what a
        real timestamp refers to and keeps the reported pose unbiased with
        respect to the blur.

        ``view`` is how a second camera is rendered without a second GL context
        (see `View` and `render_stereo.py`): it left-multiplies the model pose,
        rotates the lights to match, swaps the intrinsics, and adds occluders.
        Leaving it ``None`` is the monocular path and behaves exactly as before.

        ``body_colour`` is a linear base-colour factor. ``None`` keeps the white
        body and returns a **single-channel** frame, which is what every result
        in this package was measured on. Anything else (e.g. `RED_BODY`) returns
        **BGR**, because the whole point of a coloured body is that the chroma is
        the signal -- see `segment.score_channel`.
        """
        light = light or LightRig(dome=((60.0, 0.0),))

        if exposure is not None and exposure.blurs:
            return self._render_exposed(
                tilt_deg, azimuth_deg, center_mm, alpha, light, bg_level, exposure, spin_deg,
                view, background, body_colour,
            )

        sample = self._render_instant(
            tilt_deg, azimuth_deg, center_mm, alpha, light, bg_level, spin_deg, view,
            background, body_colour
        )
        if exposure is not None and exposure.sigma > 0:
            sample.image = _add_noise(sample.image, exposure.sigma, exposure.seed)
        return sample

    def _render_exposed(self, tilt_deg, azimuth_deg, center_mm, alpha, light, bg_level,
                        exposure, spin_deg, view=None, background=None,
                        body_colour=None):
        """Average sub-frames across the exposure window, then add read noise.

        Sub-frames are placed symmetrically about the window centre so the mean
        pose equals the reported ground-truth pose; sampling from one edge would
        bias every measurement by half an exposure of motion.
        """
        n = max(2, int(exposure.subframes))
        centre = np.asarray(center_mm, dtype=np.float64)
        vel = np.asarray(exposure.velocity_mm_s, dtype=np.float64)

        offsets = (np.arange(n) - (n - 1) / 2.0) / max(n - 1, 1) * exposure.exposure_s

        accum = None
        for dt in offsets:
            sub = self._render_instant(
                tilt_deg + exposure.tilt_rate_deg_s * dt,
                azimuth_deg,
                centre + vel * dt,
                alpha,
                light,
                bg_level,
                spin_deg + 360.0 * exposure.spin_hz * dt,
                view,
                background,
                body_colour,
            )
            accum = sub.image.astype(np.float64) if accum is None else accum + sub.image

        blurred = np.clip(accum / n, 0, 255).astype(np.uint8)

        # Both the pose and the mask are taken at mid-exposure, and they must
        # match: the mask exists to score segmentation against the pose being
        # reported. Using the union swept during the exposure instead makes IoU
        # meaningless -- it collapsed from 0.91 to 0.23 purely because the swept
        # region is far larger than any instant, not because segmentation got
        # worse (pose error was unchanged).
        truth = self._render_instant(
            tilt_deg, azimuth_deg, centre, alpha, light, bg_level, spin_deg, view,
            background, body_colour
        )
        truth.image = (
            _add_noise(blurred, exposure.sigma, exposure.seed)
            if exposure.sigma > 0 else blurred
        )
        truth.exposure = exposure
        return truth

    def _render_instant(self, tilt_deg, azimuth_deg, center_mm, alpha, light, bg_level,
                        spin_deg=0.0, view=None, background=None, body_colour=None):
        model_to_cam = pose_matrix(tilt_deg, azimuth_deg, center_mm, spin_deg)
        K = self.K
        if view is not None:
            # A second camera is simulated by moving the *world* into its frame,
            # since the pyrender camera is nailed to the origin (one GL context
            # per process). See `View`.
            model_to_cam = view.T_this_ref @ model_to_cam
            if view.K is not None:
                K = view.K

        bg = float(np.clip(bg_level, 0.0, 1.0))
        # With a supplied backdrop the scene renders onto transparency and the
        # backdrop is composited in afterwards, because pyrender's `bg_color` is
        # a single colour and these are gradients and bands.
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0] if background is not None
            else [bg, bg, bg, 1.0],
            ambient_light=[light.ambient] * 3,
        )

        body = (1.0, 1.0, 1.0) if body_colour is None else tuple(body_colour)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*body, float(alpha)],
            alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
            metallicFactor=0.0,
            roughnessFactor=0.75,
            doubleSided=True,
        )
        node = scene.add(
            pyrender.Mesh.from_trimesh(self.mesh, material=material, smooth=False),
            pose=model_to_cam,
        )

        # Occluders go in *after* the robot node, and are deliberately excluded
        # from `seg_node_map` below, so the ground-truth mask stays the true
        # silhouette of the robot. That is the point: it makes the mask the
        # thing occlusion should be scored against, rather than something
        # occlusion has already corrupted.
        if view is not None:
            for occ_mesh, occ_pose, occ_colour in view.occluders:
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        occ_mesh,
                        material=pyrender.MetallicRoughnessMaterial(
                            baseColorFactor=[*occ_colour, 1.0],
                            metallicFactor=0.0,
                            roughnessFactor=0.9,
                            doubleSided=True,
                        ),
                        smooth=False,
                    ),
                    pose=view.T_this_ref @ occ_pose,
                )

        # The camera sits at the origin looking down +z in OpenCV axes; pyrender
        # wants the OpenGL convention, hence CV_TO_GL.
        cam = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            znear=1.0, zfar=5000.0,
        )
        scene.add(cam, pose=CV_TO_GL)

        # Lights are fixed in the *world*, not in the camera, so a second view
        # must see them from its own orientation. Skipping this rotation gives
        # two individually plausible images that are jointly impossible -- both
        # lit from their own front -- which flatters the estimator precisely
        # where it is weakest (the sweep found lighting to be the single
        # dominant error driver, 3.4x spread).
        light_R = np.eye(3) if view is None else view.T_this_ref[:3, :3]
        for d in light.directions():
            scene.add(
                pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light.intensity),
                pose=CV_TO_GL @ _look_along(light_R @ d),
            )

        colour, _ = self._renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        # White body: collapse to one channel with `max`, which is what every
        # measured result here used. Coloured body: keep the channels and hand
        # back BGR, since the chroma is the entire signal.
        gray = (colour[..., :3].max(axis=2) if body_colour is None
                else colour[..., 2::-1])

        if background is not None:
            # Straight "over": `robot * a + backdrop * (1 - a)`.
            #
            # pyrender returns **un-premultiplied** alpha, which is worth stating
            # because assuming otherwise looks almost right. A dim, partially
            # covered edge pixel comes back with a *bright* RGB and a small
            # alpha -- 63 where the opaque path renders 4, because 63 x 0.063 = 4
            # -- so adding the backdrop without scaling by alpha lights up the
            # entire silhouette boundary. The robot is a thin double-sided shell,
            # so a large fraction of its pixels are partial: 6096 with alpha > 0
            # against only 3487 fully opaque.
            #
            # Doing this per sub-frame inside a motion-blur accumulation is
            # exactly right rather than merely convenient: the backdrop is
            # static, so mean(robot_i*a_i + (1-a_i)*bg) is what a real sensor
            # integrates. Blurring the backdrop along with the robot would be
            # the error.
            #
            # This path is not pixel-identical to the opaque `bg_level` one --
            # 8-bit rounding differs at partial-coverage boundaries, shifting the
            # fitted major axis by up to 0.6 px. Scored against analytic truth it
            # is the *better* of the two (bias +0.485 px against +0.713, rms
            # 0.993 against 1.122), because the slight edge darkening opposes the
            # threshold-and-hull's outward bias. So this is not a regression; it
            # does mean results on composited backdrops are a new baseline rather
            # than a continuation of the flat-background ones.
            a = colour[..., 3].astype(np.float64) / 255.0
            back = np.asarray(background, dtype=np.float64)
            if back.shape != gray.shape[:2]:
                raise ValueError(
                    f"background {back.shape} does not match frame {gray.shape[:2]}")
            if gray.ndim == 3:                      # broadcast over BGR
                a, back = a[..., None], back[..., None]
            gray = gray.astype(np.float64) * a + (1.0 - a) * back
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Ground-truth mask: re-render flat and unlit so the silhouette is exact
        # regardless of how the shading happened to fall.
        mask = self._silhouette(scene, node)

        return Sample(
            image=gray,
            mask=mask,
            # Read the centre off the pose rather than echoing the argument, so
            # a view transform lands in the ground truth automatically.
            center_mm=model_to_cam[:3, 3].copy(),
            normal=normal_from_pose(model_to_cam),
            tilt_deg=float(tilt_deg),
            azimuth_deg=float(azimuth_deg),
            alpha=float(alpha),
            bg_level=bg,
            light=light,
            K=None if view is None else K,
        )

    def _silhouette(self, scene, node):
        """Exact mesh coverage, via pyrender's flat segmentation pass."""
        seg = self._renderer.render(
            scene, flags=pyrender.RenderFlags.SEG, seg_node_map={node: (255, 255, 255)}
        )[0]
        return seg[..., 0] > 127


def _look_along(direction):
    """4x4 whose -z axis points along ``direction`` (OpenCV axes).

    pyrender's directional lights emit along their node's -z, so this is how a
    desired travel direction becomes a light pose.
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    z = -d
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([x, y, z])
    return m
