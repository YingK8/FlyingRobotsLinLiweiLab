"""Render the same robot pose through both cameras of a `rig.StereoRig`.

The whole module is a consequence of one constraint documented in `render.py`:
pyglet's Cocoa backend allows **one GL context per process**, so there is one
`Renderer` and there cannot be a second.  A stereo pair is therefore produced by
rendering twice from the same renderer, moving the *world* into each camera's
frame rather than moving a camera.  `render.View` carries that transform.

The reference frame is the **rig's world frame** -- +z up, origin at the nominal
hover point -- not either camera.  That is what makes a pose mean the same thing
in both views: ``tilt_deg`` is lean away from vertical and ``azimuth_deg`` is the
world bearing that lean points along, so one ground-truth pose drives both
renders and the two `Sample`s come back already expressed in their own camera's
coordinates.

Three things are easy to get wrong here and each produces images that look
completely plausible:

1. **A transposed extrinsic.**  ``View.T_this_ref`` maps reference (world)
   coordinates into the camera's.  Getting it backwards mirrors the pair.
   `selftest_stereo.py` checks both views against `conic.project_circle` for
   exactly this reason -- the same guard `render.py` already keeps for CV vs GL.
2. **Lights left in camera coordinates.**  A physical rig lights the robot from
   fixed world directions; if each view lights it from its own front, the pair
   is jointly impossible.  `render._render_instant` rotates them, and the
   selftest asserts a symmetric rig gives symmetric shading.
3. **Normals with opposite signs.**  `render.normal_from_pose` orients the rotor
   normal *toward the camera*, matching `conic.backproject`.  For a rig with one
   camera above and one below the rotor plane, the two views therefore report
   normals pointing opposite ways.  That is correct, not a bug, and
   `stereo.py` compares normals as lines rather than as vectors because of it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import render as rendermod  # noqa: E402
from rig import StereoRig  # noqa: E402

# The takeoff stand: a black rod below the robot.  8 mm diameter is what the
# bench uses; the length is arbitrary as long as it leaves frame.
STAND_DIAMETER_MM = 8.0
STAND_LENGTH_MM = 120.0
# Black, and it means it: the stand sits far below `segment.THRESH` = 128, so
# from above it is invisible to the segmenter rather than a distractor. From
# below it physically removes rim pixels, which is the case worth measuring.
STAND_COLOUR = (0.02, 0.02, 0.02)

# Placeholder coil geometry -- an annulus the robot climbs through.  These
# numbers are NOT measured off the rig; they exist so occlusion is a sweep axis
# rather than an assumption. Measure the real assembly before drawing any
# conclusion from a coil-occlusion result.
COIL_INNER_R_MM = 25.0
COIL_OUTER_R_MM = 45.0
COIL_THICKNESS_MM = 6.0
COIL_COLOUR = (0.05, 0.05, 0.06)


def takeoff_stand(top_z_mm=-11.0, diameter_mm=STAND_DIAMETER_MM, length_mm=STAND_LENGTH_MM):
    """The stand rod, as ``(mesh, pose, colour)`` in world coordinates.

    ``top_z_mm`` is the world height of the rod's top face.  The default puts it
    just under a robot at the origin (the body reaches -4.8 mm and the rim is
    10.2 mm across), i.e. the moment of takeoff, which is the worst case for a
    camera looking up.  Raising the robot instead of lowering the rod is the
    same thing; this parameter exists so a sweep can walk the gap.
    """
    mesh = trimesh.creation.cylinder(radius=diameter_mm / 2.0, height=length_mm)
    pose = np.eye(4)
    pose[2, 3] = top_z_mm - length_mm / 2.0  # cylinder is centred on its own origin
    return (mesh, pose, STAND_COLOUR)


def coil_ring(z_mm=40.0, inner_r=COIL_INNER_R_MM, outer_r=COIL_OUTER_R_MM,
              thickness=COIL_THICKNESS_MM):
    """A coil-assembly proxy above the robot, as ``(mesh, pose, colour)``.

    An annulus, so the robot is visible through the middle from directly above
    and progressively occluded from oblique angles -- the qualitative behaviour
    the real assembly has.  See the module constants: the dimensions are
    placeholders.
    """
    mesh = trimesh.creation.annulus(r_min=inner_r, r_max=outer_r, height=thickness)
    pose = np.eye(4)
    pose[2, 3] = z_mm
    return (mesh, pose, COIL_COLOUR)


@dataclass
class StereoSample:
    """One pose rendered through every camera, with world-frame ground truth.

    ``views`` are `render.Sample`s in camera order, each already carrying its own
    ``center_mm``, ``normal`` and ``K`` in that camera's coordinates.
    ``center_world`` and ``normal_world`` are the single truth all of them came
    from, and are what a stereo estimate should be scored against.
    """

    views: tuple
    center_world: np.ndarray
    normal_world: np.ndarray
    tilt_deg: float
    azimuth_deg: float
    rig: StereoRig = field(default=None)

    def __len__(self):
        return len(self.views)

    def __getitem__(self, i):
        return self.views[i]

    @property
    def images(self):
        return [v.image for v in self.views]

    @property
    def detected_all(self):
        """True when every view has a non-empty ground-truth silhouette.

        A pose can leave one camera's frame entirely; that is a legitimate
        outcome for a wide-baseline rig and callers must not assume otherwise.
        """
        return all(v.mask.any() for v in self.views)


class StereoRenderer:
    """One `render.Renderer`, one `rig.StereoRig`, N views per pose.

    Same one-per-process rule as `render.Renderer`, for the same reason -- this
    class holds one, it does not create more.
    """

    def __init__(self, rig, width=1024, height=768, occluders=(), mesh_path=None):
        self.rig = rig
        self.width, self.height = width, height
        self._renderer = rendermod.Renderer(
            width, height, **({} if mesh_path is None else {"mesh_path": mesh_path})
        )
        self.set_occluders(occluders)

    def set_rig(self, rig):
        """Point the same GL context at a different camera arrangement.

        The reason this exists rather than constructing a second
        `StereoRenderer`: pyglet's Cocoa backend cannot build another NSOpenGL
        pixel format once one exists, so a sweep over rig geometries that made a
        renderer per configuration dies partway through with
        ``ObjCInstance PygletDelegate has no attribute initWithAttributes_``.
        Nothing about a rig change touches the context -- only the per-view
        transforms -- so swapping is both correct and free.
        """
        self.rig = rig
        self.set_occluders(self._occluders)
        return self

    def set_occluders(self, occluders):
        """Swap the occluder set without rebuilding the GL context.

        Occluders are shared across views -- they are physical objects in the
        world, and `render.View` transforms them per camera.
        """
        self._occluders = tuple(occluders)
        self._views = tuple(
            rendermod.View(
                T_this_ref=np.linalg.inv(cam.T_world_cam),
                K=cam.K,
                occluders=self._occluders,
                name=cam.name or str(i),
            )
            for i, cam in enumerate(self.rig.cameras)
        )

    def close(self):
        self._renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def mesh(self):
        return self._renderer.mesh

    def render_pair(self, tilt_deg, azimuth_deg, center_world_mm, alpha=1.0, light=None,
                    bg_level=0.0, exposure=None, spin_deg=0.0, background=None):
        """Render one world pose through every camera.

        ``tilt_deg`` is lean away from world +z and ``azimuth_deg`` the world
        bearing it leans along -- the same parameterisation `render.pose_matrix`
        uses, read in the world frame instead of a camera frame.  A hovering
        robot is tilt 0.

        ``exposure`` velocity and tilt rate are likewise world quantities, so
        one motion produces consistent blur in both views.

        ``background`` is a greyscale field composited behind the robot. The
        **same** field goes to both views on purpose: two cameras pointed at one
        scene do not see two different backdrops, and giving them independent
        ones would let the estimator average away a nuisance that is correlated
        in reality.
        """
        centre = np.asarray(center_world_mm, dtype=np.float64).reshape(3)
        views = tuple(
            self._renderer.render(
                tilt_deg, azimuth_deg, centre,
                alpha=alpha, light=light, bg_level=bg_level,
                exposure=exposure, spin_deg=spin_deg, view=v,
                background=background,
            )
            for v in self._views
        )
        # The world-frame truth is the pose we asked for; recover the normal
        # through the same helper the views use so signs cannot drift apart.
        model_to_world = rendermod.pose_matrix(tilt_deg, azimuth_deg, centre, spin_deg)
        normal_world = model_to_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
        return StereoSample(
            views=views,
            center_world=centre,
            normal_world=normal_world / np.linalg.norm(normal_world),
            tilt_deg=float(tilt_deg),
            azimuth_deg=float(azimuth_deg),
            rig=self.rig,
        )


def default_rig(elev_deg=(45.0, 45.0), azim_deg=(0.0, 90.0), range_mm=None, scale=1.0):
    """The rig the plan settles on, optionally rescaled for a smaller image.

    Intrinsics scale with resolution; the geometry does not.
    """
    kw = {} if range_mm is None else {"range_mm": range_mm}
    rig = StereoRig.from_spherical(elev_deg=elev_deg, azim_deg=azim_deg, **kw)
    return rig if scale == 1.0 else rig.scaled(scale)


def main(argv=None):
    """Render one pair to PNGs -- the quickest way to eyeball a rig."""
    import argparse

    import cv2

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--elev", type=float, nargs=2, default=(45.0, 45.0))
    ap.add_argument("--azim", type=float, nargs=2, default=(0.0, 90.0))
    ap.add_argument("--range-mm", type=float, default=250.0)
    ap.add_argument("--tilt", type=float, default=0.0, help="robot lean from vertical, deg")
    ap.add_argument("--robot-azim", type=float, default=0.0, help="bearing of that lean, deg")
    ap.add_argument("--z", type=float, default=0.0, help="robot height above the hover point, mm")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--stand", action="store_true", help="add the takeoff rod")
    ap.add_argument("--coils", action="store_true", help="add the coil-ring proxy")
    ap.add_argument("--out", default=str(
        HERE.parents[2] / "results" / "pose_validation" / "stereo_preview"))
    args = ap.parse_args(argv)

    rig = default_rig(args.elev, args.azim, args.range_mm)
    occ = []
    if args.stand:
        occ.append(takeoff_stand())
    if args.coils:
        occ.append(coil_ring())

    for k, v in rig.summary().items():
        print(f"{k:24s} {v}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with StereoRenderer(rig, args.width, args.height, occluders=occ) as r:
        s = r.render_pair(args.tilt, args.robot_azim, (0.0, 0.0, args.z))
        for i, v in enumerate(s.views):
            name = rig.cameras[i].name or str(i)
            cv2.imwrite(str(out / f"view_{name}.png"), v.image)
            print(f"view {name}: centre {np.array2string(v.center_mm, precision=2)} mm, "
                  f"normal {np.array2string(v.normal, precision=3)}, "
                  f"mask {v.mask.sum()} px")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
