"""
Which OpenCV index is an ELP, and whether the pair is still the calibrated one.

A and B are positional everywhere downstream -- `rig.StereoRig.a` is just
`cameras[0]` -- so a pair that opens in the wrong order produces poses that are
smooth, plausible and wrong. `elp.probe_indices` answers "what opens right now";
this answers the two questions that actually gate a flight.

**Indices come from opening cameras, and from nothing else.** That is not the
obvious design and it was arrived at the expensive way. macOS offers two device
listings -- `ffmpeg -f avfoundation -list_devices` and `system_profiler
SPCameraDataType` -- and *neither* enumerates in OpenCV's order. Measured on this
bench with both ELPs and the built-in camera connected:

    ffmpeg        [0] FaceTime   [1] ELP        [2] ELP
    OpenCV         0  1280x800    1  1280x800    2  1920x1080

OpenCV's `CAP_AVFOUNDATION` puts USB cameras *before* the built-in one; ffmpeg
does not. An index read off either listing is a guess wearing a uniform, so the
ELPs are found by asking each index for the sensor's native mode and keeping the
ones that deliver it exactly. A camera that substitutes a size is not an ELP --
which is the same check `elp.open_elp(strict=True)` makes, for the same reason.

Identity, separately, is real: `system_profiler` reports a `unique-id` that for a
UVC device derives from its USB location, so it is stable across a replug into the
same port. It cannot be tied to an OpenCV index -- see above -- but it does answer
"are these the same two cameras the rig was calibrated against", which is what
`rig.StereoRig.sources` uses it for. The two ELPs are told apart by their position
in the probe order, and that holds as long as neither cable moves.

Stage 1 of 4, so nothing here may import the rig.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import elp  # noqa: E402
import sources  # noqa: E402

Device = namedtuple("Device", "name unique_id")


def native_mode():
    """
    The ELP's native ``(width, height)``, from `elp_camera.json`.
    """

    modes = elp.load_profile().get("modes", [])
    m = next((m for m in modes if m.get("verdict") == "native"), None) or modes[0]
    return int(m["width"]), int(m["height"])


def connected():
    """
    ``[Device(name, unique_id)]`` for every camera macOS reports.

        Identity only -- the order is `system_profiler`'s and means nothing to
        OpenCV. Used for the "is this the calibrated pair" check and for error
        messages, never to pick an index.
    """

    proc = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        capture_output=True, text=True, timeout=60,
    )
    cams = json.loads(proc.stdout or "{}").get("SPCameraDataType", [])
    return [Device(c.get("_name", ""), c.get("spcamera_unique-id", "")) for c in cams]


def elp_ids(name=None):
    """
    The unique IDs of the connected ELPs, by model name from `elp_camera.json`.
    """

    if name is None:
        name = elp.load_profile().get("device", {}).get("name", "")
    return [d.unique_id for d in connected() if d.name == name]


def elp_indices(n=2, max_index=None):
    """
    The OpenCV indices that deliver the ELP's native mode, in probe order.

        Opens each index and asks for the native size, because that is the only
        thing on this platform that does not lie about which camera is which. A
        device that substitutes a different size (the built-in FaceTime returns
        1920x1080) is rejected.

        Slow -- roughly a second per index -- so call it once per session, but call
        it at the point of use rather than saving the result: the device list moves,
        and a saved list is stale the same way a hardcoded index is.
    """

    want = native_mode()
    if max_index is None:
        max_index = len(connected())
    found = []
    for i in range(max_index):
        cam = None
        try:
            cam = sources.MonoCamera(index=i, width=want[0], height=want[1], grayscale=True)
            got = cam.actual
            if (got["width"], got["height"]) == want:
                found.append(i)
        except Exception:
            continue                              # index absent, or held by something else
        finally:
            if cam is not None:
                cam.close()
    if n is not None and len(found) != n:
        raise RuntimeError(
            f"need {n} camera(s) at {want[0]}x{want[1]}, found {len(found)}: {found}.\n"
            f"Connected: " + ", ".join(f"{d.name} ({d.unique_id})" for d in connected())
        )
    return found


def _self_check():
    profile = elp.load_profile()
    assert native_mode() == (1280, 800), native_mode()
    assert profile.get("device", {}).get("name") == "Global Shutter Camera"

    # connected() must not be trusted for ordering, only for identity.
    ids = elp_ids()
    assert all(isinstance(x, str) for x in ids), ids
    print(f"identify: self-check passed ({len(ids)} ELP(s) by identity)")


if __name__ == "__main__":
    _self_check()
    for d in connected():
        print(f"  {d.name:32s} {d.unique_id}")
    print("ELP indices (probed):", elp_indices(n=None))
