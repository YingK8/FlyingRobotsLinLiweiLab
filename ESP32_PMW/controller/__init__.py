"""
The live host pipeline: camera -> calib -> pose -> control, plus viz.

Import by full path so a reader can see which stage a name comes from:

    from controller.calib.rig import StereoRig
    from controller.control import z_track

Each stage's reasoning lives in its own `theory.md`.
"""
