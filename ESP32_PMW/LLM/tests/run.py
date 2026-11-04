"""Every LLM offline check: `.venv/bin/python -m LLM.tests.run`. No camera, serial or coil drive."""
import unittest


def run():
    from LLM.tests import test_sync, test_capture_pose, test_pipeline, test_world_takeoff, test_telemetry, test_start
    test_sync.run()
    test_capture_pose.run_tests()
    test_telemetry.run()
    test_start.run()
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_world_takeoff))
    if not result.wasSuccessful():
        raise AssertionError("world/takeoff checks failed")
    from controller.camera import test_portable
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_portable))
    if not result.wasSuccessful():
        raise AssertionError("portable recorder checks failed")
    from LLM.tests.test_experiment import run as experiment_checks
    experiment_checks()
    from LLM.tests import test_drive_only
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_drive_only))
    if not result.wasSuccessful():
        raise AssertionError("frequency-only bench checks failed")
    from LLM.tests import test_holder_detection, test_holder, test_holder_geometry
    for module in (test_holder_detection, test_holder, test_holder_geometry):
        result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        if not result.wasSuccessful():
            raise AssertionError(f"{module.__name__} checks failed")
    import runpy
    runpy.run_module("LLM.record.ramps", run_name="__main__")   # every LLM/ramps file loads
    test_pipeline.run()
    from LLM.tests import test_bench
    test_bench.run()


if __name__ == "__main__":
    run()
    print("All offline checks passed")
