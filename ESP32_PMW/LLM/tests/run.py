"""One offline entrypoint using the repository's self-check convention."""
import unittest


def run():
    from LLM.tests import test_sync, test_capture_pose, test_auto_pose, test_pipeline, test_world_takeoff, test_telemetry, test_reference, test_start
    test_sync.run()
    test_capture_pose.run_tests()
    test_auto_pose.run()
    test_telemetry.run()
    test_reference.run()
    test_start.run()
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_world_takeoff))
    if not result.wasSuccessful():
        raise AssertionError("world/takeoff checks failed")
    test_pipeline.run()


if __name__ == "__main__":
    run()
