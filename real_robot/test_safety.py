import math
import unittest

from real_robot.safety import SafetyConfig, SafetyState, evaluate, rate_limit


class SafetyTests(unittest.TestCase):
    def ready_state(self, **changes):
        values = dict(
            enabled=True,
            enable_stamp=10.0,
            camera_stamp=10.0,
            inference_stamp=10.0,
            scan_stamp=10.0,
            obstacle_distance=2.0,
            estop_seen=True,
            estop_stamp=10.0,
            estop_active=False,
            inference_connected=True,
            candidate=(0.1, -0.1, 0.2),
        )
        values.update(changes)
        return SafetyState(**values)

    def test_ready(self):
        allowed, reason, command = evaluate(SafetyConfig(), self.ready_state(), 10.1)
        self.assertTrue(allowed)
        self.assertEqual(reason, "ready")
        self.assertEqual(command, (0.1, -0.1, 0.2))

    def test_deadman_timeout_stops(self):
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(enable_stamp=9.0), 10.1)[1], "deadman_not_held")

    def test_estop_stops(self):
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(estop_active=True), 10.1)[1], "estop_active")
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(estop_stamp=8.0), 10.1)[1], "estop_state_stale")

    def test_optional_estop_does_not_require_a_message(self):
        config = SafetyConfig(require_estop=False)
        state = self.ready_state(estop_seen=False, estop_stamp=None, estop_active=True)
        self.assertTrue(evaluate(config, state, 10.1)[0])

    def test_obstacle_stops(self):
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(obstacle_distance=0.4), 10.1)[1], "obstacle_too_close")

    def test_stale_inputs_stop(self):
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(camera_stamp=8.0), 10.1)[1], "camera_stale")
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(scan_stamp=8.0), 10.1)[1], "scan_stale")

    def test_nonfinite_and_clipping(self):
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(candidate=(math.nan, 0, 0)), 10.1)[1], "inference_nonfinite")
        allowed, _, command = evaluate(SafetyConfig(), self.ready_state(candidate=(5, -5, 5)), 10.1)
        self.assertTrue(allowed)
        self.assertEqual(command, (0.2, -0.2, 0.35))

    def test_rate_limit(self):
        command = rate_limit((0, 0, 0), (1, -1, 1), 0.1, 0.3, 0.5)
        self.assertAlmostEqual(command[0], 0.03)
        self.assertAlmostEqual(command[1], -0.03)
        self.assertAlmostEqual(command[2], 0.05)


if __name__ == "__main__":
    unittest.main()
