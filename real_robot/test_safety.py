import math
import unittest

from real_robot.safety import SafetyConfig, SafetyState, directional_obstacle_distance, evaluate, is_fresh, rate_limit


class SafetyTests(unittest.TestCase):
    def ready_state(self, **changes):
        values = dict(
            enabled=True,
            enable_stamp=10.0,
            camera_stamp=10.0,
            inference_stamp=10.0,
            scan_stamp=10.0,
            obstacle_distance=2.0,
            rotation_obstacle_distance=2.0,
            estop_seen=True,
            estop_stamp=10.0,
            estop_active=False,
            inference_connected=True,
            target_valid=True,
            trajectory_valid=True,
            target_position_valid=True,
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

    def test_close_side_obstacle_does_not_block_forward_corridor(self):
        points = [(0.05, -0.55), (2.4, 0.0)]
        distance = directional_obstacle_distance(points, (0.2, 0.0, 0.0), 0.48, 0.397, 0.05, 60.0)
        self.assertAlmostEqual(distance, 2.4)

    def test_close_side_obstacle_blocks_motion_toward_it(self):
        points = [(0.05, -0.55), (2.4, 0.0)]
        distance = directional_obstacle_distance(points, (0.0, -0.2, 0.0), 0.48, 0.397, 0.05, 60.0)
        self.assertAlmostEqual(distance, math.hypot(0.05, 0.55))

    def test_rotation_is_suppressed_near_side_obstacle(self):
        state = self.ready_state(obstacle_distance=2.4, rotation_obstacle_distance=0.55)
        allowed, reason, command = evaluate(SafetyConfig(), state, 10.1)
        self.assertTrue(allowed)
        self.assertEqual(reason, "ready_rotation_suppressed")
        self.assertEqual(command, (0.1, -0.1, 0.0))

    def test_pure_rotation_stops_near_side_obstacle(self):
        state = self.ready_state(candidate=(0.0, 0.0, 0.2), obstacle_distance=2.4, rotation_obstacle_distance=0.55)
        self.assertEqual(evaluate(SafetyConfig(), state, 10.1)[1], "obstacle_too_close_for_rotation")

    def test_stale_inputs_stop(self):
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(camera_stamp=8.0), 10.1)[1], "camera_stale")
        self.assertEqual(evaluate(SafetyConfig(), self.ready_state(scan_stamp=8.0), 10.1)[1], "scan_stale")

    def test_remote_inference_loss_stops(self):
        config = SafetyConfig()
        self.assertEqual(evaluate(config, self.ready_state(inference_connected=False), 10.1)[1],
                         "inference_disconnected")
        self.assertEqual(evaluate(config, self.ready_state(inference_stamp=8.0), 10.1)[1],
                         "inference_stale")
        self.assertFalse(is_fresh(10.1, 9.0, config.camera_timeout))

    def test_target_lock_is_required(self):
        state = self.ready_state(target_valid=False)
        self.assertEqual(evaluate(SafetyConfig(), state, 10.1)[1], "target_not_locked")

    def test_trajectory_consistency_is_required(self):
        state = self.ready_state(trajectory_valid=False)
        self.assertEqual(evaluate(SafetyConfig(), state, 10.1)[1], "trajectory_target_mismatch")

    def test_target_position_consistency_is_required(self):
        state = self.ready_state(target_position_valid=False)
        self.assertEqual(evaluate(SafetyConfig(), state, 10.1)[1], "target_position_inconsistent")

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
