import math
import unittest

import numpy as np

from real_robot.target_geometry import (
    CameraGeometry,
    extrinsics_rotation,
    fuse_target_command,
    locate_target,
    optical_to_base,
    trajectory_follows_target,
)


K = np.array([[390.0, 0.0, 320.0], [0.0, 390.0, 240.0], [0.0, 0.0, 1.0]])


def geometry(mount=(0.3, 0.0, 1.0, 0.0, 0.0, 0.0), t=(0.0, 0.0, 0.0)):
    return CameraGeometry(K, K, np.eye(3), np.array(t), mount)


class TargetGeometryTests(unittest.TestCase):
    def test_extrinsics_are_column_major(self):
        rotation = extrinsics_rotation([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(rotation[0].tolist(), [1, 4, 7])

    def test_optical_axis_is_base_forward(self):
        point = optical_to_base(np.array([0.0, 0.0, 2.0]), (0.3, 0.0, 1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(point, [2.3, 0.0, 1.0])
        # Image right (optical +x) is base -y.
        point = optical_to_base(np.array([1.0, 0.0, 2.0]), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(point, [2.0, -1.0, 0.0])

    def test_locate_person_ignores_background(self):
        depth = np.full((480, 640), 5.0, dtype=np.float32)
        depth[100:460, 400:520] = 2.0  # person right of center, wall behind
        position, reason = locate_target(depth, [400, 100, 520, 460], geometry())
        self.assertEqual(reason, "target_located")
        self.assertAlmostEqual(position[0], 2.3, places=2)
        self.assertLess(position[1], 0.0)
        self.assertAlmostEqual(position[1], -(460 - 320) * 2.0 / 390.0, places=2)

    def test_depth_baseline_shifts_projection(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[100:460, 300:340] = 2.0
        # With a 0.06 m baseline the person projects ~12 px left in color.
        position, reason = locate_target(depth, [288, 100, 328, 460], geometry(t=(-0.06, 0.0, 0.0)), stride=1)
        self.assertEqual(reason, "target_located")
        _, reason = locate_target(depth, [340, 100, 380, 460], geometry(t=(-0.06, 0.0, 0.0)), stride=1)
        self.assertEqual(reason, "target_depth_insufficient")

    def test_too_close_person_is_not_placed_on_background(self):
        # Person has no valid depth (inside minimum range); only far wall remains.
        depth = np.full((480, 640), 3.6, dtype=np.float32)
        depth[:, 0:260] = 0.0
        _, reason = locate_target(depth, [0, 20, 380, 470], geometry())
        self.assertEqual(reason, "target_depth_sparse")
        _, reason = locate_target(depth, [0, 0, 380, 479], geometry(), image_size=(640, 480))
        self.assertEqual(reason, "target_too_close_for_depth")

    def test_mixed_depth_surfaces_are_rejected(self):
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        depth[:, ::2] = 1.0  # alternating near/far columns: no single surface
        _, reason = locate_target(depth, [200, 100, 440, 460], geometry(), stride=1)
        self.assertEqual(reason, "target_depth_inconsistent")

    def test_missing_depth_is_reported(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        self.assertEqual(locate_target(depth, [100, 100, 200, 400], geometry())[1], "depth_empty")

    def test_forward_and_detour_paths_are_accepted(self):
        forward = [[0.07, 0.0, 0.0], [0.14, 0.0, 0.0]]
        detour = [[0.05, 0.07, 0.3], [0.08, 0.14, 0.5]]
        self.assertTrue(trajectory_follows_target(forward, (3.0, 0.0))[0])
        self.assertTrue(trajectory_follows_target(detour, (3.0, 0.0))[0])

    def test_retreat_from_distant_target_is_rejected(self):
        away = [[-0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]]
        ok, reason, _, _ = trajectory_follows_target(away, (3.0, 0.0))
        self.assertFalse(ok)
        self.assertEqual(reason, "path_moves_away_from_target")
        self.assertTrue(trajectory_follows_target(away, (3.0, 0.0), reject_retreat=False)[0])

    def test_path_toward_other_side_is_rejected(self):
        # Target 2 m ahead-left; path heads hard right (toward another person).
        ok, reason, _, _ = trajectory_follows_target([[0.1, -0.2, -1.0]], (1.5, 1.3))
        self.assertFalse(ok)
        self.assertEqual(reason, "path_moves_away_from_target")

    def test_closing_inside_follow_distance_is_rejected(self):
        ok, reason, _, _ = trajectory_follows_target([[0.2, 0.0, 0.0]], (1.0, 0.0), min_follow_distance=0.9)
        self.assertFalse(ok)
        self.assertEqual(reason, "path_too_close_to_target")
        # Backing away from a too-close target is allowed.
        self.assertTrue(trajectory_follows_target([[-0.1, 0.0, 0.0]], (0.6, 0.0))[0])

    def test_distances_are_reported(self):
        _, _, current, final = trajectory_follows_target([[0.0, 0.0, 0.0]], (3.0, 4.0))
        self.assertTrue(math.isclose(current, 5.0) and math.isclose(final, 5.0))

    def test_fusion_blends_aligned_omtrack_toward_target(self):
        command, info = fuse_target_command((0.15, 0.1, 0.2), (2.0, 0.0))
        self.assertEqual(info["reason"], "blended_omtrack_toward_target")
        self.assertGreater(command[0], 0.0)
        self.assertGreater(command[1], 0.0)
        self.assertLess(command[1], 0.1)

    def test_fusion_replaces_opposite_omtrack_direction(self):
        command, info = fuse_target_command((-0.2, 0.0, -0.3), (2.0, 0.0))
        self.assertEqual(info["reason"], "replaced_misaligned_omtrack")
        self.assertGreater(command[0], 0.0)
        self.assertAlmostEqual(command[1], 0.0)
        self.assertAlmostEqual(command[2], 0.0)

    def test_fusion_steers_to_off_center_target(self):
        command, _ = fuse_target_command((0.2, 0.0, 0.0), (2.0, 1.0))
        self.assertGreater(command[0], 0.0)
        self.assertGreater(command[1], 0.0)
        self.assertGreater(command[2], 0.0)

    def test_fusion_stops_translation_at_follow_distance(self):
        command, info = fuse_target_command((0.2, 0.0, 0.2), (0.8, 0.0))
        self.assertEqual(info["reason"], "within_follow_distance")
        self.assertEqual(command, (0.0, 0.0, 0.0))

    def test_fusion_speed_is_distance_limited(self):
        command, info = fuse_target_command((0.2, 0.0, 0.0), (1.0, 0.0))
        self.assertAlmostEqual(math.hypot(command[0], command[1]), 0.04)
        self.assertAlmostEqual(info["distance_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
