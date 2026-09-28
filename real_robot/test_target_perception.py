from types import SimpleNamespace
import unittest

import numpy as np

from real_robot.target_perception import (
    TargetPerception,
    appearance_descriptor,
    box_iou,
    cosine_similarity,
    attribute_on_person,
    color_fraction,
    target_attribute,
    target_query,
    trajectory_is_directionally_consistent,
)


class TargetPerceptionHelpersTests(unittest.TestCase):
    def test_target_query(self):
        self.assertEqual(
            target_query("Follow the person holding a blue basket. Maintain a safe distance."),
            "the person holding a blue basket",
        )

    def test_box_iou(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_target_attribute(self):
        self.assertEqual(target_attribute("the person who is visibly holding a blue basket"), "blue basket")
        self.assertEqual(target_attribute("the person wearing a black jacket"), "black jacket")
        self.assertEqual(target_attribute("the man in a red shirt"), "red shirt")
        self.assertEqual(target_attribute("the person holding something blue color basket"), "blue basket")
        self.assertIsNone(target_attribute("the person directly in front of you"))

    def test_attribute_must_be_on_and_smaller_than_person(self):
        person = [0, 0, 100, 200]
        self.assertEqual(attribute_on_person([40, 80, 60, 120], person), 1.0)
        self.assertEqual(attribute_on_person([110, 80, 130, 120], person), 0.0)
        # A scene-sized spurious attribute box does not validate anyone.
        self.assertEqual(attribute_on_person([-50, -50, 300, 300], person), 0.0)

    def test_color_fraction(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :50] = (200, 60, 20)  # BGR blue
        self.assertGreater(color_fraction(image, [0, 0, 50, 100], "blue"), 0.9)
        self.assertLess(color_fraction(image, [50, 0, 100, 100], "blue"), 0.01)
        self.assertGreater(color_fraction(image, [50, 0, 100, 100], "black"), 0.9)

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([1, 0])), 1.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([0, 1])), 0.0)

    def test_center_target_accepts_curved_path(self):
        valid, reason = trajectory_is_directionally_consistent(
            [270, 100, 370, 400], 640, [[0, 0, 0], [0.1, -0.04, -0.2]]
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "target_centered")

    def test_opposite_direction_is_rejected(self):
        valid, reason = trajectory_is_directionally_consistent(
            [20, 100, 120, 400], 640, [[0, 0, 0], [0.1, -0.08, -0.2]]
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "trajectory_points_away_from_target")

    def test_same_direction_is_accepted(self):
        valid, reason = trajectory_is_directionally_consistent(
            [20, 100, 120, 400], 640, [[0, 0, 0], [0.1, 0.08, 0.2]]
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "direction_consistent")

    def test_appearance_separates_black_and_white_clothing(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, 100:] = 255
        black = appearance_descriptor(image, [0, 0, 100, 200])
        white = appearance_descriptor(image, [100, 0, 200, 200])
        self.assertAlmostEqual(cosine_similarity(black, black), 1.0, places=5)
        self.assertLess(cosine_similarity(black, white), 0.1)


BLACK = np.array([1.0, 0.0, 0.0], dtype=np.float32)
BLACK_TURNED = np.array([0.9, 0.436, 0.0], dtype=np.float32)
WHITE = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def person(track_id, feature, x=100.0):
    return {"bbox": np.array([x, 50, x + 80, 300], dtype=np.float32), "track_id": track_id, "score": 0.9, "feature": feature}


class ScriptedPerception(TargetPerception):
    """TargetPerception with scripted tracks and grounding instead of models."""

    def __init__(self, frames, grounded_ids):
        self.grounding_interval = 1
        self.acquire_frames = 3
        self.lost_frames = 4
        self.ambiguity_margin = 0.08
        self.reid_match_threshold = 0.72
        self.gallery_update_threshold = 0.85
        self.tracker = SimpleNamespace(reset=lambda: None)
        self.frames = list(frames)
        self.grounded_ids = list(grounded_ids)
        self.reset()

    def _track_people(self, bgr):
        return self.frames.pop(0)

    def _select_grounded_track(self, rgb, tracks):
        # Grounding runs only when the state machine needs it, so consume lazily.
        wanted = self.grounded_ids.pop(0) if self.grounded_ids else None
        match = self._current_track(tracks, wanted)
        return (match, "prompt_match_found") if match is not None else (None, "no_prompt_match")

    def _annotate(self, bgr, tracks, selected, valid):
        return bgr

    def run(self, prompt="Follow the person wearing a black jacket."):
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        return [self.update(image, prompt) for _ in range(len(self.frames))]


class TargetStateMachineTests(unittest.TestCase):
    def test_locks_only_after_consecutive_confirmation(self):
        frames = [[person(1, BLACK), person(2, WHITE, 400)]] * 4
        results = ScriptedPerception(frames, [1] * 4).run()
        self.assertEqual([r.state for r in results], ["SEARCHING", "SEARCHING", "LOCKED", "LOCKED"])
        self.assertEqual([r.valid for r in results], [False, False, True, True])
        self.assertEqual(results[-1].track_id, 1)

    def test_interrupted_confirmation_restarts(self):
        frames = [[person(1, BLACK)], [], [person(1, BLACK)], [person(1, BLACK)], [person(1, BLACK)]]
        results = ScriptedPerception(frames, [1, None, 1, 1, 1]).run()
        self.assertFalse(results[3].valid)
        self.assertTrue(results[4].valid)

    def test_lost_target_is_not_replaced_by_another_person(self):
        locked = [[person(1, BLACK), person(2, WHITE, 400)]] * 3
        gone = [[person(2, WHITE, 400)]] * 6
        # Even if grounding wrongly prefers the white-shirt person, recovery is refused.
        results = ScriptedPerception(locked + gone, [1] * 3 + [2] * 6).run()
        self.assertTrue(results[2].valid)
        self.assertTrue(all(not r.valid for r in results[3:]))
        self.assertEqual(results[-1].state, "LOST")

    def test_track_id_switch_to_different_appearance_stops(self):
        frames = [[person(1, BLACK)]] * 3 + [[person(1, WHITE)]]
        results = ScriptedPerception(frames, [1] * 4).run()
        self.assertEqual(results[-1].state, "UNCERTAIN")
        self.assertEqual(results[-1].reason, "appearance_mismatch")
        self.assertFalse(results[-1].valid)

    def test_original_track_recovers_after_brief_occlusion(self):
        frames = [[person(1, BLACK)]] * 3 + [[]] + [[person(1, BLACK_TURNED)]] * 3
        results = ScriptedPerception(frames, [1] * 7).run()
        self.assertEqual(results[3].state, "UNCERTAIN")
        self.assertEqual([r.valid for r in results[4:]], [False, False, True])
        self.assertEqual(results[-1].reason, "identity_recovered")

    def test_new_track_recovers_only_with_appearance_and_prompt(self):
        frames = [[person(1, BLACK)]] * 3 + [[]] + [[person(7, BLACK)]] * 3
        # Grounding is consulted on the first search frame and on each recovery frame.
        refused = ScriptedPerception(frames, [1, None, None, None]).run()
        self.assertTrue(all(not r.valid for r in refused[3:]))
        accepted = ScriptedPerception(frames, [1, 7, 7, 7]).run()
        self.assertTrue(accepted[-1].valid)
        self.assertEqual(accepted[-1].track_id, 7)


if __name__ == "__main__":
    unittest.main()
