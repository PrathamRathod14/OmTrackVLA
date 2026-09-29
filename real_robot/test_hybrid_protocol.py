import unittest

from real_robot.hybrid_protocol import combine_replies, target_payload
from real_robot.inference_server import decode_request_for_mode, validate_external_target


def perception_reply():
    return {
        "ok": True, "mode": "perception", "request_id": 7, "frame_id": 42,
        "inference_seconds": 0.02, "target_query": "person with blue basket",
        "target_state": "LOCKED", "target_valid": True,
        "target_reason": "identity_verified", "target_track_id": 3,
        "target_bbox": [10.0, 20.0, 90.0, 180.0],
        "target_grounding_score": 0.8, "target_person_score": 0.9,
        "target_reid_similarity": 0.95, "target_people_count": 2,
        "target_debug": {"people": []}, "target_image_b64": "jpeg",
    }


def planner_reply():
    return {
        "ok": True, "mode": "planner", "request_id": 7, "frame_id": 42,
        "inference_seconds": 0.31, "target_state": "LOCKED", "target_valid": True,
        "target_track_id": 3, "target_bbox": [10.0, 20.0, 90.0, 180.0],
        "planner_ran": True, "command": [0.1, 0.0, 0.0],
        "trajectory": [[0.0, 0.0, 0.0]],
    }


class HybridProtocolTests(unittest.TestCase):
    def test_image_free_planner_request_only_for_unlocked_target(self):
        target = perception_reply()
        target["target_valid"] = False
        target["target_state"] = "LOST"
        image = decode_request_for_mode({"encoding": "none", "target": target}, "planner")
        self.assertEqual(image.shape, (1, 1, 3))
        with self.assertRaisesRegex(ValueError, "invalid target"):
            decode_request_for_mode({"encoding": "none", "target": perception_reply()}, "planner")

    def test_combine_same_frame_results(self):
        local = perception_reply()
        target = target_payload(local, 7, 42)
        self.assertEqual(target["target_track_id"], 3)
        self.assertNotIn("target_image_b64", target)
        combined = combine_replies(local, planner_reply(), 7, 42)
        self.assertEqual(combined["mode"], "hybrid")
        self.assertEqual(combined["target_image_b64"], "jpeg")
        self.assertAlmostEqual(combined["inference_seconds"], 0.33)

    def test_rejects_stale_or_crossed_identity(self):
        remote = planner_reply()
        remote["frame_id"] = 41
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            combine_replies(perception_reply(), remote, 7, 42)
        remote = planner_reply()
        remote["target_track_id"] = 4
        with self.assertRaisesRegex(RuntimeError, "different target"):
            combine_replies(perception_reply(), remote, 7, 42)

    def test_rejects_unlocked_motion_and_invalid_bbox(self):
        local = perception_reply()
        local["target_state"] = "LOST"
        with self.assertRaisesRegex(RuntimeError, "non-locked"):
            target_payload(local, 7, 42)
        local = perception_reply()
        local["target_bbox"] = [10.0, 20.0, 900.0, 180.0]
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_external_target(target_payload(local, 7, 42), (480, 848, 3))
        remote = planner_reply()
        remote["planner_ran"] = False
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            combine_replies(perception_reply(), remote, 7, 42)


if __name__ == "__main__":
    unittest.main()
