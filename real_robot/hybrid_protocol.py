"""Same-frame checks for Ridgeback perception plus Thor waypoint inference."""

from __future__ import annotations


TARGET_KEYS = (
    "target_query", "target_state", "target_valid", "target_reason",
    "target_track_id", "target_bbox", "target_grounding_score",
    "target_person_score", "target_reid_similarity", "target_people_count",
)
IDENTITY_KEYS = ("target_state", "target_valid", "target_track_id", "target_bbox")


def checked_reply(reply: object, request_id: int, frame_id: int, mode: str) -> dict:
    if not isinstance(reply, dict) or reply.get("ok") is not True:
        error = reply.get("error") if isinstance(reply, dict) else "missing response"
        raise RuntimeError(f"{mode} service failed: {error}")
    if (type(reply.get("request_id")) is not int or reply["request_id"] != request_id
            or type(reply.get("frame_id")) is not int or reply["frame_id"] != frame_id
            or reply.get("mode") != mode):
        raise RuntimeError(f"{mode} response does not match this frame and service")
    return reply


def target_payload(perception: dict, request_id: int, frame_id: int) -> dict:
    checked_reply(perception, request_id, frame_id, "perception")
    state = perception.get("target_state")
    valid = perception.get("target_valid")
    if state not in ("SEARCHING", "LOCKED", "UNCERTAIN", "LOST") or type(valid) is not bool:
        raise RuntimeError("perception returned an invalid target state")
    if valid and state != "LOCKED":
        raise RuntimeError("perception marked a non-locked target valid")
    if any(key not in perception for key in TARGET_KEYS):
        raise RuntimeError("perception omitted target metadata")
    return {key: perception[key] for key in TARGET_KEYS}


def combine_replies(perception: dict, planner: dict, request_id: int, frame_id: int) -> dict:
    target = target_payload(perception, request_id, frame_id)
    checked_reply(planner, request_id, frame_id, "planner")
    if any(planner.get(key) != target[key] for key in IDENTITY_KEYS):
        raise RuntimeError("planner used a different target identity or frame")
    if (planner.get("planner_ran") is True) != target["target_valid"]:
        raise RuntimeError("planner state disagrees with target validity")
    result = dict(planner)
    result.update(target)
    result["target_debug"] = perception.get("target_debug")
    result["target_image_b64"] = perception.get("target_image_b64")
    result["inference_seconds"] = (
        float(perception.get("inference_seconds", 0.0))
        + float(planner.get("inference_seconds", 0.0))
    )
    result["mode"] = "hybrid"
    return result
