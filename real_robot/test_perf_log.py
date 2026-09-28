from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from real_robot.perf_log import PerfLog, timed
from real_robot.summarize_perf import percentile, summarize


class PerfLogTests(unittest.TestCase):
    def test_opt_in_log_and_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"OMTRACKVLA_PROFILE_DIR": directory}
        ):
            inference = PerfLog("inference")
            stages = {}
            with timed(stages, "jpeg_decode"):
                sum(range(100))
            inference.record(event="frame", request_id=1, planner_updated=True,
                             inference_ms=12.0, stages_ms=stages)
            inference.close()
            bridge = PerfLog("bridge")
            bridge.record(event="control", reason="dry_run:ready", allowed=False,
                          tick_interval_ms=50.0, tick_ms=1.0)
            bridge.close()
            report = summarize(Path(directory))
        self.assertEqual(report["counts"], {"inference.frame": 1, "bridge.control": 1})
        self.assertEqual(report["timings"]["inference.frame.inference_ms"]["median_ms"], 12.0)
        self.assertIn("inference.frame.jpeg_decode", report["timings"])
        self.assertEqual(report["control_reasons"], {"dry_run:ready": 1})

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([0.0, 10.0, 20.0], 0.95), 19.0)


if __name__ == "__main__":
    unittest.main()
