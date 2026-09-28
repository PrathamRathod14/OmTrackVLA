#!/usr/bin/env python3
"""Summarize opt-in Ridgeback JSONL timing logs without changing the robot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty values")
    rank = (len(ordered) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(path: Path) -> dict:
    samples = defaultdict(list)
    counts = Counter()
    reasons = Counter()
    timestamps = defaultdict(list)
    for name in ("inference", "bridge"):
        file_path = path / f"{name}.jsonl"
        if not file_path.is_file():
            continue
        for line in file_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            event = str(row.get("event", "unknown"))
            counts[f"{name}.{event}"] += 1
            if event == "control":
                reasons[str(row.get("reason", "unknown"))] += 1
            if event == "frame":
                group = "planner" if row.get("planner_updated") else "tracking"
                timestamps[f"{name}.{group}"].append(float(row["wall_time"]))
            for key, value in row.items():
                if key.endswith("_ms") and isinstance(value, (int, float)):
                    samples[f"{name}.{event}.{key}"].append(float(value))
            for key, value in (row.get("stages_ms") or {}).items():
                if isinstance(value, (int, float)):
                    samples[f"{name}.{event}.{key}"].append(float(value))
    metrics = {
        key: {
            "count": len(values),
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(percentile(values, 0.95), 3),
            "max_ms": round(max(values), 3),
        }
        for key, values in sorted(samples.items()) if values
    }
    cadence = {}
    for key, times in timestamps.items():
        if len(times) >= 2:
            span = max(times) - min(times)
            if span > 0:
                cadence[key] = round((len(times) - 1) / span, 3)
    return {"counts": dict(counts), "cadence_hz": cadence,
            "control_reasons": dict(reasons), "timings": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="OMTRACKVLA_PROFILE_DIR used for the run")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    report = summarize(args.directory)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
