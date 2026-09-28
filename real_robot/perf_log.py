"""Opt-in JSONL stage timings for the Ridgeback dry-run pipeline."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable, Dict, Iterator, Optional


class PerfLog:
    def __init__(self, component: str) -> None:
        directory = os.environ.get("OMTRACKVLA_PROFILE_DIR", "").strip()
        self.enabled = bool(directory)
        self._lock = threading.Lock()
        self._file = None
        if self.enabled:
            path = Path(directory).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            self._file = (path / f"{component}.jsonl").open("a", buffering=1)

    def record(self, **fields: object) -> None:
        if self._file is None:
            return
        row = {"wall_time": time.time(), **fields}
        line = json.dumps(row, separators=(",", ":"), allow_nan=False)
        with self._lock:
            self._file.write(line + "\n")

    def close(self) -> None:
        if self._file is not None:
            with self._lock:
                self._file.close()
                self._file = None


@contextmanager
def timed(
    stages: Optional[Dict[str, float]],
    name: str,
    synchronize: Optional[Callable[[], None]] = None,
) -> Iterator[None]:
    if stages is None:
        yield
        return
    if synchronize is not None:
        synchronize()
    started = time.perf_counter()
    try:
        yield
    finally:
        if synchronize is not None:
            synchronize()
        stages[name] = stages.get(name, 0.0) + (time.perf_counter() - started) * 1000.0
