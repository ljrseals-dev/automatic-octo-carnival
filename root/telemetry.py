"""Performance metrics collection."""
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from .logging_setup import setup_logging
from .paths import log_dir

log = setup_logging()

class Telemetry:
    def __init__(self, name: str = "session"):
        self.name = name
        self.metrics: Dict[str, list] = {}
        self.counters: Dict[str, int] = {}
        self._timers: Dict[str, float] = {}

    def count(self, key: str, n: int = 1):
        self.counters[key] = self.counters.get(key, 0) + n

    def record(self, key: str, value: float):
        self.metrics.setdefault(key, []).append(float(value))

    @contextmanager
    def timer(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.record(key, dt)
            self.count(f"{key}_count")

    def summary(self) -> dict:
        out = {
            'name': self.name,
            'counters': dict(self.counters),
            'metrics': {},
        }
        for key, values in self.metrics.items():
            if not values:
                continue
            import numpy as np
            arr = np.array(values)
            out['metrics'][key] = {
                'n': len(values),
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
            }
        return out

    def dump(self, path: str = None):
        if path is None:
            path = log_dir() / f"telemetry_{self.name}.json"
        Path(path).write_text(json.dumps(self.summary(), indent=2))
        log.info(f"Telemetry written: {path}")
        return str(path)

# Global session telemetry
_SESSION = Telemetry("global")

def session() -> Telemetry:
    return _SESSION
