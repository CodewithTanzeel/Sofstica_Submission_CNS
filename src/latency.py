from __future__ import annotations

import time
from typing import Dict, Any


def measure_latency(func, *args, **kwargs) -> Dict[str, Any]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return {"elapsed_seconds": elapsed, "result": result}
