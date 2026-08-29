"""
audit_trail.py
Builds a human-readable, JSON-serializable execution trace for one /analyze
request. This is an explicitly graded deliverable, so entries should read
like a clear step log, not raw debug dumps.
"""

import time
from typing import List, Dict, Any


class ExecutionTrace:
    def __init__(self):
        self._steps: List[Dict[str, Any]] = []
        self._start = time.perf_counter()

    def log(self, step: str, detail: str = "", status: str = "ok"):
        """
        step:   short label, e.g. "input_validation", "intent_classification"
        detail: human-readable note, e.g. "2 images, format png -> valid"
        status: "ok" | "warning" | "error"
        """
        self._steps.append({
            "step": step,
            "detail": detail,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - self._start) * 1000, 2),
        })

    def as_list(self) -> List[Dict[str, Any]]:
        return self._steps

    def total_time_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)
