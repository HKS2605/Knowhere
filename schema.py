"""
schema.py
Shared output contract for every task module (M2-M5) and the M1 controller.
Keep this dependency-free so any teammate can import it without extra installs.
"""

from typing import Optional, Dict, Any, List


def make_output(
    task_type: str,
    statement: str,
    confidence: float,
    overlay_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standard shape every task function must return.

    task_type: one of "single_vqa", "captioning", "grounding",
               "change_detection", "fusion", "landcover_classification"
    statement: human-readable answer/description
    confidence: float in [0, 1]
    overlay_path: path under /outputs/ (served statically), or None
    extra: any additional structured data (embeddings, raw stats, etc.)
    """
    return {
        "task": task_type,
        "statement": statement,
        "confidence": round(float(confidence), 3),
        "overlay_image": overlay_path,
        "metadata": extra or {},
    }


def make_error_output(task_type: str, error_message: str) -> Dict[str, Any]:
    """Fallback shape when a tool call fails, so the API never crashes."""
    return {
        "task": task_type,
        "statement": f"Error while processing task: {error_message}",
        "confidence": 0.0,
        "overlay_image": None,
        "metadata": {"error": True, "detail": error_message},
    }
