"""
intent_classifier.py
Rule/keyword-based task routing + input validation. No ML here on purpose —
fast, deterministic, and easy to demo/explain to judges.
"""

from typing import List, Tuple

SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff"}

# Keyword sets per task — tune these against your actual demo queries.
KEYWORDS = {
    "change_detection": ["compare", "difference", "before and after", "changed", "change"],
    "captioning": ["caption", "describe", "what is in", "summary of image"],
    "grounding": ["locate", "where is", "highlight", "point out", "show me the"],
    "fusion": ["fuse", "fusion", "combine sar", "optical and sar", "land cover"],
    "landcover_classification": ["classify", "what type of land", "land use"],
}


def validate_input(num_images: int, filenames: List[str]) -> Tuple[bool, str]:
    """Returns (is_valid, error_message)."""
    if num_images == 0:
        return False, "No image provided. At least one image is required."
    if num_images > 2:
        return False, "Too many images. Maximum of 2 images supported (bi-temporal pairs only)."

    for name in filenames:
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in SUPPORTED_FORMATS:
            return False, f"Unsupported file format '{ext}'. Allowed: {', '.join(SUPPORTED_FORMATS)}."

    return True, ""


def classify_intent(query: str, num_images: int, task_type_override: str = None) -> str:
    """
    Returns one of: single_vqa, captioning, grounding, change_detection,
    fusion, landcover_classification
    """
    if task_type_override:
        return task_type_override

    q = (query or "").lower()

    for intent, keywords in KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent

    # Fallback: no keyword matched, decide from image count.
    if num_images == 2:
        return "change_detection"
    return "single_vqa"
