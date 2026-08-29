"""
tool_registry.py
Maps each intent -> the function that handles it. All functions currently
return STUB output so the frontend (M6) can integrate against a live API
from minute one. Replace each with the real teammate function as it lands.

Every function signature: (images: list[str], query: str) -> dict
images is a list of local file paths already saved to disk by main.py.
"""

from schema import make_output, make_error_output


# ---- STUBS: replace these with real imports as teammates finish ----

def stub_single_vqa(images, query):
    # TODO: M3 replace this stub with: from vqa_module import answer_vqa
    return make_output(
        task_type="single_vqa",
        statement=f"[STUB] Answer to '{query}' would appear here.",
        confidence=0.42,
    )


def stub_captioning(images, query):
    # TODO: M4 replace this stub with real captioning pipeline
    return make_output(
        task_type="captioning",
        statement="[STUB] Generated caption would appear here.",
        confidence=0.5,
    )


def stub_grounding(images, query):
    # TODO: M4 replace this stub with real grounding + overlay generation
    return make_output(
        task_type="grounding",
        statement="[STUB] Grounding result with bounding box would appear here.",
        confidence=0.5,
        overlay_path=None,
    )


def stub_change_detection(images, query):
    # TODO: M5 replace this stub with real bi-temporal change detection
    return make_output(
        task_type="change_detection",
        statement="[STUB] Change description between the two images would appear here.",
        confidence=0.5,
        overlay_path=None,
    )


def stub_fusion(images, query):
    # TODO: M2 replace this stub with: from m2_interface import get_fusion_analysis
    return make_output(
        task_type="fusion",
        statement="[STUB] Optical-SAR fusion statement would appear here.",
        confidence=0.5,
        overlay_path=None,
    )


def stub_landcover_classification(images, query):
    # TODO: M2 replace this stub with: from m2_interface import get_landcover_label
    return make_output(
        task_type="landcover_classification",
        statement="[STUB] Predicted land cover label would appear here.",
        confidence=0.5,
    )


TOOL_REGISTRY = {
    "single_vqa": stub_single_vqa,
    "captioning": stub_captioning,
    "grounding": stub_grounding,
    "change_detection": stub_change_detection,
    "fusion": stub_fusion,
    "landcover_classification": stub_landcover_classification,
}


def run_task(intent: str, images, query: str) -> dict:
    """Looks up the registry and calls the right function. Never raises."""
    func = TOOL_REGISTRY.get(intent)
    if func is None:
        return make_error_output(intent, f"No handler registered for intent '{intent}'.")
    try:
        return func(images, query)
    except Exception as e:
        return make_error_output(intent, str(e))
