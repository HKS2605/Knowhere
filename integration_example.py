"""
How M1 (agentic controller) wires Mahek's VQA module into the tool registry,
and how M2 (RS backbone) attaches once their CLIP fine-tune is ready.

This file is documentation-as-code — show it to M1 directly, it's the
exact pattern for their tool_registry dict.
"""

from rs_vqa import RSVQASpecialist

# --- 1. M1 instantiates ONE specialist at server startup (not per-request —
#        loading BLIP-VQA is expensive, do it once in FastAPI's startup event) ---
vqa_specialist = RSVQASpecialist()

# --- 2. M1 registers it in the tool registry (task -> function mapping) ---
tool_registry = {
    "single_image_vqa": vqa_specialist.answer_vqa,
    # "captioning": jugal_caption_fn,
    # "change_detection": m5_change_fn,
    # "optical_sar_fusion": m2_fusion_fn,
}

# --- 3. When M2's fine-tuned CLIP backbone is ready, ONE call upgrades
#        Mahek's refinement stage from string-matching to real embeddings.
#        M2's function signature must be: PIL.Image -> np.ndarray
#
#   from m2_rs_backbone import embed_image
#   vqa_specialist.set_rs_backbone(embed_image)
#
#   No other code changes needed — answer_vqa() picks it up automatically.

# --- 4. Inside the /analyze FastAPI endpoint, M1 calls it like this: ---
def handle_request(task: str, image, query: str) -> dict:
    fn = tool_registry.get(task)
    if fn is None:
        return {"status": "error", "error": f"Unknown task: {task}"}
    return fn(image, query)  # already JSON-safe, already never raises


if __name__ == "__main__":
    # smoke test the wiring itself
    from PIL import Image
    dummy = Image.new("RGB", (128, 128), color=(70, 130, 180))
    out = handle_request("single_image_vqa", dummy, "Is there water in this image?")
    import json
    print(json.dumps(out, indent=2))
