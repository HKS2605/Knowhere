# SatQuery — Single-Image VQA Module (M3 / Mahek)

Implements the **"Pipeline for Single VQA"** diagram end-to-end. This is the
deliverable for the "Single-Image VQA Specialist" role: `answer_vqa(image, query)`
handed off to M1's tool registry.

## Files

| File | Purpose |
|---|---|
| `rs_vqa.py` | The module. `RSVQASpecialist` class + `answer_vqa()`. |
| `integration_example.py` | Exact pattern for M1 to register this in FastAPI, and for M2 to attach their backbone. |
| `requirements.txt` | Dependencies (install with `pip install -r requirements.txt`). |

## Pipeline stage → code mapping

| Diagram box | Method |
|---|---|
| Image + query input | `answer_vqa()` args |
| Preprocessing & validation | `preprocess_image()` |
| Question type classifier | `classify_question_type()` |
| VQA model inference (BLIP-VQA) | `run_inference()` |
| RS-domain refinement (+ RS backbone/M2) | `refine_with_rs_domain()` |
| Confidence scoring | `compute_confidence()` |
| Answer output (JSON) | `VQAResult.to_json()` |

## Quick start

```bash
pip install -r requirements.txt

# synthetic smoke test, no dataset needed
python rs_vqa.py --demo

# real query
python rs_vqa.py --image path/to/tile.png --query "Is there water in the image?"
```

## Output contract (what M1 receives)

```json
{
  "task": "single_image_vqa",
  "query": "Is there water in the image?",
  "question_type": "yes_no",
  "raw_answer": "yes",
  "refined_answer": "yes",
  "confidence": 0.87,
  "confidence_breakdown": {
    "softmax_confidence": 0.91,
    "similarity_confidence": 1.0,
    "weights": {"softmax": 0.6, "similarity": 0.4}
  },
  "model_used": "Salesforce/blip-vqa-base",
  "rs_backbone_used": false,
  "image_metadata": {"source_format": ".png", "original_size": [512, 512], "resized": false},
  "latency_ms": 812.4,
  "status": "success",
  "error": null
}
```

`status`/`error` mean **`answer_vqa` never raises** — M1's output combiner
can call it directly without wrapping every teammate's function in a
try/except.

## Handoff points with teammates

- **M2 (RS backbone):** once their CLIP fine-tune is ready, one call —
  `specialist.set_rs_backbone(m2_embed_fn)` — upgrades RS-domain
  refinement from string matching to real embedding similarity. No other
  code changes. Confirm with M2 whether their function returns
  image-only embeddings or also exposes a text encoder for the RS label
  set — the current fallback assumes image-only and needs that detail to
  finish `_refine_via_embeddings`.
- **M1 (controller):** import `RSVQASpecialist`, instantiate once at
  FastAPI startup (not per-request — model loading is expensive), assign
  `.answer_vqa` into the tool registry. See `integration_example.py`.

## Before the demo

1. Pull the actual 5–8 RSVQA/VRSBench samples (real image paths + questions).
2. Replace the synthetic samples in `rs_vqa.py`'s `__main__` block with
   the real ones and run `python rs_vqa.py --demo` to regenerate
   `vqa_demo_cache.json` — this is your fallback if live GPU inference
   is flaky during the presentation.
3. Screenshot 2–3 of the JSON outputs for the PPT slide.

## Known TODOs (flag these to the team, don't let them surprise you later)

- `_refine_via_embeddings` needs M2's exact interface confirmed (see above).
- GeoTIFF band order assumption (`_load_geotiff`) takes the first 3 bands
  as RGB — fine for BigEarthNet's default composite, but confirm with M2
  if any samples use a non-standard band order.
- Count questions currently parse digits/spelled-out numbers only up to
  ten — extend `word_to_num` if larger counts show up in your sample set.
