"""
test_backbone.py — run this after LoRA training to verify everything works
Usage: python backbone/test_backbone.py
"""

import sys, json
sys.path.insert(0, '.')

from pathlib import Path
from PIL import Image
import numpy as np

print("=" * 60)
print("BACKBONE TEST — run after LoRA training")
print("=" * 60)

# ── Test 1: Did the checkpoint load? ──────────────────────────
print("\n[1/4] Loading backbone...")
from backbone.rs_backbone import RSBackbone
b = RSBackbone()
info = b.info()
print(json.dumps(info, indent=2))

if info["lora_adapted"]:
    print("✓ LoRA checkpoint loaded — training worked")
else:
    print("✗ LoRA NOT loaded — check checkpoints/rs_clip_lora.pt exists")
    print("  (backbone still works, just using base RemoteCLIP weights)")

# ── Test 2: Does encode_image work? ───────────────────────────
print("\n[2/4] Testing encode_image...")
dummy_img = Image.fromarray(
    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
)
emb = b.encode_image(dummy_img)
norm = float(np.linalg.norm(emb))
print(f"  Embedding shape : {emb.shape}")
print(f"  Embedding norm  : {norm:.4f}  (should be ~1.0)")
print("✓ encode_image works" if abs(norm - 1.0) < 0.01 else "✗ embedding not normalised")

# ── Test 3: Does classify work on a real RS image? ────────────
print("\n[3/4] Testing classify on 3 land cover images...")

# Make 3 synthetic images with realistic colours per land cover
# so the classification result makes semantic sense
test_cases = [
    ("water",    [30,  80,  180]),   # blue-ish
    ("urban",    [160, 150, 140]),   # grey-ish
    ("cropland", [80,  140, 60]),    # green-ish
]

for label, rgb in test_cases:
    arr = np.full((224, 224, 3), rgb, dtype=np.uint8)
    # add slight noise so it's not a flat patch
    arr = np.clip(arr + np.random.randint(-15, 15, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    result = b.classify(img, top_k=3)
    top = result["top_label"]
    score = result["top_score"]
    all_s = list(result["all_scores"].items())
    print(f"\n  Input: {label} image (colour hint: {rgb})")
    print(f"  Top prediction : '{top}' ({score:.3f})")
    print(f"  Top 3          : {[(k, round(v,3)) for k,v in all_s[:3]]}")

print("\n✓ classify works")

# ── Test 4: Does the M3 adapter work? ─────────────────────────
print("\n[4/4] Testing M3 adapter (get_rs_features)...")
from backbone.output_adapter import get_rs_features, encode_text_for_vqa

img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
rs = get_rs_features(img)

required = ["embedding", "top_label", "top_score", "all_scores", "model_used"]
missing  = [k for k in required if k not in rs]

if missing:
    print(f"✗ Missing keys: {missing}")
else:
    print(f"  embedding shape : {rs['embedding'].shape}")
    print(f"  top_label       : {rs['top_label']}")
    print(f"  model_used      : {rs['model_used']}")
    print("✓ M3 adapter works — Mahek can now import get_rs_features()")

# Text embedding test
txt_emb = encode_text_for_vqa("water body")
sim = float(np.dot(rs["embedding"], txt_emb))
print(f"\n  Cosine sim (random image vs 'water body'): {sim:.4f}")
print("  (any float between -1 and 1 is correct)")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  LoRA loaded   : {info['lora_adapted']}")
print(f"  Device        : {info['device']}")
print(f"  Base model    : {info['base_model']}")
print("\nDone. If all 4 steps showed ✓, your backbone is ready.")
print("Share the checkpoints/rs_clip_lora.pt file with M3, M4, M5.")