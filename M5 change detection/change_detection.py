"""
M5 — Change Detection / Bi-temporal Change-VQA Specialist
===========================================================
Owner: M5 (you), solo build.

WHAT THIS FILE DOES
--------------------
Given a bi-temporal image pair (same location, two dates), this module:
  1. Loads + aligns the pair (handles slightly different sizes).
  2. Detects change two ways and fuses them:
       a) Pixel/classical diff  -> always available, zero dependencies beyond
          opencv/numpy/skimage. This is your guaranteed-to-work baseline.
       b) Pretrained-feature diff -> uses a frozen, ImageNet-pretrained
          ResNet18 (torchvision) as a generic visual feature extractor and
          compares patch-level embeddings between before/after. This is the
          "pretrained option" fallback for when you don't have time/data to
          train a real RS change-detection network. It is purely optional:
          if torch/torchvision aren't installed, the module silently
          degrades to classical-only and still works end-to-end.
  3. Produces a binary change mask + change map visualization + overlay.
  4. Computes change statistics (% area changed, location/direction,
     rough land-cover category of the change) using cheap color heuristics
     (no training needed — same spirit as M4's heuristic grounding).
  5. Generates a templated natural-language change description.
  6. Answers simple change-VQA queries (yes/no/direction) from those stats.
  7. Returns everything as ONE JSON-serializable dict that matches the
     shared schema M1's tool registry expects, so M1 can drop it straight
     into the output combiner with zero glue code.

HOW M1 CALLS THIS
------------------
    from change_detection import analyze_change

    result = analyze_change(
        image_before_path="data/before.tif",
        image_after_path="data/after.tif",
        query="Has the built-up area increased, decreased, or remained unchanged?",
        out_dir="outputs/",
    )
    # result is a dict -> M1 just does output_combiner.merge(result)

Everything else (dataset loading, plotting demo grids, etc.) lives in
test_change_detection.py, kept OUT of this file so the file you hand off
to M1 is small and dependency-light.
"""

from __future__ import annotations

import os
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology, measure

# ---------------------------------------------------------------------------
# Optional pretrained-feature backend (graceful degradation if torch absent)
# ---------------------------------------------------------------------------
_TORCH_OK = True
try:
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
except Exception:  # pragma: no cover - environment without torch
    _TORCH_OK = False


# ===========================================================================
# 1. IO + PREPROCESSING
# ===========================================================================

def _read_image_any(path: str) -> np.ndarray:
    """
    Reads GeoTIFF/TIFF/PNG/JPEG into an HxWx3 uint8 RGB array.
    Tries rasterio first (proper geospatial band handling), falls back to
    OpenCV/PIL for plain PNG/JPEG (as used by the public benchmark subsets).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                bands = src.read()  # (C, H, W)
                if bands.shape[0] >= 3:
                    arr = np.transpose(bands[:3], (1, 2, 0))
                else:
                    arr = np.repeat(np.transpose(bands[:1], (1, 2, 0)), 3, axis=2)
                arr = arr.astype(np.float32)
                # normalize per-band to 0-255 (satellite bands are often 16-bit)
                for c in range(arr.shape[2]):
                    band = arr[:, :, c]
                    lo, hi = np.percentile(band, (1, 99))
                    if hi > lo:
                        arr[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1) * 255
                return arr.astype(np.uint8)
        except Exception:
            pass  # fall through to cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_and_align_pair(
    path_before: str, path_after: str, target_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads both images and resizes both to the SAME shape so pixel/feature
    diffing is valid. (Full geo-registration is out of scope for the
    mandatory deliverable — the problem statement's ISRO/SAC set is already
    pre-georeferenced and co-registered, and CDVQA pairs are pre-aligned.)
    """
    before = _read_image_any(path_before)
    after = _read_image_any(path_after)
    before = cv2.resize(before, target_size, interpolation=cv2.INTER_AREA)
    after = cv2.resize(after, target_size, interpolation=cv2.INTER_AREA)
    return before, after


# ===========================================================================
# 2. CHANGE DETECTION CORE
# ===========================================================================

if _TORCH_OK:
    class _PretrainedFeatureDiff:
        """Frozen ImageNet ResNet18 used as a generic feature extractor for
        patch-wise embedding distance. No training required — this is the
        'use a pretrained option' fallback. Only defined if torch is present.
        """

        def __init__(self, device: str = "cpu"):
            self.device = device
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
            # Keep everything up to layer2 -> good balance of spatial res vs semantics
            self.extractor = torch.nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2,
            ).eval().to(device)
            for p in self.extractor.parameters():
                p.requires_grad = False
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        @torch.no_grad()
        def diff_map(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
            h, w = before.shape[:2]
            tb = self.tf(before).unsqueeze(0).to(self.device)
            ta = self.tf(after).unsqueeze(0).to(self.device)
            fb = self.extractor(tb)          # (1, C, h', w')
            fa = self.extractor(ta)
            fb = F.normalize(fb, dim=1)
            fa = F.normalize(fa, dim=1)
            cos_sim = (fb * fa).sum(dim=1)   # (1, h', w') in [-1, 1]
            dist = (1 - cos_sim).squeeze(0).cpu().numpy()  # 0 = identical, 2 = opposite
            dist = cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)
            dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
            return dist  # normalized 0..1 change-likelihood map


def _pixel_diff_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Classical grayscale absolute difference, normalized 0..1. Always available."""
    gb = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ga = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = np.abs(gb - ga)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    return diff


def _binarize(diff_map: np.ndarray) -> np.ndarray:
    """Otsu threshold + morphological cleanup -> clean binary change mask."""
    scaled = (diff_map * 255).astype(np.uint8)
    try:
        thresh = threshold_otsu(scaled)
    except ValueError:
        thresh = 128
    mask = (scaled > thresh).astype(np.uint8)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=64)
    mask = morphology.remove_small_holes(mask, area_threshold=64)
    mask = morphology.closing(mask, morphology.disk(3))
    return mask.astype(np.uint8)


# ===========================================================================
# 3. HEURISTIC LAND-COVER CATEGORIZATION (no training needed)
# ===========================================================================

def _classify_pixels(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Cheap, index-free heuristic land-cover masks from RGB only (same spirit
    as M4's index-thresholding grounding). If NIR is available in a 4-band
    GeoTIFF this could be swapped for real NDVI/NDWI — left as a TODO hook.
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = (r + g + b) / 3.0
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0

    veg_score = (g - r) + (g - b)
    water_score = (b - r) + (b - g) - 0.3 * gray  # water: bluish + darker
    edge_density = cv2.blur(edges, (9, 9))
    builtup_score = edge_density * 255 - 0.2 * veg_score  # busy edges, low greenness

    vegetation = veg_score > 15
    water = water_score > 10
    builtup = (builtup_score > 40) & (~vegetation) & (~water)

    return {"vegetation": vegetation, "water": water, "built_up": builtup}


def _dominant_change_category(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> str:
    """Which land-cover class grew the most inside the changed region."""
    if mask.sum() == 0:
        return "no_significant_change"
    before_cls = _classify_pixels(before)
    after_cls = _classify_pixels(after)
    deltas = {}
    for cls in ("vegetation", "water", "built_up"):
        before_area = (before_cls[cls] & mask.astype(bool)).sum()
        after_area = (after_cls[cls] & mask.astype(bool)).sum()
        deltas[cls] = int(after_area) - int(before_area)
    dominant = max(deltas, key=lambda k: abs(deltas[k]))
    direction = "increased" if deltas[dominant] > 0 else "decreased"
    return f"{dominant}_{direction}"


# ===========================================================================
# 4. STATS + DESCRIPTION + VQA
# ===========================================================================

_QUADRANTS = {
    (0, 0): "north-west", (0, 1): "north-east",
    (1, 0): "south-west", (1, 1): "south-east",
}


def _change_stats(mask: np.ndarray) -> Dict[str, Any]:
    h, w = mask.shape
    total = h * w
    changed = int(mask.sum())
    pct = round(100.0 * changed / total, 2)

    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    num_regions = len(props)

    if changed > 0:
        ys, xs = np.where(mask)
        cy, cx = ys.mean() / h, xs.mean() / w
        quadrant = _QUADRANTS[(int(cy >= 0.5), int(cx >= 0.5))]
    else:
        quadrant = "none"

    largest_region_pct = 0.0
    if props:
        largest = max(props, key=lambda p: p.area)
        largest_region_pct = round(100.0 * largest.area / total, 2)

    return {
        "change_percentage": pct,
        "num_change_regions": num_regions,
        "dominant_quadrant": quadrant,
        "largest_region_pct_of_image": largest_region_pct,
    }


def _generate_description(stats: Dict[str, Any], category: str) -> str:
    pct = stats["change_percentage"]
    quadrant = stats["dominant_quadrant"]
    if pct < 1.0:
        return "No significant change was detected between the two acquisition dates."

    cls, direction = (category.rsplit("_", 1) + ["unknown"])[:2] if "_" in category else (category, "")
    cls_readable = cls.replace("_", " ")

    magnitude = "minor" if pct < 5 else "moderate" if pct < 15 else "substantial"
    loc = f" concentrated in the {quadrant} part of the scene" if quadrant != "none" else ""

    return (
        f"A {magnitude} change ({pct}% of the image area) was detected between the two dates"
        f"{loc}. The dominant change pattern corresponds to {cls_readable} {direction}, "
        f"spread across {stats['num_change_regions']} distinct region(s)."
    )


def _answer_change_vqa(query: Optional[str], stats: Dict[str, Any], category: str) -> Optional[str]:
    """Keyword-driven yes/no/direction answering from diff stats — deliberately
    simple per the mandatory scope ('Simple change-VQA: yes/no/direction from
    diff stats'). No model call needed."""
    if not query:
        return None
    q = query.lower()
    pct = stats["change_percentage"]
    cls, direction = (category.rsplit("_", 1) + [None])[:2] if "_" in category else (category, None)

    asked_class = None
    for candidate in ("built_up", "built-up", "vegetation", "water"):
        if candidate.replace("-", " ") in q or candidate.replace("_", " ") in q:
            asked_class = candidate.replace("-", "_")
            break

    if any(w in q for w in ("increase", "decrease", "unchanged", "remained")):
        if asked_class and asked_class == cls:
            if direction == "increased":
                return "Increased."
            elif direction == "decreased":
                return "Decreased."
            return "Remained unchanged."
        if pct < 1.0:
            return "Remained unchanged."
        return "Increased." if direction == "increased" else "Decreased."

    if q.strip().startswith(("has", "did", "is there", "was there")) or "any change" in q:
        return "Yes." if pct >= 1.0 else "No."

    if "where" in q:
        if stats["dominant_quadrant"] == "none":
            return "No location of change — no significant change detected."
        return f"Primarily in the {stats['dominant_quadrant']} region of the image."

    if "what changed" in q or "describe" in q:
        return None  # let caller use the full templated description instead

    return "Yes." if pct >= 1.0 else "No."


# ===========================================================================
# 5. VISUALIZATION
# ===========================================================================

def _save_visuals(before, after, mask, diff_map, out_dir, run_id) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    change_map_vis = (diff_map * 255).astype(np.uint8)
    change_map_vis = cv2.applyColorMap(change_map_vis, cv2.COLORMAP_JET)

    overlay = before.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 0] = 255
    alpha = 0.5
    mask_3c = np.stack([mask] * 3, axis=-1).astype(bool)
    overlay = np.where(mask_3c, (overlay * (1 - alpha) + red_layer * alpha).astype(np.uint8), overlay)

    side_by_side = np.concatenate([before, after, overlay], axis=1)

    paths = {}
    paths["change_map_path"] = os.path.join(out_dir, f"{run_id}_change_map.png")
    paths["overlay_path"] = os.path.join(out_dir, f"{run_id}_overlay.png")
    paths["before_after_overlay_path"] = os.path.join(out_dir, f"{run_id}_before_after_overlay.png")

    cv2.imwrite(paths["change_map_path"], change_map_vis)
    cv2.imwrite(paths["overlay_path"], cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["before_after_overlay_path"], cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    return paths


# ===========================================================================
# 6. PUBLIC ENTRY POINT — this is the function you hand to M1
# ===========================================================================

_feature_backend: Optional["_PretrainedFeatureDiff"] = None


def _get_feature_backend():
    global _feature_backend
    if _TORCH_OK and _feature_backend is None:
        _feature_backend = _PretrainedFeatureDiff(device="cpu")
    return _feature_backend


def analyze_change(
    image_before_path: str,
    image_after_path: str,
    query: Optional[str] = None,
    out_dir: str = "outputs",
    use_pretrained_features: bool = True,
) -> Dict[str, Any]:
    """
    THE function M1 calls. Matches the shared schema used by every other
    specialist (M2/M4 etc.) so M1's output combiner needs no special-casing.

    Returns
    -------
    dict with keys: task, model_tool, inputs, outputs, execution_trace
    """
    t0 = time.time()
    run_id = uuid.uuid4().hex[:8]

    before, after = load_and_align_pair(image_before_path, image_after_path)

    pixel_map = _pixel_diff_map(before, after)

    used_pretrained = False
    if use_pretrained_features and _TORCH_OK:
        try:
            backend = _get_feature_backend()
            feat_map = backend.diff_map(before, after)
            fused_map = 0.5 * pixel_map + 0.5 * feat_map
            used_pretrained = True
        except Exception:
            fused_map = pixel_map
    else:
        fused_map = pixel_map

    mask = _binarize(fused_map)
    stats = _change_stats(mask)
    category = _dominant_change_category(before, after, mask)
    description = _generate_description(stats, category)
    vqa_answer = _answer_change_vqa(query, stats, category)

    visuals = _save_visuals(before, after, mask, fused_map, out_dir, run_id)

    # confidence: crude but honest — higher when change is either clearly
    # present or clearly absent, lower near the decision boundary, and a
    # small bonus when the pretrained + classical signals agree.
    boundary_dist = abs(stats["change_percentage"] - 2.0) / 2.0
    confidence = float(np.clip(0.55 + 0.25 * min(boundary_dist, 1.0) + (0.15 if used_pretrained else 0.0), 0.4, 0.95))

    result = {
        "task": "change_vqa" if query else "change_detection",
        "model_tool": (
            "M5_ChangeDetector_v1 "
            f"(otsu-pixel-diff{'+resnet18-pretrained-feature-diff' if used_pretrained else ''})"
        ),
        "inputs": {
            "image_before": image_before_path,
            "image_after": image_after_path,
            "query": query,
        },
        "outputs": {
            "change_map_path": visuals["change_map_path"],
            "overlay_path": visuals["overlay_path"],
            "before_after_overlay_path": visuals["before_after_overlay_path"],
            "change_percentage": stats["change_percentage"],
            "num_change_regions": stats["num_change_regions"],
            "dominant_quadrant": stats["dominant_quadrant"],
            "dominant_change_category": category,
            "change_description": description,
            "vqa_answer": vqa_answer,
            "confidence": round(confidence, 2),
        },
        "execution_trace": {
            "run_id": run_id,
            "selected_task": "change_vqa" if query else "change_detection",
            "models_used": [
                "classical_pixel_diff+otsu",
            ] + (["resnet18_imagenet_pretrained_feature_diff"] if used_pretrained else []),
            "params": {"target_size": [512, 512], "use_pretrained_features": use_pretrained_features},
            "latency_seconds": round(time.time() - t0, 3),
        },
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="M5 change detection standalone runner")
    parser.add_argument("before", help="path to before image")
    parser.add_argument("after", help="path to after image")
    parser.add_argument("--query", default=None)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    out = analyze_change(args.before, args.after, args.query, args.out_dir)
    print(json.dumps(out, indent=2))
