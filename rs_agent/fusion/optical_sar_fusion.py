"""
optical_sar_fusion.py  —  M2: Optical–SAR Fusion Module
=========================================================
Run standalone:
    python fusion/optical_sar_fusion.py                       # synthetic test
    python fusion/optical_sar_fusion.py --optical a.tif --sar b.tif   # real files
    python fusion/optical_sar_fusion.py --optical s2_patch/ --sar s1.tif  # BEN dir

App integration (M1 imports this):
    from fusion.optical_sar_fusion import analyze_optical_sar
    result = analyze_optical_sar(optical_path, sar_path, query, output_dir)

Output schema (every key is JSON-serialisable, no numpy arrays):
    {
        "task":            "optical_sar_fusion",
        "answer_text":     str,
        "visual_evidence": str,          # abs path to overlay PNG
        "confidence":      float,        # optical-SAR agreement ratio [0,1]
        "model_used":      str,
        "params":          dict,         # thresholds used (for audit trail)
        "stats":           dict,         # per-class pixel %
        "execution_trace": list[str],
        "error":           None | str,
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ── thresholds ─────────────────────────────────────────────────────────────────
NDVI_VEG_THRESHOLD    =  0.25    # NDVI above → vegetation
NDWI_WATER_THRESHOLD  =  0.05    # NDWI above → water (optical)
SAR_WATER_DB_THRESH   = -17.0    # primary pol below → water
SAR_BUILDUP_DB_THRESH = -10.0    # primary pol above → potential built-up
SAR_RATIO_THRESH      =  3.0     # primary/cross-pol linear ratio above → built-up

# BigEarthNet v2 per-band suffixes
BEN_BAND_SUFFIXES = {
    "blue":  "_B02.tif",
    "green": "_B03.tif",
    "red":   "_B04.tif",
    "nir":   "_B08.tif",
}

COLOUR_MAP = {
    "vegetation":         (0,   180, 0,   120),
    "water_optical_only": (100, 180, 255, 100),
    "water_sar_only":     (150, 50,  200, 100),
    "water_confirmed":    (0,   60,  200, 160),
    "buildup_confirmed":  (220, 40,  40,  150),
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _is_already_db(arr: np.ndarray) -> bool:
    """Linear SAR intensity is always ≥ 0; negatives mean already-dB."""
    return bool(arr.min() < -1.0)


def _to_db(intensity: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(intensity, eps))


def _pct(mask: np.ndarray) -> float:
    return round(float(mask.mean() * 100), 2)


def _resize_to_match(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Resize target 2-D array to match ref's (H, W).
    Works correctly for both float32 (SAR dB) and bool (masks).
    PIL mode F handles arbitrary float32 values including negatives.
    """
    if ref.shape[:2] == target.shape[:2]:
        return target
    h, w = ref.shape[:2]
    was_bool = target.dtype == bool
    # PIL mode 'F' = 32-bit float; handles negatives fine
    img = Image.fromarray(target.astype(np.float32), mode="F")
    img = img.resize((w, h), Image.NEAREST)
    result = np.array(img, dtype=np.float32)
    return result.astype(bool) if was_bool else result


def _detect_s2_band_layout(n_bands: int):
    """
    Return (blue_1idx, green_1idx, red_1idx, nir_1idx) for a rasterio dataset
    based on number of bands.  nir_1idx is None when NIR is not present.
    """
    if n_bands >= 8:
        return (2, 3, 4, 8)   # S2 L1C / L2A full product
    elif n_bands == 4:
        return (1, 2, 3, 4)   # BGRN composite
    elif n_bands == 3:
        return (1, 2, 3, None)
    else:
        return (1, 1, 1, None)


# ── loaders ────────────────────────────────────────────────────────────────────

def _load_ben_patch(path: Path) -> Dict[str, Optional[np.ndarray]]:
    """
    BigEarthNet v2 S2 patch: each band is a separate GeoTIFF.
    Accepts the patch directory or any file inside it.
    """
    import rasterio
    patch_dir = path if path.is_dir() else path.parent
    bands: Dict[str, Optional[np.ndarray]] = {}
    for band_name, suffix in BEN_BAND_SUFFIXES.items():
        hits = list(patch_dir.glob(f"*{suffix}"))
        if not hits:
            bands[band_name] = None
            continue
        with rasterio.open(hits[0]) as src:
            arr = src.read(1).astype(np.float32)
        bands[band_name] = np.clip(arr / 10000.0, 0.0, 1.0)
    if all(v is None for v in bands.values()):
        raise ValueError(f"No BigEarthNet band files found in {patch_dir}")
    logger.info("BEN patch loaded from %s | bands present: %s",
                patch_dir, {k: v is not None for k, v in bands.items()})
    return bands


def _load_optical(path: Union[str, Path]) -> Dict[str, Optional[np.ndarray]]:
    """
    Returns {"red", "green", "blue", "nir"} as float32 arrays in [0, 1].
    nir is None when not available (PNG/JPEG or 3-band TIFF).

    Detection order:
      1. BigEarthNet patch directory OR sibling _B02.tif files present
      2. Multi-band GeoTIFF (S2 L1C/L2A, Cartosat-2S, any 3-4-8-13 band product)
      3. PNG / JPEG (RGB only)
    """
    path = Path(path)

    # ── BigEarthNet per-band directory ─────────────────────────────────────────
    is_dir = path.is_dir()
    has_ben_siblings = (
        path.suffix.lower() in (".tif", ".tiff")
        and any(path.parent.glob("*_B02.tif"))
    )
    if is_dir or has_ben_siblings:
        try:
            return _load_ben_patch(path)
        except Exception as e:
            logger.warning("BEN load failed (%s) — trying multi-band fallback", e)

    # ── Multi-band GeoTIFF ─────────────────────────────────────────────────────
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                n = src.count
                bi, gi, ri, niri = _detect_s2_band_layout(n)
                # Probe DN range to decide scale factor
                probe = src.read(1).astype(np.float32)
                scale = 10000.0 if probe.max() > 1000.0 else 255.0

                def _rn(idx: int) -> np.ndarray:
                    return np.clip(src.read(idx).astype(np.float32) / scale, 0.0, 1.0)

                red   = _rn(ri)
                green = _rn(gi)
                blue  = _rn(bi)
                nir   = _rn(niri) if niri else None

            logger.info("Optical GeoTIFF: %d bands, scale=%.0f, nir=%s", n, scale, niri)
            return {"red": red, "green": green, "blue": blue, "nir": nir}
        except ImportError:
            logger.warning("rasterio not installed — falling back to PIL")

    # ── PNG / JPEG ─────────────────────────────────────────────────────────────
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    logger.info("Optical PNG/JPEG loaded from %s — NIR unavailable", path)
    return {"red": arr[:, :, 0], "green": arr[:, :, 1], "blue": arr[:, :, 2], "nir": None}


def _load_sar(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Returns {"vv_linear", "vh_linear", "_already_db"}.
    vv_linear / vh_linear are float32 arrays (linear intensity OR dB, flagged by _already_db).

    Handles:
      - Sentinel-1 GRD (linear or already-dB, auto-detected)
      - RISAT-1 HH/HV  (mapped to vv/vh slots)
      - Single-band SAR (vh duplicated from vv)
      - PNG/JPEG amplitude [0,255] (squared to intensity)
    """
    path = Path(path)

    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                n = src.count
                b1 = src.read(1).astype(np.float32)
                b2 = src.read(2).astype(np.float32) if n >= 2 else b1.copy()

            already_db = _is_already_db(b1)
            logger.info("SAR GeoTIFF: %d bands, already_db=%s, range=[%.3f, %.3f]",
                        n, already_db, b1.min(), b1.max())
            return {"vv_linear": b1, "vh_linear": b2, "_already_db": already_db}
        except ImportError:
            logger.warning("rasterio not installed — falling back to PIL for SAR")

    # PNG/JPEG: assume amplitude [0,255], square to intensity
    arr = np.array(Image.open(path).convert("L")).astype(np.float32)
    vv = (arr / 255.0) ** 2
    logger.info("SAR PNG/JPEG loaded from %s", path)
    return {"vv_linear": vv, "vh_linear": vv, "_already_db": False}


# ── index / mask computation ───────────────────────────────────────────────────

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    out = np.where(denom > 0.001, (nir - red) / denom, 0.0)
    return out.astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = green + nir
    out = np.where(denom > 0.001, (green - nir) / denom, 0.0)
    return out.astype(np.float32)


def compute_sar_masks(vv_db: np.ndarray, vh_db: np.ndarray) -> Dict[str, np.ndarray]:
    water = vv_db < SAR_WATER_DB_THRESH
    vv_lin = 10.0 ** (vv_db / 10.0)
    vh_lin = 10.0 ** (vh_db / 10.0)
    ratio  = vv_lin / np.maximum(vh_lin, 1e-10)
    buildup = (vv_db > SAR_BUILDUP_DB_THRESH) & (ratio > SAR_RATIO_THRESH)
    return {"water": water, "buildup": buildup}


def fuse_masks(
    opt_water: np.ndarray,
    opt_veg: np.ndarray,
    sar_water: np.ndarray,
    sar_buildup: np.ndarray,
) -> Dict[str, Any]:
    water_confirmed    = opt_water & sar_water
    water_optical_only = opt_water & ~sar_water
    water_sar_only     = sar_water & ~opt_water
    buildup_confirmed  = sar_buildup & ~opt_water
    vegetation         = opt_veg & ~opt_water

    total    = int(water_confirmed.sum() + water_optical_only.sum()
                   + water_sar_only.sum() + buildup_confirmed.sum() + vegetation.sum())
    confirmed = int(water_confirmed.sum() + buildup_confirmed.sum())
    agreement_ratio = confirmed / max(total, 1)

    H, W = opt_water.shape
    amap = np.zeros((H, W), dtype=np.uint8)
    amap[vegetation]         = 1
    amap[water_optical_only] = 2
    amap[water_sar_only]     = 3
    amap[water_confirmed]    = 4
    amap[buildup_confirmed]  = 5

    return {
        "water_confirmed":    water_confirmed,
        "water_optical_only": water_optical_only,
        "water_sar_only":     water_sar_only,
        "buildup_confirmed":  buildup_confirmed,
        "vegetation":         vegetation,
        "agreement_ratio":    agreement_ratio,
        "agreement_map":      amap,
    }


# ── overlay ────────────────────────────────────────────────────────────────────

def generate_overlay(
    optical_rgb: np.ndarray,
    fusion: Dict[str, Any],
    out_path: Path,
) -> Path:
    H, W = optical_rgb.shape[:2]
    if optical_rgb.dtype != np.uint8:
        optical_rgb = np.clip(optical_rgb * 255, 0, 255).astype(np.uint8)
    if optical_rgb.ndim == 2:
        optical_rgb = np.stack([optical_rgb] * 3, axis=-1)

    base   = Image.fromarray(optical_rgb).convert("RGBA")
    mask_a = np.zeros((H, W, 4), dtype=np.uint8)

    for cls, colour in COLOUR_MAP.items():
        m = fusion.get(cls)
        if m is None or not isinstance(m, np.ndarray):
            continue
        m = _resize_to_match(optical_rgb[:, :, 0], m.astype(np.float32)).astype(bool)
        mask_a[m] = colour

    fused = Image.alpha_composite(base, Image.fromarray(mask_a, mode="RGBA")).convert("RGB")

    panel = Image.new("RGB", (W * 2 + 10, H + 100), (30, 30, 30))
    panel.paste(Image.fromarray(optical_rgb), (0, 0))
    panel.paste(fused, (W + 10, 0))

    draw = ImageDraw.Draw(panel)
    draw.text((4, H + 4),       "Optical (original)", fill=(200, 200, 200))
    draw.text((W + 14, H + 4),  "Fused overlay",      fill=(200, 200, 200))

    legend = [
        ((0, 180, 0),    "Vegetation (NDVI)"),
        ((0, 60, 200),   "Water — confirmed (optical+SAR)"),
        ((100, 180, 255),"Water — optical only"),
        ((150, 50, 200), "Water — SAR only"),
        ((220, 40, 40),  "Built-up (SAR double-bounce)"),
    ]
    x0, y0 = W + 14, H + 22
    for (r, g, b), label in legend:
        draw.rectangle([x0, y0, x0 + 12, y0 + 12], fill=(r, g, b))
        draw.text((x0 + 16, y0), label, fill=(220, 220, 220))
        y0 += 16

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(str(out_path))
    logger.info("Overlay saved → %s", out_path)
    return out_path


# ── answer text builder ────────────────────────────────────────────────────────

def _build_answer(stats: Dict[str, float], agreement: float, nir_available: bool) -> str:
    parts = []

    if not nir_available:
        parts.append(
            "[Note: NIR band unavailable — optical water/vegetation detection skipped; "
            "results based on SAR only.]"
        )

    if stats["water_confirmed_pct"] > 0.5:
        parts.append(
            f"Water bodies confirmed by both optical and SAR cover "
            f"~{stats['water_confirmed_pct']:.1f}% of the scene."
        )
    if stats["water_optical_only_pct"] > 1.0:
        parts.append(
            f"Additional optical-only water ({stats['water_optical_only_pct']:.1f}%) — "
            f"possibly shallow water, turbid water, or cloud shadow."
        )
    if stats["water_sar_only_pct"] > 1.0:
        parts.append(
            f"SAR-only water signal over {stats['water_sar_only_pct']:.1f}% — "
            f"likely rough or dark surface attenuating optical return."
        )
    if stats["buildup_confirmed_pct"] > 0.5:
        parts.append(
            f"Built-up / urban area (SAR double-bounce) covers "
            f"~{stats['buildup_confirmed_pct']:.1f}% of the scene."
        )
    if stats["vegetation_pct"] > 1.0:
        parts.append(
            f"Vegetation (NDVI > {NDVI_VEG_THRESHOLD}) covers "
            f"~{stats['vegetation_pct']:.1f}% of the scene."
        )

    if not parts or (len(parts) == 1 and parts[0].startswith("[Note")):
        parts.append(
            "No dominant water, built-up, or vegetation signatures detected. "
            "The scene may be predominantly bare soil, mixed agricultural land, or desert."
        )

    parts.append(
        f"Cross-modal agreement ratio: {agreement:.2f} "
        f"({'high' if agreement > 0.5 else 'moderate' if agreement > 0.2 else 'low'} "
        f"optical–SAR corroboration)."
    )
    return " ".join(parts)


# ── main public function ───────────────────────────────────────────────────────

def analyze_optical_sar(
    optical_path: Union[str, Path],
    sar_path: Union[str, Path],
    query: str = "Use the optical and SAR images together to identify built-up and water-covered regions.",
    output_dir: Union[str, Path] = "outputs",
) -> Dict[str, Any]:
    """
    Called by M1 (agentic controller) for optical-SAR cross-modal tasks.
    Also usable standalone — see module docstring.
    """
    output_dir  = Path(output_dir)
    optical_path = Path(optical_path)
    sar_path     = Path(sar_path)
    trace: list[str] = []

    try:
        # Step 1 ── Load ───────────────────────────────────────────────────────
        trace.append(f"Step 1: Loading images | optical={optical_path.name} sar={sar_path.name}")
        opt = _load_optical(optical_path)
        sar = _load_sar(sar_path)

        # Step 2A ── Optical indices ───────────────────────────────────────────
        trace.append("Step 2A: Computing optical indices (NDVI, NDWI)")
        red, green, nir = opt["red"], opt["green"], opt.get("nir")
        nir_available = nir is not None

        if nir_available:
            ndvi      = compute_ndvi(red, nir)
            ndwi      = compute_ndwi(green, nir)
            opt_veg   = ndvi > NDVI_VEG_THRESHOLD
            opt_water = ndwi > NDWI_WATER_THRESHOLD
            trace.append(f"  NDVI veg>{NDVI_VEG_THRESHOLD}  NDWI water>{NDWI_WATER_THRESHOLD}")
        else:
            H, W = red.shape
            opt_veg   = np.zeros((H, W), dtype=bool)
            opt_water = np.zeros((H, W), dtype=bool)
            trace.append("  NIR unavailable — optical masks set to zero")

        # Step 2B ── SAR backscatter ───────────────────────────────────────────
        already_db = sar.get("_already_db", False)
        if already_db:
            trace.append("Step 2B: SAR already in dB — no conversion needed")
            vv_db = sar["vv_linear"]
            vh_db = sar["vh_linear"]
        else:
            trace.append("Step 2B: Converting SAR linear intensity → dB")
            vv_db = _to_db(sar["vv_linear"])
            vh_db = _to_db(sar["vh_linear"])

        vv_db = np.clip(vv_db, -35.0, 5.0)
        vh_db = np.clip(vh_db, -35.0, 5.0)
        vv_db = _resize_to_match(red, vv_db)
        vh_db = _resize_to_match(red, vh_db)
        trace.append(f"  SAR water<{SAR_WATER_DB_THRESH}dB  built-up>{SAR_BUILDUP_DB_THRESH}dB  ratio>{SAR_RATIO_THRESH}")

        sar_masks = compute_sar_masks(vv_db, vh_db)

        # Step 3 ── Fusion ─────────────────────────────────────────────────────
        trace.append("Step 3: Fusing optical and SAR masks via agreement logic")
        fusion = fuse_masks(opt_water, opt_veg, sar_masks["water"], sar_masks["buildup"])
        agreement = fusion["agreement_ratio"]
        trace.append(f"  Agreement ratio: {agreement:.3f}")

        # Step 4 ── Overlay ────────────────────────────────────────────────────
        trace.append("Step 4: Generating colour overlay image")
        rgb = np.stack([
            np.clip(red * 255, 0, 255).astype(np.uint8),
            np.clip(green * 255, 0, 255).astype(np.uint8),
            np.clip(opt["blue"] * 255, 0, 255).astype(np.uint8),
        ], axis=-1)
        overlay_path = output_dir / f"fusion_overlay_{optical_path.stem}.png"
        generate_overlay(rgb, fusion, overlay_path)

        # Step 5 ── Stats + answer text ────────────────────────────────────────
        trace.append("Step 5: Building stats and answer text")
        stats = {
            "water_confirmed_pct":    _pct(fusion["water_confirmed"]),
            "water_optical_only_pct": _pct(fusion["water_optical_only"]),
            "water_sar_only_pct":     _pct(fusion["water_sar_only"]),
            "buildup_confirmed_pct":  _pct(fusion["buildup_confirmed"]),
            "vegetation_pct":         _pct(fusion["vegetation"]),
        }
        answer_text = _build_answer(stats, agreement, nir_available)
        trace.append("Step 6: Returning JSON to controller")

        return {
            "task":            "optical_sar_fusion",
            "answer_text":     answer_text,
            "visual_evidence": str(overlay_path.resolve()),
            "confidence":      round(agreement, 4),
            "model_used":      "optical-sar-fusion-heuristic",
            "params": {
                "ndvi_veg_threshold":    NDVI_VEG_THRESHOLD,
                "ndwi_water_threshold":  NDWI_WATER_THRESHOLD,
                "sar_water_db_thresh":   SAR_WATER_DB_THRESH,
                "sar_buildup_db_thresh": SAR_BUILDUP_DB_THRESH,
                "sar_ratio_thresh":      SAR_RATIO_THRESH,
            },
            "stats":           stats,
            "execution_trace": trace,
            "error":           None,
        }

    except Exception as exc:
        logger.exception("Fusion module error: %s", exc)
        return {
            "task":            "optical_sar_fusion",
            "answer_text":     f"Fusion analysis failed: {exc}",
            "visual_evidence": None,
            "confidence":      0.0,
            "model_used":      "optical-sar-fusion-heuristic",
            "params":          {},
            "stats":           {},
            "execution_trace": trace,
            "error":           str(exc),
        }


# ── synthetic test-data generator ─────────────────────────────────────────────

def _make_synthetic_test_data(out_dir: Path):
    """
    Creates realistic synthetic test images with known land-cover regions.
    Used when running standalone without real data.

    Regions planted (so we can assert results):
      top-left  80×80  → water  (low NIR, low SAR)
      bottom-right 76×76 → vegetation  (high NIR, mid SAR)
      centre 80×80     → built-up  (low NDWI/NDVI, high SAR)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = 256, 256
    rng = np.random.default_rng(42)

    # ── Optical: 4-band BGRN GeoTIFF ─────────────────────────────────────────
    # We generate a proper 4-band GeoTIFF (blue, green, red, NIR) so that
    # the optical code path exercises NIR-based NDVI/NDWI.
    opt_path = out_dir / "synthetic_optical_4band.tif"
    try:
        import rasterio
        from rasterio.transform import from_bounds
        blue  = rng.uniform(0.05, 0.25, (H, W)).astype(np.float32)
        green = rng.uniform(0.10, 0.35, (H, W)).astype(np.float32)
        red   = rng.uniform(0.08, 0.30, (H, W)).astype(np.float32)
        nir   = rng.uniform(0.20, 0.55, (H, W)).astype(np.float32)

        # water: low NIR, moderate green → NDWI > 0
        nir[:80, :80]   = 0.04
        red[:80, :80]   = 0.06
        green[:80, :80] = 0.30

        # vegetation: high NIR, low red → NDVI > 0.3
        nir[180:, 180:] = 0.72
        red[180:, 180:] = 0.09
        green[180:, 180:] = 0.18

        # built-up: mid optical, handled mainly by SAR
        # (no special optical treatment needed)

        # Scale to S2 DN [0,10000] so the probe-based scale detection works
        def dn(arr): return (arr * 10000).astype(np.float32)

        transform = from_bounds(0, 0, 1, 1, W, H)
        with rasterio.open(
            opt_path, "w", driver="GTiff",
            height=H, width=W, count=4,
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(dn(blue),  1)
            dst.write(dn(green), 2)
            dst.write(dn(red),   3)
            dst.write(dn(nir),   4)
        logger.info("Synthetic 4-band optical GeoTIFF written: %s", opt_path)
    except ImportError:
        # rasterio not available — fall back to PNG (NIR will be unavailable)
        logger.warning("rasterio not installed — writing PNG fallback for optical")
        opt_path = out_dir / "synthetic_optical.png"
        rgb = np.stack([
            np.clip(red * 255, 0, 255),
            np.clip(green * 255, 0, 255),
            np.clip(blue * 255, 0, 255),
        ], axis=-1).astype(np.uint8)
        # plant obvious colours so the visual is readable
        rgb[:80, :80]   = [30, 100, 200]   # blue = water
        rgb[180:, 180:] = [40, 160, 50]    # green = vegetation
        rgb[88:168, 88:168] = [160, 150, 140]  # grey = built-up
        Image.fromarray(rgb).save(opt_path)

    # ── SAR: 2-band (VV, VH) GeoTIFF in linear intensity ─────────────────────
    sar_path = out_dir / "synthetic_sar_linear.tif"
    vv = rng.uniform(0.02, 0.20, (H, W)).astype(np.float32)
    vh = rng.uniform(0.01, 0.10, (H, W)).astype(np.float32)

    # water: very low VV (specular → low backscatter) — will be < -17 dB
    vv[:80, :80] = 0.002
    vh[:80, :80] = 0.001

    # built-up: high VV + high VV/VH ratio (double-bounce)
    vv[88:168, 88:168] = 0.25
    vh[88:168, 88:168] = 0.04   # ratio ≈ 6.25 → > SAR_RATIO_THRESH=3

    # vegetation: moderate VV
    vv[180:, 180:] = 0.08
    vh[180:, 180:] = 0.05

    try:
        import rasterio
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 1, 1, W, H)
        with rasterio.open(
            sar_path, "w", driver="GTiff",
            height=H, width=W, count=2,
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(vv, 1)
            dst.write(vh, 2)
        logger.info("Synthetic 2-band SAR GeoTIFF written: %s", sar_path)
    except ImportError:
        logger.warning("rasterio not installed — writing grayscale PNG for SAR")
        sar_path = out_dir / "synthetic_sar.png"
        amp = np.sqrt(vv) * 255
        Image.fromarray(amp.astype(np.uint8)).save(sar_path)

    return opt_path, sar_path


# ── CLI entry point ────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="M2 Optical-SAR Fusion — run standalone or as module"
    )
    parser.add_argument("--optical", type=Path, default=None,
                        help="Path to optical image or BigEarthNet patch dir")
    parser.add_argument("--sar",     type=Path, default=None,
                        help="Path to SAR GeoTIFF or PNG")
    parser.add_argument("--query",   type=str,
                        default="Use the optical and SAR images together to identify "
                                "built-up and water-covered regions.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-data-dir", type=Path, default=Path("test_data"),
                        help="Where to write synthetic test images (used when --optical/--sar omitted)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── resolve input paths ────────────────────────────────────────────────────
    if args.optical is None or args.sar is None:
        print("No --optical/--sar provided → generating synthetic test data...\n")
        opt_path, sar_path = _make_synthetic_test_data(args.test_data_dir)
    else:
        opt_path, sar_path = args.optical, args.sar
        if not opt_path.exists():
            parser.error(f"Optical path not found: {opt_path}")
        if not sar_path.exists():
            parser.error(f"SAR path not found: {sar_path}")

    print(f"Optical : {opt_path}")
    print(f"SAR     : {sar_path}")
    print(f"Output  : {args.output_dir}\n")

    # ── run fusion ─────────────────────────────────────────────────────────────
    result = analyze_optical_sar(
        optical_path=opt_path,
        sar_path=sar_path,
        query=args.query,
        output_dir=args.output_dir,
    )

    # ── print result ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("FUSION RESULT")
    print("=" * 60)
    display = {k: v for k, v in result.items() if k != "execution_trace"}
    print(json.dumps(display, indent=2))
    print("\nExecution trace:")
    for step in result["execution_trace"]:
        print(" ", step)

    if result["error"]:
        print(f"\n[ERROR] {result['error']}")
        return 1

    if result["visual_evidence"]:
        print(f"\nOverlay PNG → {result['visual_evidence']}")

    # Quick sanity check on synthetic data
    if args.optical is None:
        stats = result["stats"]
        issues = []
        # water region should be detected somewhere (SAR at minimum)
        water_total = (stats["water_confirmed_pct"]
                       + stats["water_optical_only_pct"]
                       + stats["water_sar_only_pct"])
        if water_total < 1.0:
            issues.append(f"Expected water detection, got {water_total:.2f}%")
        if issues:
            print("\n[WARN] Sanity check issues:")
            for i in issues:
                print("  !", i)
        else:
            print("\n[OK] Sanity checks passed on synthetic data")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())