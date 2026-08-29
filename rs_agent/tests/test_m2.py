"""
tests/test_m2.py — M2 Integration + Unit Tests
Run from anywhere:
    pytest tests/test_m2.py -v
    pytest tests/test_m2.py -v -k "not backbone"   # skip model-download tests
"""
import json
import sys
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# conftest.py handles sys.path — no manual insert needed here


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_tif_pair(tmp_path_factory):
    """
    Create a synthetic 4-band optical GeoTIFF + 2-band SAR GeoTIFF.
    Used for tests that exercise the GeoTIFF loaders.
    Skipped if rasterio is not installed.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    out = tmp_path_factory.mktemp("tif_data")
    H, W = 128, 128
    rng  = np.random.default_rng(0)

    # ── optical ────────────────────────────────────────────────────────────────
    blue  = rng.uniform(0.05, 0.25, (H, W)).astype(np.float32)
    green = rng.uniform(0.10, 0.35, (H, W)).astype(np.float32)
    red   = rng.uniform(0.08, 0.30, (H, W)).astype(np.float32)
    nir   = rng.uniform(0.20, 0.55, (H, W)).astype(np.float32)
    nir[:40, :40]  = 0.04   # water blob
    red[:40, :40]  = 0.06
    green[:40, :40] = 0.30
    nir[90:, 90:]  = 0.72   # vegetation blob
    red[90:, 90:]  = 0.09

    def dn(a): return (a * 10000).astype(np.float32)
    opt_path = out / "opt_4band.tif"
    with rasterio.open(opt_path, "w", driver="GTiff", height=H, width=W,
                       count=4, dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(0, 0, 1, 1, W, H)) as dst:
        dst.write(dn(blue), 1);  dst.write(dn(green), 2)
        dst.write(dn(red),  3);  dst.write(dn(nir),   4)

    # ── SAR (linear intensity) ─────────────────────────────────────────────────
    vv = rng.uniform(0.02, 0.20, (H, W)).astype(np.float32)
    vh = rng.uniform(0.01, 0.10, (H, W)).astype(np.float32)
    vv[:40, :40]  = 0.002;  vh[:40, :40]  = 0.001  # water: low backscatter
    vv[50:90, 50:90] = 0.25; vh[50:90, 50:90] = 0.04  # built-up

    sar_path = out / "sar_2band.tif"
    with rasterio.open(sar_path, "w", driver="GTiff", height=H, width=W,
                       count=2, dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(0, 0, 1, 1, W, H)) as dst:
        dst.write(vv, 1);  dst.write(vh, 2)

    # ── SAR already-in-dB variant ──────────────────────────────────────────────
    vv_db = np.clip(10.0 * np.log10(np.maximum(vv, 1e-10)), -35, 5)
    vh_db = np.clip(10.0 * np.log10(np.maximum(vh, 1e-10)), -35, 5)
    sar_db_path = out / "sar_already_db.tif"
    with rasterio.open(sar_db_path, "w", driver="GTiff", height=H, width=W,
                       count=2, dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(0, 0, 1, 1, W, H)) as dst:
        dst.write(vv_db, 1);  dst.write(vh_db, 2)

    return {"opt": opt_path, "sar_linear": sar_path, "sar_db": sar_db_path, "H": H, "W": W}


@pytest.fixture(scope="session")
def synthetic_png_pair(tmp_path_factory):
    """Simple RGB PNG pair — no rasterio needed."""
    out = tmp_path_factory.mktemp("png_data")
    H, W = 128, 128
    rng = np.random.default_rng(1)

    rgb = rng.integers(80, 200, (H, W, 3), dtype=np.uint8)
    rgb[:40, :40] = [30, 120, 210]      # blue = water region
    rgb[90:, 90:] = [40, 160, 55]       # green = vegetation
    Image.fromarray(rgb).save(out / "optical.png")

    amp = rng.integers(100, 200, (H, W), dtype=np.uint8)
    amp[:40, :40] = 8       # low → water
    amp[50:90, 50:90] = 240 # high → built-up
    Image.fromarray(amp).save(out / "sar.png")

    return {"opt": out / "optical.png", "sar": out / "sar.png"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_RESULT_KEYS = [
    "task", "answer_text", "visual_evidence", "confidence",
    "model_used", "params", "stats", "execution_trace", "error",
]

def _assert_m1_schema(result: dict):
    """Assert the output matches the contract M1 expects."""
    for k in REQUIRED_RESULT_KEYS:
        assert k in result, f"Missing key in result: '{k}'"
    assert result["task"] == "optical_sar_fusion"
    assert isinstance(result["answer_text"], str) and len(result["answer_text"]) > 10
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_used"] == "optical-sar-fusion-heuristic"
    assert isinstance(result["execution_trace"], list) and len(result["execution_trace"]) > 0
    for k, v in result["stats"].items():
        assert 0.0 <= v <= 100.0, f"stat {k}={v} out of [0,100]"
    # Full result must be JSON-serialisable (no numpy arrays leaking through)
    clean = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    json.dumps(clean)   # raises TypeError if anything is not serialisable


# ─────────────────────────────────────────────────────────────────────────────
# Index math unit tests (no I/O, no model loading)
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexMath:
    def test_ndvi_range(self):
        from fusion.optical_sar_fusion import compute_ndvi
        rng = np.random.default_rng(0)
        r = rng.uniform(0, 1, (50, 50)).astype(np.float32)
        n = rng.uniform(0, 1, (50, 50)).astype(np.float32)
        v = compute_ndvi(r, n)
        assert v.min() >= -1.0 and v.max() <= 1.0

    def test_ndvi_water_is_negative(self):
        from fusion.optical_sar_fusion import compute_ndvi
        # water: red ≈ NIR → NDVI ≈ 0; actually if NIR < red → negative
        r = np.full((10, 10), 0.15, dtype=np.float32)
        n = np.full((10, 10), 0.05, dtype=np.float32)
        assert compute_ndvi(r, n).mean() < 0

    def test_ndwi_water_is_positive(self):
        from fusion.optical_sar_fusion import compute_ndwi
        g = np.full((10, 10), 0.30, dtype=np.float32)
        n = np.full((10, 10), 0.04, dtype=np.float32)
        assert compute_ndwi(g, n).mean() > 0

    def test_ndwi_range(self):
        from fusion.optical_sar_fusion import compute_ndwi
        rng = np.random.default_rng(1)
        g = rng.uniform(0, 1, (50, 50)).astype(np.float32)
        n = rng.uniform(0, 1, (50, 50)).astype(np.float32)
        v = compute_ndwi(g, n)
        assert v.min() >= -1.0 and v.max() <= 1.0

    def test_sar_water_mask_fires_for_low_vv(self):
        from fusion.optical_sar_fusion import compute_sar_masks
        vv_db = np.full((20, 20), -22.0, dtype=np.float32)   # below -17 threshold
        vh_db = np.full((20, 20), -25.0, dtype=np.float32)
        masks = compute_sar_masks(vv_db, vh_db)
        assert masks["water"].all(), "All pixels should be detected as water"

    def test_sar_buildup_mask_fires_for_high_vv_ratio(self):
        from fusion.optical_sar_fusion import compute_sar_masks
        vv_db = np.full((20, 20), -6.0,  dtype=np.float32)   # above -10
        vh_db = np.full((20, 20), -14.0, dtype=np.float32)   # ratio > 3
        masks = compute_sar_masks(vv_db, vh_db)
        assert masks["buildup"].any(), "Built-up mask should fire"

    def test_to_db_known_value(self):
        from fusion.optical_sar_fusion import _to_db
        val = _to_db(np.array([1.0], dtype=np.float32))
        assert abs(val[0]) < 0.01   # 10*log10(1) = 0

    def test_is_already_db(self):
        from fusion.optical_sar_fusion import _is_already_db
        linear = np.array([0.001, 0.1, 0.5], dtype=np.float32)
        db     = np.array([-20.0, -15.0, -8.0], dtype=np.float32)
        assert not _is_already_db(linear)
        assert     _is_already_db(db)

    def test_resize_to_match_float32_negative(self):
        """_resize_to_match must preserve negative dB values."""
        from fusion.optical_sar_fusion import _resize_to_match
        ref    = np.zeros((64, 64), dtype=np.float32)
        target = np.full((32, 32), -18.5, dtype=np.float32)
        result = _resize_to_match(ref, target)
        assert result.shape == (64, 64)
        assert abs(result[0, 0] - (-18.5)) < 0.01, "dB value must be preserved after resize"

    def test_resize_to_match_bool_mask(self):
        from fusion.optical_sar_fusion import _resize_to_match
        ref    = np.zeros((64, 64), dtype=np.float32)
        target = np.zeros((32, 32), dtype=bool)
        target[8:16, 8:16] = True
        result = _resize_to_match(ref, target)
        assert result.dtype == bool
        assert result.shape == (64, 64)
        assert result.any()


# ─────────────────────────────────────────────────────────────────────────────
# Loader unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoaders:
    def test_load_optical_png(self, synthetic_png_pair, tmp_path):
        from fusion.optical_sar_fusion import _load_optical
        bands = _load_optical(synthetic_png_pair["opt"])
        assert "red" in bands and "green" in bands and "blue" in bands
        assert bands["nir"] is None
        for k in ("red", "green", "blue"):
            assert bands[k].dtype == np.float32
            assert 0.0 <= bands[k].min() and bands[k].max() <= 1.0

    def test_load_optical_4band_tif(self, synthetic_tif_pair):
        from fusion.optical_sar_fusion import _load_optical
        bands = _load_optical(synthetic_tif_pair["opt"])
        assert bands["nir"] is not None, "4-band TIF must yield NIR"
        for k in ("red", "green", "blue", "nir"):
            assert 0.0 <= bands[k].min() and bands[k].max() <= 1.01

    def test_load_sar_linear_tif(self, synthetic_tif_pair):
        from fusion.optical_sar_fusion import _load_sar
        sar = _load_sar(synthetic_tif_pair["sar_linear"])
        assert sar["_already_db"] is False
        assert sar["vv_linear"].min() >= 0.0, "Linear SAR must be non-negative"

    def test_load_sar_already_db_tif(self, synthetic_tif_pair):
        from fusion.optical_sar_fusion import _load_sar
        sar = _load_sar(synthetic_tif_pair["sar_db"])
        assert sar["_already_db"] is True

    def test_load_sar_png(self, synthetic_png_pair):
        from fusion.optical_sar_fusion import _load_sar
        sar = _load_sar(synthetic_png_pair["sar"])
        assert sar["_already_db"] is False
        assert sar["vv_linear"].shape[0] > 0

    def test_load_optical_detects_ben_sibling(self, tmp_path):
        """If a _B02.tif sibling exists, loader must treat this as a BEN patch."""
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.transform import from_bounds

        H, W = 64, 64
        transform = from_bounds(0, 0, 1, 1, W, H)
        data = np.random.default_rng(5).uniform(0, 5000, (H, W)).astype(np.float32)

        for suffix in ("_B02.tif", "_B03.tif", "_B04.tif", "_B08.tif"):
            p = tmp_path / f"S2A_patch{suffix}"
            with rasterio.open(p, "w", driver="GTiff", height=H, width=W,
                               count=1, dtype="float32", crs="EPSG:4326",
                               transform=transform) as dst:
                dst.write(data, 1)

        # pass any of the band files as the optical path
        bands = __import__("fusion.optical_sar_fusion", fromlist=["_load_optical"])._load_optical(
            tmp_path / "S2A_patch_B04.tif"
        )
        assert bands["nir"] is not None, "BEN sibling detection must load NIR"


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — PNG inputs (no rasterio needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionPipelinePNG:
    def test_schema_ok(self, synthetic_png_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_png_pair["opt"], synthetic_png_pair["sar"],
            output_dir=tmp_path,
        )
        _assert_m1_schema(result)

    def test_no_error(self, synthetic_png_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_png_pair["opt"], synthetic_png_pair["sar"],
            output_dir=tmp_path,
        )
        assert result["error"] is None, f"Unexpected error: {result['error']}"

    def test_overlay_png_created(self, synthetic_png_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_png_pair["opt"], synthetic_png_pair["sar"],
            output_dir=tmp_path,
        )
        assert result["visual_evidence"] is not None
        assert Path(result["visual_evidence"]).exists()

    def test_sar_water_detected(self, synthetic_png_pair, tmp_path):
        """The synthetic SAR has a clear low-backscatter water blob."""
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_png_pair["opt"], synthetic_png_pair["sar"],
            output_dir=tmp_path,
        )
        water_total = (result["stats"]["water_confirmed_pct"]
                       + result["stats"]["water_optical_only_pct"]
                       + result["stats"]["water_sar_only_pct"])
        assert water_total > 1.0, f"Expected water detection, got {water_total:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — GeoTIFF inputs (rasterio required)
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionPipelineTIF:
    def test_schema_ok_with_tif(self, synthetic_tif_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_tif_pair["opt"], synthetic_tif_pair["sar_linear"],
            output_dir=tmp_path,
        )
        _assert_m1_schema(result)

    def test_nir_band_used(self, synthetic_tif_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_tif_pair["opt"], synthetic_tif_pair["sar_linear"],
            output_dir=tmp_path,
        )
        # NIR available → at least one optical-based class should appear
        opt_water = result["stats"]["water_optical_only_pct"]
        veg       = result["stats"]["vegetation_pct"]
        assert opt_water + veg > 0.0, "NIR-based indices should produce some detections"

    def test_water_detected_in_planted_region(self, synthetic_tif_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_tif_pair["opt"], synthetic_tif_pair["sar_linear"],
            output_dir=tmp_path,
        )
        water = (result["stats"]["water_confirmed_pct"]
                 + result["stats"]["water_optical_only_pct"]
                 + result["stats"]["water_sar_only_pct"])
        assert water > 1.0, f"Water region planted but detected {water:.2f}%"

    def test_already_db_sar_gives_same_result(self, synthetic_tif_pair, tmp_path):
        """Linear and already-dB SAR from the same scene → same stats."""
        from fusion.optical_sar_fusion import analyze_optical_sar
        r_linear = analyze_optical_sar(
            synthetic_tif_pair["opt"], synthetic_tif_pair["sar_linear"],
            output_dir=tmp_path / "linear",
        )
        r_db = analyze_optical_sar(
            synthetic_tif_pair["opt"], synthetic_tif_pair["sar_db"],
            output_dir=tmp_path / "db",
        )
        for k in r_linear["stats"]:
            diff = abs(r_linear["stats"][k] - r_db["stats"][k])
            assert diff < 0.5, f"Linear vs already-dB mismatch in {k}: {diff:.2f}%"

    def test_overlay_file_is_valid_image(self, synthetic_tif_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_tif_pair["opt"], synthetic_tif_pair["sar_linear"],
            output_dir=tmp_path,
        )
        p = Path(result["visual_evidence"])
        assert p.exists()
        img = Image.open(p)
        assert img.width > 0 and img.height > 0
        assert img.mode == "RGB"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic GeoTIFF generator (tests _make_synthetic_test_data)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticGenerator:
    def test_generates_valid_files(self, tmp_path):
        from fusion.optical_sar_fusion import _make_synthetic_test_data
        opt, sar = _make_synthetic_test_data(tmp_path)
        assert opt.exists(), f"Optical file not created: {opt}"
        assert sar.exists(), f"SAR file not created: {sar}"

    def test_full_pipeline_on_generated_data(self, tmp_path):
        from fusion.optical_sar_fusion import _make_synthetic_test_data, analyze_optical_sar
        opt, sar = _make_synthetic_test_data(tmp_path / "data")
        result = analyze_optical_sar(opt, sar, output_dir=tmp_path / "out")
        _assert_m1_schema(result)
        assert result["error"] is None


# ─────────────────────────────────────────────────────────────────────────────
# M1 integration contract tests (these are what M1 depends on)
# ─────────────────────────────────────────────────────────────────────────────

class TestM1Contract:
    """
    These tests define the exact contract between M2 and M1.
    If any of these break, M1's tool registry will stop working.
    """

    def test_function_is_importable(self):
        from fusion.optical_sar_fusion import analyze_optical_sar
        assert callable(analyze_optical_sar)

    def test_accepts_str_paths(self, synthetic_png_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            str(synthetic_png_pair["opt"]),
            str(synthetic_png_pair["sar"]),
            output_dir=str(tmp_path),
        )
        assert result["task"] == "optical_sar_fusion"

    def test_never_raises_always_returns_dict(self, tmp_path):
        """M1 must always get a dict back, even for invalid inputs."""
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            "/nonexistent/optical.tif",
            "/nonexistent/sar.tif",
            output_dir=tmp_path,
        )
        assert isinstance(result, dict)
        assert result["error"] is not None
        assert result["task"] == "optical_sar_fusion"

    def test_visual_evidence_is_abs_path_or_none(self, synthetic_png_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_png_pair["opt"], synthetic_png_pair["sar"],
            output_dir=tmp_path,
        )
        if result["visual_evidence"] is not None:
            assert Path(result["visual_evidence"]).is_absolute(), \
                "visual_evidence must be an absolute path so M1 can serve it"

    def test_result_fully_json_serialisable(self, synthetic_png_pair, tmp_path):
        from fusion.optical_sar_fusion import analyze_optical_sar
        result = analyze_optical_sar(
            synthetic_png_pair["opt"], synthetic_png_pair["sar"],
            output_dir=tmp_path,
        )
        try:
            json.dumps(result)
        except TypeError as e:
            pytest.fail(f"Result is not JSON-serialisable: {e}")

    def test_m1_registry_call_pattern(self, synthetic_png_pair, tmp_path):
        """Simulate exactly how M1's tool registry calls this module."""
        from fusion.optical_sar_fusion import analyze_optical_sar

        registry = {
            "optical_sar_fusion": lambda optical, sar, query, output_dir: analyze_optical_sar(
                optical, sar, query, output_dir
            )
        }
        result = registry["optical_sar_fusion"](
            str(synthetic_png_pair["opt"]),
            str(synthetic_png_pair["sar"]),
            "Use the optical and SAR images together to identify water and built-up regions.",
            str(tmp_path),
        )
        _assert_m1_schema(result)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone / M2→M3 adapter tests (marked slow — require model download)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestRSBackbone:
    @pytest.fixture(scope="class")
    def backbone(self):
        try:
            from backbone.rs_backbone import RSBackbone
            return RSBackbone(use_lora=False)
        except ImportError as e:
            pytest.skip(f"open_clip_torch not installed: {e}")

    def test_encode_image_shape_and_dtype(self, backbone):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        emb = backbone.encode_image(img)
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert emb.dtype == np.float32

    def test_embedding_is_normalised(self, backbone):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        emb = backbone.encode_image(img)
        assert abs(np.linalg.norm(emb) - 1.0) < 0.01

    def test_classify_schema(self, backbone):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = backbone.classify(img)
        for k in ("top_label", "top_score", "all_scores", "embedding", "model_used"):
            assert k in result, f"backbone.classify() missing key: {k}"
        assert 0.0 <= result["top_score"] <= 1.0

    def test_numpy_array_input(self, backbone):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        emb = backbone.encode_image(arr)
        assert emb.ndim == 1

    def test_m3_adapter_schema(self):
        try:
            from backbone.m2_output_adapter import get_rs_features
        except ImportError as e:
            pytest.skip(f"Dependency missing: {e}")
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = get_rs_features(img)
        for k in ("embedding", "top_label", "top_score", "all_scores", "model_used"):
            assert k in result, f"M3 adapter missing key: {k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])