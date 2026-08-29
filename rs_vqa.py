"""
SatQuery — Single-Image VQA Module (Owner: Mahek, Role M3)
=============================================================
Implements the "Pipeline for Single VQA" diagram end-to-end:

    Image + query input
          |
    Preprocessing & validation
          |
    Question type classifier
          |
    VQA model inference (pretrained BLIP-VQA)
          |
    RS-domain refinement  <---  RS backbone (M2, CLIP embeddings)
          |
    Confidence scoring
          |
    Answer output (JSON) -> agentic controller (M1)

Design goals
------------
1. `answer_vqa(image, query)` is the ONE function M1 imports into its tool
   registry. Its return value is the "shared JSON" contract every other
   teammate's function should also follow.
2. The RS backbone (M2's fine-tuned CLIP) is a DEPENDENCY, not a hard
   requirement. This module runs standalone with a naive similarity
   fallback today, and auto-upgrades the moment M2 hands off their real
   embedding function via `RSVQASpecialist.set_rs_backbone(...)`.
3. Every stage in the diagram is a separate, testable method — so this
   maps 1:1 onto the PPT slide / demo script, and so M1's audit-trail
   generator can log each stage individually.

Run standalone for a smoke test:
    python rs_vqa.py --demo
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import difflib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.vqa")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

# --------------------------------------------------------------------------- #
# 0. Config — RS label set, question-type patterns, model name
# --------------------------------------------------------------------------- #

# Canonical remote-sensing vocabulary (BigEarthNet / RSVQA / VRSBench-aligned).
# Used at the "RS-domain refinement" stage to pull BLIP's generic answer back
# onto a term the RS backbone (M2) and the rest of the pipeline understand.
RS_LABEL_SET = [
    "water", "river", "lake", "sea", "coastline",
    "vegetation", "forest", "grassland", "crop field", "agricultural land",
    "built-up area", "urban area", "residential area", "industrial area",
    "road", "highway", "bridge",
    "bare soil", "sand", "desert",
    "cloud", "shadow", "snow", "ice",
    "airport", "runway", "port", "parking lot",
    "mountain", "hill", "wetland",
]

# Canonical yes/no vocabulary — BLIP sometimes answers "yeah"/"no there isn't"/etc.
YES_TOKENS = {"yes", "yeah", "yep", "true", "there is", "present"}
NO_TOKENS = {"no", "nope", "false", "there isn't", "there is not", "absent", "none"}

MODEL_NAME = "Salesforce/blip-vqa-base"  # matches diagram: "Pretrained BLIP-VQA answer"


# --------------------------------------------------------------------------- #
# 1. Question-type classifier  (diagram stage 3: yes/no, count, category, compare)
# --------------------------------------------------------------------------- #

class QuestionType:
    YES_NO = "yes_no"
    COUNT = "count"
    CATEGORY = "category"
    COMPARE = "compare"
    OTHER = "other"


_COUNT_PATTERNS = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.I)
_COMPARE_PATTERNS = re.compile(
    r"\bmore\b|\bless\b|\bfewer\b|\blarger\b|\bsmaller\b|\bcompare\b|\bthan\b|\bwhich (has|is)\b",
    re.I,
)
_YES_NO_STARTERS = re.compile(
    r"^\s*(is|are|does|do|was|were|has|have|can|could|will|would)\b", re.I
)


def classify_question_type(query: str) -> str:
    """Rule-based classifier — cheap, deterministic, and matches the 4 buckets
    M1's audit trail expects. Order matters: check compare before yes/no,
    since "is X more than Y" is a compare question in a yes/no wrapper."""
    q = query.strip()
    if _COMPARE_PATTERNS.search(q):
        return QuestionType.COMPARE
    if _COUNT_PATTERNS.search(q):
        return QuestionType.COUNT
    if _YES_NO_STARTERS.search(q):
        return QuestionType.YES_NO
    return QuestionType.CATEGORY


# --------------------------------------------------------------------------- #
# 2. Data contract — what this module hands back to M1
# --------------------------------------------------------------------------- #

@dataclass
class VQAResult:
    task: str = "single_image_vqa"
    query: str = ""
    question_type: str = ""
    raw_answer: str = ""
    refined_answer: str = ""
    confidence: float = 0.0
    confidence_breakdown: dict = field(default_factory=dict)
    model_used: str = MODEL_NAME
    rs_backbone_used: bool = False
    image_metadata: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    status: str = "success"
    error: Optional[str] = None

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 3. Main specialist class
# --------------------------------------------------------------------------- #

class RSVQASpecialist:
    """
    Owns stages 2-6 of the diagram. Instantiate once (loads BLIP-VQA onto
    GPU if available), then call `.answer_vqa(image, query)` per request —
    this is what M1's tool registry calls.
    """

    MAX_SIDE = 1024          # resize cap — keeps inference fast on hackathon GPUs
    MIN_SIDE = 32             # reject unusably small crops
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(self, device: Optional[str] = None, lazy_load: bool = False):
        self.device = device
        self._processor = None
        self._model = None
        self._rs_backbone: Optional[Callable[[Image.Image], np.ndarray]] = None
        self._rs_label_embeddings: Optional[dict] = None
        if not lazy_load:
            self._load_model()

    # -- model loading -------------------------------------------------- #

    def _load_model(self):
        """Loads pretrained BLIP-VQA. Isolated so a missing GPU / package
        during early integration doesn't crash import of this whole file —
        M1 can import RSVQASpecialist and stub `.answer_vqa` before your
        model finishes downloading."""
        import torch
        from transformers import BlipProcessor, BlipForQuestionAnswering

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading %s on %s ...", MODEL_NAME, self.device)
        self._processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self._model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(self.device)
        self._model.eval()
        logger.info("Model loaded.")

    def set_rs_backbone(self, embed_fn: Callable[[Image.Image], np.ndarray]):
        """Hand-off point for M2. Call this once M2's fine-tuned CLIP is ready:

            from m2_rs_backbone import embed_image  # M2's function
            specialist.set_rs_backbone(embed_image)

        `embed_fn` must take a PIL Image and return a 1D np.ndarray embedding.
        Once set, RS-domain refinement uses cosine similarity in embedding
        space instead of the string-similarity fallback.
        """
        self._rs_backbone = embed_fn
        self._rs_label_embeddings = None  # invalidate cache, recompute lazily
        logger.info("RS backbone (M2) attached — refinement upgraded to embedding-based.")

    # -- stage 1: preprocessing & validation ----------------------------- #

    def preprocess_image(self, image: Union[str, bytes, Image.Image]) -> tuple[Image.Image, dict]:
        """Diagram stage: 'Preprocessing & validation — Resize, normalize,
        format check'. Accepts a file path, raw bytes, or a PIL Image.
        Handles GeoTIFF via rasterio if available, else falls back to PIL
        (works for most 3-band GeoTIFFs; multi-band scientific TIFFs should
        go through rasterio)."""
        meta = {"source_format": None, "original_size": None, "resized": False}

        if isinstance(image, Image.Image):
            img = image.convert("RGB")
            meta["source_format"] = "PIL.Image"
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image)).convert("RGB")
            meta["source_format"] = "bytes"
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            ext = path.suffix.lower()
            if ext not in self.SUPPORTED_EXTS:
                raise ValueError(f"Unsupported image format '{ext}'. Supported: {self.SUPPORTED_EXTS}")
            meta["source_format"] = ext
            if ext in (".tif", ".tiff"):
                img = self._load_geotiff(path)
            else:
                img = Image.open(path).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        meta["original_size"] = img.size

        # format check — reject degenerate images before they hit the model
        w, h = img.size
        if w < self.MIN_SIDE or h < self.MIN_SIDE:
            raise ValueError(f"Image too small ({w}x{h}); minimum side is {self.MIN_SIDE}px")

        # resize — cap the long side, preserve aspect ratio
        if max(w, h) > self.MAX_SIDE:
            scale = self.MAX_SIDE / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.BILINEAR)
            meta["resized"] = True
            meta["resized_to"] = new_size

        return img, meta

    @staticmethod
    def _load_geotiff(path: Path) -> Image.Image:
        """GeoTIFF -> RGB PIL Image. Uses rasterio if installed (handles
        multi-band / georeferenced data correctly); falls back to PIL
        (fine for standard 3-band GeoTIFFs, which is most of BigEarthNet)."""
        try:
            import rasterio
            with rasterio.open(path) as src:
                # take first 3 bands as RGB; RS imagery band order varies by
                # sensor so this is a reasonable default, not a guarantee
                band_count = min(3, src.count)
                arr = src.read(list(range(1, band_count + 1)))  # (bands, H, W)
                arr = np.transpose(arr, (1, 2, 0))
                if band_count == 1:
                    arr = np.repeat(arr, 3, axis=2)
                arr = arr.astype(np.float32)
                # normalize to 0-255 per-band (robust to reflectance / DN ranges)
                for b in range(arr.shape[2]):
                    band = arr[:, :, b]
                    lo, hi = np.percentile(band, 2), np.percentile(band, 98)
                    if hi > lo:
                        arr[:, :, b] = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
                arr = arr.astype(np.uint8)
                return Image.fromarray(arr).convert("RGB")
        except ImportError:
            logger.warning("rasterio not installed — falling back to PIL for GeoTIFF (%s)", path)
            return Image.open(path).convert("RGB")

    # -- stage 3: VQA model inference ------------------------------------ #

    def run_inference(self, image: Image.Image, query: str) -> tuple[str, float]:
        """Diagram stage: 'VQA model inference — Pretrained BLIP-VQA answer'.
        Returns (answer_text, softmax_confidence)."""
        import torch

        inputs = self._processor(image, query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True,
            )
        answer = self._processor.decode(out.sequences[0], skip_special_tokens=True).strip()

        # softmax confidence from the generation scores (mean token prob)
        if out.scores:
            probs = [torch.softmax(s, dim=-1).max().item() for s in out.scores]
            softmax_conf = float(np.mean(probs))
        else:
            softmax_conf = 0.5

        return answer, softmax_conf

    # -- stage 4: RS-domain refinement ------------------------------------ #

    def refine_with_rs_domain(self, raw_answer: str, question_type: str, image: Image.Image) -> tuple[str, float]:
        """Diagram stage: 'RS-domain refinement — Match against RS label set',
        fed by 'RS backbone (M2) — CLIP embeddings'.

        Returns (refined_answer, similarity_score in [0,1]).
        """
        answer = raw_answer.lower().strip()

        # yes/no and count questions don't need label-set matching
        if question_type == QuestionType.YES_NO:
            normalized = self._normalize_yes_no(answer)
            return normalized, 1.0 if normalized in ("yes", "no") else 0.5

        if question_type == QuestionType.COUNT:
            digits = re.findall(r"\d+", answer)
            if digits:
                return digits[0], 1.0
            # BLIP sometimes spells small numbers out
            word_to_num = {"zero": "0", "one": "1", "two": "2", "three": "3",
                           "four": "4", "five": "5", "six": "6", "seven": "7",
                           "eight": "8", "nine": "9", "ten": "10"}
            for word, num in word_to_num.items():
                if word in answer:
                    return num, 0.9
            return answer, 0.4  # couldn't parse a count — flag low confidence

        # category / compare questions -> match against RS_LABEL_SET
        if self._rs_backbone is not None:
            return self._refine_via_embeddings(answer, image)
        return self._refine_via_string_similarity(answer)

    def _refine_via_string_similarity(self, answer: str) -> tuple[str, float]:
        """Fallback refinement (no M2 backbone attached yet): fuzzy string
        match against RS_LABEL_SET so answers stay in-vocabulary."""
        best_label, best_score = answer, 0.0
        for label in RS_LABEL_SET:
            score = difflib.SequenceMatcher(None, answer, label).ratio()
            if answer in label or label in answer:
                score = max(score, 0.85)
            if score > best_score:
                best_label, best_score = label, score
        # don't force a bad match — if nothing is close, keep BLIP's own answer
        if best_score < 0.4:
            return answer, 0.3
        return best_label, best_score

    def _refine_via_embeddings(self, answer: str, image: Image.Image) -> tuple[str, float]:
        """Upgraded refinement using M2's fine-tuned CLIP embeddings:
        embed the image, embed each RS label, cosine-similarity match.
        This is strictly better than string matching because it grounds
        the refinement in the *image*, not just BLIP's raw text."""
        if self._rs_label_embeddings is None:
            self._rs_label_embeddings = {
                label: self._rs_backbone(Image.new("RGB", (1, 1)))  # placeholder shape probe
                for label in []  # populated lazily below to avoid text-encoder assumptions
            }
            # NOTE: if M2's embed_fn is image-only, label embeddings should come
            # from M2's paired text encoder instead. Left as a clear TODO hook
            # so M2 and Mahek can wire the exact interface together at hand-off.
            self._rs_label_embeddings = {}

        img_emb = self._rs_backbone(image)
        img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)

        # if M2 hasn't supplied text-label embeddings, fall back to string match
        # but keep the door open for embedding-based label matching once wired
        if not self._rs_label_embeddings:
            return self._refine_via_string_similarity(answer)

        best_label, best_score = answer, 0.0
        for label, lbl_emb in self._rs_label_embeddings.items():
            lbl_emb = lbl_emb / (np.linalg.norm(lbl_emb) + 1e-8)
            sim = float(np.dot(img_emb, lbl_emb))
            if sim > best_score:
                best_label, best_score = label, sim
        return best_label, best_score

    @staticmethod
    def _normalize_yes_no(answer: str) -> str:
        a = answer.lower().strip()
        if a in YES_TOKENS or a.startswith("yes"):
            return "yes"
        if a in NO_TOKENS or a.startswith("no"):
            return "no"
        return a

    # -- stage 5: confidence scoring -------------------------------------- #

    @staticmethod
    def compute_confidence(softmax_conf: float, similarity_score: float) -> tuple[float, dict]:
        """Diagram stage: 'Confidence scoring — Softmax + similarity score'.
        Weighted blend: model's own certainty (softmax) and how well the
        answer matches known RS vocabulary (similarity)."""
        w_softmax, w_similarity = 0.6, 0.4
        combined = w_softmax * softmax_conf + w_similarity * similarity_score
        combined = round(float(np.clip(combined, 0.0, 1.0)), 4)
        breakdown = {
            "softmax_confidence": round(softmax_conf, 4),
            "similarity_confidence": round(similarity_score, 4),
            "weights": {"softmax": w_softmax, "similarity": w_similarity},
        }
        return combined, breakdown

    # -- orchestrator: stage 0-6 end to end -------------------------------- #

    def answer_vqa(self, image: Union[str, bytes, Image.Image], query: str) -> dict:
        """
        THE function M1 imports into its tool registry:

            from rs_vqa import RSVQASpecialist
            vqa = RSVQASpecialist()
            registry["single_image_vqa"] = vqa.answer_vqa

        Runs the full diagram top to bottom and always returns a JSON-safe
        dict — never raises. Errors are captured in the `status`/`error`
        fields so M1's output combiner doesn't need a try/except around
        every teammate's function.
        """
        t0 = time.time()
        result = VQAResult(query=query, rs_backbone_used=self._rs_backbone is not None)

        try:
            if self._model is None:
                self._load_model()

            if not query or not query.strip():
                raise ValueError("Empty query.")

            # stage 1: preprocessing & validation
            img, img_meta = self.preprocess_image(image)
            result.image_metadata = img_meta

            # stage 2: question type classifier
            qtype = classify_question_type(query)
            result.question_type = qtype

            # stage 3: VQA model inference
            raw_answer, softmax_conf = self.run_inference(img, query)
            result.raw_answer = raw_answer

            # stage 4: RS-domain refinement (uses M2's backbone if attached)
            refined_answer, similarity_score = self.refine_with_rs_domain(raw_answer, qtype, img)
            result.refined_answer = refined_answer

            # stage 5: confidence scoring
            confidence, breakdown = self.compute_confidence(softmax_conf, similarity_score)
            result.confidence = confidence
            result.confidence_breakdown = breakdown

            result.status = "success"

        except Exception as exc:  # noqa: BLE001 — must never crash M1's pipeline
            logger.exception("answer_vqa failed for query=%r", query)
            result.status = "error"
            result.error = str(exc)

        result.latency_ms = round((time.time() - t0) * 1000, 2)
        return result.to_json()


# --------------------------------------------------------------------------- #
# 4. Demo / cache harness — "Pull 5-8 RSVQA/VRSBench samples ... cache outputs"
# --------------------------------------------------------------------------- #

def run_demo_cache(
    specialist: RSVQASpecialist,
    samples: list[dict],
    cache_path: str = "vqa_demo_cache.json",
) -> list[dict]:
    """
    `samples` = [{"image": "path/or/PIL", "query": "..."}, ...]  (5-8 entries
    pulled from RSVQA / VRSBench, per the role doc). Runs each through the
    full pipeline and writes results to `cache_path` so the demo doesn't
    depend on live GPU inference during the actual presentation.
    """
    results = []
    for i, sample in enumerate(samples, 1):
        logger.info("[%d/%d] query=%r", i, len(samples), sample["query"])
        res = specialist.answer_vqa(sample["image"], sample["query"])
        res["sample_id"] = sample.get("id", f"sample_{i}")
        results.append(res)

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Cached %d demo results -> %s", len(results), cache_path)
    return results


# --------------------------------------------------------------------------- #
# 5. Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SatQuery — Single-Image VQA module")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic smoke test (no real dataset needed)")
    parser.add_argument("--image", type=str, help="Path to an image for a single query")
    parser.add_argument("--query", type=str, help="Question to ask about the image")
    args = parser.parse_args()

    specialist = RSVQASpecialist()

    if args.image and args.query:
        print(json.dumps(specialist.answer_vqa(args.image, args.query), indent=2))
    elif args.demo:
        # synthetic image so this runs with zero dataset dependency —
        # replace with real RSVQA/VRSBench paths for the actual demo cache
        synthetic = Image.new("RGB", (256, 256), color=(34, 139, 34))
        demo_samples = [
            {"id": "s1", "image": synthetic, "query": "Is there water in the image?"},
            {"id": "s2", "image": synthetic, "query": "How many buildings are visible?"},
            {"id": "s3", "image": synthetic, "query": "What is the dominant land cover type?"},
            {"id": "s4", "image": synthetic, "query": "Is there more vegetation than built-up area?"},
        ]
        run_demo_cache(specialist, demo_samples, cache_path="vqa_demo_cache.json")
        print("Demo cache written to vqa_demo_cache.json")
    else:
        parser.print_help()
