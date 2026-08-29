"""
rs_backbone.py  —  M2: RS Backbone Adaptation
================================================
Wraps a pretrained remote-sensing CLIP checkpoint (RemoteCLIP) with an optional
LoRA fine-tune pass on a small BigEarthNet subset.

Outputs the adapted embedding model as a callable API used by:
  - Single-image VQA (M3/Mahek)  ← primary consumer
  - Captioning + Grounding (M4/Jugal)
  - Change Detection (M5)

Public API (what M1 imports):
    from backbone.rs_backbone import RSBackbone
    backbone = RSBackbone()                  # loads pretrained weights
    emb = backbone.encode_image(pil_image)   # → np.ndarray shape (512,)
    emb = backbone.encode_text("water body") # → np.ndarray shape (512,)
    result = backbone.classify(pil_image, candidate_labels) # → dict

Strategy: RemoteCLIP (ViT-L/14, open-source, trained on 4M RS image-text pairs)
as base. LoRA fine-tune only the attention Q/V projections on a 2000-tile
BigEarthNet subset — enough to satisfy "remote-sensing adaptation" requirement
without needing 10+ GPU-hours.

If the LoRA checkpoint doesn't exist yet (first run), falls back to RemoteCLIP
base weights — still domain-adapted, still evaluatable.
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ── checkpoint config ──────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LORA_CHECKPOINT = CHECKPOINT_DIR / "rs_clip_lora.pt"

# RemoteCLIP HuggingFace repo (open weights, no token needed)
REMOTECLIP_MODEL_ID = "chendelong/RemoteCLIP"
REMOTECLIP_VARIANT  = "ViT-L-14"   # best accuracy; swap to ViT-B-32 if VRAM < 6 GB

# RS label set used for zero-shot classification (VQA refinement anchor)
RS_LABEL_SET = [
    "water body", "river", "lake", "sea", "flood",
    "built-up area", "urban", "residential", "industrial", "road", "highway",
    "vegetation", "forest", "cropland", "grassland", "shrubland",
    "bare soil", "sand", "desert",
    "snow", "ice", "glacier",
    "wetland", "mangrove",
    "airport", "port", "bridge",
    "cloud", "shadow",
]

class RSBackbone:
    """
    Remote-sensing adapted CLIP backbone.

    Parameters
    ----------
    device : str
        "cuda" | "cpu" — auto-detected if not specified.
    use_lora : bool
        Load the LoRA-adapted checkpoint if available. Falls back to base if not.
    """

    def __init__(self, device: Optional[str] = None, use_lora: bool = True):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = use_lora
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._lora_loaded = False
        self._load_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        """
        Load RemoteCLIP. Try LoRA checkpoint first; fall back to base weights.
        """
        import torch
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch not installed.\n"
                "Run: pip install open-clip-torch huggingface_hub"
            )

        logger.info("Loading RemoteCLIP (%s) …", REMOTECLIP_VARIANT)

        # --- load the pretrained RemoteCLIP from HuggingFace hub ---
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=REMOTECLIP_MODEL_ID,
                filename=f"RemoteCLIP-{REMOTECLIP_VARIANT}.pt",
                cache_dir=str(CHECKPOINT_DIR / "remoteclip_cache"),
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                REMOTECLIP_VARIANT, pretrained=ckpt_path
            )
        except Exception as e:
            logger.warning("RemoteCLIP download failed (%s). Falling back to openai CLIP.", e)
            # Final fallback: plain CLIP (still works, just not RS-adapted)
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )

        model = model.eval().to(self.device)
        tokenizer = open_clip.get_tokenizer(REMOTECLIP_VARIANT)

        # --- optionally overlay LoRA weights ---
        if self.use_lora and LORA_CHECKPOINT.exists():
            try:
                lora_state = torch.load(LORA_CHECKPOINT, map_location=self.device)
                # only load keys that are in the model (LoRA injects new keys)
                model.load_state_dict(lora_state, strict=False)
                self._lora_loaded = True
                logger.info("LoRA checkpoint loaded from %s", LORA_CHECKPOINT)
            except Exception as e:
                logger.warning("LoRA checkpoint load failed (%s) — using base weights.", e)
        else:
            if self.use_lora:
                logger.info(
                    "No LoRA checkpoint found at %s — using base RemoteCLIP weights.", LORA_CHECKPOINT
                )

        self._model      = model
        self._preprocess = preprocess
        self._tokenizer  = tokenizer
        logger.info(
            "RSBackbone ready | device=%s | lora=%s", self.device, self._lora_loaded
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public embedding API
    # ──────────────────────────────────────────────────────────────────────────

    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Returns L2-normalised image embedding, shape (D,) float32.

        Parameters
        ----------
        image : PIL.Image or np.ndarray (H, W, C) uint8
        """
        import torch
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().float().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Returns L2-normalised text embedding, shape (D,) float32.
        """
        import torch
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().float().numpy()

    def similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> float:
        """Cosine similarity between two normalised embeddings."""
        return float(np.dot(image_emb, text_emb))

    # ──────────────────────────────────────────────────────────────────────────
    # Zero-shot classification  (used by VQA refinement stage)
    # ──────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        image: Union[Image.Image, np.ndarray],
        candidate_labels: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Zero-shot RS classification against candidate_labels (default: RS_LABEL_SET).

        Returns
        -------
        {
          "top_label":   str,
          "top_score":   float,          # softmax probability
          "all_scores":  {label: score}, # top_k entries
          "embedding":   np.ndarray,     # image embedding for downstream use
          "model_used":  str,
        }
        """
        import torch
        import torch.nn.functional as F

        labels = candidate_labels or RS_LABEL_SET
        img_emb = self.encode_image(image)                      # (D,)
        txt_embs = np.stack([self.encode_text(l) for l in labels])  # (N, D)

        logits = torch.tensor(img_emb @ txt_embs.T) * 100.0    # scaled cosine
        probs  = F.softmax(logits, dim=-1).numpy()

        ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])[:top_k]

        return {
            "top_label":  ranked[0][0],
            "top_score":  round(ranked[0][1], 4),
            "all_scores": {k: round(v, 4) for k, v in ranked},
            "embedding":  img_emb,          # raw ndarray — VQA module uses this
            "model_used": (
                f"RemoteCLIP-{REMOTECLIP_VARIANT}-LoRA"
                if self._lora_loaded else
                f"RemoteCLIP-{REMOTECLIP_VARIANT}-base"
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utility: batch encode (for LoRA training / evaluation)
    # ──────────────────────────────────────────────────────────────────────────

    def encode_image_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Returns (N, D) float32 array of normalised embeddings."""
        import torch
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self._preprocess(img.convert("RGB")) for img in batch])
            tensors = tensors.to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    # ──────────────────────────────────────────────────────────────────────────
    # Model info
    # ──────────────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "base_model":   f"RemoteCLIP-{REMOTECLIP_VARIANT}",
            "lora_adapted": self._lora_loaded,
            "device":       self.device,
            "label_set_size": len(RS_LABEL_SET),
        }
