"""
m2_output_adapter.py  —  M2 → M3 (VQA) Interface
===================================================
This is the ONLY file Mahek (M3/VQA) needs from M2.

It exposes a clean function get_rs_features(image) that:
  1. Runs the RSBackbone to get a CLIP embedding
  2. Returns a structured dict with the embedding + zero-shot classification
     that Mahek's answer_vqa() function can directly use for RS-domain refinement

Schema matches what the single-image VQA pipeline's
"RS-domain refinement" box expects (per the pipeline diagram):
  - embedding: np.ndarray (D,) float32
  - top_label: str
  - top_score: float
  - all_scores: {label: prob}
  - model_used: str

Usage in M3 (Mahek's code):
    from backbone.m2_output_adapter import get_rs_features
    rs = get_rs_features(pil_image)
    # rs["embedding"] → use for CLIP similarity matching
    # rs["top_label"] → RS-domain answer refinement anchor
    # rs["all_scores"] → softmax probabilities for confidence
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Singleton — load once per process, not per call
_BACKBONE = None

def _get_backbone():
    global _BACKBONE
    if _BACKBONE is None:
        from backbone.rs_backbone import RSBackbone
        _BACKBONE = RSBackbone()
    return _BACKBONE


def get_rs_features(
    image: Union[Image.Image, np.ndarray, str, Path],
    candidate_labels: Optional[List[str]] = None,
) -> dict:
    """
    Primary M2 → M3 interface.

    Parameters
    ----------
    image            : PIL Image, numpy array (H,W,C), or path to image file
    candidate_labels : Optional custom label set; uses default RS_LABEL_SET if None

    Returns
    -------
    {
        "embedding":   np.ndarray shape (D,),   # L2-normalised CLIP embedding
        "top_label":   str,                      # best RS class label
        "top_score":   float,                    # softmax probability
        "all_scores":  {label: float},           # full probability distribution
        "model_used":  str,                      # model name for audit trail
    }
    """
    backbone = _get_backbone()

    # Accept file paths
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    result = backbone.classify(image, candidate_labels=candidate_labels, top_k=10)
    # embedding is already inside result from classify()
    return result


def encode_text_for_vqa(text: str) -> np.ndarray:
    """
    Convenience: encode a text string with the RS backbone.
    Mahek's RS-domain refinement step uses this to match VQA answers
    against the RS label space by cosine similarity.

    Returns L2-normalised embedding, shape (D,) float32.
    """
    return _get_backbone().encode_text(text)


def backbone_info() -> dict:
    """Returns model metadata for the audit trail."""
    return _get_backbone().info()