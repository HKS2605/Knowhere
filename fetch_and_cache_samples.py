"""
Pulls 5-8 real samples from VRSBench (via Hugging Face `datasets`, streaming
mode — no full download needed) and runs them through the existing
answer_vqa() pipeline, writing vqa_demo_cache.json.

Run this INSTEAD of `python rs_vqa.py --demo` once you have real samples —
same output format, just backed by real data instead of a synthetic image.

Usage (Kaggle or local, GPU recommended):
    pip install datasets
    python fetch_and_cache_samples.py --n 8
"""

import argparse
import logging

from rs_vqa import RSVQASpecialist, run_demo_cache

logger = logging.getLogger("satquery.fetch_samples")


def load_vrsbench_samples(n: int = 8) -> list[dict]:
    """Streams the first `n` VRSBench samples — doesn't download the full
    dataset. Field names below (`image`/`question`) are VRSBench's typical
    VQA config; if your first run prints different keys, adjust the two
    lines marked below rather than guessing blindly."""
    from datasets import load_dataset

    ds = load_dataset("xiang709/VRSBench", split="train", streaming=True)

    samples = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        if i == 0:
            logger.info("First example fields: %s", list(ex.keys()))

        # --- adjust these two lines if the printed fields above differ ---
        image = ex.get("image")
        query = ex.get("question") or ex.get("query")
        # -------------------------------------------------------------

        if image is None or not query:
            logger.warning("Skipping sample %d — missing image or question field", i)
            continue

        samples.append({"id": f"vrsbench_{i}", "image": image, "query": query})

    if not samples:
        raise RuntimeError(
            "No usable samples pulled — check the field names printed above "
            "and adjust load_vrsbench_samples()."
        )
    return samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Number of samples to pull (5-8 per role doc)")
    parser.add_argument("--out", type=str, default="vqa_demo_cache.json")
    args = parser.parse_args()

    specialist = RSVQASpecialist()
    samples = load_vrsbench_samples(args.n)
    logger.info("Pulled %d real samples, running through pipeline...", len(samples))
    run_demo_cache(specialist, samples, cache_path=args.out)
    print(f"Done — {len(samples)} real samples cached to {args.out}")
