"""
Test / demo harness for change_detection.py

Two modes:
  1. --synthetic  : generates fake before/after satellite-like image pairs
                     on the fly (no dataset download needed). Use this FIRST
                     to prove your pipeline works end-to-end, completely solo.
  2. --data_dir    : points at a folder of real CDVQA-style pairs, structured as
                     data_dir/
                         pair_001/before.png
                         pair_001/after.png
                         pair_002/before.png
                         pair_002/after.png
                         ...
                     (This matches how CDVQA / LEVIR-CD style sets are usually
                     unzipped. Adjust _iter_pairs() if your download differs.)

Run:
    python test_change_detection.py --synthetic
    python test_change_detection.py --data_dir ./cdvqa_samples
"""

import os
import argparse
import json
import numpy as np
import cv2

from change_detection import analyze_change

REPRESENTATIVE_QUERIES = [
    "What changed between these two dates, and where did the change occur?",
    "Has the built-up area increased, decreased, or remained unchanged?",
    "Is there any change between the two images?",
    "Where did the vegetation loss happen?",
]


def _make_synthetic_pair(seed: int, out_dir: str):
    """Creates a plausible-looking before/after 'satellite tile': green
    (vegetation) base, a blue water patch, and in 'after' we bulldoze part
    of the vegetation into a gray built-up block — so every heuristic in
    change_detection.py has something real to detect."""
    rng = np.random.RandomState(seed)
    h, w = 512, 512
    before = np.zeros((h, w, 3), dtype=np.uint8)
    before[:, :] = [60, 140, 60]  # vegetation green
    # water body
    cv2.circle(before, (380, 120), 70, (40, 60, 180), -1)
    # some texture noise so it's not flat
    before = np.clip(before.astype(np.int16) + rng.randint(-10, 10, before.shape), 0, 255).astype(np.uint8)

    after = before.copy()
    # simulate built-up expansion in a specific quadrant (south-east-ish)
    cv2.rectangle(after, (300, 300), (470, 470), (120, 120, 120), -1)
    # add fake "roads" edges so builtup heuristic (edge density) fires
    for i in range(300, 470, 20):
        cv2.line(after, (300, i), (470, i), (90, 90, 90), 1)
        cv2.line(after, (i, 300), (i, 470), (90, 90, 90), 1)
    after = np.clip(after.astype(np.int16) + rng.randint(-10, 10, after.shape), 0, 255).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    before_path = os.path.join(out_dir, "before.png")
    after_path = os.path.join(out_dir, "after.png")
    cv2.imwrite(before_path, cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(after_path, cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
    return before_path, after_path


def _iter_pairs(data_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    for name in sorted(os.listdir(data_dir)):
        pdir = os.path.join(data_dir, name)
        if not os.path.isdir(pdir):
            continue
        before_path = after_path = None
        for f in os.listdir(pdir):
            low = f.lower()
            if low.startswith("before") and low.endswith(exts):
                before_path = os.path.join(pdir, f)
            elif low.startswith("after") and low.endswith(exts):
                after_path = os.path.join(pdir, f)
        if before_path and after_path:
            yield name, before_path, after_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use generated fake pairs")
    ap.add_argument("--data_dir", default=None, help="folder of real pair_xxx/before.png,after.png")
    ap.add_argument("--n", type=int, default=5, help="number of synthetic pairs to generate")
    ap.add_argument("--out_dir", default="test_outputs")
    args = ap.parse_args()

    results = []

    if args.data_dir:
        pairs = list(_iter_pairs(args.data_dir))
        if not pairs:
            print(f"No pairs found under {args.data_dir}. Expected pair_xxx/before.png + after.png")
            return
    else:
        args.synthetic = True
        pairs = []
        for i in range(args.n):
            b, a = _make_synthetic_pair(seed=i, out_dir=os.path.join("synthetic_pairs", f"pair_{i:03d}"))
            pairs.append((f"pair_{i:03d}", b, a))

    for idx, (name, before_path, after_path) in enumerate(pairs):
        query = REPRESENTATIVE_QUERIES[idx % len(REPRESENTATIVE_QUERIES)]
        print(f"\n=== {name} | query: '{query}' ===")
        result = analyze_change(before_path, after_path, query=query, out_dir=args.out_dir)
        print(f"  change %       : {result['outputs']['change_percentage']}")
        print(f"  category       : {result['outputs']['dominant_change_category']}")
        print(f"  quadrant       : {result['outputs']['dominant_quadrant']}")
        print(f"  description    : {result['outputs']['change_description']}")
        print(f"  vqa_answer     : {result['outputs']['vqa_answer']}")
        print(f"  confidence     : {result['outputs']['confidence']}")
        print(f"  models_used    : {result['execution_trace']['models_used']}")
        print(f"  latency (s)    : {result['execution_trace']['latency_seconds']}")
        results.append(result)

    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} full result JSONs -> {os.path.join(args.out_dir, 'all_results.json')}")
    print(f"Visuals saved under -> {args.out_dir}/")


if __name__ == "__main__":
    main()
