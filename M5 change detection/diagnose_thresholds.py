"""
diagnose_thresholds.py
------------------------
Run this ONCE you have real data organized (via organize_real_data.py or
manually) into pair_xxx/before.*, after.* folders. It prints the raw
veg_score / water_score / builtup_score distributions from
change_detection._classify_pixels() across your real images, so you can see
whether the hardcoded thresholds (15 / 10 / 40) actually make sense for
YOUR dataset's color range, instead of guessing.

USAGE
    python diagnose_thresholds.py --data_dir ./cdvqa_samples

If the printed percentiles are way off from the current thresholds in
change_detection.py's _classify_pixels(), update those threshold constants
to roughly the real data's ~60-80th percentile for each score.
"""

import argparse
import os
import numpy as np
import cv2

from change_detection import _read_image_any, _classify_pixels


def score_maps(rgb):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = (r + g + b) / 3.0
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    veg_score = (g - r) + (g - b)
    water_score = (b - r) + (b - g) - 0.3 * gray
    edge_density = cv2.blur(edges, (9, 9))
    veg_for_builtup = veg_score  # reused as in main module
    builtup_score = edge_density * 255 - 0.2 * veg_for_builtup
    return veg_score, water_score, builtup_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    all_veg, all_water, all_built = [], [], []
    n = 0
    for name in sorted(os.listdir(args.data_dir)):
        pdir = os.path.join(args.data_dir, name)
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            if fname.lower().startswith(("before", "after")):
                path = os.path.join(pdir, fname)
                img = _read_image_any(path)
                img = cv2.resize(img, (512, 512))
                v, w, bl = score_maps(img)
                all_veg.append(v.flatten())
                all_water.append(w.flatten())
                all_built.append(bl.flatten())
                n += 1

    if n == 0:
        print("No before/after images found under", args.data_dir)
        return

    all_veg = np.concatenate(all_veg)
    all_water = np.concatenate(all_water)
    all_built = np.concatenate(all_built)

    print(f"Scanned {n} images.\n")
    for label, arr, current_threshold in [
        ("veg_score", all_veg, 15),
        ("water_score", all_water, 10),
        ("builtup_score", all_built, 40),
    ]:
        p50, p70, p85, p95 = np.percentile(arr, [50, 70, 85, 95])
        print(f"{label}: current threshold={current_threshold}")
        print(f"  min={arr.min():.1f} p50={p50:.1f} p70={p70:.1f} p85={p85:.1f} p95={p95:.1f} max={arr.max():.1f}")
        if current_threshold < p50:
            print(f"  -> WARNING: current threshold is below the median. Almost everything will be classified")
            print(f"     as this class. Consider raising the threshold toward p70-p85 ({p70:.0f}-{p85:.0f}).")
        elif current_threshold > p95:
            print(f"  -> WARNING: current threshold is above the 95th percentile. Almost nothing will be")
            print(f"     classified as this class. Consider lowering toward p85-p95 ({p85:.0f}-{p95:.0f}).")
        else:
            print(f"  -> Looks reasonable (falls within the real data's spread).")
        print()


if __name__ == "__main__":
    main()
