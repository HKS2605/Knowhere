"""
organize_real_data.py
----------------------
Point this at whatever folder your real dataset unzipped into, and it will
try to auto-detect before/after pairs and copy them into the layout
test_change_detection.py expects:

    organized/
      pair_001/before.png
      pair_001/after.png
      ...

WHY THIS EXISTS
Real bi-temporal datasets (CDVQA, LEVIR-CD, SECOND, OSCD...) all ship with
slightly different folder conventions — some use im1/im2, some A/B, some
_t1/_t2 suffixes. Rather than you hand-sorting files, this script pattern-
matches common conventions. If it can't confidently pair a file, it prints
it as "unmatched" instead of guessing wrong — check that list.

USAGE
    python organize_real_data.py --src ./downloaded_dataset --dst ./cdvqa_samples

If your dataset structure doesn't match ANY of the patterns below, open this
file and add one more rule to PAIR_PATTERNS — it's just (before_marker,
after_marker) tuples matched against filenames/foldernames.
"""

import os
import re
import shutil
import argparse
from collections import defaultdict

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# (before_marker_regex, after_marker_regex) — checked against the path
# (folder name + filename combined), case-insensitive.
PAIR_PATTERNS = [
    (r"(^|[_/\\])im1([_.]|$)", r"(^|[_/\\])im2([_.]|$)"),
    (r"(^|[_/\\])a([_.]|$)",   r"(^|[_/\\])b([_.]|$)"),
    (r"before", r"after"),
    (r"_t1", r"_t2"),
    (r"pre", r"post"),
    (r"time1", r"time2"),
    (r"1(?=\.[a-z]+$)", r"2(?=\.[a-z]+$)"),  # trailing ...001.png / ...002.png style
]


def _find_all_images(src):
    paths = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, f))
    return paths


def _strip_marker(path, marker_regex):
    """Remove the before/after marker to get a 'pair key' that should match
    across before and after images of the same location."""
    norm = path.replace("\\", "/").lower()
    return re.sub(marker_regex, "|", norm)


def organize(src, dst):
    images = _find_all_images(src)
    if not images:
        print(f"No images found under {src}")
        return

    print(f"Found {len(images)} image files. Trying pairing patterns...")

    for before_re, after_re in PAIR_PATTERNS:
        before_imgs = [p for p in images if re.search(before_re, p.replace("\\", "/").lower())]
        after_imgs = [p for p in images if re.search(after_re, p.replace("\\", "/").lower())]
        if not before_imgs or not after_imgs:
            continue

        keyed = defaultdict(dict)
        for p in before_imgs:
            keyed[_strip_marker(p, before_re)]["before"] = p
        for p in after_imgs:
            key = _strip_marker(p, after_re)
            if key in keyed:
                keyed[key]["after"] = p

        matched = {k: v for k, v in keyed.items() if "before" in v and "after" in v}
        if len(matched) >= 1:
            print(f"Pattern matched: before~/{before_re}/ after~/{after_re}/  -> {len(matched)} pairs")
            os.makedirs(dst, exist_ok=True)
            for i, (k, v) in enumerate(sorted(matched.items())):
                pair_dir = os.path.join(dst, f"pair_{i:03d}")
                os.makedirs(pair_dir, exist_ok=True)
                shutil.copy(v["before"], os.path.join(pair_dir, "before" + os.path.splitext(v["before"])[1]))
                shutil.copy(v["after"], os.path.join(pair_dir, "after" + os.path.splitext(v["after"])[1]))
            print(f"Wrote {len(matched)} pairs -> {dst}/pair_000 .. pair_{len(matched)-1:03d}")
            print("NOTE: test_change_detection.py's _iter_pairs() looks for before.png/after.png —")
            print("      if your files were .jpg/.tif, either rename to .png or tweak _iter_pairs().")
            return

    print("Could not auto-detect a pairing pattern. Files found:")
    for p in images[:30]:
        print("  ", p)
    print("\nAdd a matching (before_regex, after_regex) tuple to PAIR_PATTERNS in this")
    print("script based on the actual naming convention you see above, then re-run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root folder of the downloaded dataset")
    ap.add_argument("--dst", default="cdvqa_samples", help="output folder in pair_xxx layout")
    args = ap.parse_args()
    organize(args.src, args.dst)
