# M5 — Change Detection / Bi-temporal Change-VQA: Solo Build Guide

You are building this **completely alone**, so the plan below assumes zero
help, zero guarantee of GPU access, and no dependency on teammates' code
until the very last step. Every phase produces something that runs and is
demoable on its own.

---

## Phase 0 (10 min) — Environment, no excuses

```bash
mkdir m5 && cd m5
python3 -m venv venv && source venv/bin/activate   # or use conda
pip install numpy opencv-python scikit-image scipy Pillow
# try these too, but don't block on them:
pip install torch torchvision
pip install rasterio   # only if you'll touch real GeoTIFFs
```

Drop `change_detection.py` and `test_change_detection.py` (given to you)
into this folder.

**Checkpoint:** run
```bash
python test_change_detection.py --synthetic --n 5
```
This generates 5 fake before/after pairs internally — you don't need any
dataset yet — and prints change %, category, description, VQA answer,
confidence for each. If this runs clean, your whole pipeline logic is
already proven correct. Do this literally first, before touching real data.

---

## Phase 1 (20–30 min) — Get real bi-temporal data

You don't need to train anything (per team decision: pretrained/classical
methods only where full training isn't feasible solo). You need real pairs
to demo against.

1. **CDVQA** — this is the dataset named in the problem statement for
   change-based VQA evaluation. Search for the public CDVQA repo/release
   (it's built on top of the SECOND change-detection dataset). Download a
   small subset (10–20 pairs is plenty for a demo).
2. Fallback if CDVQA is slow to get: **LEVIR-CD** or **SECOND** dataset —
   both are standard public bi-temporal building/land-cover change sets,
   PNG format, no special reader needed. Use these to demo the *mechanism*
   even if final scoring uses CDVQA.
3. Restructure whatever you download into:
   ```
   cdvqa_samples/
     pair_001/before.png
     pair_001/after.png
     pair_002/before.png
     pair_002/after.png
     ...
   ```
   (Write a tiny one-off script to rename/copy files into this layout —
   don't do it by hand for more than a few pairs.)

**Checkpoint:**
```bash
python test_change_detection.py --data_dir ./cdvqa_samples
```

---

## Phase 2 (30–40 min) — Tune the heuristics against real data

Classical diff + Otsu threshold works out of the box but WILL need
calibration once you see real satellite pairs (they're noisier than my
synthetic ones). In `change_detection.py`, the knobs you'll likely touch:

- `_binarize()` — if you're getting too much noise flagged as "change,"
  increase `min_size` in `remove_small_objects`, or add a second
  `cv2.medianBlur` pass on `pixel_map` before thresholding.
- `_classify_pixels()` — the vegetation/water/built-up thresholds
  (`veg_score > 15`, `water_score > 10`, `builtup_score > 40`) are tuned on
  my synthetic tile. Print `veg_score.mean()`, `water_score.mean()` etc. on
  a few real pairs and adjust the cutoffs so the categories look right on
  visual inspection — this is deliberately a heuristic, not a trained
  model, so "looks right by eye" is the correct bar, not a metric.
- If pixel-diff is too noisy on real SAR-adjacent optical data, that's
  exactly when the pretrained ResNet18 feature-diff branch (already wired
  in, auto-enabled if torch is installed) earns its keep — it's more
  robust to illumination/speckle noise than raw pixel differencing because
  it compares semantic features, not raw brightness.

**Do NOT try to train a change-detection network from scratch solo in this
timeframe.** If you want a step up from feature-diff without training,
the next tier (only if you have slack time) is downloading a *pretrained*
checkpoint for an off-the-shelf Siamese change-detection model (e.g. BIT or
ChangeFormer, from the `open-cd` / MMSegmentation-based repos) and running
inference only — still zero training, just swapping the diff backend. Not
required for the mandatory deliverable.

---

## Phase 3 (15 min) — Change-VQA answer quality pass

Run the 5 representative queries from the problem statement against a few
real pairs and read the answers out loud — are they defensible?

```python
from change_detection import analyze_change
r = analyze_change("cdvqa_samples/pair_001/before.png",
                    "cdvqa_samples/pair_001/after.png",
                    query="Has the built-up area increased, decreased, or remained unchanged?")
print(r["outputs"]["vqa_answer"])
print(r["outputs"]["change_description"])
```

If an answer looks obviously wrong, it's almost always the category
heuristic (Phase 2), not the VQA logic — the VQA function only reads stats
that were already computed, it doesn't reinterpret the image.

---

## Phase 4 (10 min) — Package for handoff to M1

M1 needs exactly one thing from you: a callable function that returns a
JSON-serializable dict in the shared schema. That's `analyze_change()` —
already done. Confirm the contract:

```python
result = analyze_change(before_path, after_path, query=None, out_dir="outputs/")
# result["task"], result["model_tool"], result["inputs"], result["outputs"], result["execution_trace"]
```

If M1's registry expects a slightly different key name, that's a 2-line
rename in the `result = {...}` block at the bottom of `analyze_change()` —
don't renegotiate your whole pipeline over a naming mismatch, just adapt
the dict shape at the boundary.

**Self-test the handoff without M1's code existing yet:** write a 5-line
mock that pretends to be M1's output combiner:

```python
def fake_output_combiner(result):
    assert "outputs" in result and "change_description" in result["outputs"]
    print("M1-compatible ✅", result["outputs"]["change_description"])

fake_output_combiner(analyze_change(before, after, query="What changed?"))
```

This means you can prove your module is integration-ready **before**
anyone else's code exists — critical since you're working with zero
support right now.

---

## Phase 5 (10 min) — PPT assets

Your slide needs (per the plan): before/after images + change-map visuals.
`analyze_change()` already saves three PNGs per run into `out_dir`:
- `*_change_map.png` — heatmap of the raw diff signal (JET colormap)
- `*_overlay.png` — before-image with the detected change region highlighted in red
- `*_before_after_overlay.png` — the 3-panel strip (before | after | overlay), the single best image for a slide

Run once on your best-looking real pair and pull `*_before_after_overlay.png`
straight into the deck.

---

## What to do if you get stuck with zero time left

Priority order — the mandatory deliverable is satisfied at Phase 0/1
already (classical diff + templated description + simple VQA on real
pairs). Everything past that is quality polish:

1. **Must have:** synthetic test passes (Phase 0). Proves the mechanism.
2. **Must have:** runs on ≥5 real CDVQA pairs (Phase 1) with non-garbage
   output.
3. **Should have:** heuristics tuned so categories look sane by eye (Phase 2).
4. **Nice to have:** pretrained ResNet18 feature-diff enabled (already
   automatic if torch installs — don't spend time here if torch install
   fails, classical-only still satisfies the mandatory requirement).
5. **Nice to have:** off-the-shelf pretrained Siamese CD model swap-in.

If Phase 1 (real data) is the blocker close to deadline, demo on synthetic
pairs and say so plainly in the audit trail / PPT — an honest, working
classical pipeline beats a broken attempt at something fancier.
