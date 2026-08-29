# SatQuery AI — Frontend/Backend Integration Guide

Purpose: one file the whole team can point to so nobody has to guess where
their code goes, how the pieces talk to each other, or what the shared data
contract looks like. Written from M5's side (change detection) but covers
the whole repo layout since integration only works if everyone follows the
same structure.

---

## 1. Repo folder structure (recommended)

```
Knowhere/
├── backend/
│   ├── main.py                  <- M1: FastAPI app, /analyze endpoint
│   ├── registry.py               <- M1: task -> function mapping
│   ├── output_combiner.py        <- M1: merges specialist outputs
│   ├── specialists/
│   │   ├── vqa.py                <- Mahek: single-image VQA
│   │   ├── captioning_grounding.py <- M4: captioning + grounding
│   │   ├── change_detection.py   <- M5 (you): this module
│   │   ├── fusion.py              <- M2: optical-SAR fusion
│   │   └── rs_backbone.py         <- M2: RS-adapted backbone
│   ├── outputs/                  <- generated images served via /outputs/ static route
│   ├── requirements.txt
│   └── venv/                     (gitignored)
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx                <- Urvashi: upload + query UI, fetch calls
│   │   ├── components/
│   │   │   ├── UploadPanel.jsx
│   │   │   ├── ResultsPanel.jsx   <- renders image evidence, confidence, trace
│   │   │   └── ExecutionTrace.jsx
│   │   └── main.jsx
│   └── node_modules/             (gitignored)
│
├── docs/
│   ├── M5_STEP_BY_STEP_GUIDE.md
│   └── architecture.png          (for the final PPT / README)
│
├── .gitignore
└── README.md
```

**Where your M5 files go:** everything currently in your
`m5_change_detection/` folder becomes `backend/specialists/change_detection.py`
(just the core module — test/diagnostic/organize scripts can live in a
`backend/specialists/tools/` subfolder or stay at repo root under `scripts/`,
your call, they're dev utilities, not part of the running app).

---

## 2. The shared contract (this is what actually makes integration work)

Every specialist function — yours included — takes specialist-specific
inputs and returns ONE dict shape. M1's `output_combiner.py` and Urvashi's
`ResultsPanel.jsx` are both written against this shape, so as long as you
match it, nobody needs to touch your code to wire you in.

```json
{
  "task": "change_detection" | "change_vqa",
  "model_tool": "string describing which model/method ran",
  "inputs": { "...": "whatever the specialist needs, echoed back" },
  "outputs": {
    "...": "specialist-specific result fields",
    "confidence": 0.0
  },
  "execution_trace": {
    "run_id": "string",
    "selected_task": "string",
    "models_used": ["list", "of", "strings"],
    "params": { "...": "..." },
    "latency_seconds": 0.0
  }
}
```

`analyze_change()` in `change_detection.py` already returns exactly this.
Do not rename top-level keys without telling M1 — that will break
`output_combiner.py` silently returning `None`/`KeyError` for your task.

---

## 3. How M1 wires you in (`backend/registry.py`)

```python
from specialists.change_detection import analyze_change
from specialists.vqa import answer_vqa
# ... etc

TASK_REGISTRY = {
    "change_detection": analyze_change,
    "change_vqa": analyze_change,   # same function; query=None vs query=str
    "single_vqa": answer_vqa,
    # ...
}

def route_and_run(task_name, **kwargs):
    fn = TASK_REGISTRY[task_name]
    return fn(**kwargs)
```

M1's intent classifier decides `task_name` from the user's query + how many
images were uploaded (2 images + a change-y query -> `change_detection` or
`change_vqa`). You don't write any of that routing logic — you only need to
make sure `analyze_change()` keeps accepting
`(image_before_path, image_after_path, query=None, out_dir=...)`.

---

## 4. How M1 serves your output images to the frontend

Your `analyze_change()` writes PNGs to `out_dir` (e.g.
`backend/outputs/<run_id>_overlay.png`). M1's FastAPI app mounts that folder
as a static route:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
```

So a path you return like `outputs/abcd1234_overlay.png` becomes reachable
at `http://localhost:8000/outputs/abcd1234_overlay.png` — that's the URL
Urvashi's frontend puts directly into an `<img src=...>` tag. **Make sure
the `out_dir` you pass into `analyze_change()` in production is the same
`outputs/` folder M1 mounts** — coordinate this path with M1 directly, don't
assume; the sample scripts default to `test_outputs/` for your own local
testing, which is fine standalone but should change to `outputs/` (or
whatever M1's app.py uses) once you're wired into the real backend.

---

## 5. How the frontend actually calls this (Urvashi's side, for your awareness)

```javascript
// frontend/src/App.jsx (simplified)
const formData = new FormData();
formData.append("image_before", beforeFile);
formData.append("image_after", afterFile);
formData.append("query", queryText);

const res = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  body: formData,
});
const result = await res.json();

// result.outputs.change_description   -> shown as text
// result.outputs.overlay_path         -> shown as <img>, prefixed with backend origin
// result.outputs.confidence           -> shown as a progress bar
// result.execution_trace              -> shown inside a <details> collapsible
```

You don't need to write this — just know that every field name in your
`outputs` dict is a field Urvashi's UI is (or will be) reading directly. If
you ever add a new output field, tell her the exact key name so she can
render it; don't rename existing ones without a heads-up, same rule as
Section 2.

---

## 6. Local dev — running the whole stack together

Two terminals, per the team's agreed stack:

**Terminal 1 — backend**
```powershell
cd backend
venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend**
```powershell
cd frontend
npm run dev
```
Vite serves on port 5173 by default and proxies/fetches to `localhost:8000`.

You (M5) don't need the frontend running to test your own code — everything
in `test_change_detection.py` runs standalone. You only need this section
once you're doing the final integration pass with M1 and Urvashi.

---

## 7. Integration checklist before demo day

- [ ] `change_detection.py` copied into `backend/specialists/`
- [ ] `out_dir` default matches M1's actual `outputs/` folder path
- [ ] M1's `registry.py` imports `analyze_change` and it runs without error
      inside the FastAPI process (not just your standalone script — test
      this together, import errors from relative paths are the #1
      integration bug)
- [ ] Run one of the 5 representative queries end-to-end: upload two real
      images via the actual frontend, confirm the overlay image renders
      and the description/VQA answer text shows up correctly
- [ ] Confirm `execution_trace.models_used` shows up in Urvashi's collapsible
      trace panel — this is an explicitly graded deliverable, don't skip
      checking it renders
