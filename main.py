"""
main.py
M1 — Agentic Controller + FastAPI Backend.
Single entrypoint: POST /analyze. Everything else (M2-M5) plugs in via
tool_registry.py. Frontend (M6) should point at this from minute one.
"""

import os
import shutil
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from intent_classifier import classify_intent, validate_input
from tool_registry import run_task
from audit_trail import ExecutionTrace
from schema import make_error_output

OUTPUTS_DIR = "outputs"
UPLOADS_DIR = "uploads"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="RS Multi-Modal VQA Controller (M1)")

# Hackathon CORS: wide open, tighten later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve overlay images etc. at /outputs/<filename>
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    task_type: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
):
    trace = ExecutionTrace()

    filenames = [img.filename for img in images]
    trace.log("input_received", f"{len(images)} image(s): {filenames}, query='{query}'")

    # ---- 1. Validate input ----
    is_valid, error_msg = validate_input(len(images), filenames)
    if not is_valid:
        trace.log("input_validation", error_msg, status="error")
        result = make_error_output(task_type or "unknown", error_msg)
        result["execution_trace"] = trace.as_list()
        return result

    trace.log("input_validation", "passed")

    # ---- 2. Save uploads to disk, get local paths ----
    saved_paths = []
    for img in images:
        ext = img.filename.rsplit(".", 1)[-1].lower()
        local_name = f"{uuid.uuid4().hex}.{ext}"
        local_path = os.path.join(UPLOADS_DIR, local_name)
        with open(local_path, "wb") as f:
            shutil.copyfileobj(img.file, f)
        saved_paths.append(local_path)

    trace.log("file_save", f"saved {len(saved_paths)} file(s) to {UPLOADS_DIR}/")

    # ---- 3. Classify intent ----
    intent = classify_intent(query, len(images), task_type)
    trace.log("intent_classification", f"routed to '{intent}'")

    # ---- 4. Run the matching tool ----
    result = run_task(intent, saved_paths, query)
    status = "error" if result.get("metadata", {}).get("error") else "ok"
    trace.log("tool_execution", f"'{intent}' returned confidence={result.get('confidence')}", status=status)

    # ---- 5. Attach trace + return ----
    result["execution_trace"] = trace.as_list()
    result["total_time_ms"] = trace.total_time_ms()
    return result
