from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from m4_pipeline import caption_and_ground
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("sample_images", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="sample_images"), name="outputs")

@app.post("/analyze")
async def analyze(image: UploadFile, query: str = Form(...)):
    temp_path = f"sample_images/uploaded_{image.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    result = caption_and_ground(temp_path, query)
    return result