from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from single_music import generate_embedding
import shutil
import os
import numpy as np
from recommend import recommend_song

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".mp3"):
        return JSONResponse(status_code=400, content={"error": "O arquivo precisa ser .mp3"})

    input_path = os.path.join(INPUT_DIR, "music.mp3")
    output_path = os.path.join(OUTPUT_DIR, "embedding.npy")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        generate_embedding(input_path, output_path)
        new_embedding = np.load(output_path)

        recommendations = recommend_song(new_embedding, n=3)
        recommended_songs = recommendations.to_dict(orient="records")

        return {
            "message": "Embedding gerado com sucesso",
            "output": output_path,
            "recommendations": recommended_songs
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
