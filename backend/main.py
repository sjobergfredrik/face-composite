import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from swapper import FaceSwapper

app = FastAPI(title="Face Composite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

swapper: FaceSwapper | None = None


@app.on_event("startup")
async def startup():
    global swapper
    print("Loading models...")
    swapper = FaceSwapper()
    print("Models loaded.")


def read_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def encode_png(img: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode result.")
    return buf.tobytes()


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": swapper is not None}


@app.post("/detect-faces")
async def detect_faces(image: UploadFile = File(...)):
    img = read_image(await image.read())
    faces = swapper.detect_faces(img)
    return {"faces": faces, "image_width": img.shape[1], "image_height": img.shape[0]}


@app.post("/swap")
async def swap_face(
    scene: UploadFile = File(...),
    actor: UploadFile = File(...),
    target_face_index: int = Form(0),
    model: str = Form("hyperswap_1c_256"),
):
    scene_img = read_image(await scene.read())
    actor_img = read_image(await actor.read())

    try:
        result = swapper.swap(scene_img, actor_img, target_face_index, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(content=encode_png(result), media_type="image/png")
