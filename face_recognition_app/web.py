from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from face_recognition_app.database import FaceDatabase, load_database
from face_recognition_app.embedding import UNKNOWN_NAME, find_best_match
from face_recognition_app.insightface_backend import InsightFaceBackend


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_DATABASE_PATH = Path("face_database.npz")
DEFAULT_THRESHOLD = 0.35
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(
    title="FaceID InsightFace API",
    description="Upload an image and receive InsightFace detection and recognition data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0),
    database: str = Query(str(DEFAULT_DATABASE_PATH)),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)

        try:
            return _analyze_path(
                image_path=tmp_path,
                original_filename=file.filename or "upload",
                database_path=Path(database),
                threshold=threshold,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _analyze_path(
    image_path: Path,
    original_filename: str,
    database_path: Path,
    threshold: float,
) -> dict[str, Any]:
    database = _load_database_if_available(database_path)
    faces = _backend().get_faces(image_path)

    return {
        "filename": original_filename,
        "faces_detected": len(faces),
        "database_loaded": database is not None,
        "database_path": str(database_path),
        "threshold": threshold,
        "faces": [
            _serialize_face(face, database=database, threshold=threshold)
            for face in faces
        ],
    }


def _load_database_if_available(database_path: Path) -> FaceDatabase | None:
    if not database_path.exists():
        return None
    return load_database(database_path)


def _serialize_face(
    face: dict[str, Any],
    database: FaceDatabase | None,
    threshold: float,
) -> dict[str, Any]:
    embedding = face["embedding"]
    match = (
        find_best_match(embedding, database, threshold)
        if database
        else None
    )
    bbox = [round(float(value), 2) for value in face["bbox"]]
    x1, y1, x2, y2 = bbox

    return {
        "bbox": bbox,
        "width": round(max(0.0, x2 - x1), 2),
        "height": round(max(0.0, y2 - y1), 2),
        "name": match.name if match else UNKNOWN_NAME,
        "score": round(match.score, 4) if match else None,
        "is_unknown": match.is_unknown if match else True,
        "embedding_size": int(embedding.shape[0]),
    }


@lru_cache(maxsize=1)
def _backend() -> InsightFaceBackend:
    return InsightFaceBackend()


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str) -> FileResponse:
    index_path = FRONTEND_DIST / "index.html"
    requested_path = (FRONTEND_DIST / full_path).resolve()

    if requested_path.is_file() and FRONTEND_DIST in requested_path.parents:
        return FileResponse(requested_path)
    if index_path.exists():
        return FileResponse(index_path)

    raise HTTPException(
        status_code=404,
        detail="React build not found. Run `cd frontend && npm install && npm run build`.",
    )
