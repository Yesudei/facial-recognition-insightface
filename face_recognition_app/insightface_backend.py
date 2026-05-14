from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class InsightFaceBackend:
    """Adapter around InsightFace's FaceAnalysis API."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed. Install the project dependencies with "
                "`pip install -r requirements.txt` inside a Python 3.10 or 3.11 "
                "virtual environment."
            ) from exc

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers or ["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_faces(self, image_path: str | Path) -> list[dict[str, Any]]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self.app.get(image)
        return [
            {
                "bbox": face.bbox.astype(float).tolist(),
                "embedding": np.asarray(face.embedding, dtype=np.float32),
            }
            for face in faces
        ]
