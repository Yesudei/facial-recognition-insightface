from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from warnings import warn

import numpy as np

from face_recognition_app.database import FaceDatabase
from face_recognition_app.embedding import UNKNOWN_NAME, find_best_match, normalize_embedding


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceBackend(Protocol):
    def get_faces(self, image_path: str | Path) -> list[dict[str, Any]]:
        ...


class ImageIO(Protocol):
    def read(self, image_path: str | Path) -> np.ndarray:
        ...

    def draw_face(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        label: str,
        score: float,
        is_unknown: bool,
    ) -> np.ndarray:
        ...

    def write(self, image_path: str | Path, image: np.ndarray) -> None:
        ...


@dataclass(frozen=True)
class RecognizedFace:
    bbox: list[float]
    name: str
    score: float

    @property
    def is_unknown(self) -> bool:
        return self.name == UNKNOWN_NAME


@dataclass(frozen=True)
class ImageRecognitionResult:
    image_path: Path
    output_path: Path
    matches: list[RecognizedFace]


class OpenCVImageIO:
    def read(self, image_path: str | Path) -> np.ndarray:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image

    def draw_face(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        label: str,
        score: float,
        is_unknown: bool,
    ) -> np.ndarray:
        import cv2

        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        color = (0, 0, 255) if is_unknown else (0, 180, 0)
        text = f"{label} ({score:.2f})" if not is_unknown else label

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_y = max(20, y1 - 10)
        cv2.putText(
            image,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        return image

    def write(self, image_path: str | Path, image: np.ndarray) -> None:
        import cv2

        output_path = Path(image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise ValueError(f"Could not write output image: {output_path}")


def enroll_known_faces(known_dir: str | Path, backend: FaceBackend) -> FaceDatabase:
    root = Path(known_dir)
    if not root.exists():
        raise FileNotFoundError(f"Known faces folder not found: {root}")

    database: FaceDatabase = {}
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        identity_embeddings: list[np.ndarray] = []
        for image_path in _iter_image_files(identity_dir):
            faces = backend.get_faces(image_path)
            if not faces:
                warn(f"No face found in known image: {image_path}", stacklevel=2)
                continue
            face = _largest_face(faces)
            identity_embeddings.append(normalize_embedding(face["embedding"]))

        if identity_embeddings:
            database[identity_dir.name] = identity_embeddings

    if not database:
        raise ValueError(
            f"No known faces were enrolled from {root}. Add images under "
            "known_faces/PersonName/image.jpg."
        )

    return database


def recognize_images(
    test_dir: str | Path,
    database: Mapping[str, Sequence[np.ndarray]],
    backend: FaceBackend,
    output_dir: str | Path,
    threshold: float,
    image_io: ImageIO | None = None,
) -> list[ImageRecognitionResult]:
    root = Path(test_dir)
    if not root.exists():
        raise FileNotFoundError(f"Test images folder not found: {root}")

    io = image_io or OpenCVImageIO()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[ImageRecognitionResult] = []

    for image_path in _iter_image_files(root):
        image = io.read(image_path)
        recognized_faces: list[RecognizedFace] = []

        for face in backend.get_faces(image_path):
            match = find_best_match(face["embedding"], database, threshold)
            bbox = [float(value) for value in face["bbox"]]
            recognized_face = RecognizedFace(
                bbox=bbox,
                name=match.name,
                score=match.score,
            )
            recognized_faces.append(recognized_face)
            image = io.draw_face(
                image,
                bbox,
                match.name,
                match.score,
                match.is_unknown,
            )

        output_path = output_root / f"{image_path.stem}_recognized{image_path.suffix}"
        io.write(output_path, image)
        results.append(
            ImageRecognitionResult(
                image_path=image_path,
                output_path=output_path,
                matches=recognized_faces,
            )
        )

    return results


def _iter_image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _largest_face(faces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(faces, key=lambda face: _bbox_area(face["bbox"]))


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
