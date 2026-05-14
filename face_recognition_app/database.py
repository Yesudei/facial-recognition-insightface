from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from face_recognition_app.embedding import normalize_embedding


FaceDatabase = dict[str, list[np.ndarray]]


def save_database(database: Mapping[str, Sequence[np.ndarray]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = sorted(database)
    arrays: dict[str, np.ndarray] = {"names": np.array(names, dtype=str)}
    for index, name in enumerate(names):
        embeddings = [normalize_embedding(embedding) for embedding in database[name]]
        if not embeddings:
            continue
        arrays[f"embeddings_{index}"] = np.vstack(embeddings).astype(np.float32)

    np.savez_compressed(output_path, **arrays)


def load_database(path: str | Path) -> FaceDatabase:
    database_path = Path(path)
    if not database_path.exists():
        raise FileNotFoundError(f"Face database not found: {database_path}")

    loaded = np.load(database_path, allow_pickle=False)
    names = [str(name) for name in loaded["names"]]

    database: FaceDatabase = {}
    for index, name in enumerate(names):
        key = f"embeddings_{index}"
        if key not in loaded:
            database[name] = []
            continue
        database[name] = [row.astype(np.float32) for row in loaded[key]]

    return database
