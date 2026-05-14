from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


UNKNOWN_NAME = "Unknown"


@dataclass(frozen=True)
class FaceMatch:
    name: str
    score: float

    @property
    def is_unknown(self) -> bool:
        return self.name == UNKNOWN_NAME


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Face embedding cannot be a zero vector.")
    return vector / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_normalized = normalize_embedding(first)
    second_normalized = normalize_embedding(second)
    score = float(np.dot(first_normalized, second_normalized))
    return max(-1.0, min(1.0, score))


def find_best_match(
    query_embedding: np.ndarray,
    database: Mapping[str, Sequence[np.ndarray]],
    threshold: float,
) -> FaceMatch:
    best_name = UNKNOWN_NAME
    best_score = -1.0

    for name, known_embeddings in database.items():
        for known_embedding in known_embeddings:
            score = cosine_similarity(query_embedding, known_embedding)
            if score > best_score:
                best_name = name
                best_score = score

    if best_score < threshold:
        return FaceMatch(name=UNKNOWN_NAME, score=max(0.0, best_score))

    return FaceMatch(name=best_name, score=best_score)
