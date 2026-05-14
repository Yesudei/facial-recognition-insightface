# InsightFace Face Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a command-line facial recognition assignment demo using InsightFace/ArcFace embeddings.

**Architecture:** Pure matching and database logic are separated from the heavy InsightFace runtime. Unit tests exercise embedding math, database persistence, and pipeline behavior with a fake backend, while the real backend is used only when the user runs the assignment.

**Tech Stack:** Python, NumPy, OpenCV, InsightFace, ONNX Runtime, standard-library `unittest`.

---

## File Structure

- Create `face_recognition_app/__init__.py`: package metadata.
- Create `face_recognition_app/embedding.py`: normalize vectors, compute cosine similarity, choose best identity match.
- Create `face_recognition_app/database.py`: save/load compressed embedding databases.
- Create `face_recognition_app/pipeline.py`: enroll known faces and annotate recognition results.
- Create `face_recognition_app/insightface_backend.py`: lazy import and configure InsightFace.
- Create `main.py`: command-line interface.
- Create `tests/test_embedding.py`: tests for matching behavior.
- Create `tests/test_database.py`: tests for database persistence.
- Create `tests/test_pipeline.py`: tests for enrollment/recognition flow with fake backend.
- Create `requirements.txt`: runtime dependencies.
- Create `README.md`: assignment explanation and commands.

### Task 1: Embedding Logic

**Files:**
- Test: `tests/test_embedding.py`
- Create: `face_recognition_app/embedding.py`
- Create: `face_recognition_app/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
import unittest
import numpy as np

from face_recognition_app.embedding import find_best_match, normalize_embedding


class EmbeddingTests(unittest.TestCase):
    def test_normalize_embedding_returns_unit_vector(self):
        vector = np.array([3.0, 4.0])
        normalized = normalize_embedding(vector)
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0)

    def test_find_best_match_returns_identity_when_score_meets_threshold(self):
        query = np.array([1.0, 0.0])
        database = {
            "Alice": [np.array([0.95, 0.05])],
            "Bob": [np.array([0.0, 1.0])],
        }
        match = find_best_match(query, database, threshold=0.5)
        self.assertEqual(match.name, "Alice")
        self.assertGreater(match.score, 0.9)

    def test_find_best_match_returns_unknown_below_threshold(self):
        query = np.array([1.0, 0.0])
        database = {"Bob": [np.array([0.0, 1.0])]}
        match = find_best_match(query, database, threshold=0.5)
        self.assertEqual(match.name, "Unknown")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_embedding -v`

Expected: failure because `face_recognition_app.embedding` does not exist yet.

- [ ] **Step 3: Implement minimal embedding code**

Create a dataclass `FaceMatch`, `normalize_embedding`, `cosine_similarity`, and `find_best_match`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m unittest tests.test_embedding -v`

Expected: 3 tests pass.

### Task 2: Database Persistence

**Files:**
- Test: `tests/test_database.py`
- Create: `face_recognition_app/database.py`

- [ ] **Step 1: Write failing round-trip test**

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_recognition_app.database import load_database, save_database


class DatabaseTests(unittest.TestCase):
    def test_database_round_trip_preserves_names_and_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faces.npz"
            database = {
                "Alice": [np.array([1.0, 0.0]), np.array([0.9, 0.1])],
                "Bob": [np.array([0.0, 1.0])],
            }
            save_database(database, path)
            loaded = load_database(path)
            self.assertEqual(set(loaded), {"Alice", "Bob"})
            self.assertEqual(len(loaded["Alice"]), 2)
            np.testing.assert_allclose(loaded["Bob"][0], np.array([0.0, 1.0]))
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m unittest tests.test_database -v`

Expected: failure because `face_recognition_app.database` does not exist yet.

- [ ] **Step 3: Implement save/load**

Store names and per-person embeddings in a compressed `.npz` file.

- [ ] **Step 4: Run test to verify pass**

Run: `python -m unittest tests.test_database -v`

Expected: database test passes.

### Task 3: Pipeline With Fake Backend

**Files:**
- Test: `tests/test_pipeline.py`
- Create: `face_recognition_app/pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

```python
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from face_recognition_app.pipeline import enroll_known_faces, recognize_images


class FakeBackend:
    def __init__(self):
        self.embeddings = {
            "alice.jpg": np.array([1.0, 0.0]),
            "scene.jpg": np.array([0.99, 0.01]),
        }

    def get_faces(self, image_path):
        embedding = self.embeddings[Path(image_path).name]
        return [{"bbox": [1, 1, 8, 8], "embedding": embedding}]


class PipelineTests(unittest.TestCase):
    def test_enroll_known_faces_uses_folder_names_as_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            known = root / "known_faces" / "Alice"
            known.mkdir(parents=True)
            cv2.imwrite(str(known / "alice.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

            database = enroll_known_faces(known.parent, FakeBackend())

            self.assertEqual(list(database), ["Alice"])
            self.assertEqual(len(database["Alice"]), 1)

    def test_recognize_images_writes_annotated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_images = root / "test_images"
            outputs = root / "outputs"
            test_images.mkdir()
            cv2.imwrite(str(test_images / "scene.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

            results = recognize_images(
                test_images,
                {"Alice": [np.array([1.0, 0.0])]},
                FakeBackend(),
                outputs,
                threshold=0.5,
            )

            self.assertEqual(results[0].matches[0].name, "Alice")
            self.assertTrue((outputs / "scene_recognized.jpg").exists())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_pipeline -v`

Expected: failure because `face_recognition_app.pipeline` does not exist yet.

- [ ] **Step 3: Implement pipeline**

Create enrollment and recognition functions using injected backend objects.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m unittest tests.test_pipeline -v`

Expected: pipeline tests pass.

### Task 4: Real InsightFace Backend, CLI, and Documentation

**Files:**
- Create: `face_recognition_app/insightface_backend.py`
- Create: `main.py`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Implement lazy InsightFace backend**

The backend imports `insightface` only when the assignment command is run, initializes `FaceAnalysis(name="buffalo_l")`, and returns dictionaries with `bbox` and `embedding`.

- [ ] **Step 2: Implement CLI**

Commands:

```powershell
python main.py enroll --known-dir known_faces --database face_database.npz
python main.py recognize --test-dir test_images --database face_database.npz --output-dir outputs --threshold 0.35
python main.py run-demo
```

- [ ] **Step 3: Write documentation**

README includes assignment explanation, setup commands, folder examples, run commands, troubleshooting for Python 3.14 dependency failures, and ArcFace/InsightFace citations.

- [ ] **Step 4: Verify full project**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.
