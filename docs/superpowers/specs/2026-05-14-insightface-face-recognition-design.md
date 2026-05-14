# InsightFace Face Recognition Design

## Goal

Build a small computer vision assignment project that demonstrates face recognition with InsightFace and ArcFace embeddings. The project should be easy to run from the command line, easy to explain in a report or presentation, and practical for a short deadline.

## Scope

The project recognizes people from saved images. A user places one or more reference photos for each person under `known_faces/<person_name>/`, places unknown images under `test_images/`, enrolls known people into a local embedding database, and runs recognition to produce annotated output images under `outputs/`.

The project does not train a neural network. It uses InsightFace's pretrained `buffalo_l` model package to detect faces and extract ArcFace-style face embeddings.

## Architecture

The code is split into small modules:

- `face_recognition_app/embedding.py` handles vector normalization, cosine similarity, and best-match decisions.
- `face_recognition_app/database.py` stores and loads known face embeddings in a compressed NumPy file.
- `face_recognition_app/insightface_backend.py` wraps the actual InsightFace `FaceAnalysis` model.
- `face_recognition_app/pipeline.py` handles enrolling known faces and recognizing faces in test images.
- `main.py` exposes command-line commands.

This keeps the pure matching logic testable without requiring InsightFace to be installed during unit tests.

## Data Flow

1. Enrollment scans `known_faces/<person_name>/*`.
2. InsightFace detects faces and extracts one embedding per usable face.
3. Embeddings are normalized and saved to `face_database.npz`.
4. Recognition loads the database and scans `test_images/*`.
5. For each detected test face, the project compares the embedding against all known embeddings with cosine similarity.
6. If the best score is at least the threshold, the result uses the matching name; otherwise it labels the face `Unknown`.
7. The result image is written to `outputs/` with bounding boxes, labels, and similarity scores.

## Error Handling

The command line should give clear messages for common student-demo problems:

- Missing `known_faces` or `test_images` folders.
- No usable face found in a known-face image.
- Multiple faces found in a known-face image, where the largest face is used.
- Missing database file before recognition.
- Failed image reads.

## Testing

Automated tests cover the pure project logic:

- Embedding normalization.
- Cosine similarity.
- Threshold behavior for known versus unknown faces.
- Database save/load round trip.
- Recognition pipeline behavior with a fake backend.

The tests intentionally avoid importing InsightFace so they can run quickly on a normal Python installation.

## Running the Project

The README will include:

- Recommended Python version: 3.10 or 3.11.
- Virtual environment setup.
- `pip install -r requirements.txt`.
- Folder structure.
- Commands for enrollment and recognition.
- Explanation of ArcFace and InsightFace in assignment-friendly language.

## Self-Review

This spec has no placeholder sections. The scope is a single, focused command-line image recognition demo. The architecture matches the requested assignment topic and avoids training or unrelated features.
