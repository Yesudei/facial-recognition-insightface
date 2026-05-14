# Facial Recognition with InsightFace

Computer vision assignment project using InsightFace and ArcFace-style face embeddings.

## References

- Paper: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- GitHub: [deepinsight/insightface](https://github.com/deepinsight/insightface)
- PyPI package: [insightface 0.7.3](https://pypi.org/project/insightface/)

## What This Project Does

This project recognizes faces from images using a pretrained InsightFace model. It does not train a neural network. Instead, it:

1. Detects faces in known images.
2. Extracts an embedding vector for each face.
3. Saves those vectors in `face_database.npz`.
4. Detects faces in test images.
5. Compares test embeddings with known embeddings using cosine similarity.
6. Draws names, similarity scores, and bounding boxes on output images.

There is also an assignment report draft in `ASSIGNMENT_REPORT.md`.

## Folder Structure

```text
Vision/
  main.py
  requirements.txt
  face_recognition_app/
  known_faces/
    PersonName/
      photo1.jpg
      photo2.jpg
  test_images/
    group_photo.jpg
  outputs/
```

Example:

```text
known_faces/
  Alice/
    alice1.jpg
    alice2.jpg
  Bob/
    bob1.jpg

test_images/
  class_photo.jpg
```

## Important Python Note

This computer currently has Python 3.14 installed as the default. The full InsightFace dependency stack may not install cleanly on Python 3.14.

Recommended: install **Python 3.10 or Python 3.11** from https://www.python.org/downloads/ and check **Add python.exe to PATH** during installation.

After installing Python 3.11, open PowerShell in this folder:

```powershell
cd "C:\Users\esude\OneDrive\Desktop\Vision"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## How To Run

### 1. Add Known People

Create one folder per person:

```text
known_faces/
  YourName/
    image1.jpg
    image2.jpg
```

Use clear front-facing photos. One face per known image is best.

### 2. Add Test Images

Put images to recognize in:

```text
test_images/
```

### 3. Enroll Known Faces

```powershell
python main.py enroll --known-dir known_faces --database face_database.npz
```

The first InsightFace run may download the `buffalo_l` model package, so internet access may be needed.

### 4. Recognize Faces

```powershell
python main.py recognize --test-dir test_images --database face_database.npz --output-dir outputs --threshold 0.35
```

Annotated result images will appear in:

```text
outputs/
```

### 5. One-Command Demo

After adding images to `known_faces/` and `test_images/`, you can also run:

```powershell
python main.py run-demo
```

## Threshold Tuning

Default threshold: `0.35`

- If known people are labeled `Unknown`, try `--threshold 0.30`.
- If the program gives wrong names too easily, try `--threshold 0.40` or `0.45`.

## Run Tests

These tests check the project logic without requiring InsightFace to be installed:

```powershell
python -m unittest discover -s tests -v
```

## How ArcFace Fits This Assignment

ArcFace is a face recognition method that trains a deep network to produce highly discriminative face embeddings. The main idea is the **additive angular margin loss**, which encourages embeddings from the same identity to be close together and embeddings from different identities to be far apart on a normalized hypersphere.

In this project, InsightFace provides a pretrained model based on this idea. For every detected face, the model returns an embedding vector. The project compares embeddings using cosine similarity:

- High similarity means the faces are likely the same identity.
- Low similarity means the face should be labeled `Unknown`.

## Troubleshooting

### `No module named insightface`

Activate your virtual environment and run:

```powershell
pip install -r requirements.txt
```

### `pip install insightface` fails on Windows

Use Python 3.10 or 3.11. If it still fails, install Microsoft C++ Build Tools and CMake, then retry:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### No face detected

Use clearer images with visible faces. For `known_faces`, use one person per photo.

### Wrong person recognized

Increase the threshold:

```powershell
python main.py recognize --threshold 0.45
```

## Ethics Note

Face recognition should be used carefully. For an assignment demo, use photos you have permission to use and avoid deploying the system for surveillance or real identity decisions.
