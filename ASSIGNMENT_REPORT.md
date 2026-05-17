# Facial Recognition with InsightFace

## 1. Introduction

This assignment demonstrates face recognition using InsightFace, a deep learning toolkit for face analysis. The project uses a pretrained ArcFace-style model to detect faces, extract face embeddings, and compare those embeddings to recognize known people from images.

The goal is not to train a new model. Instead, the system uses transfer learning through a pretrained model and focuses on the practical recognition pipeline: enrollment, embedding extraction, similarity comparison, and result visualization.

## 2. Background

ArcFace, introduced in the paper **"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"**, improves face recognition by learning discriminative face embeddings. It uses an additive angular margin loss so that embeddings of the same person become close together, while embeddings of different people become farther apart.

InsightFace provides pretrained models based on this research. In this project, the `buffalo_l` model package is used through the `FaceAnalysis` API. The model detects faces and returns embedding vectors that can be compared using cosine similarity.

## 3. Methodology

The project follows these steps:

1. Images of known people are placed in folders under `known_faces/`.
2. InsightFace detects faces in those known images.
3. The program extracts one embedding vector for each known face.
4. The embeddings are saved in a local database file named `face_database.npz`.
5. Test images are placed in `test_images/`.
6. The program detects faces in each test image.
7. Each detected face embedding is compared with the saved known embeddings.
8. If the highest cosine similarity is above the selected threshold, the face is labeled with the matched person's name.
9. If the score is below the threshold, the face is labeled `Unknown`.
10. Annotated output images are saved in `outputs/`.

## 4. Implementation

The project is implemented in Python. The main libraries are:

- `insightface` for face detection and embedding extraction.
- `onnxruntime` for running the pretrained model.
- `opencv-python` for reading, drawing, and saving images.
- `numpy` for vector operations and cosine similarity.

The code is split into small modules:

- `embedding.py` handles normalization and cosine similarity.
- `database.py` saves and loads known face embeddings.
- `pipeline.py` handles enrollment and recognition workflow.
- `insightface_backend.py` connects the project to InsightFace.
- `web.py` exposes a FastAPI endpoint for image uploads from the web interface.
- `main.py` provides the command-line interface.

The project also includes a React + TypeScript web interface in `frontend/`.
It lets a user upload a photo, preview it, call the local `/api/analyze`
endpoint, inspect face boxes and match confidence, and view the raw JSON
response.

## 5. Commands Used

Install dependencies:

```powershell
pip install -r requirements.txt
```

Enroll known faces:

```powershell
python main.py enroll --known-dir known_faces --database face_database.npz
```

Recognize test images:

```powershell
python main.py recognize --test-dir test_images --database face_database.npz --output-dir outputs --threshold 0.35
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Run the web interface:

```powershell
python main.py serve
cd frontend
npm install
npm run dev
```

For a single server, run `npm run build` in `frontend/` and then open the
FastAPI server URL.

The repository also includes a `Dockerfile` that builds the React frontend and
runs the Python backend on Python 3.11, which is the recommended runtime for
the InsightFace dependency stack.

## 6. Results

After running recognition, annotated images are generated in the `outputs/` folder. Each detected face is marked with a bounding box and either the recognized name with similarity score or the label `Unknown`.

In the web interface, the uploaded image is displayed with face bounding boxes
overlaid in the browser. When a face database exists, the interface also shows
the best identity match, confidence score, face size, and embedding dimension.

The threshold controls how strict the recognition is. A lower threshold recognizes more faces but can increase false matches. A higher threshold is stricter but can label real known people as unknown.

## 7. Conclusion

This project shows how modern face recognition systems use pretrained deep learning models to extract embeddings instead of comparing raw images directly. ArcFace improves recognition by making face embeddings more separable in angular space. InsightFace makes this practical by providing pretrained models and a simple API for detection and recognition.

## 8. References

1. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**. arXiv:1801.07698. https://arxiv.org/abs/1801.07698
2. InsightFace GitHub repository. https://github.com/deepinsight/insightface
3. InsightFace PyPI package. https://pypi.org/project/insightface/
