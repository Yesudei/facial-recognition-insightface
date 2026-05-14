import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_recognition_app.pipeline import enroll_known_faces, recognize_images


class FakeBackend:
    def __init__(self):
        self.embeddings = {
            "alice.jpg": np.array([1.0, 0.0]),
            "scene.jpg": np.array([0.99, 0.01]),
            "stranger.jpg": np.array([0.0, 1.0]),
        }

    def get_faces(self, image_path):
        embedding = self.embeddings[Path(image_path).name]
        return [{"bbox": [1, 1, 8, 8], "embedding": embedding}]


class FakeImageIO:
    def read(self, image_path):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def draw_face(self, image, bbox, label, score, is_unknown):
        image[0, 0, 0] = 255
        return image

    def write(self, image_path, image):
        Path(image_path).write_bytes(b"fake image")


class PipelineTests(unittest.TestCase):
    def test_enroll_known_faces_uses_folder_names_as_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            known = Path(directory) / "known_faces" / "Alice"
            known.mkdir(parents=True)
            (known / "alice.jpg").write_bytes(b"fake image")

            database = enroll_known_faces(known.parent, FakeBackend())

            self.assertEqual(list(database), ["Alice"])
            self.assertEqual(len(database["Alice"]), 1)
            np.testing.assert_allclose(database["Alice"][0], np.array([1.0, 0.0]))

    def test_recognize_images_writes_annotated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_images = root / "test_images"
            outputs = root / "outputs"
            test_images.mkdir()
            (test_images / "scene.jpg").write_bytes(b"fake image")

            results = recognize_images(
                test_images,
                {"Alice": [np.array([1.0, 0.0])]},
                FakeBackend(),
                outputs,
                threshold=0.5,
                image_io=FakeImageIO(),
            )

            self.assertEqual(results[0].matches[0].name, "Alice")
            self.assertTrue((outputs / "scene_recognized.jpg").exists())

    def test_recognize_images_labels_faces_unknown_below_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_images = root / "test_images"
            outputs = root / "outputs"
            test_images.mkdir()
            (test_images / "stranger.jpg").write_bytes(b"fake image")

            results = recognize_images(
                test_images,
                {"Alice": [np.array([1.0, 0.0])]},
                FakeBackend(),
                outputs,
                threshold=0.5,
                image_io=FakeImageIO(),
            )

            self.assertEqual(results[0].matches[0].name, "Unknown")
            self.assertTrue(results[0].matches[0].is_unknown)


if __name__ == "__main__":
    unittest.main()
