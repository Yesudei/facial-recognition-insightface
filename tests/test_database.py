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

    def test_load_database_raises_for_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.npz"

            with self.assertRaises(FileNotFoundError):
                load_database(path)


if __name__ == "__main__":
    unittest.main()
