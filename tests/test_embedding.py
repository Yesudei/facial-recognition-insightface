import unittest

import numpy as np

from face_recognition_app.embedding import (
    cosine_similarity,
    find_best_match,
    normalize_embedding,
)


class EmbeddingTests(unittest.TestCase):
    def test_normalize_embedding_returns_unit_vector(self):
        vector = np.array([3.0, 4.0])

        normalized = normalize_embedding(vector)

        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0)
        np.testing.assert_allclose(normalized, np.array([0.6, 0.8]))

    def test_normalize_embedding_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            normalize_embedding(np.array([0.0, 0.0]))

    def test_cosine_similarity_is_high_for_similar_vectors(self):
        score = cosine_similarity(np.array([1.0, 0.0]), np.array([0.9, 0.1]))

        self.assertGreater(score, 0.99)

    def test_find_best_match_returns_identity_when_score_meets_threshold(self):
        query = np.array([1.0, 0.0])
        database = {
            "Alice": [np.array([0.95, 0.05])],
            "Bob": [np.array([0.0, 1.0])],
        }

        match = find_best_match(query, database, threshold=0.5)

        self.assertEqual(match.name, "Alice")
        self.assertGreater(match.score, 0.9)
        self.assertFalse(match.is_unknown)

    def test_find_best_match_returns_unknown_below_threshold(self):
        query = np.array([1.0, 0.0])
        database = {"Bob": [np.array([0.0, 1.0])]}

        match = find_best_match(query, database, threshold=0.5)

        self.assertEqual(match.name, "Unknown")
        self.assertTrue(match.is_unknown)


if __name__ == "__main__":
    unittest.main()
