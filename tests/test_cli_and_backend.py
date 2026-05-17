import importlib
import unittest

from main import build_parser


class CliAndBackendTests(unittest.TestCase):
    def test_parser_supports_enroll_command(self):
        parser = build_parser()

        args = parser.parse_args(
            ["enroll", "--known-dir", "known_faces", "--database", "faces.npz"]
        )

        self.assertEqual(args.command, "enroll")
        self.assertEqual(args.known_dir, "known_faces")
        self.assertEqual(args.database, "faces.npz")

    def test_parser_supports_recognize_command_with_threshold(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "recognize",
                "--test-dir",
                "test_images",
                "--database",
                "faces.npz",
                "--output-dir",
                "outputs",
                "--threshold",
                "0.42",
            ]
        )

        self.assertEqual(args.command, "recognize")
        self.assertAlmostEqual(args.threshold, 0.42)

    def test_parser_supports_serve_command(self):
        parser = build_parser()

        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])

        self.assertEqual(args.command, "serve")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)

    def test_insightface_backend_module_imports_without_loading_model(self):
        module = importlib.import_module("face_recognition_app.insightface_backend")

        self.assertTrue(hasattr(module, "InsightFaceBackend"))


if __name__ == "__main__":
    unittest.main()
