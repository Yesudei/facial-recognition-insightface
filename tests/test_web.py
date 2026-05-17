import asyncio
import io
import unittest
from unittest.mock import patch

from starlette.datastructures import Headers

from face_recognition_app import web


class WebTests(unittest.TestCase):
    def test_analyze_upload_closes_temp_file_before_analysis(self):
        def read_uploaded_file(image_path, original_filename, database_path, threshold):
            self.assertEqual(original_filename, "blank.jpg")
            with open(image_path, "rb") as image_file:
                payload = image_file.read()
            self.assertEqual(payload, b"fake image")
            return {
                "filename": original_filename,
                "faces_detected": 0,
                "database_loaded": False,
                "database_path": str(database_path),
                "threshold": threshold,
                "faces": [],
            }

        upload = web.UploadFile(
            file=io.BytesIO(b"fake image"),
            filename="blank.jpg",
            headers=Headers({"content-type": "image/jpeg"}),
        )

        with patch.object(web, "_analyze_path", side_effect=read_uploaded_file):
            response = asyncio.run(
                web.analyze_image(
                    file=upload,
                    threshold=web.DEFAULT_THRESHOLD,
                    database=str(web.DEFAULT_DATABASE_PATH),
                )
            )

        self.assertEqual(response["faces_detected"], 0)


if __name__ == "__main__":
    unittest.main()
