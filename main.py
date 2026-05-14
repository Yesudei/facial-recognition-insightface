from __future__ import annotations

import argparse
import sys
from pathlib import Path

from face_recognition_app.database import load_database, save_database
from face_recognition_app.insightface_backend import InsightFaceBackend
from face_recognition_app.pipeline import enroll_known_faces, recognize_images


DEFAULT_THRESHOLD = 0.35


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Facial recognition assignment demo using InsightFace and ArcFace."
    )
    subparsers = parser.add_subparsers(dest="command")

    enroll = subparsers.add_parser(
        "enroll",
        help="Create a face database from known_faces/<person_name>/ images.",
    )
    enroll.add_argument("--known-dir", default="known_faces")
    enroll.add_argument("--database", default="face_database.npz")

    recognize = subparsers.add_parser(
        "recognize",
        help="Recognize faces in test images and write annotated output images.",
    )
    recognize.add_argument("--test-dir", default="test_images")
    recognize.add_argument("--database", default="face_database.npz")
    recognize.add_argument("--output-dir", default="outputs")
    recognize.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    demo = subparsers.add_parser(
        "run-demo",
        help="Run enrollment and recognition with the default assignment folders.",
    )
    demo.add_argument("--known-dir", default="known_faces")
    demo.add_argument("--test-dir", default="test_images")
    demo.add_argument("--database", default="face_database.npz")
    demo.add_argument("--output-dir", default="outputs")
    demo.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "enroll":
            _run_enroll(args)
        elif args.command == "recognize":
            _run_recognize(args)
        elif args.command == "run-demo":
            _run_enroll(args)
            _run_recognize(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def _run_enroll(args: argparse.Namespace) -> None:
    backend = InsightFaceBackend()
    database = enroll_known_faces(Path(args.known_dir), backend)
    save_database(database, Path(args.database))
    total_embeddings = sum(len(embeddings) for embeddings in database.values())
    print(
        f"Enrolled {len(database)} people with {total_embeddings} face embeddings. "
        f"Saved database to {args.database}."
    )


def _run_recognize(args: argparse.Namespace) -> None:
    backend = InsightFaceBackend()
    database = load_database(Path(args.database))
    results = recognize_images(
        Path(args.test_dir),
        database,
        backend,
        Path(args.output_dir),
        threshold=args.threshold,
    )

    print(f"Processed {len(results)} images. Annotated outputs saved to {args.output_dir}.")
    for result in results:
        if not result.matches:
            print(f"- {result.image_path.name}: no faces detected")
            continue
        labels = ", ".join(
            f"{match.name} ({match.score:.2f})" if not match.is_unknown else "Unknown"
            for match in result.matches
        )
        print(f"- {result.image_path.name}: {labels}")


if __name__ == "__main__":
    raise SystemExit(main())
