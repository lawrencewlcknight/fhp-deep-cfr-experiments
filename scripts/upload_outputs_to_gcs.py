"""Upload Batch experiment outputs to Cloud Storage.

This helper intentionally runs from the experiment virtual environment rather
than through ``gsutil``. On some Batch images the system Cloud SDK can be tied
to an older Python runtime, while the experiment venv is already known to be
healthy by the time outputs are ready to copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple
from urllib.parse import urlparse


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected a gs://bucket[/prefix] URI, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    return parsed.netloc, prefix.rstrip("/")


def iter_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            yield path


def destination_blob_name(source_dir: Path, file_path: Path, prefix: str) -> str:
    relative = file_path.relative_to(source_dir).as_posix()
    parts = [part for part in (prefix, source_dir.name, relative) if part]
    return "/".join(parts)


def upload_directory(source_dir: Path, destination: str) -> int:
    from google.cloud import storage

    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    bucket_name, prefix = parse_gcs_uri(destination)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    count = 0
    for file_path in iter_files(source_dir):
        blob_name = destination_blob_name(source_dir, file_path, prefix)
        print(f"Uploading {file_path} -> gs://{bucket_name}/{blob_name}", flush=True)
        bucket.blob(blob_name).upload_from_filename(str(file_path))
        count += 1

    print(
        f"Uploaded {count} file(s) from {source_dir} to gs://{bucket_name}/{prefix}/",
        flush=True,
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("destination", help="Destination gs://bucket[/prefix] URI")
    args = parser.parse_args()

    upload_directory(args.source_dir, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
