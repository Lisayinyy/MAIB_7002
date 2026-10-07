#!/usr/bin/env python3
"""Download and verify the public, revision-pinned FreshRetailNet-50K files.

Pins match outputs/final_protocol/locked_protocol.json. This script intentionally
has no third-party dependencies and requires no credentials. Existing files are
replaced only after a complete temporary download passes SHA-256 validation.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
BASE_URL = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve"
SOURCE_HASHES = {
    "train": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    "eval": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
}
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_split(split: str, data_dir: Path, *, check_only: bool, timeout: float) -> Path:
    """Return a hash-verified file, retaining any original on download failure."""
    destination = data_dir / f"{split}.parquet"
    expected = SOURCE_HASHES[split]
    if destination.is_file():
        if sha256_file(destination) == expected:
            print(f"{split}: existing file verified; skipped download.", flush=True)
            return destination
        if check_only:
            raise ValueError(f"{split}: existing file failed SHA-256 verification.")
        print(f"{split}: existing file has a different hash; downloading a verified replacement.", flush=True)
    elif check_only:
        raise FileNotFoundError(f"{split}: data file is missing.")

    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{REVISION}/data/{split}.parquet"
    request = urllib.request.Request(url, headers={"User-Agent": "FreshRetailNet-course-project/1.0"})
    fd, temporary_name = tempfile.mkstemp(prefix=f".{split}.", suffix=".part", dir=data_dir)
    temporary_path = Path(temporary_name)
    written = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected_length = response.headers.get("Content-Length")
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if expected_length is not None and written != int(expected_length):
            raise ValueError(f"{split}: incomplete download; existing data retained.")
        if digest.hexdigest() != expected:
            raise ValueError(f"{split}: downloaded file failed SHA-256 verification; existing data retained.")
        os.replace(temporary_path, destination)
        print(f"{split}: downloaded {written:,} bytes and verified SHA-256.", flush=True)
        return destination
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--splits", nargs="+", choices=tuple(SOURCE_HASHES), default=list(SOURCE_HASHES))
    parser.add_argument("--check-only", action="store_true", help="Verify existing files without network access.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network operation timeout in seconds (default: 60).")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    failed = False
    for split in dict.fromkeys(args.splits):
        try:
            ensure_split(split, args.data_dir, check_only=args.check_only, timeout=args.timeout)
        except (OSError, ValueError, urllib.error.URLError) as error:
            # Do not print exception URLs, redirect locations, response bodies, or credentials.
            if isinstance(error, urllib.error.HTTPError):
                detail = f"HTTP {error.code}"
            elif isinstance(error, ValueError):
                detail = str(error)
            elif isinstance(error, FileNotFoundError) and args.check_only:
                detail = "data file is missing"
            else:
                detail = type(error).__name__
            print(f"ERROR {split}: {detail}. No unverified download was installed.", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
