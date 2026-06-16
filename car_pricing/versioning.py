from __future__ import annotations

import hashlib
import os


def compute_file_hash(file_path: str, chunk_size: int = 65536) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_data_version(csv_path: str) -> str:
    return compute_file_hash(csv_path)[:12]
