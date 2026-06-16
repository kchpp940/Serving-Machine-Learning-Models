from __future__ import annotations

import hashlib
import os
from typing import Optional


def compute_file_hash(file_path: str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def compute_data_version(csv_path: str) -> str:
    return compute_file_hash(csv_path)


def compute_content_hash(content: str, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()[:16]
