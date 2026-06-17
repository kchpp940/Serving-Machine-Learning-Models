from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: str
    metadata_path: str
    status_path: str

    def model_exists(self) -> bool:
        return Path(self.model_path).exists()

    def metadata_exists(self) -> bool:
        return Path(self.metadata_path).exists()

    def status_exists(self) -> bool:
        return Path(self.status_path).exists()


@dataclass(frozen=True)
class RuntimeConfig:
    repo_root: str
    model_dir: str
    data_csv_path: str
    api_base_url: str
    streamlit_dir: str
    legacy_fastapi_dir: str
    request_timeout: int = 10

    @classmethod
    def from_env(cls, repo_root: Optional[str] = None) -> "RuntimeConfig":
        root = Path(repo_root or os.environ.get("REPO_ROOT", str(_REPO_ROOT))).resolve()
        return cls(
            repo_root=str(root),
            model_dir=str(root / "car_pricing_api" / "models"),
            data_csv_path=str(root / "Data" / "cars.csv"),
            api_base_url=os.environ.get("API_BASE_URL", "http://localhost:8000"),
            streamlit_dir=str(root / "streamlitapp"),
            legacy_fastapi_dir=str(root / "fastapi"),
            request_timeout=int(os.environ.get("API_REQUEST_TIMEOUT", "10")),
        )

    def artifact_paths(self, model_filename: str = "sklearn_gbr.pkl") -> ArtifactPaths:
        model_dir = Path(self.model_dir)
        return ArtifactPaths(
            model_path=str(model_dir / model_filename),
            metadata_path=str(model_dir / "model_metadata.json"),
            status_path=str(model_dir / "model_status.json"),
        )
