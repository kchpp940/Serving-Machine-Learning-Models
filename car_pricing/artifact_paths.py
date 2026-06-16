from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List


MODEL_FILENAME = "sklearn_gbr.pkl"
METADATA_FILENAME = "model_metadata.json"
STATUS_FILENAME = "model_status.json"
SCHEMA_SNAPSHOT_FILENAME = "schema_snapshot.json"
LINEAGE_FILENAME = "model_lineage.json"
DEFAULT_MODEL_DIRNAME = "models"
DEFAULT_DATA_DIRNAME = "Data"
DEFAULT_DATA_FILENAME = "cars.csv"
BENTOML_MODEL_TAG = "gbr"


@dataclass
class ArtifactPaths:
    model_dir: str
    training_data_dir: str
    model_filename: str = MODEL_FILENAME
    metadata_filename: str = METADATA_FILENAME
    status_filename: str = STATUS_FILENAME
    schema_snapshot_filename: str = SCHEMA_SNAPSHOT_FILENAME
    lineage_filename: str = LINEAGE_FILENAME
    data_filename: str = DEFAULT_DATA_FILENAME

    @property
    def model_file(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.model_dir, self.metadata_filename)

    @property
    def status_file(self) -> str:
        return os.path.join(self.model_dir, self.status_filename)

    @property
    def schema_snapshot_file(self) -> str:
        return os.path.join(self.model_dir, self.schema_snapshot_filename)

    @property
    def lineage_file(self) -> str:
        return os.path.join(self.model_dir, self.lineage_filename)

    @property
    def training_data_file(self) -> str:
        return os.path.join(self.training_data_dir, self.data_filename)

    def ensure_model_dir(self, exist_ok: bool = True) -> str:
        os.makedirs(self.model_dir, exist_ok=exist_ok)
        return self.model_dir

    def ensure_training_data_dir(self, exist_ok: bool = True) -> str:
        os.makedirs(self.training_data_dir, exist_ok=exist_ok)
        return self.training_data_dir

    def ensure_all_dirs(self, exist_ok: bool = True) -> None:
        self.ensure_model_dir(exist_ok=exist_ok)
        self.ensure_training_data_dir(exist_ok=exist_ok)

    def model_file_exists(self) -> bool:
        return os.path.isfile(self.model_file)

    def metadata_file_exists(self) -> bool:
        return os.path.isfile(self.metadata_file)

    def status_file_exists(self) -> bool:
        return os.path.isfile(self.status_file)

    def schema_snapshot_file_exists(self) -> bool:
        return os.path.isfile(self.schema_snapshot_file)

    def lineage_file_exists(self) -> bool:
        return os.path.isfile(self.lineage_file)

    def training_data_file_exists(self) -> bool:
        return os.path.isfile(self.training_data_file)

    def assert_model_file_exists(self) -> None:
        if not self.model_file_exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_file}")

    def assert_training_data_file_exists(self) -> None:
        if not self.training_data_file_exists():
            raise FileNotFoundError(f"训练数据文件不存在: {self.training_data_file}")

    @classmethod
    def from_script_dir(
        cls,
        script_file: str,
        model_dirname: str = DEFAULT_MODEL_DIRNAME,
        data_dirname: str = DEFAULT_DATA_DIRNAME,
    ) -> "ArtifactPaths":
        script_dir = os.path.dirname(os.path.abspath(script_file))
        return cls(
            model_dir=os.path.join(script_dir, model_dirname),
            training_data_dir=os.path.join(script_dir, data_dirname),
        )

    @classmethod
    def from_parent_dir(
        cls,
        script_file: str,
        model_dirname: str = DEFAULT_MODEL_DIRNAME,
        data_dirname: str = DEFAULT_DATA_DIRNAME,
        parent_levels: int = 1,
    ) -> "ArtifactPaths":
        script_dir = os.path.dirname(os.path.abspath(script_file))
        parent_dir = script_dir
        for _ in range(parent_levels):
            parent_dir = os.path.dirname(parent_dir)
        return cls(
            model_dir=os.path.join(script_dir, model_dirname),
            training_data_dir=os.path.join(parent_dir, data_dirname),
        )

    @classmethod
    def from_service_dir(
        cls,
        service_file: str,
        model_dirname: str = DEFAULT_MODEL_DIRNAME,
        data_dirname: str = DEFAULT_DATA_DIRNAME,
        parent_levels: int = 2,
    ) -> "ArtifactPaths":
        service_dir = os.path.dirname(os.path.abspath(service_file))
        api_dir = service_dir
        for _ in range(parent_levels):
            api_dir = os.path.dirname(api_dir)
        return cls(
            model_dir=os.path.join(os.path.dirname(service_dir), model_dirname),
            training_data_dir=os.path.join(api_dir, data_dirname),
        )

    @classmethod
    def from_dir(
        cls,
        model_dir: str,
        training_data_dir: Optional[str] = None,
    ) -> "ArtifactPaths":
        abs_model_dir = os.path.abspath(model_dir)
        if training_data_dir is None:
            training_data_dir = os.path.join(os.path.dirname(abs_model_dir), DEFAULT_DATA_DIRNAME)
        return cls(
            model_dir=abs_model_dir,
            training_data_dir=os.path.abspath(training_data_dir),
        )

    @classmethod
    def for_car_pricing_api_train(cls, script_file: str) -> "ArtifactPaths":
        return cls.from_parent_dir(script_file, parent_levels=1)

    @classmethod
    def for_car_pricing_api_service(cls, service_file: str) -> "ArtifactPaths":
        return cls.from_service_dir(service_file, parent_levels=2)

    @classmethod
    def for_fastapi_train(cls, script_file: str) -> "ArtifactPaths":
        return cls.from_parent_dir(script_file, parent_levels=1)

    @classmethod
    def for_fastapi_app(cls, app_file: str) -> "ArtifactPaths":
        return cls.from_script_dir(app_file)

    @classmethod
    def for_bentoml(cls, script_file: str) -> "ArtifactPaths":
        return cls.from_script_dir(script_file)

    @classmethod
    def for_flaskapp(cls, app_file: str) -> "ArtifactPaths":
        return cls.from_script_dir(app_file)

    def find_existing_model_file(self, fallback_dirs: Optional[List[str]] = None) -> Optional[str]:
        if self.model_file_exists():
            return self.model_file
        if fallback_dirs:
            for d in fallback_dirs:
                candidate = os.path.join(d, self.model_filename)
                if os.path.isfile(candidate):
                    return candidate
        return None
