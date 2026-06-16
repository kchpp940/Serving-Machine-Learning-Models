from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from car_pricing.config import RuntimeConfig


@dataclass
class ArtifactPaths:
    config: RuntimeConfig
    model_dir: str
    training_data_dir: str

    @property
    def model_file(self) -> str:
        return os.path.join(self.model_dir, self.config.model_filename)

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.model_dir, self.config.model_metadata_filename)

    @property
    def status_file(self) -> str:
        return os.path.join(self.model_dir, self.config.model_status_filename)

    @property
    def schema_snapshot_file(self) -> str:
        return os.path.join(self.model_dir, self.config.schema_snapshot_filename)

    @property
    def lineage_file(self) -> str:
        return os.path.join(self.model_dir, self.config.lineage_filename)

    @property
    def training_data_file(self) -> str:
        return os.path.join(self.training_data_dir, self.config.data_csv_filename)

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
    def from_config(
        cls,
        config: Optional[RuntimeConfig] = None,
        base_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
    ) -> "ArtifactPaths":
        if config is None:
            config = RuntimeConfig.from_env(base_dir=base_dir)
        if model_dir is not None:
            resolved_model_dir = os.path.abspath(model_dir)
        else:
            resolved_model_dir = config.resolve_model_dir(base_dir=base_dir)
        resolved_data_dir = config.resolve_data_dir(base_dir=base_dir)
        return cls(
            config=config,
            model_dir=resolved_model_dir,
            training_data_dir=resolved_data_dir,
        )

    @classmethod
    def from_dir(
        cls,
        model_dir: str,
        training_data_dir: Optional[str] = None,
        config: Optional[RuntimeConfig] = None,
    ) -> "ArtifactPaths":
        if config is None:
            config = RuntimeConfig.default()
        abs_model_dir = os.path.abspath(model_dir)
        if training_data_dir is None:
            training_data_dir = config.resolve_data_dir(
                base_dir=os.path.dirname(abs_model_dir)
            )
        return cls(
            config=config,
            model_dir=abs_model_dir,
            training_data_dir=os.path.abspath(training_data_dir),
        )
