from pydantic import BaseModel, validator, ValidationError
import os
from pathlib import Path
from main.common.constants import PROJECT_ROOT_DIR


class PathConfig(BaseModel):
    ROOT_DIR: Path
    RAW_DATA_DIR: Path
    PREPARED_DATA_DIR: Path

    @validator("*", pre=True)
    def create_dir(cls, v):
        if isinstance(v, Path) and not str(v).endswith('.csv') and not str(v).endswith('.yaml') and not str(v).endswith('.json'):
            os.makedirs(v, exist_ok=True)
            return v
        elif isinstance(v, Path) and str(v).endswith('.csv'):
            return v
        elif isinstance(v, Path) and str(v).endswith('.yaml'):
            return v
        elif isinstance(v, Path) and str(v).endswith('.json'):
            return v
        raise ValidationError


def initialize_paths():
    root_dir = PROJECT_ROOT_DIR.parent.parent.parent.parent
    raw_data_dir = root_dir / "dstc8-schema-guided-dialogue"
    prepared_data_dir = raw_data_dir / "sgd_supervision"

    return PathConfig(
        ROOT_DIR=root_dir,
        RAW_DATA_DIR=raw_data_dir,
        PREPARED_DATA_DIR=prepared_data_dir
    )
