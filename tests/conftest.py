import os
import shutil
from pathlib import Path

import pytest

MODEL_DATA_ENV_VAR = "NLMOD_TEST_MODEL_DATA_DIR"


@pytest.fixture(scope="session", autouse=True)
def session_model_data_dir(tmp_path_factory):
    """Use one temp folder for model-data generated and shared during a test session."""
    source_dir = Path(__file__).parent / "data"
    model_data_dir = tmp_path_factory.mktemp("model_data")

    # Seed the shared temp directory with checked-in netcdf fixtures.
    for src in source_dir.glob("*.nc"):
        shutil.copy2(src, model_data_dir / src.name)

    old_value = os.environ.get(MODEL_DATA_ENV_VAR)
    os.environ[MODEL_DATA_ENV_VAR] = str(model_data_dir)
    try:
        yield model_data_dir
    finally:
        if old_value is None:
            os.environ.pop(MODEL_DATA_ENV_VAR, None)
        else:
            os.environ[MODEL_DATA_ENV_VAR] = old_value