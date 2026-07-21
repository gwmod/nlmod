import gc
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

try:
    from xarray.backends.file_manager import FILE_CACHE
except Exception:  # pragma: no cover - defensive for xarray internals
    FILE_CACHE = None

MODEL_DATA_ENV_VAR = "NLMOD_TEST_MODEL_DATA_DIR"


@pytest.fixture(scope="session", autouse=True)
def session_model_data_dir(tmp_path_factory):
    """Use one temp folder for model-data generated and shared during a test session."""
    source_dir = Path(__file__).parent / "data"
    model_data_dir = tmp_path_factory.mktemp("model_data")

    # Seed the shared temp directory with checked-in fixture files.
    for pattern in ("*.nc", "*.zip", "*.tar"):
        for src in source_dir.glob(pattern):
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


@pytest.fixture(autouse=True)
def close_all_matplotlib_figures():
    """Close figures and backend file handles created during a test."""
    yield
    plt.close("all")
    gc.collect()
    if FILE_CACHE is not None:
        FILE_CACHE.clear()
