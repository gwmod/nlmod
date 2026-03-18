import os

import util


def test_model_data_dir_fallback_for_interactive(monkeypatch):
    monkeypatch.delenv("NLMOD_TEST_MODEL_DATA_DIR", raising=False)

    model_data_dir = util.get_model_data_dir()

    assert os.path.isdir(model_data_dir)
    assert os.environ.get("NLMOD_TEST_MODEL_DATA_DIR") == model_data_dir
