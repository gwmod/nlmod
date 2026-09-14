import pytest

from nlmod import config
from nlmod.version import __version__, show_versions


@pytest.fixture(autouse=True)
def reset_config_options_after_test():
    config.reset_options()
    yield
    config.reset_options()


def test_set_get_and_reset_options():
    config.set_options(nc_hash=False, dataset_coords_hash=False)

    assert config.get_options("nc_hash") == {"nc_hash": False}
    assert config.get_options("dataset_coords_hash") == {
        "dataset_coords_hash": False,
    }

    config.reset_options(["nc_hash", "dataset_coords_hash"])
    assert config.get_options("nc_hash") == {"nc_hash": True}
    assert config.get_options("dataset_coords_hash") == {
        "dataset_coords_hash": True,
    }


def test_set_options_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown option"):
        config.set_options(does_not_exist=True)


def test_cache_options_context_restores_on_exit():
    original = config.get_options()["dataset_data_vars_hash"]

    with config.cache_options(dataset_data_vars_hash=not original) as opts:
        assert opts["dataset_data_vars_hash"] is (not original)

    assert config.get_options()["dataset_data_vars_hash"] is original


def test_cache_options_restores_on_exception():
    original = config.get_options()["explicit_dataset_coordinate_comparison"]

    with pytest.raises(RuntimeError, match="boom"):
        with config.cache_options(explicit_dataset_coordinate_comparison=not original):
            raise RuntimeError("boom")

    assert config.get_options()["explicit_dataset_coordinate_comparison"] is original


def test_show_versions_prints_expected_lines(capsys):
    show_versions()
    out = capsys.readouterr().out

    assert "Python version" in out
    assert "NumPy version" in out
    assert "Xarray version" in out
    assert "Matplotlib version" in out
    assert "Flopy version" in out
    assert f"nlmod version      : {__version__}" in out
