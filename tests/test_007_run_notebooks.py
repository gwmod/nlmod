"""run notebooks in the examples directory."""
import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

tst_dir = os.path.dirname(os.path.realpath(__file__))
nbdir = os.path.join(tst_dir, "..", "docs", "examples")


def _run_notebook(nbdir, fname):

    fname_nb = os.path.join(nbdir, fname)
    with open(fname_nb) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=6000)
    out = ep.preprocess(nb, {"metadata": {"path": nbdir}})

    return out


@pytest.mark.notebooks
def test_run_notebook_01_basic_model():
    _run_notebook(nbdir, "01_basic_model.ipynb")


@pytest.mark.notebooks
def test_run_notebook_02_surface_water():
    _run_notebook(nbdir, "02_surface_water.ipynb")


@pytest.mark.notebooks
def test_run_notebook_03_local_grid_refinement():
    _run_notebook(nbdir, "03_local_grid_refinement.ipynb")


@pytest.mark.notebooks
@pytest.mark.skip("requires art_tools")
def test_run_notebook_04_modifying_layermodels():
    _run_notebook(nbdir, "04_modifying_layermodels.ipynb")


@pytest.mark.notebooks
def test_run_notebook_05_caching():
    _run_notebook(nbdir, "05_caching.ipynb")


@pytest.mark.notebooks
def test_run_notebook_06_compare_layermodels():
    _run_notebook(nbdir, "06_compare_layermodels.ipynb")


@pytest.mark.notebooks
def test_run_notebook_07_resampling():
    _run_notebook(nbdir, "07_resampling.ipynb")


@pytest.mark.notebooks
def test_run_notebook_08_gis():
    _run_notebook(nbdir, "08_gis.ipynb")


@pytest.mark.notebooks
def test_run_notebook_09_schoonhoven():
    _run_notebook(nbdir, "09_schoonhoven.ipynb")


@pytest.mark.notebooks
def test_run_notebook_10_modpath():
    _run_notebook(nbdir, "10_modpath.ipynb")


@pytest.mark.notebooks
def test_run_notebook_11_grid_rotation():
    _run_notebook(nbdir, "11_grid_rotation.ipynb")


@pytest.mark.notebooks
def test_run_notebook_12_layer_generation():
    _run_notebook(nbdir, "12_layer_generation.ipynb")
