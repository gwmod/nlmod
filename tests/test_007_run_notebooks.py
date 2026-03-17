"""Run all notebooks in the docs directory recursively."""

# ruff: noqa: D103
import re
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

tst_dir = Path(__file__).resolve().parent
docs_dir = (tst_dir / ".." / "docs").resolve()


def _is_dated_notebook_copy(path):
    return re.match(r"^\d{8,10}_", path.name) is not None


def _iter_notebooks(base_dir):
    for path in sorted(base_dir.rglob("*.ipynb")):
        rel_parts = path.relative_to(base_dir).parts
        if rel_parts and rel_parts[0] == "build":
            continue
        if ".ipynb_checkpoints" in rel_parts:
            continue
        if _is_dated_notebook_copy(path):
            continue
        yield path


NOTEBOOKS = list(_iter_notebooks(docs_dir))
NOTEBOOKS_BY_REL = {path.relative_to(docs_dir).as_posix(): path for path in NOTEBOOKS}

# Some notebooks consume artifacts produced by other notebooks.
NOTEBOOK_DEPENDENCIES = {
    "examples/09_schoonhoven.ipynb": ["data_sources/02_surface_water.ipynb"],
    "examples/14_stromingen_example.ipynb": ["data_sources/02_surface_water.ipynb"],
    "utilities/13_plot_methods.ipynb": ["examples/09_schoonhoven.ipynb"],
    "workflows/03_aggregating_surface_water.ipynb": ["data_sources/02_surface_water.ipynb"],
    "workflows/10_modpath.ipynb": ["examples/03_local_grid_refinement.ipynb"],
    "workflows/11_particle_tracking_prt.ipynb": ["examples/00_model_from_scratch.ipynb"],
    "workflows/18_observations.ipynb": ["examples/03_local_grid_refinement.ipynb"],
}

_EXECUTED_NOTEBOOKS = set()


def _run_notebook(path):
    with path.open(encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=6000)
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})


def _run_with_dependencies(path):
    rel = path.relative_to(docs_dir).as_posix()
    for dep_rel in NOTEBOOK_DEPENDENCIES.get(rel, []):
        dep_path = NOTEBOOKS_BY_REL.get(dep_rel)
        if dep_path is None:
            raise RuntimeError(f"Notebook dependency not found: {dep_rel}")
        if dep_path not in _EXECUTED_NOTEBOOKS:
            _run_with_dependencies(dep_path)

    if path not in _EXECUTED_NOTEBOOKS:
        _run_notebook(path)
        _EXECUTED_NOTEBOOKS.add(path)


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOKS,
    ids=[str(path.relative_to(docs_dir)) for path in NOTEBOOKS],
)
def test_run_notebook(notebook_path):
    _run_with_dependencies(notebook_path)
