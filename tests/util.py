import os
import tempfile

from shapely.geometry import LineString

import nlmod

MODEL_DATA_ENV_VAR = "NLMOD_TEST_MODEL_DATA_DIR"


def get_model_data_dir():
    model_data_dir = os.environ.get(MODEL_DATA_ENV_VAR)
    if model_data_dir is None:
        # In interactive windows there is no pytest fixture lifecycle, so provide
        # a deterministic temp fallback to make single-test debugging possible.
        model_data_dir = os.path.join(
            tempfile.gettempdir(), "nlmod_test_model_data_interactive"
        )
        os.environ[MODEL_DATA_ENV_VAR] = model_data_dir
    os.makedirs(model_data_dir, exist_ok=True)
    return model_data_dir


def get_ds_structured(extent=None, model_name="test", **kwargs):
    if extent is None:
        extent = [0, 1000, 0, 1000]
    model_ws = os.path.join(get_model_data_dir(), model_name)
    ds = nlmod.get_ds(extent, model_name=model_name, model_ws=model_ws, **kwargs)
    return ds


def get_ds_vertex(extent=None, line=None, **kwargs):
    if line is None:
        line = [(0, 1000), (1000, 0)]
    ds = get_ds_structured(extent=extent, **kwargs)
    model_ws = os.path.join(get_model_data_dir(), "gridgen")
    refinement_features = [([LineString(line)], "line", 1)]
    ds = nlmod.grid.refine(ds, model_ws, refinement_features=refinement_features)
    return ds


def get_gwf(ds):
    sim = nlmod.sim.sim(ds)
    if "time" in ds.variables:
        nlmod.sim.tdis(ds, sim)
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.dis(ds, gwf)
    return gwf
