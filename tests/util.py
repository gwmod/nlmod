from shapely.geometry import LineString
import os
import nlmod


def get_ds_structured(extent=None, model_name="test", **kwargs):
    if extent is None:
        extent = [0, 1000, 0, 1000]
    model_ws = os.path.join("data", model_name)
    ds = nlmod.get_ds(extent, model_name=model_name, model_ws=model_ws, **kwargs)
    return ds


def get_ds_vertex(extent=None, line=None, **kwargs):
    if line is None:
        line = [(0, 1000), (1000, 0)]
    ds = get_ds_structured(extent=extent, **kwargs)
    model_ws = os.path.join("data", "gridgen")
    refinement_features = [([LineString(line)], "line", 1)]
    ds = nlmod.grid.refine(ds, model_ws, refinement_features=refinement_features)
    return ds
