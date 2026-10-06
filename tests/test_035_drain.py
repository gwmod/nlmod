import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import test_010_wells
from shapely.geometry import LineString, MultiPoint, box

import nlmod


def _drn_spd(drains, **kwargs):
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drn = nlmod.gwf.drain.drain_from_df(drains, gwf, ds, silent=True, **kwargs)
    return drn, drn.stress_period_data.array[0]


def test_drain_from_df_line_and_polygon():
    """Test vector drain conductance, elevation and boundnames."""
    drains = gpd.GeoDataFrame(
        {
            "name": ["line-drain", "area-drain"],
            "elevation": [-1.0, -3.0],
            "conductance_per_meter": [2.0, np.nan],
            "conductance_per_squared_meter": [np.nan, 3.0],
        },
        geometry=[
            LineString([(-499.0, 499.0), (-497.0, 499.0)]),
            box(-499.0, 497.0, -497.0, 498.0),
        ],
    )

    drn, spd = _drn_spd(drains, boundnames="name", pname="drn_test")

    assert drn.package_name == "drn_test"
    assert spd["boundname"].tolist() == ["line-drain", "area-drain"]
    assert spd["elev"].tolist() == [-1.0, -3.0]
    assert spd["cond"].tolist() == pytest.approx([4.0, 6.0])


def test_drain_from_df_keeps_elevations_separate_within_cell():
    """Test drains in one cell keep separate elevations instead of the minimum."""
    drains = gpd.GeoDataFrame(
        {
            "elevation": [-1.0, -3.0],
            "conductance_per_meter": [2.0, 2.0],
        },
        geometry=[
            LineString([(-499.0, 499.0), (-497.0, 499.0)]),
            LineString([(-499.0, 498.0), (-497.0, 498.0)]),
        ],
    )

    _, spd = _drn_spd(drains)

    assert spd["cellid"][0] == spd["cellid"][1]
    assert spd["elev"].tolist() == [-1.0, -3.0]
    assert spd["cond"].tolist() == pytest.approx([4.0, 4.0])


def test_drain_from_df_uses_clipped_line_and_polygon_measures():
    """Test vector conductance uses clipped geometry measure per cell."""
    drains = gpd.GeoDataFrame(
        {
            "elevation": [-1.0, -2.0],
            "conductance_per_meter": [2.0, np.nan],
            "conductance_per_squared_meter": [np.nan, 2.0],
        },
        geometry=[
            LineString([(-499.0, 499.0), (-481.0, 499.0)]),
            box(-499.0, 495.0, -481.0, 498.0),
        ],
    )

    _, spd = _drn_spd(drains)

    assert sorted(spd["cond"].tolist()) == pytest.approx([18.0, 18.0, 54.0, 54.0])


def test_drain_from_df_preserves_point_conductance():
    """Test that point drains use supplied integrated conductance unchanged."""
    drains = pd.DataFrame(
        {
            "x": [-495.0],
            "y": [495.0],
            "elevation": [-1.0],
            "cond": [7.0],
        }
    )

    _, spd = _drn_spd(drains)

    assert spd["cond"].tolist() == [7.0]
    assert spd["elev"].tolist() == [-1.0]


def test_drain_from_df_rejects_multipoint_conductance():
    """Test MultiPoint drains are rejected to avoid duplicating conductance."""
    drains = gpd.GeoDataFrame(
        {
            "elevation": [-1.0],
            "cond": [7.0],
        },
        geometry=[MultiPoint([(-495.0, 495.0), (-485.0, 495.0)])],
    )

    with pytest.raises(TypeError, match="Unsupported drain geometry types"):
        nlmod.gwf.drain.drain_from_df(drains, gwf=None, ds=None, silent=True)
