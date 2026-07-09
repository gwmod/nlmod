import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import test_010_wells
import util
from shapely.geometry import LineString, MultiPoint, Polygon

import nlmod


def test_drain_from_df_vector_keeps_drain_thresholds_and_mvr_mapping():
    """Test vector drain conductance and MVR provider metadata."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = gpd.GeoDataFrame(
        {
            "name": ["line-drain", "area-drain"],
            "elevation": [-1.0, -3.0],
            "conductance_per_meter": [2.0, np.nan],
            "conductance_per_squared_meter": [np.nan, 3.0],
            "mover_lake_name": ["lake-1", "lake-2"],
        },
        geometry=[
            LineString([(-499.0, 499.0), (-497.0, 499.0)]),
            Polygon(
                [
                    (-499.0, 498.0),
                    (-497.0, 498.0),
                    (-497.0, 497.0),
                    (-499.0, 497.0),
                ]
            ),
        ],
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        boundnames="name",
        mover_destinations="mover_lake_name",
        mover=True,
        pname="drn_test",
        silent=True,
        return_provider_mapping=True,
    )

    assert drn.package_name == "drn_test"
    assert drn.mover.array is True
    assert provider_mapping["mvr_provider_id"].tolist() == [0, 1]
    assert provider_mapping["mover_destination"].tolist() == ["lake-1", "lake-2"]
    assert provider_mapping["boundname"].tolist() == ["line-drain", "area-drain"]
    assert provider_mapping["elev"].tolist() == [-1.0, -3.0]
    assert provider_mapping["cond"].tolist() == pytest.approx([4.0, 6.0])
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)

    head = -2.0
    generated_flux = (
        provider_mapping["cond"] * np.maximum(head - provider_mapping["elev"], 0.0)
    ).sum()
    min_elevation_collapsed_flux = provider_mapping["cond"].sum() * max(
        head - -3.0, 0.0
    )
    assert generated_flux == pytest.approx(6.0)
    assert generated_flux < min_elevation_collapsed_flux


def test_drain_from_df_keeps_same_geometry_thresholds_separate():
    """Test same-geometry drains in one cell keep separate elevations."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
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

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_same_geom",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping["elev"].tolist() == [-1.0, -3.0]
    assert provider_mapping["cond"].tolist() == pytest.approx([4.0, 4.0])
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)

    head = -2.0
    generated_flux = (
        provider_mapping["cond"] * np.maximum(head - provider_mapping["elev"], 0.0)
    ).sum()
    min_elevation_collapsed_flux = provider_mapping["cond"].sum() * max(
        head - -3.0, 0.0
    )
    assert generated_flux == pytest.approx(4.0)
    assert generated_flux < min_elevation_collapsed_flux


def test_drain_from_df_uses_clipped_line_and_polygon_measures():
    """Test vector conductance uses clipped geometry measure per cell."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = gpd.GeoDataFrame(
        {
            "elevation": [-1.0, -2.0],
            "conductance_per_meter": [2.0, np.nan],
            "conductance_per_squared_meter": [np.nan, 2.0],
        },
        geometry=[
            LineString([(-499.0, 499.0), (-481.0, 499.0)]),
            Polygon(
                [
                    (-499.0, 498.0),
                    (-481.0, 498.0),
                    (-481.0, 495.0),
                    (-499.0, 495.0),
                ]
            ),
        ],
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_clipped",
        silent=True,
        return_provider_mapping=True,
    )

    assert sorted(provider_mapping["cond"].tolist()) == pytest.approx(
        [18.0, 18.0, 54.0, 54.0]
    )
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_mvr_perioddata_from_provider_mapping():
    """Test conversion from DRN provider mapping to MVR perioddata."""
    provider_mapping = pd.DataFrame(
        {
            "package": ["drn_test", "drn_test"],
            "mvr_provider_id": [0, 1],
            "mover_destination": ["lake-1", "lake-2"],
        }
    )

    perioddata = nlmod.gwf.drain.mvr_perioddata_from_provider_mapping(
        provider_mapping,
        receiver_package="lak",
        receiver_id_map={"lake-1": 0, "lake-2": 1},
    )

    assert perioddata == [
        ("drn_test", 0, "lak", 0, "FACTOR", 1.0),
        ("drn_test", 1, "lak", 1, "FACTOR", 1.0),
    ]

    provider_mapping = pd.DataFrame(
        {
            "package": ["drn_test", "drn_test", "drn_test"],
            "mvr_provider_id": [0, 1, 2],
            "mover_destination": ["lake-1", np.nan, "lake-2"],
        },
        index=[10, 20, 30],
    )
    perioddata = nlmod.gwf.drain.mvr_perioddata_from_provider_mapping(
        provider_mapping,
        receiver_package="lak",
        receiver_id_map={"lake-1": 0, "lake-2": 1},
    )
    assert perioddata == [
        ("drn_test", 0, "lak", 0, "FACTOR", 1.0),
        ("drn_test", 2, "lak", 1, "FACTOR", 1.0),
    ]


def test_drain_from_df_preserves_direct_3d_cellids():
    """Test that explicit 3D cell IDs are not prefixed with another layer."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0, 0)],
            "elevation": [-12.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_3d",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (0, 0, 0)
    assert provider_mapping.loc[0, "elev"] == -12.0
    assert provider_mapping.loc[0, "cond"] == 5.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_preserves_active_3d_cellid_with_timeseries_elevation():
    """Test active explicit 3D cell IDs can keep timeseries elevations."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0, 0)],
            "elevation": ["stage_ts"],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_3d_ts",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (0, 0, 0)
    assert provider_mapping.loc[0, "elev"] == "stage_ts"
    assert provider_mapping.loc[0, "cond"] == 5.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_omits_3d_cellids_without_elevation():
    """Test explicit 3D cell IDs without elevation are omitted."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0, 0)],
            "elevation": [np.nan],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_3d_nan",
        silent=True,
        return_provider_mapping=True,
    )

    assert drn is None
    assert provider_mapping.empty


def test_drain_from_df_places_2d_cellids_in_layer_from_elevation():
    """Test that 2D cell IDs use drain elevation for layer placement."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0)],
            "elevation": [-12.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_2d",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (1, 0, 0)
    assert provider_mapping.loc[0, "elev"] == -12.0
    assert provider_mapping.loc[0, "cond"] == 5.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_places_2d_cellids_below_inactive_top_layer():
    """Test 2D cell IDs skip inactive layers during layer placement."""
    ds = test_010_wells.get_model_ds()
    ds["active_domain"] = ds["botm"].notnull()
    ds["active_domain"].data[0, 0, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [0, 1, 1]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0)],
            "elevation": [-1.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_inactive_top",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (1, 0, 0)
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_keeps_valid_2d_cellids_after_omitting_inactive_column():
    """Test omitted 2D rows do not suppress later valid rows."""
    ds = test_010_wells.get_model_ds()
    ds["active_domain"] = ds["top"].notnull()
    ds["active_domain"].data[0, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [0, 0, 0]
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 1].tolist() == [1, 1, 1]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0), (0, 1)],
            "elevation": [-1.0, -1.0],
            "cond": [5.0, 7.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_mixed_active",
        silent=True,
        return_provider_mapping=True,
    )

    assert drn is not None
    assert provider_mapping["mvr_provider_id"].tolist() == [0]
    assert provider_mapping.loc[0, "cellid"] == (0, 0, 1)
    assert provider_mapping.loc[0, "cond"] == 7.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_places_2d_cellids_below_pass_through_layer():
    """Test 2D cell IDs skip pass-through layers during layer placement."""
    ds = test_010_wells.get_model_ds()
    ds["botm"].data[1, 0, 0] = ds["botm"].data[0, 0, 0]
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [1, -1, 1]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0)],
            "elevation": [-12.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_pass_through",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (2, 0, 0)
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_omits_2d_cellids_without_active_layers():
    """Test 2D cell IDs are omitted when the column has no active layers."""
    ds = test_010_wells.get_model_ds()
    ds["active_domain"] = ds["top"].notnull()
    ds["active_domain"].data[0, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [0, 0, 0]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0)],
            "elevation": [-1.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_no_active_layers",
        silent=True,
        return_provider_mapping=True,
    )

    assert drn is None
    assert provider_mapping.empty


@pytest.mark.parametrize(
    ("cellid", "setup_idomain", "elevation", "expected_idomain", "expected_cellid"),
    [
        ((0, 0, 0), "inactive_top", -1.0, [0, 1, 1], (1, 0, 0)),
        ((1, 0, 0), "pass_through_middle", -12.0, [1, -1, 1], (2, 0, 0)),
        ((1, 0, 0), "inactive_middle", -12.0, [1, 0, 1], (0, 0, 0)),
        ((1, 0, 0), "inactive_middle", -14.0, [1, 0, 1], (2, 0, 0)),
    ],
)
def test_drain_from_df_remaps_3d_inactive_or_pass_through_cellids(
    cellid, setup_idomain, elevation, expected_idomain, expected_cellid
):
    """Test explicit 3D cell IDs are remapped to the nearest active layer."""
    ds = test_010_wells.get_model_ds()
    if setup_idomain == "inactive_top":
        ds["botm"].data[0, 0, 0] = ds["top"].data[0, 0]
    elif setup_idomain == "pass_through_middle":
        ds["botm"].data[1, 0, 0] = ds["botm"].data[0, 0, 0]
    elif setup_idomain == "inactive_middle":
        ds["active_domain"] = ds["botm"].notnull()
        ds["active_domain"].data[1, 0, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == expected_idomain
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [cellid],
            "elevation": [elevation],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_remapped_3d_idomain",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == expected_cellid
    assert provider_mapping.loc[0, "elev"] == elevation
    assert provider_mapping.loc[0, "cond"] == 5.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_raises_for_ambiguous_nearest_active_layer():
    """Test unresolved nearest-active ties require explicit modeler input."""
    ds = test_010_wells.get_model_ds()
    ds["active_domain"] = ds["botm"].notnull()
    ds["active_domain"].data[1, 0, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [1, 0, 1]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(1, 0, 0)],
            "elevation": [-12.5],
            "cond": [5.0],
        }
    )

    with pytest.raises(ValueError, match="equally near"):
        nlmod.gwf.drain.drain_from_df(
            drains,
            gwf,
            ds,
            pname="drn_ambiguous_remap",
            silent=True,
        )


def test_drain_from_df_remaps_multiple_3d_cellids_in_one_call():
    """Test batched explicit 3D remapping keeps row-specific layer choices."""
    ds = test_010_wells.get_model_ds()
    ds["active_domain"] = ds["botm"].notnull()
    ds["active_domain"].data[1, 0, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [1, 0, 1]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0, 0), (1, 0, 0), (1, 0, 0)],
            "elevation": [-1.0, -12.0, -14.0],
            "cond": [3.0, 5.0, 7.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_multi_remap",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping["cellid"].tolist() == [(0, 0, 0), (0, 0, 0), (2, 0, 0)]
    assert provider_mapping["elev"].tolist() == [-1.0, -12.0, -14.0]
    assert provider_mapping["cond"].tolist() == [3.0, 5.0, 7.0]
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_remaps_3d_cellids_with_layered_top():
    """Test remapping uses layer-specific top when top has a layer dimension."""
    ds = test_010_wells.get_model_ds()
    ds["botm"].data[1, 0, 0] = ds["botm"].data[0, 0, 0]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    ds = nlmod.layers.add_layer_dim_to_top(ds)
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [1, -1, 1]
    drains = pd.DataFrame(
        {
            "cellid": [(1, 0, 0)],
            "elevation": [-12.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_layered_top",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (2, 0, 0)
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_raises_for_3d_cellids_without_active_layers():
    """Test explicit 3D cell IDs still require an active column."""
    ds = test_010_wells.get_model_ds()
    ds["active_domain"] = ds["top"].notnull()
    ds["active_domain"].data[0, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [0, 0, 0]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0, 0)],
            "elevation": [-1.0],
            "cond": [5.0],
        }
    )

    with pytest.raises(ValueError, match="no active layers"):
        nlmod.gwf.drain.drain_from_df(
            drains,
            gwf,
            ds,
            pname="drn_bad_3d_idomain",
            silent=True,
        )


def test_drain_from_df_requires_numeric_elevation_to_remap_3d_cellids():
    """Test inactive explicit 3D cell IDs need numeric elevations for remapping."""
    ds = test_010_wells.get_model_ds()
    ds["botm"].data[0, 0, 0] = ds["top"].data[0, 0]
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0, 0].tolist() == [0, 1, 1]
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 0, 0)],
            "elevation": ["stage_ts"],
            "cond": [5.0],
        }
    )

    with pytest.raises(ValueError, match="numeric"):
        nlmod.gwf.drain.drain_from_df(
            drains,
            gwf,
            ds,
            pname="drn_non_numeric_remap",
            silent=True,
        )


def test_drain_from_df_remaps_vertex_3d_cellids_to_nearest_active_layer():
    """Test explicit vertex 3D cell IDs are remapped like structured cell IDs."""
    ds = util.get_ds_vertex(
        model_name="drain_vertex",
        top=0.0,
        botm=[-10.0, -15.0, -30.0],
        kh=[10.0, 0.1, 20.0],
        kv=[5.0, 0.05, 10.0],
    )
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    ds["active_domain"] = ds["botm"].notnull()
    ds["active_domain"].data[1, 0] = False
    assert nlmod.dims.layers.get_idomain(ds).data[:, 0].tolist() == [1, 0, 1]
    gwf = util.get_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(1, 0)],
            "elevation": [-14.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_vertex_remap",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (2, 0)
    assert provider_mapping.loc[0, "elev"] == -14.0
    assert provider_mapping.loc[0, "cond"] == 5.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_preserves_active_vertex_3d_cellids():
    """Test active explicit vertex 3D cell IDs are preserved exactly."""
    ds = util.get_ds_vertex(
        model_name="drn_vtx_active",
        top=0.0,
        botm=[-10.0, -15.0, -30.0],
        kh=[10.0, 0.1, 20.0],
        kv=[5.0, 0.05, 10.0],
    )
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    assert nlmod.dims.layers.get_idomain(ds).data[:, 1].tolist() == [1, 1, 1]
    gwf = util.get_gwf(ds)
    drains = pd.DataFrame(
        {
            "cellid": [(0, 1)],
            "elevation": [-14.0],
            "cond": [5.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_vertex_active",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cellid"] == (0, 1)
    assert provider_mapping.loc[0, "elev"] == -14.0
    assert provider_mapping.loc[0, "cond"] == 5.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_preserves_point_conductance():
    """Test that point drains use supplied integrated conductance unchanged."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = pd.DataFrame(
        {
            "x": [-495.0],
            "y": [495.0],
            "elevation": [-1.0],
            "cond": [7.0],
        }
    )

    drn, provider_mapping = nlmod.gwf.drain.drain_from_df(
        drains,
        gwf,
        ds,
        pname="drn_point",
        silent=True,
        return_provider_mapping=True,
    )

    assert provider_mapping.loc[0, "cond"] == 7.0
    _assert_mapping_matches_stress_period_data(drn, provider_mapping)


def test_drain_from_df_rejects_multipoint_conductance():
    """Test MultiPoint drains are rejected to avoid duplicating conductance."""
    ds = test_010_wells.get_model_ds()
    _, gwf = test_010_wells.get_sim_and_gwf(ds)
    drains = gpd.GeoDataFrame(
        {
            "elevation": [-1.0],
            "cond": [7.0],
        },
        geometry=[MultiPoint([(-495.0, 495.0), (-485.0, 495.0)])],
    )

    with pytest.raises(TypeError, match="Unsupported drain geometry types"):
        nlmod.gwf.drain.drain_from_df(
            drains,
            gwf,
            ds,
            pname="drn_multipoint",
            silent=True,
        )


def _assert_mapping_matches_stress_period_data(drn, provider_mapping):
    spd = drn.stress_period_data.array[0]
    assert len(spd) == len(provider_mapping)
    assert sorted(provider_mapping["mvr_provider_id"].astype(int)) == list(
        range(len(spd))
    )
    for _, row in provider_mapping.iterrows():
        record = spd[int(row["mvr_provider_id"])]
        assert record["cellid"] == row["cellid"]
        if isinstance(row["elev"], str):
            assert record["elev"] == row["elev"]
        else:
            assert record["elev"] == pytest.approx(row["elev"])
        assert record["cond"] == pytest.approx(row["cond"])
        if "boundname" in record.dtype.names and pd.notna(row["boundname"]):
            assert record["boundname"] == row["boundname"]
