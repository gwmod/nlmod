# ruff: noqa: D103
import matplotlib
import pytest

matplotlib.use("Agg")

import flopy
import geopandas as gpd
import pandas as pd
import util
from flopy.utils import make_hfb_array
from shapely.geometry import LineString, MultiLineString, Point, Polygon

import nlmod
from nlmod.dims.grid import modelgrid_from_ds
from nlmod.dims.layers import get_idomain


def _normalize_spd(spd):
    return [
        (tuple(int(i) for i in cellid1), tuple(int(i) for i in cellid2), float(hydchr))
        for cellid1, cellid2, hydchr in spd
    ]


def _structured_diagonal_expected_spd(hydchr=1 / 100.0):
    expected_spd = []
    for ilay, layer_hydchr in ((0, hydchr), (1, hydchr), (2, hydchr * 0.5)):
        for idx in range(9):
            expected_spd.append(((ilay, idx, idx), (ilay, idx + 1, idx), layer_hydchr))
            expected_spd.append(
                ((ilay, idx + 1, idx + 1), (ilay, idx + 1, idx), layer_hydchr)
            )
    return expected_spd


def _flopy_pair_set(ds, geometry, layer=0):
    modelgrid = modelgrid_from_ds(ds, idomain=get_idomain(ds).values)
    return {
        frozenset(
            (
                tuple(int(i) for i in row.cellid1),
                tuple(int(i) for i in row.cellid2),
            )
        )
        for row in make_hfb_array(modelgrid, geometry)
        if int(row.cellid1[0]) == layer
    }


def _flopy_depth5_top_layer_spd(ds, geometries, hydchr):
    if not isinstance(geometries, (list, tuple)):
        geometries = [geometries]
    modelgrid = modelgrid_from_ds(ds, idomain=get_idomain(ds).values)
    expected_spd = []
    for geometry in geometries:
        for row in make_hfb_array(modelgrid, geometry):
            cellid1 = tuple(int(i) for i in row.cellid1)
            cellid2 = tuple(int(i) for i in row.cellid2)
            if cellid1[0] == 0:
                expected_spd.append((cellid1, cellid2, hydchr * 0.5))
    return expected_spd


def test_flopy_make_hfb_array_available():
    assert callable(make_hfb_array)


def test_get_hfb_spd_uses_flopy_make_hfb_array(monkeypatch):
    ds = util.get_ds_structured()
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 1000), (1000, 0)])]})
    calls = []
    original_make_hfb_array = nlmod.gwf.hfb.make_hfb_array

    def spy_make_hfb_array(modelgrid, geometry):
        calls.append((modelgrid, geometry))
        return original_make_hfb_array(modelgrid, geometry)

    monkeypatch.setattr(nlmod.gwf.hfb, "make_hfb_array", spy_make_hfb_array)

    nlmod.gwf.hfb.get_hfb_spd(ds, gdf, hydchr=1 / 100.0, depth=5.0)

    assert len(calls) == 1


def test_get_hfb_spd_vertex():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    geometry = LineString([(0, 1000), (1000, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [geometry]})
    spd = nlmod.gwf.hfb.get_hfb_spd(ds, gdf, hydchr=1 / 100.0, depth=5.0)
    assert {
        frozenset((cellid1, cellid2)) for cellid1, cellid2, _ in _normalize_spd(spd)
    } == _flopy_pair_set(ds, geometry)
    assert {hydchr for _, _, hydchr in _normalize_spd(spd)} == {0.005}
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: spd})
    # also test the plot method
    ax = gdf.plot()
    nlmod.gwf.hfb.plot_hfb(hfb, gwf, ax=ax)


def test_get_hfb_spd_structured():
    ds = util.get_ds_structured()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    coords = [(0, 1000), (1000, 0)]
    gdf = gpd.GeoDataFrame({"geometry": [LineString(coords)]})
    spd = nlmod.gwf.hfb.get_hfb_spd(ds, gdf, hydchr=1 / 100.0, depth=25.0)
    assert _normalize_spd(spd) == _structured_diagonal_expected_spd()
    elevation_spd = nlmod.gwf.hfb.get_hfb_spd(
        ds, gdf, hydchr=1 / 100.0, elevation=-25.0
    )
    assert _normalize_spd(elevation_spd) == _structured_diagonal_expected_spd()
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: spd})
    # also test the plot method
    ax = gdf.plot()
    nlmod.gwf.hfb.plot_hfb(hfb, gwf, ax=ax)


def test_get_hfb_spd_skips_inactive_and_passthrough_cells():
    ds = util.get_ds_structured()
    ds["botm"].values[1] = ds["botm"].values[0]
    active_domain = ds["top"] == ds["top"]
    active_domain.values[0, 0] = False
    ds["active_domain"] = active_domain
    idomain = get_idomain(ds)
    geometry = LineString([(0, 1000), (1000, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [geometry]})

    spd = _normalize_spd(
        nlmod.gwf.hfb.get_hfb_spd(ds, gdf, hydchr=1 / 100.0, depth=25.0)
    )
    expected_spd = []
    for ilay, layer_hydchr in ((0, 1 / 100.0), (2, 0.0075)):
        for idx in range(9):
            expected_spd.append(((ilay, idx, idx), (ilay, idx + 1, idx), layer_hydchr))
            expected_spd.append(
                ((ilay, idx + 1, idx + 1), (ilay, idx + 1, idx), layer_hydchr)
            )
    expected_spd = [
        row for row in expected_spd if row[0][1:] != (0, 0) and row[1][1:] != (0, 0)
    ]

    assert spd == expected_spd
    assert all(
        idomain.values[cellid1] > 0 and idomain.values[cellid2] > 0
        for cellid1, cellid2, _ in spd
    )
    assert all(cellid1[0] != 1 and cellid2[0] != 1 for cellid1, cellid2, _ in spd)
    assert all(
        cellid1[1:] != (0, 0) and cellid2[1:] != (0, 0) for cellid1, cellid2, _ in spd
    )


def test_hfb_from_df_vertex():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "hydchr": [1 / 100.0, 1 / 200.0],
            "depth": [5.0, None],
            "elevation": [None, -2.0],
            "geometry": [
                LineString([(0, 1000), (1000, 0)]),
                LineString([(100, 1000), (1000, 100)]),
            ],
        },
    )

    hfb = nlmod.gwf.hfb.hfb_from_df(gdf, gwf, ds, pname="hfb_test")

    assert hfb.package_name == "hfb_test"
    expected_spd = []
    for row_number, row in enumerate(gdf.itertuples()):
        if pd.notna(row.depth):
            expected_spd += nlmod.gwf.hfb.get_hfb_spd(
                ds,
                gdf.iloc[[row_number]],
                hydchr=row.hydchr,
                depth=row.depth,
            )
        else:
            expected_spd += nlmod.gwf.hfb.get_hfb_spd(
                ds,
                gdf.iloc[[row_number]],
                hydchr=row.hydchr,
                elevation=row.elevation,
            )
    expected_spd = [
        [cellid1, cellid2, float(hydchr)]
        for cellid1, cellid2, hydchr in expected_spd
        if float(hydchr) > 0
    ]
    actual_spd = [
        [row[0], row[1], float(row[2])] for row in hfb.stress_period_data.data[0]
    ]
    assert actual_spd == expected_spd


def test_hfb_from_df_requires_depth_or_elevation():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "hydchr": [1 / 100.0],
            "depth": [None],
            "elevation": [None],
            "geometry": [LineString([(0, 1000), (1000, 0)])],
        },
    )

    with pytest.raises(ValueError, match="Exactly one"):
        nlmod.gwf.hfb.hfb_from_df(gdf, gwf, ds)

    gdf.loc[0, "depth"] = 5.0
    gdf.loc[0, "elevation"] = -2.0
    with pytest.raises(ValueError, match="Exactly one"):
        nlmod.gwf.hfb.hfb_from_df(gdf, gwf, ds)


@pytest.mark.parametrize("hydchr", [0.0, -0.01])
def test_hfb_from_df_rejects_nonpositive_hydchr(hydchr):
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "hydchr": [hydchr],
            "depth": [5.0],
            "geometry": [LineString([(0, 1000), (1000, 0)])],
        },
    )

    with pytest.raises(ValueError, match="hydchr must be positive"):
        nlmod.gwf.hfb.hfb_from_df(gdf, gwf, ds, elevation=None)


def test_hfb_from_df_returns_none_when_all_hydchr_rows_are_zero():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "hydchr": [1 / 100.0],
            "depth": [0.0],
            "geometry": [LineString([(0, 1000), (1000, 0)])],
        },
    )

    assert nlmod.gwf.hfb.hfb_from_df(gdf, gwf, ds, elevation=None) is None


def test_hfb_from_df_rejects_non_line_geometries():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "hydchr": [1 / 100.0],
            "depth": [5.0],
            "geometry": [Point(0, 1000)],
        },
    )

    with pytest.raises(ValueError, match="LineString"):
        nlmod.gwf.hfb.hfb_from_df(gdf, gwf, ds, elevation=None)


def test_hfb_from_df_accepts_scalar_values():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 1000), (1000, 0)])]})

    hfb = nlmod.gwf.hfb.hfb_from_df(
        gdf,
        gwf,
        ds,
        hydchr=1 / 100.0,
        depth=5.0,
        elevation=None,
    )

    expected_spd = nlmod.gwf.hfb.get_hfb_spd(
        ds,
        gdf,
        hydchr=1 / 100.0,
        depth=5.0,
    )
    expected_spd = [
        [cellid1, cellid2, float(hydchr)]
        for cellid1, cellid2, hydchr in expected_spd
        if float(hydchr) > 0
    ]
    actual_spd = [
        [row[0], row[1], float(row[2])] for row in hfb.stress_period_data.data[0]
    ]
    assert actual_spd == expected_spd


def test_hfb_from_df_scalar_values_apply_to_all_features():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 1000), (1000, 0)]),
                LineString([(100, 1000), (1000, 100)]),
            ],
        },
    )

    hfb = nlmod.gwf.hfb.hfb_from_df(
        gdf,
        gwf,
        ds,
        hydchr=1 / 100.0,
        depth=5.0,
        elevation=None,
    )

    expected_spd = []
    for row_number in range(len(gdf)):
        expected_spd += nlmod.gwf.hfb.get_hfb_spd(
            ds,
            gdf.iloc[[row_number]],
            hydchr=1 / 100.0,
            depth=5.0,
        )
    expected_spd = [
        [cellid1, cellid2, float(hydchr)]
        for cellid1, cellid2, hydchr in expected_spd
        if float(hydchr) > 0
    ]
    actual_spd = [
        [row[0], row[1], float(row[2])] for row in hfb.stress_period_data.data[0]
    ]
    assert actual_spd == expected_spd


def test_hfb_from_df_accepts_custom_column_names():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "resistance_inverse": [1 / 100.0],
            "barrier_depth": [5.0],
            "geometry": [LineString([(0, 1000), (1000, 0)])],
        },
    )

    hfb = nlmod.gwf.hfb.hfb_from_df(
        gdf,
        gwf,
        ds,
        hydchr="resistance_inverse",
        depth="barrier_depth",
        elevation=None,
    )

    expected_spd = nlmod.gwf.hfb.get_hfb_spd(
        ds,
        gdf,
        hydchr=1 / 100.0,
        depth=5.0,
    )
    expected_spd = [
        [cellid1, cellid2, float(hydchr)]
        for cellid1, cellid2, hydchr in expected_spd
        if float(hydchr) > 0
    ]
    actual_spd = [
        [row[0], row[1], float(row[2])] for row in hfb.stress_period_data.data[0]
    ]
    assert actual_spd == expected_spd


def test_hfb_from_df_accepts_multilinestring():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                MultiLineString(
                    [
                        [(0, 1000), (500, 500)],
                        [(500, 500), (1000, 0)],
                    ],
                ),
            ],
        },
    )

    hfb = nlmod.gwf.hfb.hfb_from_df(
        gdf,
        gwf,
        ds,
        hydchr=1 / 100.0,
        depth=5.0,
        elevation=None,
    )

    expected_spd = _flopy_depth5_top_layer_spd(
        ds, list(gdf.geometry.iloc[0].geoms), 1 / 100.0
    )
    actual_spd = [
        (tuple(row[0]), tuple(row[1]), float(row[2]))
        for row in hfb.stress_period_data.data[0]
    ]
    assert actual_spd == expected_spd


def test_hfb_from_df_accepts_scalar_elevation():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 1000), (1000, 0)])]})

    hfb = nlmod.gwf.hfb.hfb_from_df(
        gdf,
        gwf,
        ds,
        hydchr=1 / 200.0,
        depth=None,
        elevation=-2.0,
    )

    expected_spd = nlmod.gwf.hfb.get_hfb_spd(
        ds,
        gdf,
        hydchr=1 / 200.0,
        elevation=-2.0,
    )
    expected_spd = [
        [cellid1, cellid2, float(hydchr)]
        for cellid1, cellid2, hydchr in expected_spd
        if float(hydchr) > 0
    ]
    actual_spd = [
        [row[0], row[1], float(row[2])] for row in hfb.stress_period_data.data[0]
    ]
    assert actual_spd == expected_spd


def test_polygon_to_hfb_vertex():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    coords = [(135, 230), (568, 170), (778, 670), (260, 786)]
    gdf = gpd.GeoDataFrame({"geometry": [Polygon(coords)]}).reset_index()
    hfb = nlmod.gwf.hfb.polygon_to_hfb(gdf, ds, hydchr=1 / 100.0, gwf=gwf)
    # also test the plot method
    ax = gdf.plot()
    nlmod.gwf.hfb.plot_hfb(hfb, gwf, ax=ax)


def test_polygon_to_hfb_structured():
    ds = util.get_ds_structured()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    coords = [(135, 230), (568, 170), (778, 670), (260, 786)]
    gdf = gpd.GeoDataFrame({"geometry": [Polygon(coords)]}).reset_index()
    hfb = nlmod.gwf.hfb.polygon_to_hfb(gdf, ds, hydchr=1 / 100.0, gwf=gwf)
    # also test the plot method
    ax = gdf.plot()
    nlmod.gwf.hfb.plot_hfb(hfb, gwf, ax=ax)


def test_line_to_hfb_deprecated():
    ds = util.get_ds_structured()
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(100, 1000), (225.0, 425.1)])]})

    with pytest.warns(DeprecationWarning, match="line_to_hfb.*deprecated"):
        cellids = nlmod.gwf.hfb.line_to_hfb(gdf, ds)

    assert cellids


def test_line2hfb_deprecated_without_duplicate_line_to_hfb_warning():
    ds = util.get_ds_structured()
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(100, 1000), (225.0, 425.1)])]})

    with pytest.warns(
        DeprecationWarning, match="line2hfb.*deprecated"
    ) as warning_records:
        cellids = nlmod.gwf.hfb.line2hfb(gdf, ds)

    messages = [str(record.message) for record in warning_records]
    assert any("line2hfb" in message for message in messages)
    assert not any("line_to_hfb" in message for message in messages)
    assert cellids


def test_line_to_hfb_buffer_structured():
    ds = util.get_ds_structured()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "name": ["hfb1", "hfb2"],
            "geometry": [
                LineString([(100, 1000), (225.0, 425.1)]),
                LineString([(225.0, 425.1), (225.0, 0)]),
            ],
        },
    )
    with pytest.warns(DeprecationWarning, match="line_to_hfb_buffer.*deprecated"):
        cellids = nlmod.gwf.hfb.line_to_hfb_buffer(gdf, ds)
    # also test the plot method
    ax = gdf.plot(column="name")
    gwf.modelgrid.plot(ax=ax)
    nlmod.gwf.hfb.plot_hfb(cellids, gwf, ax=ax)


def test_line_to_hfb_buffer_vertex():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    gdf = gpd.GeoDataFrame(
        {
            "name": ["hfb1", "hfb2"],
            "geometry": [
                LineString([(100, 1000), (225.0, 425.1)]),
                LineString([(225.0, 425.1), (225.0, 0)]),
            ],
        },
    )
    with pytest.warns(DeprecationWarning, match="line_to_hfb_buffer.*deprecated"):
        cellids = nlmod.gwf.hfb.line_to_hfb_buffer(gdf, ds)
    # also test the plot method
    ax = gdf.plot(column="name")
    gwf.modelgrid.plot(ax=ax)
    nlmod.gwf.hfb.plot_hfb(cellids, gwf, ax=ax)
