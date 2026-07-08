# ruff: noqa: D103
import matplotlib
import pytest

matplotlib.use("Agg")

import flopy
import geopandas as gpd
import pandas as pd
import util
from shapely.geometry import LineString, MultiLineString, Point, Polygon

import nlmod


def test_get_hfb_spd_vertex():
    # this test also tests line_to_hfb
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    coords = [(0, 1000), (1000, 0)]
    gdf = gpd.GeoDataFrame({"geometry": [LineString(coords)]})
    spd = nlmod.gwf.hfb.get_hfb_spd(ds, gdf, hydchr=1 / 100.0, depth=5.0)
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: spd})
    # also test the plot method
    ax = gdf.plot()
    nlmod.gwf.hfb.plot_hfb(hfb, gwf, ax=ax)


def test_get_hfb_spd_structured():
    # this test also tests line_to_hfb
    ds = util.get_ds_structured()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)
    coords = [(0, 1000), (1000, 0)]
    gdf = gpd.GeoDataFrame({"geometry": [LineString(coords)]})
    spd = nlmod.gwf.hfb.get_hfb_spd(ds, gdf, hydchr=1 / 100.0, depth=5.0)
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: spd})
    # also test the plot method
    ax = gdf.plot()
    nlmod.gwf.hfb.plot_hfb(hfb, gwf, ax=ax)


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


def test_line_to_hfb_buffer_structured():
    # this test also tests line_to_hfb
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
    cellids = nlmod.gwf.hfb.line_to_hfb_buffer(gdf, ds)
    # also test the plot method
    ax = gdf.plot(column="name")
    gwf.modelgrid.plot(ax=ax)
    nlmod.gwf.hfb.plot_hfb(cellids, gwf, ax=ax)


def test_line_to_hfb_buffer_vertex():
    # this test also tests line_to_hfb
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
    cellids = nlmod.gwf.hfb.line_to_hfb_buffer(gdf, ds)
    # also test the plot method
    ax = gdf.plot(column="name")
    gwf.modelgrid.plot(ax=ax)
    nlmod.gwf.hfb.plot_hfb(cellids, gwf, ax=ax)
