# ruff: noqa: D103
import matplotlib
import pytest

matplotlib.use("Agg")

import flopy
import geopandas as gpd
import numpy as np
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


def _expected_partial_hydchr(ds, cellid1, cellid2, hydchr, frac):
    # Mirrors the parallel-path equivalent of nlmod.gwf.hfb._append_partial_hydchr.
    # The physics of that equivalent is validated independently in
    # test_get_hfb_spd_partial_penetration_matches_resolved_barrier.
    x = ds["x"].values
    y = ds["y"].values
    if len(cellid1) == 3:
        x1, y1 = x[cellid1[2]], y[cellid1[1]]
        x2, y2 = x[cellid2[2]], y[cellid2[1]]
    else:
        x1, y1 = x[cellid1[1]], y[cellid1[1]]
        x2, y2 = x[cellid2[1]], y[cellid2[1]]
    distance = float(np.hypot(x1 - x2, y1 - y2))
    kh1 = ds["kh"].values[cellid1]
    kh2 = ds["kh"].values[cellid2]
    open_face_conductance = 2.0 * kh1 * kh2 / ((kh1 + kh2) * distance)
    return float((hydchr + (1.0 - frac) * open_face_conductance) / frac)


def _structured_diagonal_expected_spd(ds, hydchr=1 / 100.0):
    expected_spd = []
    for ilay in (0, 1, 2):
        for idx in range(9):
            for cellid1, cellid2 in (
                ((ilay, idx, idx), (ilay, idx + 1, idx)),
                ((ilay, idx + 1, idx + 1), (ilay, idx + 1, idx)),
            ):
                if ilay == 2:
                    # barrier penetrates half of the bottom layer
                    layer_hydchr = _expected_partial_hydchr(
                        ds, cellid1, cellid2, hydchr, 0.5
                    )
                else:
                    layer_hydchr = hydchr
                expected_spd.append((cellid1, cellid2, layer_hydchr))
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
                # depth 5.0 penetrates half of the 10-m top layer
                expected_spd.append(
                    (
                        cellid1,
                        cellid2,
                        _expected_partial_hydchr(ds, cellid1, cellid2, hydchr, 0.5),
                    )
                )
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
    # depth 5.0 penetrates half of the 10-m top layer
    for cellid1, cellid2, hydchr_value in _normalize_spd(spd):
        assert hydchr_value == _expected_partial_hydchr(
            ds, cellid1, cellid2, 1 / 100.0, 0.5
        )
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
    assert _normalize_spd(spd) == _structured_diagonal_expected_spd(ds)
    elevation_spd = nlmod.gwf.hfb.get_hfb_spd(
        ds, gdf, hydchr=1 / 100.0, elevation=-25.0
    )
    assert _normalize_spd(elevation_spd) == _structured_diagonal_expected_spd(ds)
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
    for ilay in (0, 2):
        for idx in range(9):
            for cellid1, cellid2 in (
                ((ilay, idx, idx), (ilay, idx + 1, idx)),
                ((ilay, idx + 1, idx + 1), (ilay, idx + 1, idx)),
            ):
                if ilay == 2:
                    # with layer 1 collapsed, depth 25.0 penetrates 15 of the
                    # 20 m of layer 2
                    layer_hydchr = _expected_partial_hydchr(
                        ds, cellid1, cellid2, 1 / 100.0, 0.75
                    )
                else:
                    layer_hydchr = 1 / 100.0
                expected_spd.append((cellid1, cellid2, layer_hydchr))
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


def _run_two_column_flow(ws, nlay, top, botm, hfb_spd):
    """Steady two-column CHD model; returns the flow crossing the shared face."""
    sim = flopy.mf6.MFSimulation(
        sim_name="t", sim_ws=str(ws), exe_name=nlmod.util.get_exe_path("mf6")
    )
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-9, inner_dvclose=1e-10)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="t", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=1, ncol=2, delr=100.0, delc=100.0, top=top, botm=botm
    )
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfic(gwf, strt=1.0)
    chd_spd = [[(ilay, 0, 0), 1.0] for ilay in range(nlay)]
    chd_spd += [[(ilay, 0, 1), 0.0] for ilay in range(nlay)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)
    flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: hfb_spd})
    flopy.mf6.ModflowGwfoc(
        gwf, budget_filerecord="t.cbc", saverecord=[("BUDGET", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    assert success
    cbc = flopy.utils.CellBudgetFile(str(ws / "t.cbc"))
    chd_flows = cbc.get_data(text="CHD")[0]
    return chd_flows["q"][chd_flows["q"] > 0].sum()


def test_get_hfb_spd_partial_penetration_matches_resolved_barrier(tmp_path):
    # A barrier penetrating the upper half of a single 10-m layer. The reference run
    # resolves that layer into a walled and an open sublayer; the equivalent
    # single-layer HYDCHR produced by get_hfb_spd must reproduce the reference flow.
    ds = util.get_ds_structured(
        extent=[0, 200, 0, 100],
        model_name="hfb_partial",
        top=10.0,
        botm=[0.0],
        kh=10.0,
        kv=1.0,
    )
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(100, 0), (100, 100)])]})
    spd = nlmod.gwf.hfb.get_hfb_spd(ds, gdf, hydchr=1 / 100.0, depth=5.0)
    spd = [[cellid1, cellid2, float(hydchr)] for cellid1, cellid2, hydchr in spd]
    assert len(spd) == 1
    q_equivalent = _run_two_column_flow(
        tmp_path / "eq", nlay=1, top=10.0, botm=[0.0], hfb_spd=spd
    )
    q_reference = _run_two_column_flow(
        tmp_path / "ref",
        nlay=2,
        top=10.0,
        botm=[5.0, 0.0],
        hfb_spd=[[(0, 0, 0), (0, 0, 1), 1 / 100.0]],
    )
    assert q_equivalent == pytest.approx(q_reference, rel=0.01)
