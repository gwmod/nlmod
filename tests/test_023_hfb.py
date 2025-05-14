# ruff: noqa: D103
import flopy
import geopandas as gpd
import util
from shapely.geometry import LineString, Polygon

import nlmod


def test_get_hfb_spd():
    # this test also tests line2hfb
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


def test_polygon_to_hfb():
    ds = util.get_ds_vertex()
    ds = nlmod.time.set_ds_time(ds, "2023", time="2024")
    gwf = util.get_gwf(ds)

    coords = [(135, 230), (568, 170), (778, 670), (260, 786)]
    gdf = gpd.GeoDataFrame({"geometry": [Polygon(coords)]}).reset_index()

    hfb = nlmod.gwf.hfb.polygon_to_hfb(gdf, ds, hydchr=1 / 100.0, gwf=gwf)

    # also test the plot method
    ax = gdf.plot()
    nlmod.gwf.hfb.plot_hfb(hfb, gwf, ax=ax)
