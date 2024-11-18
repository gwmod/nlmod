import os

import geopandas as gpd
import pandas as pd

import nlmod


def test_gdf_to_seasonal_pkg():
    model_name = "sw"
    model_ws = os.path.join("data", model_name)
    extent = [119000, 120000, 523000, 524000]
    ds = nlmod.get_ds(extent, model_ws=model_ws, model_name=model_name)
    ds = nlmod.time.set_ds_time(ds, time=[365.0], start=pd.Timestamp.today())
    gdf = nlmod.gwf.surface_water.get_gdf(ds)

    sim = nlmod.sim.sim(ds)
    nlmod.sim.tdis(ds, sim)
    nlmod.sim.ims(sim)
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.dis(ds, gwf)
    nlmod.gwf.npf(ds, gwf)
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)
    nlmod.gwf.oc(ds, gwf)

    nlmod.gwf.surface_water.gdf_to_seasonal_pkg(gdf, gwf, ds, pkg="DRN")


def test_get_seaonal_timeseries():
    extent = [119000, 120000, 523000, 524000]
    ds = nlmod.get_ds(extent)
    time = pd.date_range("2020", "2025", freq="MS")
    ds = nlmod.time.set_ds_time(ds, start="2019", time=time)
    s = nlmod.gwf.surface_water.get_seaonal_timeseries(ds, 1.0, 0.0)
    assert s.index[0] <= pd.to_datetime(ds.time.start)
    assert s.index[-1] >= ds.time[-1]


def test_gdf_lake():
    model_name = "la"
    model_ws = os.path.join("data", model_name)
    ds = nlmod.get_ds(
        [170000, 171000, 550000, 551000], model_ws=model_ws, model_name=model_name
    )
    ds = nlmod.time.set_ds_time(ds, time=[1], start=pd.Timestamp.today())
    ds = nlmod.dims.refine(ds)

    sim = nlmod.sim.sim(ds)
    nlmod.sim.tdis(ds, sim)
    nlmod.sim.ims(sim)
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.dis(ds, gwf)

    ds["evap"] = (("time",), [0.0004])

    # add lake with outlet and evaporation
    gdf_lake = gpd.GeoDataFrame(
        {
            "name": ["lake_0", "lake_0", "lake_1"],
            "strt": [1.0, 1.0, 2.0],
            "clake": [10.0, 10.0, 10.0],
            "EVAPORATION": ["evap", "evap", "evap"],
            "lakeout": ["lake_1", "lake_1", None],
            "outlet_invert": ["use_elevation", "use_elevation", None],
        },
        index=[14, 15, 16],
    )

    nlmod.gwf.lake_from_gdf(gwf, gdf_lake, ds, boundname_column="name")

    # remove lake package
    gwf.remove_package("LAK_0")

    # add lake with outlet and inflow
    ds["inflow"] = (("time",), [100.0])
    gdf_lake = gpd.GeoDataFrame(
        {
            "name": ["lake_0", "lake_0", "lake_1"],
            "strt": [1.0, 1.0, 2.0],
            "clake": [10.0, 10.0, 10.0],
            "INFLOW": ["inflow", "inflow", None],
            "lakeout": [
                "lake_1",
                "lake_1",
                -1,
            ],  # lake 0 overflows in lake 1, the outlet from lake 1 is removed from the model
            "outlet_invert": [0, 0, None],
        },
        index=[14, 15, 16],
    )

    nlmod.gwf.lake_from_gdf(gwf, gdf_lake, ds, boundname_column="name")
