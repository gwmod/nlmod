import pandas as pd
import geopandas as gpd
import os
import nlmod


def test_gdf_to_seasonal_pkg():
    model_name = "sw"
    model_ws = os.path.join("data", model_name)
    ds = nlmod.get_ds(
        [170000, 171000, 550000, 551000], model_ws=model_ws, model_name=model_name
    )
    ds = nlmod.time.set_ds_time(ds, time=pd.Timestamp.today())
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


def test_gdf_lake():
    model_name = "la"
    model_ws = os.path.join("data", model_name)
    ds = nlmod.get_ds(
        [170000, 171000, 550000, 551000], model_ws=model_ws, model_name=model_name
    )
    ds = nlmod.time.set_ds_time(ds, time=pd.Timestamp.today())
    ds = nlmod.dims.refine(ds)

    sim = nlmod.sim.sim(ds)
    nlmod.sim.tdis(ds, sim)
    nlmod.sim.ims(sim)
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.dis(ds, gwf)

    ds['evap'] = (('time',), [0.0004])

    gdf_lake = gpd.GeoDataFrame(
        {
            "name": ["0", "0", "1"],
            "lakeno": [0, 0, 1],
            "strt": [1.0, 1.0, 2.0],
            "clake": [10.0, 10.0, 10.0],
            'EVAPORATION': ['evap', 'evap', 'evap'],
            "lakeout": [1, 1, None],
            "outlet_invert": [0, 0, None],
        },
        index=[14, 15, 16],
    )

    # add lake to groundwaterflow model
    nlmod.gwf.lake_from_gdf(
        gwf, gdf_lake, ds, boundname_column="name", recharge=False)
