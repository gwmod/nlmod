import nlmod
import pandas as pd


def test_gdf_to_seasonal_pkg():
    ds = nlmod.get_ds([170000, 171000, 550000, 551000], model_ws="sw", model_name="sw")
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
