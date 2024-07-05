import os

import flopy
import xarray as xr

import nlmod


def test_modpath():
    # start with runned model from test_001_model.test_create_sea_model
    model_ws = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    ds = xr.open_dataset(os.path.join(model_ws, "basic_sea_model.nc"))

    sim = flopy.mf6.MFSimulation.load("mfsim.nam", sim_ws=ds.model_ws)
    gwf = sim.get_model(ds.model_name)

    xy_start = [
        (float(ds.x.mean()), float(ds.y.mean())),
        (float(ds.x.mean()) + 100, float(ds.y.mean())),
    ]

    # create a modpath model
    mpf = nlmod.modpath.mpf(gwf)

    # create the basic modpath package
    nlmod.modpath.bas(mpf)

    # find the nodes for given xy
    nodes = nlmod.modpath.xy_to_nodes(xy_start, mpf, ds, layer=5)

    # create a particle tracking group at the cell faces
    pg = nlmod.modpath.pg_from_fdt(nodes)

    # create the modpath simulation file
    nlmod.modpath.sim(mpf, pg, "backward", gwf=gwf)

    # run modpath model
    nlmod.modpath.write_and_run(mpf)

    # test reading of pathline file
    nlmod.modpath.load_pathline_data(mpf)

    # get the nodes from a package
    nodes = nlmod.modpath.package_to_nodes(gwf, "GHB", mpf)

    # get nodes of all cells in the top modellayer
    nodes = nlmod.modpath.layer_to_nodes(mpf, 0)
