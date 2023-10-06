import numpy as np
import pandas as pd
import nlmod
import util


def test_uzf_structured():
    # %% create model Dataset
    extent = [200_000, 202_000, 400_000, 403_000]
    ds = util.get_ds_structured(extent, top=0, botm=np.linspace(-1, -10, 10))

    time = pd.date_range("2022", "2023", freq="D")
    ds = nlmod.time.set_ds_time(ds, start="2021", time=time)

    ds.update(nlmod.read.knmi.get_recharge(ds, method="separate"))

    # %% generate sim and gwf
    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.sim.ims(sim)

    # Create discretization
    _ = nlmod.gwf.dis(ds, gwf)

    # create node property flow
    _ = nlmod.gwf.npf(ds, gwf)

    # Create the initial conditions package
    _ = nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    _ = nlmod.gwf.oc(ds, gwf)

    bhead = ds["botm"][1]
    cond = ds["area"] * 1
    _ = nlmod.gwf.ghb(ds, gwf, bhead=bhead, cond=cond, layer=len(ds.layer) - 1)

    # create recharge package
    _ = nlmod.gwf.uzf(ds, gwf)

    # %% run
    # _ = nlmod.sim.write_and_run(sim, ds)
