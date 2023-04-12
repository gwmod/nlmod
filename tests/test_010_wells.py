import pandas as pd
import nlmod


def get_model_ds():
    kh = [10, 0.1, 20]
    kv = [0.5 * k for k in kh]

    ds = nlmod.get_ds(
        [-500, 500, -500, 500],
        delr=10.0,
        top=0.0,
        botm=[-10, -15, -30],
        kh=kh,
        kv=kv,
        model_ws="./scratch_model",
        model_name="from_scratch",
    )

    ds = nlmod.time.set_ds_time(ds, time=pd.Timestamp.today())

    return ds


def get_sim_and_gwf(ds=None):
    if ds is None:
        ds = get_model_ds()
    sim = nlmod.sim.sim(ds)
    nlmod.sim.tdis(ds, sim)
    nlmod.sim.ims(sim, complexity="SIMPLE")
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.dis(ds, gwf)
    nlmod.gwf.npf(ds, gwf)
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)
    nlmod.gwf.oc(ds, gwf, save_head=True)
    return sim, gwf


def test_wel_from_df():
    wells = pd.DataFrame(columns=["x", "y", "top", "botm", "Q"], index=range(2))
    wells.loc[0] = 100, -50, -5, -10, -100.0
    wells.loc[1] = 200, 150, -20, -30, -300.0

    sim, gwf = get_sim_and_gwf()
    nlmod.gwf.wells.wel_from_df(wells, gwf)


def test_maw_from_df():
    wells = pd.DataFrame(columns=["x", "y", "top", "botm", "rw", "Q"], index=range(2))
    wells.loc[0] = 100, -50, -5, -10, 0.1, -100.0
    wells.loc[1] = 200, 150, -20, -30, 0.1, -300.0

    sim, gwf = get_sim_and_gwf()
    nlmod.gwf.wells.maw_from_df(wells, gwf)
