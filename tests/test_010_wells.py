import pandas as pd
import pytest

import nlmod


def get_model_ds(time=None, start=None):
    if time is None:
        time = [1]
    if start is None:
        start = pd.Timestamp.today()
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

    ds = nlmod.time.set_ds_time(ds, time=time, start=start)

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


def test_wel_from_df_no_multiplier():
    wells = pd.DataFrame(columns=["x", "y", "top", "botm", "Q"], index=range(2))
    wells.loc[0] = 100, -50, -5, -10, -100.0
    wells.loc[1] = 200, 150, -20, -30, -300.0

    sim, gwf = get_sim_and_gwf()
    nlmod.gwf.wells.wel_from_df(wells, gwf, auxmultname=None)


def test_maw_from_df():
    wells = pd.DataFrame(columns=["x", "y", "top", "botm", "rw", "Q"], index=range(2))
    wells.loc[0] = 100, -50, -5, -10, 0.1, -100.0
    wells.loc[1] = 200, 150, -20, -30, 0.1, -300.0

    sim, gwf = get_sim_and_gwf()
    nlmod.gwf.wells.maw_from_df(wells, gwf)


def test_wel_from_df_transient():
    time = pd.date_range("2000", "2001", freq="MS")
    ds = get_model_ds(start="1990", time=time)

    Q = pd.DataFrame(-100.0, index=time, columns=["Q1", "Q2"])
    wells = pd.DataFrame(columns=["x", "y", "top", "botm", "Q"], index=range(2))
    wells.loc[0] = 100, -50, -5, -10, "Q1"
    wells.loc[1] = 200, 150, -20, -30, "Q2"

    sim, gwf = get_sim_and_gwf()
    wel = nlmod.gwf.wells.wel_from_df(wells, gwf)

    nlmod.time.dataframe_to_flopy_timeseries(Q, ds, package=wel)


def test_maw_from_df_transient():
    time = pd.date_range("2000", "2001", freq="MS")
    ds = get_model_ds(start="1990", time=time)

    Q = pd.DataFrame(-100.0, index=time, columns=["Q1", "Q2"])
    wells = pd.DataFrame(columns=["x", "y", "top", "botm", "rw", "Q"], index=range(2))
    wells.loc[0] = 100, -50, -5, -10, 0.1, "Q1"
    wells.loc[1] = 200, 150, -20, -30, 0.1, "Q2"

    sim, gwf = get_sim_and_gwf()
    maw = nlmod.gwf.wells.maw_from_df(wells, gwf)

    nlmod.time.dataframe_to_flopy_timeseries(Q, ds, package=maw)


def test_maw_from_df_with_skin():
    wells = pd.DataFrame(
        columns=["x", "y", "top", "botm", "rw", "Q", "hk_skin", "radius_skin"],
        index=range(2),
    )
    wells.loc[0] = 100, -50, -5, -10, 0.1, -100.0, 1.0, 0.15
    wells.loc[1] = 200, 150, -20, -30, 0.1, -300.0, 2.0, 0.2

    sim, gwf = get_sim_and_gwf()
    maw = nlmod.gwf.wells.maw_from_df(
        wells, gwf, hk_skin="hk_skin", radius_skin="radius_skin", condeqn="SKIN"
    )
    # 2 wells × 1 layer each (based on actual well depths)
    assert len(maw.connectiondata.array) == 2


def test_maw_from_df_grouped_wells():
    wells = pd.DataFrame(
        columns=["x", "y", "top", "botm", "rw", "Q", "group"], index=range(4)
    )
    wells.loc[0] = 100, -50, -5, -10, 0.1, -200.0, "group1"
    wells.loc[1] = 150, -50, -5, -10, 0.1, -200.0, "group1"
    wells.loc[2] = 200, 150, -20, -30, 0.1, -300.0, "group2"
    wells.loc[3] = 250, 150, -20, -30, 0.1, -300.0, "group2"

    sim, gwf = get_sim_and_gwf()
    maw = nlmod.gwf.wells.maw_from_df(wells, gwf, group="group")

    assert maw.nmawwells.array == 2  # 2 groups
    assert len(maw.perioddata.array[0]) == 2  # 2 rate settings


def test_maw_from_df_grouped_wells_mixed():
    wells = pd.DataFrame(
        columns=["x", "y", "top", "botm", "rw", "Q", "group"], index=range(4)
    )
    wells.loc[0] = 100, -50, -5, -10, 0.1, -200.0, "shared_group"
    wells.loc[1] = 150, -50, -5, -10, 0.1, -200.0, "shared_group"
    wells.loc[2] = 200, 150, -20, -30, 0.1, -300.0, ""  # empty group -> individual
    wells.loc[3] = 250, 150, -20, -30, 0.1, -400.0, ""  # empty group -> individual

    sim, gwf = get_sim_and_gwf()
    maw = nlmod.gwf.wells.maw_from_df(wells, gwf, group="group")

    assert maw.nmawwells.array == 3  # 1 shared group + 2 individual wells


def test_maw_from_df_grouped_wells_different_q_error():
    wells = pd.DataFrame(
        columns=["x", "y", "top", "botm", "rw", "Q", "group"], index=range(2)
    )
    wells.loc[0] = 100, -50, -5, -10, 0.1, -200.0, "group1"
    wells.loc[1] = 150, -50, -5, -10, 0.1, -300.0, "group1"  # Different Q

    sim, gwf = get_sim_and_gwf()

    with pytest.raises(ValueError, match="Group flow rate"):
        nlmod.gwf.wells.maw_from_df(wells, gwf, group="group")
