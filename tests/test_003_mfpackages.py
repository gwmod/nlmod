# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 16:24:03 2021.

@author: oebbe
"""
import nlmod
import pytest

import test_001_model


def test_sim_tdis_gwf_ims_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create ims
    _ = nlmod.sim.ims(sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    return sim, gwf


def dis_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("small_model")

    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)

    dis = nlmod.gwf.dis(ds, gwf)

    return dis


@pytest.mark.slow
def disv_from_ds(tmpdir):
    ds, gwf, gridprops = test_001_model.test_create_inf_panden_model(tmpdir)

    disv = nlmod.gwf.disv(ds, gwf, gridprops)

    return disv


def npf_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("small_model")
    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)
    nlmod.gwf.dis(ds)
    npf = nlmod.gwf.npf(ds, gwf)

    return npf


def oc_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("small_model")
    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)
    oc = nlmod.gwf.oc(ds, gwf)

    return oc


def sto_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("small_model")
    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)
    sto = nlmod.gwf.sto(ds, gwf)

    return sto


def ghb_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("full_sea_model")
    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    ghb = nlmod.gwf.ghb(ds, gwf, "surface_water")

    return ghb


def rch_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("full_sea_model")
    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    rch = nlmod.gwf.rch(ds, gwf)

    return rch


def drn_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("full_sea_model")
    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    drn = nlmod.gwf.surface_drain_from_ds(ds, gwf)

    return drn


def chd_from_ds(tmpdir):
    ds = test_001_model.test_get_ds_from_cache("small_model")
    _, gwf = test_sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    _ = nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # add constant head cells at model boundaries
    ds.update(nlmod.mgrid.mask_model_edge(ds, ds["idomain"]))
    chd = nlmod.gwf.chd(ds, gwf, chd="edge_mask", head="starting_head")

    return chd
