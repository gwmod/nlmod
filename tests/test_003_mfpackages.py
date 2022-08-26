# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 16:24:03 2021.

@author: oebbe
"""

import datetime as dt

import nlmod
import pytest

import test_001_model


def test_sim_tdis_gwf_ims_from_model_ds(tmpdir):

    model_ds = test_001_model.test_get_model_ds_from_cache("basic_sea_model")

    # create simulation
    sim = nlmod.gwf.sim_from_model_ds(model_ds)

    # create time discretisation
    tdis = nlmod.gwf.tdis_from_model_ds(model_ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf_from_model_ds(model_ds, sim)

    # create ims
    ims = nlmod.gwf.ims_to_sim(sim)

    return sim, gwf


def dis_from_model_ds(tmpdir):

    model_ds = test_001_model.test_get_model_ds_from_cache("small_model")

    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)

    dis = nlmod.gwf.dis_from_model_ds(model_ds, gwf)

    return dis


@pytest.mark.slow
def disv_from_model_ds(tmpdir):

    model_ds, gwf, gridprops = test_001_model.test_create_inf_panden_model(tmpdir)

    disv = nlmod.gwf.disv_from_model_ds(model_ds, gwf, gridprops)

    return disv


def npf_from_model_ds(tmpdir):

    model_ds = test_001_model.test_get_model_ds_from_cache("small_model")
    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)
    nlmod.gwf.dis_from_model_ds(model_ds)

    npf = nlmod.gwf.npf_from_model_ds(model_ds, gwf)

    return npf


def oc_from_model_ds(tmpdir):

    model_ds = test_001_model.test_get_model_ds_from_cache("small_model")
    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)

    oc = nlmod.gwf.oc_from_model_ds(model_ds, gwf)

    return oc


def sto_from_model_ds(tmpdir):

    model_ds = test_001_model.test_get_model_ds_from_cache("small_model")
    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)
    sto = nlmod.gwf.sto_from_model_ds(model_ds, gwf)

    return sto


def ghb_from_model_ds(tmpdir):

    model_ds = test_001_model.test_get_model_ds_from_cache("full_sea_model")
    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)
    nlmod.gwf.dis_from_model_ds(model_ds, gwf)

    ghb = nlmod.gwf.ghb_from_model_ds(model_ds, gwf, "surface_water")

    return ghb


def rch_from_model_ds(tmpdir):
    model_ds = test_001_model.test_get_model_ds_from_cache("full_sea_model")
    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)
    nlmod.gwf.dis_from_model_ds(model_ds, gwf)

    rch = nlmod.gwf.rch_from_model_ds(model_ds, gwf)

    return rch


def drn_from_model_ds(tmpdir):
    model_ds = test_001_model.test_get_model_ds_from_cache("full_sea_model")
    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)
    nlmod.gwf.dis_from_model_ds(model_ds, gwf)

    drn = nlmod.gwf.surface_drain_from_model_ds(model_ds, gwf)

    return drn


def chd_from_model_ds(tmpdir):
    model_ds = test_001_model.test_get_model_ds_from_cache("small_model")
    sim, gwf = test_sim_tdis_gwf_ims_from_model_ds(tmpdir)
    nlmod.gwf.dis_from_model_ds(model_ds, gwf)

    nlmod.gwf.ic_from_model_ds(model_ds, gwf, starting_head=1.0)

    # add constant head cells at model boundaries
    model_ds.update(
        nlmod.gwf.constant_head.get_chd_at_model_edge(model_ds, model_ds["idomain"])
    )
    chd = nlmod.gwf.chd_from_model_ds(model_ds, gwf, head="starting_head")

    return chd
