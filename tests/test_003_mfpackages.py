# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 16:24:03 2021.

@author: oebbe
"""
import test_001_model

import nlmod


def sim_tdis_gwf_ims_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

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
    ds = test_001_model.get_ds_from_cache("small_model")

    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)

    nlmod.gwf.dis(ds, gwf)


def npf_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    nlmod.gwf.dis(ds)
    nlmod.gwf.npf(ds, gwf)


def oc_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    nlmod.gwf.oc(ds, gwf)


def sto_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    nlmod.gwf.sto(ds, gwf)


def ghb_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("full_sea_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    nlmod.gwf.ghb(ds, gwf, "surface_water")


def rch_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("full_sea_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    nlmod.gwf.rch(ds, gwf)


def drn_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("full_sea_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)
    nlmod.gwf.surface_drain_from_ds(ds, gwf, 1.0)


def chd_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    _ = nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # add constant head cells at model boundaries
    ds.update(nlmod.grid.mask_model_edge(ds, ds["idomain"]))
    nlmod.gwf.chd(ds, gwf, chd="edge_mask", head="starting_head")
