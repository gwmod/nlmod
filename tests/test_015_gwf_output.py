import os
import tempfile

import numpy as np
import pytest
import test_001_model

import nlmod
from nlmod.dims.grid import refine
from nlmod.gwf import get_budget_da, get_heads_da

tmpdir = tempfile.gettempdir()
tst_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

grberror = "Cannot create budget data-array without grid information."


def test_create_small_model_grid_only(tmpdir, model_name="test"):
    model_name = "test"
    extent = [98700.0, 99000.0, 489500.0, 489700.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, regis_botm_layer="KRz5", use_regis=True, use_geotop=True
    )
    model_ws = os.path.join(tmpdir, model_name)
    ds = nlmod.to_model_ds(
        regis_geotop_ds, model_name, model_ws, delr=100.0, delc=100.0
    )
    assert ds.sizes["layer"] == 5

    ds = nlmod.time.set_ds_time(ds, time=[1, 2, 3], start="2015-1-1", steady=[1, 0, 0])

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create ims
    nlmod.sim.ims(sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # Create discretization
    nlmod.gwf.dis(ds, gwf)

    # create node property flow
    nlmod.gwf.npf(ds, gwf, save_flows=True, save_specific_discharge=True)

    # Create the initial conditions package
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)
    nlmod.gwf.oc(ds, gwf)

    ds.update(nlmod.grid.mask_model_edge(ds))
    nlmod.gwf.chd(ds, gwf, mask="edge_mask", head="starting_head")

    nlmod.sim.write_and_run(sim, ds)

    heads_correct = np.ones((3, 5, 2, 3))
    heads_correct[:, 3, :, 1:] = np.nan

    da = get_heads_da(ds=ds, gwf=None, fname=None)  # ds
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    da = get_heads_da(ds=None, gwf=gwf, fname=None)  # gwf
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")
    grb_file = os.path.join(ds.model_ws, ds.model_name + ".dis.grb")
    da = get_heads_da(ds=None, gwf=None, fname=fname_hds, grb_file=grb_file)  # fname
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    # budget
    da = get_budget_da("CHD", ds=ds, gwf=None, fname=None)  # ds
    da = get_budget_da("CHD", ds=None, gwf=gwf, fname=None)  # gwf
    fname_cbc = os.path.join(ds.model_ws, ds.model_name + ".cbc")
    get_budget_da("CHD", ds=None, gwf=None, fname=fname_cbc, grb_file=grb_file)  # fname
    get_budget_da(
        "DATA-SPDIS", column="qz", ds=None, gwf=None, fname=fname_cbc, grb_file=grb_file
    )  # fname

    # unstructured
    ds_unstr = refine(
        ds,
        model_ws=None,
        refinement_features=None,
        exe_name=None,
        remove_nan_layers=True,
        model_coordinates=False,
    )

    # create simulation
    sim = nlmod.sim.sim(ds_unstr)

    # create time discretisation
    _ = nlmod.sim.tdis(ds_unstr, sim)

    # create ims
    nlmod.sim.ims(sim)

    # create groundwater flow model
    gwf_unstr = nlmod.gwf.gwf(ds_unstr, sim)

    # Create discretization
    nlmod.gwf.dis(ds_unstr, gwf_unstr)

    # create node property flow
    nlmod.gwf.npf(ds_unstr, gwf_unstr, save_flows=True, save_specific_discharge=True)

    # Create the initial conditions package
    nlmod.gwf.ic(ds_unstr, gwf_unstr, starting_head=1.0)
    nlmod.gwf.oc(ds_unstr, gwf_unstr)

    ds_unstr.update(nlmod.grid.mask_model_edge(ds_unstr))
    nlmod.gwf.chd(ds_unstr, gwf_unstr, mask="edge_mask", head="starting_head")

    nlmod.sim.write_and_run(sim, ds_unstr)

    heads_correct = np.ones((3, 5, 6))
    heads_correct[:, 3, [1, 2, 4, 5]] = np.nan

    da = get_heads_da(ds=ds_unstr, gwf=None, fname=None)  # ds
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    da = get_heads_da(ds=None, gwf=gwf_unstr, fname=None)  # gwf
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")
    grb_file = os.path.join(ds.model_ws, ds.model_name + ".disv.grb")
    da = get_heads_da(ds=None, gwf=None, fname=fname_hds, grb_file=grb_file)  # fname
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    # budget
    da = get_budget_da("CHD", ds=ds_unstr, gwf=None, fname=None)  # ds
    da = get_budget_da("CHD", ds=None, gwf=gwf_unstr, fname=None)  # gwf
    da = get_budget_da(
        "CHD", ds=None, gwf=None, fname=fname_cbc, grb_file=grb_file
    )  # fname
    _ = get_budget_da(
        "DATA-SPDIS", column="qz", ds=None, gwf=None, fname=fname_cbc, grb_file=grb_file
    )  # fname


def test_get_heads_da_from_file_structured_no_grb():
    fname_hds = "./tests/data/mf6output/structured/test.hds"
    with pytest.warns(UserWarning):
        nlmod.gwf.output.get_heads_da(fname=fname_hds)


def test_get_heads_da_from_file_structured_with_grb():
    fname_hds = "./tests/data/mf6output/structured/test.hds"
    grb_file = "./tests/data/mf6output/structured/test.dis.grb"
    nlmod.gwf.output.get_heads_da(fname=fname_hds, grb_file=grb_file)


def test_get_budget_da_from_file_structured_no_grb():
    fname_cbc = "./tests/data/mf6output/structured/test.cbc"
    with pytest.raises(ValueError, match=grberror):
        nlmod.gwf.output.get_budget_da("CHD", fname=fname_cbc)


def test_get_budget_da_from_file_structured_with_grb():
    fname_cbc = "./tests/data/mf6output/structured/test.cbc"
    grb_file = "./tests/data/mf6output/structured/test.dis.grb"
    nlmod.gwf.output.get_budget_da("CHD", fname=fname_cbc, grb_file=grb_file)


def test_get_heads_da_from_file_vertex_no_grb():
    fname_hds = "./tests/data/mf6output/vertex/test.hds"
    with pytest.warns(UserWarning):
        nlmod.gwf.output.get_heads_da(fname=fname_hds)


def test_get_heads_da_from_file_vertex_with_grb():
    fname_hds = "./tests/data/mf6output/vertex/test.hds"
    grb_file = "./tests/data/mf6output/vertex/test.disv.grb"
    nlmod.gwf.output.get_heads_da(fname=fname_hds, grb_file=grb_file)


def test_get_budget_da_from_file_vertex_no_grb():
    fname_cbc = "./tests/data/mf6output/vertex/test.cbc"
    with pytest.raises(ValueError, match=grberror):
        nlmod.gwf.output.get_budget_da("CHD", fname=fname_cbc)


def test_get_budget_da_from_file_vertex_with_grb():
    fname_cbc = "./tests/data/mf6output/vertex/test.cbc"
    grb_file = "./tests/data/mf6output/vertex/test.disv.grb"
    nlmod.gwf.output.get_budget_da("CHD", fname=fname_cbc, grb_file=grb_file)


def test_postprocess_head():
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    head = nlmod.gwf.get_heads_da(ds)

    nlmod.gwf.calculate_gxg(head)

    nlmod.gwf.get_gwl_from_wet_cells(head, botm=ds["botm"])

    nlmod.gwf.get_head_at_point(head, float(ds.x.mean()), float(ds.y.mean()), ds=ds)


def test_get_flow_residuals():
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    da = nlmod.gwf.output.get_flow_residuals(ds)
    assert "time" in da.dims
    da = nlmod.gwf.output.get_flow_residuals(ds, kstpkper=(0, 0))
    assert "time" not in da.dims


def test_get_flow_lower_face():
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    da = nlmod.gwf.output.get_flow_lower_face(ds)
    assert "time" in da.dims
    da = nlmod.gwf.output.get_flow_lower_face(ds, kstpkper=(0, 0))
    assert "time" not in da.dims
