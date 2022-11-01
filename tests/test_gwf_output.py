import os
import tempfile

import nlmod
import numpy as np
from nlmod.gwf import get_heads_da
from nlmod.mdims import refine

tmpdir = tempfile.gettempdir()
tst_model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data"
)


def test_create_small_model_grid_only(tmpdir, model_name="test"):
    model_name = "test"
    extent = [98700.0, 99000.0, 489500.0, 489700.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, regis_botm_layer="KRz5", use_regis=True, use_geotop=True
    )
    model_ws = os.path.join(tmpdir, model_name)
    ds = nlmod.mdims.to_model_ds(
        regis_geotop_ds, model_name, model_ws, delr=100.0, delc=100.0
    )
    assert ds.dims["layer"] == 5

    ds = nlmod.mdims.set_ds_time(
        ds,
        start_time="2015-1-1",
        steady_state=False,
        steady_start=True,
        transient_timesteps=2,
    )

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
    nlmod.gwf.npf(ds, gwf)

    # Create the initial conditions package
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)
    nlmod.gwf.oc(ds, gwf)

    ds.update(nlmod.mgrid.mask_model_edge(ds, ds["idomain"]))
    nlmod.gwf.chd(ds, gwf, chd="edge_mask", head="starting_head")

    nlmod.sim.write_and_run(gwf, ds)

    heads_correct = np.ones((3, 5, 2, 3))
    heads_correct[:, 3, :, 1:] = np.nan

    da = get_heads_da(ds=ds, gwf=None, fname_hds=None)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    da = get_heads_da(ds=None, gwf=gwf, fname_hds=None)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")
    da = get_heads_da(ds=ds, gwf=None, fname_hds=fname_hds)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")
    da = get_heads_da(ds=None, gwf=gwf, fname_hds=fname_hds)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

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
    (_,) = nlmod.sim.tdis(ds_unstr, sim)

    # create ims
    nlmod.sim.ims(sim)

    # create groundwater flow model
    gwf_unstr = nlmod.gwf.gwf(ds_unstr, sim)

    # Create discretization
    nlmod.gwf.dis(ds_unstr, gwf_unstr)

    # create node property flow
    nlmod.gwf.npf(ds_unstr, gwf_unstr)

    # Create the initial conditions package
    nlmod.gwf.ic(ds_unstr, gwf_unstr, starting_head=1.0)
    nlmod.gwf.oc(ds_unstr, gwf_unstr)

    ds_unstr.update(nlmod.mgrid.mask_model_edge(ds_unstr, ds_unstr["idomain"]))
    nlmod.gwf.chd(ds_unstr, gwf_unstr, chd="edge_mask", head="starting_head")

    nlmod.sim.write_and_run(gwf_unstr, ds_unstr)

    heads_correct = np.ones((3, 5, 6))
    heads_correct[:, 3, [1, 2, 4, 5]] = np.nan

    da = get_heads_da(ds=ds_unstr, gwf=None, fname_hds=None)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    da = get_heads_da(ds=None, gwf=gwf_unstr, fname_hds=None)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")
    da = get_heads_da(ds=ds_unstr, gwf=None, fname_hds=fname_hds)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)

    fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")
    da = get_heads_da(ds=None, gwf=gwf_unstr, fname_hds=fname_hds)
    assert np.array_equal(da.values, heads_correct, equal_nan=True)
