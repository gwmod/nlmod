import os
import tempfile

import nlmod
import pytest
import xarray as xr

tmpdir = tempfile.gettempdir()
tst_model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data"
)


def test_model_directories(tmpdir):
    model_ws = os.path.join(tmpdir, "test_model")
    figdir, cachedir = nlmod.util.get_model_dirs(model_ws)

    return model_ws, figdir, cachedir


def test_ds_time_steady(tmpdir, modelname="test"):
    model_ws = os.path.join(tmpdir, "test_model")
    ds = nlmod.mdims.set_ds_attrs(xr.Dataset(), modelname, model_ws)
    ds = nlmod.mdims.set_ds_time(ds, start_time="2015-1-1", steady_state=True)
    return ds


def test_ds_time_transient(tmpdir, modelname="test"):
    model_ws = os.path.join(tmpdir, "test_model")
    ds = nlmod.mdims.set_ds_attrs(xr.Dataset(), modelname, model_ws)
    ds = nlmod.mdims.set_ds_time(
        ds,
        start_time="2015-1-1",
        steady_state=False,
        steady_start=True,
        transient_timesteps=10,
    )
    return ds


@pytest.mark.slow
def test_create_seamodel_grid_only_without_northsea(tmpdir, model_name="test"):
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    # extent, _, _ = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )

    ds = nlmod.mdims.to_model_ds(
        regis_geotop_ds, model_name, str(tmpdir), delr=100.0, delc=100.0
    )

    ds = nlmod.mdims.set_ds_time(
        ds,
        start_time="2015-1-1",
        steady_state=False,
        steady_start=True,
        transient_timesteps=10,
    )

    # save ds
    ds.to_netcdf(os.path.join(tst_model_dir, "basic_sea_model.nc"))

    return ds


@pytest.mark.slow
def test_create_small_model_grid_only(tmpdir, model_name="test"):
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
        transient_timesteps=10,
    )

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.gwf.ims(sim)

    # Create discretization
    _ = nlmod.gwf.dis(ds, gwf)

    # save ds
    ds.to_netcdf(os.path.join(tst_model_dir, "small_model.nc"))

    return ds, gwf


@pytest.mark.slow
def test_create_sea_model_grid_only(tmpdir, model_name="test"):
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    model_ws = os.path.join(tmpdir, model_name)
    ds = nlmod.mdims.to_model_ds(
        regis_geotop_ds, model_name, model_ws, delr=100.0, delc=100.0
    )

    ds = nlmod.mdims.set_ds_time(
        ds,
        start_time="2015-1-1",
        steady_state=False,
        steady_start=True,
        transient_timesteps=10,
    )

    # save ds
    ds.to_netcdf(os.path.join(tst_model_dir, "sea_model_grid.nc"))

    return ds


@pytest.mark.slow
def test_create_sea_model_grid_only_delr_delc_50(tmpdir, model_name="test"):
    ds = test_ds_time_transient(tmpdir)
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 50.0, 50.0)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    model_ws = os.path.join(tmpdir, model_name)
    ds = nlmod.mdims.to_model_ds(
        regis_geotop_ds, model_name, model_ws, delr=50.0, delc=50.0
    )

    # save ds
    ds.to_netcdf(os.path.join(tst_model_dir, "sea_model_grid_50.nc"))

    return ds


@pytest.mark.slow
def test_create_sea_model(tmpdir):
    ds = xr.open_dataset(
        os.path.join(tst_model_dir, "basic_sea_model.nc"), mask_and_scale=False
    )
    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.gwf.ims(sim)

    # Create discretization
    _ = nlmod.gwf.dis(ds, gwf)

    # create node property flow
    _ = nlmod.gwf.npf(ds, gwf)

    # Create the initial conditions package
    _ = nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    _ = nlmod.gwf.oc(ds, gwf)

    # voeg grote oppervlaktewaterlichamen toe
    da_name = "surface_water"
    ds.update(nlmod.read.rws.get_surface_water(ds, da_name))
    _ = nlmod.gwf.ghb(ds, gwf, da_name)

    # surface level drain
    ds.update(nlmod.read.ahn.get_ahn(ds))
    _ = nlmod.gwf.surface_drain_from_ds(ds, gwf)

    # add constant head cells at model boundaries
    ds.update(nlmod.gwf.constant_head.chd_at_model_edge(ds, ds["idomain"]))
    _ = nlmod.gwf.chd(ds, gwf, head="starting_head")

    # add knmi recharge to the model datasets
    ds.update(nlmod.read.knmi.get_recharge(ds))
    # create recharge package
    _ = nlmod.gwf.rch(ds, gwf)

    _ = nlmod.sim.write_and_run(sim, ds)

    return ds, gwf


@pytest.mark.slow
def test_create_sea_model_perlen_list(tmpdir):
    ds = xr.open_dataset(os.path.join(tst_model_dir, "basic_sea_model.nc"))

    # create transient with perlen list
    perlen = [3650, 14, 10, 11]  # length of the time steps
    transient_timesteps = 3

    # update current ds with new time dicretisation
    model_ws = os.path.join(tmpdir, "test_model")
    new_ds = nlmod.mdims.set_ds_attrs(xr.Dataset(), "test", model_ws)
    new_ds = nlmod.mdims.set_ds_time(
        new_ds,
        start_time=ds.time.start,
        steady_state=False,
        steady_start=True,
        perlen=perlen,
        transient_timesteps=transient_timesteps,
    )

    # modfiy time
    ds = ds.drop_dims("time")
    ds.update(new_ds)

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.gwf.ims(sim)

    # Create discretization
    nlmod.gwf.dis(ds, gwf)

    # create node property flow
    nlmod.gwf.npf(ds, gwf)

    # Create the initial conditions package
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    nlmod.gwf.oc(ds, gwf)

    # voeg grote oppervlaktewaterlichamen toe
    da_name = "surface_water"
    ds.update(nlmod.read.rws.get_surface_water(ds, da_name))
    nlmod.gwf.ghb(ds, gwf, da_name)

    # surface level drain
    ds.update(nlmod.read.ahn.get_ahn(ds))
    nlmod.gwf.surface_drain_from_ds(ds, gwf)

    # add constant head cells at model boundaries
    ds.update(nlmod.gwf.constant_head.chd_at_model_edge(ds, ds["idomain"]))
    nlmod.gwf.chd(ds, gwf, head="starting_head")

    # add knmi recharge to the model datasets
    ds.update(nlmod.read.knmi.get_recharge(ds))
    # create recharge package
    nlmod.gwf.rch(ds, gwf)

    nlmod.sim.write_and_run(sim, ds)

    return ds, gwf


@pytest.mark.slow
def test_create_sea_model_perlen_14(tmpdir):
    ds = xr.open_dataset(os.path.join(tst_model_dir, "basic_sea_model.nc"))

    # create transient with perlen list
    perlen = 14  # length of the time steps
    transient_timesteps = 3

    # update current ds with new time dicretisation
    model_ws = os.path.join(tmpdir, "test_model")
    new_ds = nlmod.mdims.set_ds_attrs(xr.Dataset(), "test", model_ws)
    new_ds = nlmod.mdims.set_ds_time(
        new_ds,
        start_time=ds.time.start,
        steady_state=False,
        steady_start=True,
        perlen=perlen,
        transient_timesteps=transient_timesteps,
    )

    ds = ds.drop_dims("time")
    ds.update(new_ds)

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.gwf.ims(sim)

    # Create discretization
    nlmod.gwf.dis(ds, gwf)

    # create node property flow
    nlmod.gwf.npf(ds, gwf)

    # Create the initial conditions package
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    nlmod.gwf.oc(ds, gwf)

    # voeg grote oppervlaktewaterlichamen toe
    da_name = "surface_water"
    ds.update(nlmod.read.rws.get_surface_water(ds, da_name))
    nlmod.gwf.ghb(ds, gwf, da_name)

    # surface level drain
    ds.update(nlmod.read.ahn.get_ahn(ds))
    nlmod.gwf.surface_drain_from_ds(ds, gwf)

    # add constant head cells at model boundaries
    ds.update(nlmod.gwf.constant_head.chd_at_model_edge(ds, ds["idomain"]))
    nlmod.gwf.chd(ds, gwf, head="starting_head")

    # add knmi recharge to the model datasets
    ds.update(nlmod.read.knmi.get_recharge(ds))
    # create recharge package
    nlmod.gwf.rch(ds, gwf)

    nlmod.sim.write_and_run(sim, ds)

    return ds, gwf


# obtaining the test models
def test_get_ds_from_cache(name="small_model"):

    ds = xr.open_dataset(os.path.join(tst_model_dir, name + ".nc"))

    return ds


# other functions
def _check_tmpdir(tmpdir):

    # pytest uses a LocalPath object for the tmpdir argument when testing
    # this function convert a LocalPath object to a string

    if isinstance(tmpdir, str):
        return tmpdir
    else:
        return str(tmpdir)
