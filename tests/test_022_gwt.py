import os
import tempfile

import pandas as pd
import xarray as xr

import nlmod


def test_gwt_model():
    extent = [103700, 106700, 527500, 528500]

    tmpdir = tempfile.gettempdir()
    model_name = "trnsprt_tst"
    model_ws = os.path.join(tmpdir, model_name)

    layer_model = nlmod.read.get_regis(extent, botm_layer="MSz1")
    # create a model ds
    ds = nlmod.to_model_ds(
        layer_model,
        model_name,
        model_ws,
        delr=100.0,
        delc=100.0,
        transport=True,
    )

    # add time discretisation
    time = pd.date_range("2000", "2023", freq="YS")
    ds = nlmod.time.set_ds_time(ds, start=time[0], time=time[1:])

    ds = nlmod.gwt.prepare.set_default_transport_parameters(
        ds, transport_type="chloride"
    )

    # We download the digital terrain model (AHN4)
    ahn = nlmod.read.ahn.get_ahn4(ds.extent)
    # calculate the average surface level in each cell
    ds["ahn"] = nlmod.resample.structured_da_to_ds(ahn, ds, method="average")

    # we then determine the part of each cell that is covered by sea from the original fine ahn
    ds["sea"] = nlmod.read.rws.calculate_sea_coverage(ahn, ds=ds, method="average")

    # download knmi recharge data
    knmi_ds = nlmod.read.knmi.get_recharge(ds, method="separate")

    # update model dataset
    ds.update(knmi_ds)

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    nlmod.sim.tdis(ds, sim)

    # create ims
    ims = nlmod.sim.ims(sim, complexity="MODERATE")

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # Create discretization
    nlmod.gwf.dis(ds, gwf)

    # create node property flow
    nlmod.gwf.npf(ds, gwf)

    # create storage
    nlmod.gwf.sto(ds, gwf)

    # Create the initial conditions package
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    nlmod.gwf.oc(ds, gwf)

    # build ghb package
    nlmod.gwf.ghb(
        ds,
        gwf,
        bhead=0.0,
        cond=ds["sea"] * ds["area"] / 1.0,
        auxiliary=18_000.0,
    )

    # build surface level drain package
    elev = ds["ahn"].where(ds["sea"] == 0)
    nlmod.gwf.surface_drain_from_ds(ds, gwf, elev=elev, resistance=10.0)

    # create recharge package
    nlmod.gwf.rch(ds, gwf, mask=ds["sea"] == 0)

    # create evapotranspiration package
    nlmod.gwf.evt(ds, gwf, mask=ds["sea"] == 0)

    # BUY: buoyancy package for GWF model
    nlmod.gwf.buy(ds, gwf)

    # GWT: groundwater transport model
    gwt = nlmod.gwt.gwt(ds, sim)

    # add IMS for GWT model and register it
    ims = nlmod.sim.ims(sim, pname="ims_gwt", filename=f"{gwt.name}.ims")
    nlmod.sim.register_ims_package(sim, gwt, ims)

    # DIS: discretization package
    nlmod.gwt.dis(ds, gwt)

    # IC: initial conditions package
    nlmod.gwt.ic(ds, gwt, strt=18_000)

    # ADV: advection package
    nlmod.gwt.adv(ds, gwt)

    # DSP: dispersion package
    nlmod.gwt.dsp(ds, gwt)

    # MST: mass transfer package
    nlmod.gwt.mst(ds, gwt)

    # SSM: source-sink mixing package
    nlmod.gwt.ssm(ds, gwt)

    # OC: output control
    nlmod.gwt.oc(ds, gwt)

    # GWF-GWT Exchange
    nlmod.gwt.gwfgwt(ds, sim)

    nlmod.sim.write_and_run(sim, ds)

    h = nlmod.gwf.output.get_heads_da(ds)
    c = nlmod.gwt.output.get_concentration_da(ds)

    # calculate concentration at groundwater surface
    nlmod.gwt.get_concentration_at_gw_surface(c)

    # test isosurface: first elevation where 10_000 mg/l is reached
    z = xr.DataArray(
        gwf.modelgrid.zcellcenters,
        coords={"layer": c.layer, "y": c.y, "x": c.x},
        dims=("layer", "y", "x"),
    )
    nlmod.layers.get_isosurface(c, z, 10_000.0)

    # Convert calculated heads to equivalent freshwater heads, and vice versa
    hf = nlmod.gwt.output.freshwater_head(ds, h, c)
    hp = nlmod.gwt.output.pointwater_head(ds, hf, c)

    xr.testing.assert_allclose(h, hp)
