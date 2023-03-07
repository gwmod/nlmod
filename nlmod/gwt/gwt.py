import logging
import numbers

import numpy as np
import xarray as xr
import flopy

from ..dims import grid

logger = logging.getLogger(__name__)


def gwt(ds, sim, **kwargs):
    """create groundwater transport model from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start,
        perlen, nstp, tsmult
    sim : flopy MFSimulation
        simulation object.

    Returns
    -------
    gwt : flopy ModflowGwt
        groundwater transport object.
    """

    # start creating model
    logger.info("creating modflow GWT")

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"gwt_{ds.model_name}.nam"

    gwt = flopy.mf6.ModflowGwt(
        sim, modelname=f"gwt_{ds.model_name}", model_nam_file=model_nam_file, **kwargs
    )

    return gwt


def dis(ds, gwt, length_units="METERS", pname="dis", **kwargs):
    """get discretisation package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwt : flopy ModflowGwf
        groundwater transport object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    dis : TYPE
        discretisation package.
    """
    logger.info("creating modflow DIS")

    if ds.gridtype == "vertex":
        return disv(ds, gwt, length_units=length_units)

    # check attributes
    for att in ["delr", "delc"]:
        if att in ds.attrs:
            if isinstance(ds.attrs[att], np.float32):
                ds.attrs[att] = float(ds.attrs[att])

    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        xorigin = ds.attrs["xorigin"]
        yorigin = ds.attrs["yorigin"]
        angrot = ds.attrs["angrot"]
    else:
        xorigin = ds.extent[0]
        yorigin = ds.extent[2]
        angrot = 0.0

    dis = flopy.mf6.ModflowGwtdis(
        gwt,
        pname=pname,
        length_units=length_units,
        xorigin=xorigin,
        yorigin=yorigin,
        angrot=angrot,
        nlay=ds.dims["layer"],
        nrow=ds.dims["y"],
        ncol=ds.dims["x"],
        delr=ds["delr"].values if "delr" in ds else ds.delr,
        delc=ds["delc"].values if "delc" in ds else ds.delc,
        top=ds["top"].data,
        botm=ds["botm"].data,
        idomain=ds["idomain"].data,
        filename=f"gwt_{ds.model_name}.dis",
        **kwargs,
    )

    return dis

def disv(ds, gwt, length_units="METERS", pname="disv", **kwargs):
    """create transport discretisation vertices package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwt : flopy ModflowGwt
        groundwater transport object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    disv : flopy ModflowGwtdisv
        disv package
    """
    logger.info("creating modflow DISV")

    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        xorigin = ds.attrs["xorigin"]
        yorigin = ds.attrs["yorigin"]
        angrot = ds.attrs["angrot"]
    elif "extent" in ds.attrs.keys():
        xorigin = ds.attrs["extent"][0]
        yorigin = ds.attrs["extent"][2]
        angrot = 0.0
    else:
        xorigin = 0.0
        yorigin = 0.0
        angrot = 0.0

    vertices = grid.get_vertices_from_ds(ds)
    cell2d = grid.get_cell2d_from_ds(ds)
    disv = flopy.mf6.ModflowGwtdisv(
        gwt,
        idomain=ds["idomain"].data,
        xorigin=xorigin,
        yorigin=yorigin,
        length_units=length_units,
        angrot=angrot,
        nlay=len(ds.layer),
        ncpl=len(ds.icell2d),
        nvert=len(ds.iv),
        top=ds["top"].data,
        botm=ds["botm"].data,
        vertices=vertices,
        cell2d=cell2d,
        pname=pname,
        **kwargs,
    )
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        gwt.modelgrid.set_coord_info(xoff=xorigin, yoff=yorigin, angrot=angrot)

    return disv


def adv(ds, gwt, scheme="UPSTREAM", **kwargs):
    """create advection package for groundwater transport model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwt : flopy ModflowGwt
        groundwater transport object.
    scheme : str, optional
        advection scheme to use, by default "UPSTREAM", options are
        ("UPSTREAM", "CENTRAL", "TVD").

    Returns
    -------
    adv : flopy ModflowGwtadv
        adv package
    """
    ds.attrs["advection_scheme"] = scheme
    adv = flopy.mf6.ModflowGwtadv(gwt, scheme=scheme, **kwargs)
    return adv


def dsp(ds, gwt, **kwargs):
    alpha_l = 1.0  # Longitudinal dispersivity ($m$)
    alpha_th = 0.1  # Transverse horizontal dispersivity ($m$)
    alpha_tv = 0.1  # Transverse vertical dispersivity ($m$)
    dsp = flopy.mf6.ModflowGwtdsp(
        gwt, xt3d_off=True, alh=alpha_l, ath1=alpha_th, atv=alpha_tv
    )
    return dsp


def ssm(ds, gwt, pkg_sources=None, **kwargs):
    logger.info("creating modflow SSM")

    if pkg_sources is not None:
        sources = [(ipkg, "AUX", "CONCENTRATION") for ipkg in pkg_sources]
    else:
        sources = kwargs.pop("sources")

    ssm = flopy.mf6.ModflowGwtssm(gwt, sources=sources, **kwargs)
    return ssm


def mst(ds, gwt, **kwargs):
    logger.info("creating modflow MST")

    porosity = kwargs.get("porosity", ds.porosity)
    mst = flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
    return mst


def cnc(ds, gwt, da_mask, da_conc, pname="cnc", **kwargs):
    logger.info("creating modflow CNC")

    cnc_rec = grid.da_to_reclist(ds, da_mask, col1=da_conc, layer=None)
    cnc_spd = {0: cnc_rec}
    cnc = flopy.mf6.ModflowGwtcnc(
        gwt, stress_period_data=cnc_spd, pname=pname, **kwargs
    )
    return cnc


def _set_record(conc, budget):
    record = []
    if isinstance(conc, bool):
        if conc:
            conc = "LAST"
        else:
            conc = None
    if conc is not None:
        record.append(("CONCENTRATION", conc))
    if isinstance(budget, bool):
        if budget:
            budget = "LAST"
        else:
            budget = None
    if budget is not None:
        record.append(("BUDGET", budget))
    return record


def oc(
    ds,
    gwt,
    save_concentration=True,
    save_budget=True,
    print_concentration=False,
    print_budget=False,
    pname="oc",
    **kwargs,
):
    logger.info("creating modflow OC")

    # Create the output control package
    concfile = f"gwt_{ds.model_name}.ucn"
    conc_filerecord = [concfile]
    budgetfile = f"gwt_{ds.model_name}.cbc"
    budget_filerecord = [budgetfile]
    saverecord = _set_record(save_concentration, save_budget)
    printrecord = _set_record(print_concentration, print_budget)

    oc = flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord=conc_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=saverecord,
        printrecord=printrecord,
        pname=pname,
        **kwargs,
    )
    return oc


def ic(ds, gwt, strt="chloride", pname="ic", **kwargs):
    """get initial condictions package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwt : flopy ModflowGwf
        groundwater transport object.
    strt : str, float or int, optional
        if type is int or float this is the starting concentration for all cells
        If the type is str the data variable from ds is used as starting
        concentration. The default is 'chloride'.
    pname : str, optional
        package name

    Returns
    -------
    ic : flopy ModflowGwtic
        ic package
    """
    logger.info("creating modflow IC")

    if isinstance(strt, str):
        pass
    elif isinstance(strt, numbers.Number):
        ds["gwt_strt"] = strt * xr.ones_like(ds["idomain"])
        # ds["starting_conc"].attrs["units"] = ""
        strt = "gwt_strt"

    ic = flopy.mf6.ModflowGwtic(gwt, strt=ds[strt].data, pname=pname, **kwargs)

    return ic

def gwfgwt(ds, sim, exgtype="GWF6-GWT6", **kwargs):
    exgnamea = ds.model_name 
    exgnameb = f"gwt_{ds.model_name}"
    # exchange
    gwfgwt = flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype=exgtype,
        exgmnamea=exgnamea,
        exgmnameb=exgnameb,
        **kwargs,
    )
    return gwfgwt
