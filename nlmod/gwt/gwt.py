import logging
import numbers

import flopy
import numpy as np
import xarray as xr

from ..dims import grid
from ..gwf.gwf import _set_record, dis, disv

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
    model_nam_file = f"{ds.model_name}_gwt.nam"

    gwt = flopy.mf6.ModflowGwt(
        sim, modelname=f"{ds.model_name}_gwt", model_nam_file=model_nam_file, **kwargs
    )

    return gwt


def adv(ds, gwt, scheme=None, **kwargs):
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
    logger.info("creating modflow ADV")
    if scheme is None:
        scheme = ds.attrs.get("adv_scheme")
    else:
        ds.attrs["adv_scheme"] = scheme
    adv = flopy.mf6.ModflowGwtadv(gwt, scheme=scheme, **kwargs)
    return adv


def dsp(ds, gwt, **kwargs):
    logger.info("creating modflow DSP")

    alh = kwargs.pop("alh", ds.dsp_alh)
    ath1 = kwargs.pop("ath1", ds.dsp_ath1)
    atv = kwargs.pop("atv", ds.dsp_atv)

    if "alh" not in kwargs:
        ds.attrs["dsp_alh"] = alh
    if "ath1" not in kwargs:
        ds.attrs["dsp_ath1"] = ath1
    if "atv" not in kwargs:
        ds.attrs["dsp_atv"] = atv

    dsp = flopy.mf6.ModflowGwtdsp(gwt, alh=alh, ath1=ath1, atv=atv, **kwargs)
    return dsp


def ssm(ds, gwt, pkg_sources=None, sources=None, **kwargs):
    logger.info("creating modflow SSM")

    if pkg_sources is not None and sources is None:
        sources = [(ipkg, "AUX", "CONCENTRATION") for ipkg in pkg_sources]
        ds.attrs["ssm_sources"] = pkg_sources
    elif sources is None:
        sources = [
            (ipkg.upper(), "AUX", "CONCENTRATION") for ipkg in ds.attrs["ssm_sources"]
        ]

    ssm = flopy.mf6.ModflowGwtssm(gwt, sources=sources, **kwargs)
    return ssm


def mst(ds, gwt, porosity=None, **kwargs):
    logger.info("creating modflow MST")

    if porosity is None:
        porosity = ds.porosity
    else:
        if isinstance(porosity, float):
            ds.attrs["porosity"] = porosity
        else:
            logger.warn("the porosity passed to mst pkg is not stored in ds")

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
    concfile = f"{gwt.name}.ucn"
    conc_filerecord = [concfile]
    budgetfile = f"{gwt.name}.cbc"
    budget_filerecord = [budgetfile]
    saverecord = _set_record(save_concentration, save_budget, output="concentration")
    printrecord = _set_record(print_concentration, print_budget, output="concentration")

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
    logger.info("creating modflow exchange GWFGWT")

    exgnamea = ds.model_name
    exgnameb = f"{ds.model_name}_gwt"
    # exchange
    gwfgwt = flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype=exgtype,
        exgmnamea=exgnamea,
        exgmnameb=exgnameb,
        **kwargs,
    )
    return gwfgwt
