import logging
import numbers

import flopy
import xarray as xr

from ..dims import grid
from ..gwf.gwf import _dis, _disv, _set_record

logger = logging.getLogger(__name__)


def _get_value_from_ds_attr(ds, varname, attr=None, value=None, warn=True):
    """Internal function to get value from dataset attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing model data
    varname : str
        name of the variable in flopy package
    attr : str, optional
        name of the attribute in dataset (is sometimes different to varname)
    value : Any, optional
        variable value, by default None
    warn : bool, optional
        log warning if value not found

    Returns
    -------
    value : Any
        returns variable value, if value was None, attempts to obtain
        variable from dataset attributes.
    """
    if attr is None:
        attr = varname

    if value is not None and (attr in ds.attrs):
        logger.info(
            f"Using user-provided '{varname}' and not stored attribute 'ds.{attr}'"
        )
    elif value is None and (attr in ds.attrs):
        value = ds.attrs[attr]
    elif value is None:
        if warn:
            msg = (
                f"No value found for '{varname}', passing None to flopy. "
                f"To fix this error pass '{varname}' to function or set 'ds.{attr}'."
            )
            logger.warning(msg)
        # raise ValueError(msg)
    return value


def _get_value_from_ds_datavar(ds, varname, datavar=None, value=None, warn=True):
    """Internal function to get value from dataset data variables.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing model data
    varname : str
        name of the variable in flopy package
    datavar : str, optional
        name of the data variable (is sometimes different to varname) in dataset
    value : Any, optional
        variable value, by default None
    warn : bool, optional
        log warning if value not found

    Returns
    -------
    value : Any
        returns variable value, if value was None, attempts to obtain
        variable from dataset data variables.
    """
    if datavar is None:
        datavar = varname

    if (value is not None) and (datavar in ds):
        logger.info(
            f"Using user-provided '{varname}' and not"
            f" stored data variable 'ds.{datavar}'"
        )
    elif value is None and (datavar in ds):
        value = ds[datavar]
    elif value is None:
        if warn:
            msg = (
                f"No value found for '{varname}', passing None to flopy. "
                f"To fix this error pass '{varname}' to function or set 'ds.{datavar}'."
            )
            logger.warning(msg)
        # raise ValueError(msg)
    return value


def gwt(ds, sim, modelname=None, **kwargs):
    """create groundwater transport model from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start,
        perlen, nstp, tsmult
    sim : flopy MFSimulation
        simulation object.
    modelname : str
        name of the transport model

    Returns
    -------
    gwt : flopy ModflowGwt
        groundwater transport object.
    """

    # start creating model
    logger.info("creating modflow GWT")

    # Create the Flopy groundwater flow (gwf) model object
    if modelname is None:
        modelname = f"{ds.model_name}_gwt"
    model_nam_file = f"{modelname}.nam"

    gwt = flopy.mf6.ModflowGwt(
        sim, modelname=modelname, model_nam_file=model_nam_file, **kwargs
    )

    return gwt


def dis(ds, gwt, length_units="METERS", pname="dis", **kwargs):
    """create discretisation package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwt : flopy ModflowGwf
        groundwater transport object
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    dis : flopy ModflowGwtdis
        discretisation package.
    """
    return _dis(ds, gwt, length_units, pname, **kwargs)


def disv(ds, gwt, length_units="METERS", pname="disv", **kwargs):
    """create discretisation vertices package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    model : flopy ModflowGwt
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
    return _disv(ds, gwt, length_units, pname, **kwargs)


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
    scheme = _get_value_from_ds_attr(ds, "scheme", "adv_scheme", value=scheme)
    adv = flopy.mf6.ModflowGwtadv(gwt, scheme=scheme, **kwargs)
    return adv


def dsp(ds, gwt, **kwargs):
    """create dispersion package for groundwater transport model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data
    gwt : flopy ModflowGwt
        groundwater transport object

    Returns
    -------
    dsp : flopy ModflowGwtdsp
        dsp package
    """
    logger.info("creating modflow DSP")
    alh = _get_value_from_ds_attr(ds, "alh", "dsp_alh", value=kwargs.pop("alh", None))
    ath1 = _get_value_from_ds_attr(
        ds, "ath1", "dsp_ath1", value=kwargs.pop("ath1", None)
    )
    atv = _get_value_from_ds_attr(ds, "atv", "dsp_atv", value=kwargs.pop("atv", None))
    dsp = flopy.mf6.ModflowGwtdsp(gwt, alh=alh, ath1=ath1, atv=atv, **kwargs)
    return dsp


def ssm(ds, gwt, sources=None, **kwargs):
    """create source-sink mixing package for groundwater transport model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data
    gwt : flopy ModflowGwt
        groundwater transport object
    sources : list of tuple, None
        list of tuple(s) with packages that function as source in model,
        e.g. [("GHB", "AUX", "CONCENTRATION")]. If None, sources is derived
        from model dataset attribute `ds.ssm_sources`.

    Returns
    -------
    ssm : flopy ModflowGwtssm
        ssm package
    """
    logger.info("creating modflow SSM")

    build_tuples = False
    if sources is None:
        build_tuples = True

    sources = _get_value_from_ds_attr(ds, "sources", "ssm_sources", value=sources)

    if build_tuples and sources is not None:
        sources = [(ipkg, "AUX", "CONCENTRATION") for ipkg in sources]

    ssm = flopy.mf6.ModflowGwtssm(gwt, sources=sources, **kwargs)
    return ssm


def mst(ds, gwt, porosity=None, **kwargs):
    """create mass storage transfer package for groundwater transport model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data
    gwt : flopy ModflowGwt
        groundwater transport object
    porosity : Any, optional
        porosity, can be passed as float, array-like or string. If passed as string
        data is taken from dataset.

    Returns
    -------
    mst : flopy ModflowGwtmst
        mst package
    """
    logger.info("creating modflow MST")
    if isinstance(porosity, str):
        porosity = None
    # NOTE: attempting to look for porosity in attributes first, then data variables.
    # If both are defined, the attribute value will be used. The log message in this
    # case is not entirely correct. This is something we may need to sort out, and 
    # also think about the order we do this search.
    porosity = _get_value_from_ds_attr(ds, "porosity", value=porosity, warn=False)
    porosity = _get_value_from_ds_datavar(ds, "porosity", value=porosity)
    mst = flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, **kwargs)
    return mst


def cnc(ds, gwt, da_mask, da_conc, pname="cnc", **kwargs):
    """create constant concentration package for groundwater transport model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data
    gwt : flopy ModflowGwt
        groundwater transport object
    da_mask : str
        data array containing mask where to create constant concentration cells
    da_conc : str
        data array containing concentration data

    Returns
    -------
    cnc : flopy ModflowGwtcnc
        cnc package
    """
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
    """create output control package for groundwater transport model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwt : flopy ModflowGwt
        groundwater transport object.
    pname : str, optional
        package name

    Returns
    -------
    oc : flopy ModflowGwtoc
        oc package
    """
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


def ic(ds, gwt, strt, pname="ic", **kwargs):
    """create initial condictions package for groundwater transport model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwt : flopy ModflowGwf
        groundwater transport object.
    strt : str, float or int
        if type is int or float this is the starting concentration for all cells
        If the type is str the data variable from ds is used as starting
        concentration.
    pname : str, optional
        package name

    Returns
    -------
    ic : flopy ModflowGwtic
        ic package
    """
    logger.info("creating modflow IC")
    if isinstance(strt, numbers.Number):
        strt = strt * xr.ones_like(ds["idomain"])
    ic = flopy.mf6.ModflowGwtic(gwt, strt=ds[strt].data, pname=pname, **kwargs)

    return ic


def gwfgwt(ds, sim, exgtype="GWF6-GWT6", **kwargs):
    """create GWF-GWT exchange package for modflow simulation.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    sim : flopy MFSimulation
        simulation object
    exgtype : str, optional
        exchange type, by default "GWF6-GWT6"

    Returns
    -------
    gwfgwt :
        _description_
    """
    logger.info("creating modflow exchange GWFGWT")
    type_name_dict = {}
    for name, mod in sim.model_dict.items():
        type_name_dict[mod.model_type] = name
    exgnamea = kwargs.pop("exgnamea", type_name_dict["gwf6"])
    exgnameb = kwargs.pop("exgnameb", type_name_dict["gwt6"])
    # exchange
    gwfgwt = flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype=exgtype,
        exgmnamea=exgnamea,
        exgmnameb=exgnameb,
        **kwargs,
    )
    return gwfgwt
