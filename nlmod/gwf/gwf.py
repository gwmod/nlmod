# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 17:20:34 2021.

@author: oebbe
"""
import logging
import numbers

import flopy
import numpy as np
import xarray as xr

from ..dims import grid
from ..sim import ims, sim, tdis
from . import recharge

logger = logging.getLogger(__name__)


def gwf(ds, sim, **kwargs):
    """create groundwater flow model from the model dataset.

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
    gwf : flopy ModflowGwf
        groundwaterflow object.
    """

    # start creating model
    logger.info("creating modflow GWF")

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{ds.model_name}.nam"

    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=ds.model_name, model_nam_file=model_nam_file, **kwargs
    )

    return gwf


def dis(ds, gwf, length_units="METERS", pname="dis", **kwargs):
    """get discretisation package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
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
        return disv(ds, gwf, length_units=length_units)

    # check attributes
    for att in ["delr", "delc"]:
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

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        pname=pname,
        length_units=length_units,
        xorigin=xorigin,
        yorigin=yorigin,
        angrot=angrot,
        nlay=ds.dims["layer"],
        nrow=ds.dims["y"],
        ncol=ds.dims["x"],
        delr=ds.delr,
        delc=ds.delc,
        top=ds["top"].data,
        botm=ds["botm"].data,
        idomain=ds["idomain"].data,
        filename=f"{ds.model_name}.dis",
        **kwargs,
    )

    return dis


def disv(ds, gwf, length_units="METERS", pname="disv", **kwargs):
    """get discretisation vertices package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    disv : flopy ModflowGwfdisv
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
    disv = flopy.mf6.ModflowGwfdisv(
        gwf,
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
        gwf.modelgrid.set_coord_info(xoff=xorigin, yoff=yorigin, angrot=angrot)

    return disv


def npf(ds, gwf, icelltype=0, save_flows=False, pname="npf", **kwargs):
    """get node property flow package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    icelltype : int or str, optional
        celltype, if int the icelltype for all layer, if str the icelltype from
        the model ds is used. The default is 0.
    save_flows : bool, optional
        value is passed to flopy.mf6.ModflowGwfnpf() to determine if cell by
        cell flows should be saved to the cbb file. Default is False
    pname : str, optional
        package name

    Raises
    ------
    NotImplementedError
        only icelltype 0 is implemented.

    Returns
    -------
    npf : flopy ModflowGwfnpf
        npf package.
    """
    logger.info("creating modflow NPF")

    if isinstance(icelltype, str):
        icelltype = ds[icelltype]

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        pname=pname,
        icelltype=icelltype,
        k=ds["kh"].data,
        k33=ds["kv"].data,
        save_flows=save_flows,
        **kwargs,
    )

    return npf


def ghb(ds, gwf, da_name, pname="ghb", **kwargs):
    """get general head boundary from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    da_name : str
        name of the ghb files in the model dataset.
    pname : str, optional
        package name

    Raises
    ------
    ValueError
        raised if gridtype is not structured or vertex.

    Returns
    -------
    ghb : flopy ModflowGwfghb
        ghb package
    """
    logger.info("creating modflow GHB")

    ghb_rec = grid.da_to_reclist(
        ds,
        ds[f"{da_name}_cond"] != 0,
        col1=f"{da_name}_peil",
        col2=f"{da_name}_cond",
        first_active_layer=True,
        only_active_cells=False,
        layer=0,
    )

    if len(ghb_rec) > 0:
        ghb = flopy.mf6.ModflowGwfghb(
            gwf,
            print_input=True,
            maxbound=len(ghb_rec),
            stress_period_data=ghb_rec,
            save_flows=True,
            pname=pname,
            **kwargs,
        )
        return ghb

    else:
        print("no ghb cells added")

        return None


def ic(ds, gwf, starting_head="starting_head", pname="ic", **kwargs):
    """get initial condictions package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    starting_head : str, float or int, optional
        if type is int or float this is the starting head for all cells
        If the type is str the data variable from ds is used as starting
        head. The default is 'starting_head'.
    pname : str, optional
        package name

    Returns
    -------
    ic : flopy ModflowGwfic
        ic package
    """
    logger.info("creating modflow IC")
    
    if isinstance(starting_head, str):
        pass
    elif isinstance(starting_head, numbers.Number):
        ds["starting_head"] = starting_head * xr.ones_like(ds["idomain"])
        ds["starting_head"].attrs["units"] = "mNAP"
        starting_head = "starting_head"

    ic = flopy.mf6.ModflowGwfic(gwf, pname=pname, strt=ds[starting_head].data, **kwargs)

    return ic


def sto(
    ds,
    gwf,
    sy=0.2,
    ss=0.000001,
    iconvert=1,
    save_flows=False,
    pname="sto",
    **kwargs,
):
    """get storage package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    sy : float, optional
        specific yield. The default is 0.2.
    ss : float, optional
        specific storage. The default is 0.000001.
    iconvert : int, optional
        See description in ModflowGwfsto. The default is 1.
    save_flows : bool, optional
        value is passed to flopy.mf6.ModflowGwfsto() to determine if flows
        should be saved to the cbb file. Default is False
    pname : str, optional
        package name

    Returns
    -------
    sto : flopy ModflowGwfsto
        sto package
    """
    logger.info("creating modflow STO")
    
    if ds.time.steady_state:
        return None
    else:
        if ds.time.steady_start:
            sts_spd = {0: True}
            trn_spd = {1: True}
        else:
            sts_spd = None
            trn_spd = {0: True}

        if "sy" in ds:
            sy = ds["sy"].data

        if "ss" in ds:
            ss = ds["ss"].data

        sto = flopy.mf6.ModflowGwfsto(
            gwf,
            pname=pname,
            save_flows=save_flows,
            iconvert=iconvert,
            ss=ss,
            sy=sy,
            steady_state=sts_spd,
            transient=trn_spd,
            **kwargs,
        )
        return sto


def chd(ds, gwf, chd="chd", head="starting_head", pname="chd", **kwargs):
    """get constant head boundary at the model's edges from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    chd : str, optional
        name of data variable in ds that is 1 for cells with a constant
        head and zero for all other cells. The default is 'chd'.
    head : str, optional
        name of data variable in ds that is used as the head in the chd
        cells. The default is 'starting_head'.
    pname : str, optional
        package name

    Returns
    -------
    chd : flopy ModflowGwfchd
        chd package
    """
    logger.info("creating modflow CHD")

    # get the stress_period_data
    chd_rec = grid.da_to_reclist(ds, ds[chd] != 0, col1=head)

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname=pname,
        maxbound=len(chd_rec),
        stress_period_data=chd_rec,
        save_flows=True,
        **kwargs,
    )

    return chd


def surface_drain_from_ds(ds, gwf, resistance, pname="drn", **kwargs):
    """get surface level drain (maaivelddrainage in Dutch) from the model
    dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    resistance : int or float
        resistance of the surface drain, scaled with cell area to
        calculate drain conductance.
    pname : str, optional
        package name

    Returns
    -------
    drn : flopy ModflowGwfdrn
        drn package
    """

    ds.attrs["surface_drn_resistance"] = resistance
    mask = ds["ahn"].notnull()
    drn_rec = grid.da_to_reclist(
        ds,
        mask,
        col1="ahn",
        col2=ds["area"] / ds.surface_drn_resistance,
        first_active_layer=True,
        only_active_cells=False,
    )

    drn = flopy.mf6.ModflowGwfdrn(
        gwf,
        pname=pname,
        print_input=True,
        maxbound=len(drn_rec),
        stress_period_data={0: drn_rec},
        save_flows=True,
        **kwargs,
    )

    return drn


def rch(ds, gwf, pname="rch", **kwargs):
    """get recharge package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    pname : str, optional
        package name

    Returns
    -------
    rch : flopy ModflowGwfrch
        rch package
    """
    logger.info("creating modflow RCH")
    # create recharge package
    rch = recharge.model_datasets_to_rch(gwf, ds, pname=pname, **kwargs)

    return rch


def evt(ds, gwf, pname="evt", **kwargs):
    """get evapotranspiration package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    pname : str, optional
        package name

    Returns
    -------
    evt : flopy ModflowGwfevt
        rch package
    """
    logger.info("creating modflow EVT")

    # create recharge package
    evt = recharge.model_datasets_to_evt(gwf, ds, pname=pname, **kwargs)

    return evt


def _set_record(head, budget):
    record = []
    if isinstance(head, bool):
        if head:
            head = "LAST"
        else:
            head = None
    if head is not None:
        record.append(("HEAD", head))
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
    gwf,
    save_head=True,
    save_budget=True,
    print_head=False,
    print_budget=False,
    pname="oc",
    **kwargs,
):
    """get output control package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    pname : str, optional
        package name

    Returns
    -------
    oc : flopy ModflowGwfoc
        oc package
    """
    logger.info("creating modflow OC")
    
    # Create the output control package
    headfile = f"{ds.model_name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{ds.model_name}.cbc"
    budget_filerecord = [budgetfile]
    saverecord = _set_record(save_head, save_budget)
    printrecord = _set_record(print_head, print_budget)

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname=pname,
        saverecord=saverecord,
        printrecord=printrecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        **kwargs,
    )

    return oc


def ds_to_gwf(ds):
    """Generate Simulation and GWF model from model DataSet.

    Builds the following packages:
    - sim
    - tdis
    - ims
    - gwf
      - dis
      - npf
      - ic
      - oc
      - rch if "recharge" is present in DataSet
      - evt if "evaporation" is present in DataSet

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset

    Returns
    -------
    flopy.mf6.ModflowGwf
        MODFLOW6 GroundwaterFlow model object.
    """

    # create simulation
    mf_sim = sim(ds)

    # create time discretisation
    tdis(ds, mf_sim)

    # create ims
    ims(mf_sim)

    # create groundwater flow model
    mf_gwf = gwf(ds, mf_sim)

    # Create discretization
    if ds.gridtype == "structured":
        dis(ds, mf_gwf)
    elif ds.gridtype == "vertex":
        disv(ds, mf_gwf)
    else:
        raise TypeError("gridtype not recognized.")

    # create node property flow
    npf(ds, mf_gwf)

    # Create the initial conditions package
    starting_head = "starting_head"
    if starting_head not in ds:
        starting_head = 0.0
    ic(ds, mf_gwf, starting_head=starting_head)

    # Create the output control package
    oc(ds, mf_gwf)

    if "recharge" in ds:
        rch(ds, mf_gwf)

    if "evaporation" in ds:
        evt(ds, mf_gwf)

    return mf_gwf
