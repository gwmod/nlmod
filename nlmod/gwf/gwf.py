# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 17:20:34 2021.

@author: oebbe
"""
import logging
import numbers
import os
import sys

import flopy
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt

from shutil import copyfile

from .. import mdims
from . import recharge

logger = logging.getLogger(__name__)


def write_and_run_model(gwf, model_ds, write_model_ds=True, nb_path=None):
    """write modflow files and run the model.

    2 extra options:
        1. write the model dataset to cache
        2. copy the modelscript (typically a Jupyter Notebook) to the model
           workspace with a timestamp.


    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        groundwater flow model.
    model_ds : xarray.Dataset
        dataset with model data.
    write_model_ds : bool, optional
        if True the model dataset is cached. The default is True.
    nb_path : str or None, optional
        full path of the Jupyter Notebook (.ipynb) with the modelscript. The
        default is None. Preferably this path does not have to be given
        manually but there is currently no good option to obtain the filename
        of a Jupyter Notebook from within the notebook itself.
    """

    if nb_path is not None:
        new_nb_fname = (
            f'{dt.datetime.now().strftime("%Y%m%d")}' + os.path.split(nb_path)[-1]
        )
        dst = os.path.join(model_ds.model_ws, new_nb_fname)
        logger.info(f"write script {new_nb_fname} to model workspace")
        copyfile(nb_path, dst)

    if write_model_ds:
        logger.info("write model dataset to cache")
        model_ds.attrs["model_dataset_written_to_disk_on"] = dt.datetime.now().strftime(
            "%Y%m%d_%H:%M:%S"
        )
        model_ds.to_netcdf(os.path.join(model_ds.attrs["cachedir"], "full_model_ds.nc"))

    logger.info("write modflow files to model workspace")
    gwf.simulation.write_simulation()
    model_ds.attrs["model_data_written_to_disk_on"] = dt.datetime.now().strftime(
        "%Y%m%d_%H:%M:%S"
    )

    logger.info("run model")
    assert gwf.simulation.run_simulation()[0], "Modflow run not succeeded"
    model_ds.attrs["model_ran_on"] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")


def gwf_from_model_ds(model_ds, sim):
    """create groundwater flow model from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start_time,
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
    model_nam_file = f"{model_ds.model_name}.nam"
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=model_ds.model_name, model_nam_file=model_nam_file
    )

    return gwf


def ims_to_sim(sim, complexity="MODERATE"):
    """create IMS package


    Parameters
    ----------
    sim : flopy MFSimulation
        simulation object.
    complexity : str, optional
        solver complexity for default settings. The default is "MODERATE".

    Returns
    -------
    ims : flopy ModflowIms
        ims object.

    """

    logger.info("creating modflow IMS")

    # Create the Flopy iterative model solver (ims) Package object
    ims = flopy.mf6.modflow.mfims.ModflowIms(
        sim, pname="ims", print_option="summary", complexity=complexity
    )

    return ims


def dis_from_model_ds(model_ds, gwf, length_units="METERS", angrot=0):
    """get discretisation package from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    angrot : int or float, optional
        rotation angle. The default is 0.

    Returns
    -------
    dis : TYPE
        discretisation package.
    """

    if model_ds.gridtype == "vertex":
        return disv_from_model_ds(
            model_ds, gwf, length_units=length_units, angrot=angrot
        )

    # check attributes
    for att in ["delr", "delc"]:
        if isinstance(model_ds.attrs[att], np.float32):
            model_ds.attrs[att] = float(model_ds.attrs[att])

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        length_units=length_units,
        xorigin=model_ds.extent[0],
        yorigin=model_ds.extent[2],
        angrot=angrot,
        nlay=model_ds.dims["layer"],
        nrow=model_ds.dims["y"],
        ncol=model_ds.dims["x"],
        delr=model_ds.delr,
        delc=model_ds.delc,
        top=model_ds["top"].data,
        botm=model_ds["botm"].data,
        idomain=model_ds["idomain"].data,
        filename=f"{model_ds.model_name}.dis",
    )

    return dis


def disv_from_model_ds(model_ds, gwf, length_units="METERS", angrot=0):
    """get discretisation vertices package from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    angrot : int or float, optional
        rotation angle. The default is 0.

    Returns
    -------
    disv : flopy ModflowGwfdisv
        disv package
    """

    vertices = mdims.mgrid.get_vertices_from_model_ds(model_ds)
    cell2d = mdims.mgrid.get_cell2d_from_model_ds(model_ds)
    disv = flopy.mf6.ModflowGwfdisv(
        gwf,
        idomain=model_ds["idomain"].data,
        xorigin=model_ds.extent[0],
        yorigin=model_ds.extent[2],
        length_units=length_units,
        angrot=angrot,
        nlay=len(model_ds.layer),
        ncpl=len(model_ds.icell2d),
        nvert=len(model_ds.iv),
        top=model_ds["top"].data,
        botm=model_ds["botm"].data,
        vertices=vertices,
        cell2d=cell2d,
    )

    return disv


def npf_from_model_ds(model_ds, gwf, icelltype=0, save_flows=False, **kwargs):
    """get node property flow package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    icelltype : int, optional
        celltype. The default is 0.
    save_flows : bool, optional
        value is passed to flopy.mf6.ModflowGwfnpf() to determine if cell by
        cell flows should be saved to the cbb file. Default is False

    Raises
    ------
    NotImplementedError
        only icelltype 0 is implemented.

    Returns
    -------
    npf : flopy ModflowGwfnpf
        npf package.
    """

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=icelltype,
        k=model_ds["kh"].data,
        k33=model_ds["kv"].data,
        save_flows=save_flows,
        **kwargs,
    )

    return npf


def ghb_from_model_ds(model_ds, gwf, da_name):
    """get general head boundary from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    da_name : str
        name of the ghb files in the model dataset.

    Raises
    ------
    ValueError
        raised if gridtype is not structured or vertex.

    Returns
    -------
    ghb : flopy ModflowGwfghb
        ghb package
    """

    if model_ds.gridtype == "structured":
        ghb_rec = mdims.data_array_2d_to_rec_list(
            model_ds,
            model_ds[f"{da_name}_cond"] != 0,
            col1=f"{da_name}_peil",
            col2=f"{da_name}_cond",
            first_active_layer=True,
            only_active_cells=False,
            layer=0,
        )
    elif model_ds.gridtype == "vertex":
        ghb_rec = mdims.data_array_1d_vertex_to_rec_list(
            model_ds,
            model_ds[f"{da_name}_cond"] != 0,
            col1=f"{da_name}_peil",
            col2=f"{da_name}_cond",
            first_active_layer=True,
            only_active_cells=False,
            layer=0,
        )
    else:
        raise ValueError(f"did not recognise gridtype {model_ds.gridtype}")

    if len(ghb_rec) > 0:
        ghb = flopy.mf6.ModflowGwfghb(
            gwf,
            print_input=True,
            maxbound=len(ghb_rec),
            stress_period_data=ghb_rec,
            save_flows=True,
        )
        return ghb

    else:
        print("no ghb cells added")

        return None


def ic_from_model_ds(model_ds, gwf, starting_head="starting_head"):
    """get initial condictions package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    starting_head : str, float or int, optional
        if type is int or float this is the starting head for all cells
        If the type is str the data variable from model_ds is used as starting
        head. The default is 'starting_head'.

    Returns
    -------
    ic : flopy ModflowGwfic
        ic package
    """
    if isinstance(starting_head, str):
        pass
    elif isinstance(starting_head, numbers.Number):
        model_ds["starting_head"] = starting_head * xr.ones_like(model_ds["idomain"])
        model_ds["starting_head"].attrs["units"] = "mNAP"
        starting_head = "starting_head"

    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=model_ds[starting_head].data)

    return ic


def sto_from_model_ds(model_ds, gwf, sy=0.2, ss=0.000001, iconvert=1, save_flows=False):
    """get storage package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
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

    Returns
    -------
    sto : flopy ModflowGwfsto
        sto package
    """

    if model_ds.time.steady_state:
        return None
    else:
        if model_ds.time.steady_start:
            sts_spd = {0: True}
            trn_spd = {1: True}
        else:
            sts_spd = None
            trn_spd = {0: True}

        sto = flopy.mf6.ModflowGwfsto(
            gwf,
            pname="sto",
            save_flows=save_flows,
            iconvert=iconvert,
            ss=ss,
            sy=sy,
            steady_state=sts_spd,
            transient=trn_spd,
        )
        return sto


def chd_from_model_ds(model_ds, gwf, chd="chd", head="starting_head"):
    """get constant head boundary at the model's edges from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    chd : str, optional
        name of data variable in model_ds that is 1 for cells with a constant
        head and zero for all other cells. The default is 'chd'.
    head : str, optional
        name of data variable in model_ds that is used as the head in the chd
        cells. The default is 'starting_head'.

    Returns
    -------
    chd : flopy ModflowGwfchd
        chd package
    """
    # get the stress_period_data
    if model_ds.gridtype == "structured":
        chd_rec = mdims.data_array_3d_to_rec_list(
            model_ds, model_ds[chd] != 0, col1=head
        )
    elif model_ds.gridtype == "vertex":
        cellids = np.where(model_ds[chd])
        chd_rec = list(zip(zip(cellids[0], cellids[1]), [1.0] * len(cellids[0])))

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname=chd,
        maxbound=len(chd_rec),
        stress_period_data=chd_rec,
        save_flows=True,
    )

    return chd


def surface_drain_from_model_ds(model_ds, gwf, surface_drn_cond=1000):
    """get surface level drain (maaivelddrainage in Dutch) from the model
    dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    surface_drn_cond : int or float, optional
        conductivity of the surface drain. The default is 1000.

    Returns
    -------
    drn : flopy ModflowGwfdrn
        drn package
    """

    model_ds.attrs["surface_drn_cond"] = surface_drn_cond
    mask = model_ds["ahn"].notnull()
    if model_ds.gridtype == "structured":
        drn_rec = mdims.data_array_2d_to_rec_list(
            model_ds,
            mask,
            col1="ahn",
            first_active_layer=True,
            only_active_cells=False,
            col2=model_ds.surface_drn_cond,
        )
    elif model_ds.gridtype == "vertex":
        drn_rec = mdims.data_array_1d_vertex_to_rec_list(
            model_ds,
            mask,
            col1="ahn",
            col2=model_ds.surface_drn_cond,
            first_active_layer=True,
            only_active_cells=False,
        )

    drn = flopy.mf6.ModflowGwfdrn(
        gwf,
        print_input=True,
        maxbound=len(drn_rec),
        stress_period_data={0: drn_rec},
        save_flows=True,
    )

    return drn


def rch_from_model_ds(model_ds, gwf):
    """get recharge package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.

    Returns
    -------
    rch : flopy ModflowGwfrch
        rch package
    """

    # create recharge package
    rch = recharge.model_datasets_to_rch(gwf, model_ds)

    return rch


def oc_from_model_ds(model_ds, gwf, save_budget=True, print_head=True):
    """get output control package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.

    Returns
    -------
    oc : flopy ModflowGwfoc
        oc package
    """
    # Create the output control package
    headfile = f"{model_ds.model_name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{model_ds.model_name}.cbc"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "LAST")]
    if save_budget:
        saverecord.append(("BUDGET", "ALL"))
    if print_head:
        printrecord = [("HEAD", "LAST")]
    else:
        printrecord = None

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )

    return oc
