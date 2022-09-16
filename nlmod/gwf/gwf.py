# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 17:20:34 2021.

@author: oebbe
"""
import logging
import numbers
import os

import flopy
import numpy as np
import xarray as xr
import datetime as dt

from shutil import copyfile

from .. import mdims
from . import recharge

logger = logging.getLogger(__name__)


def write_and_run_model(gwf, ds, write_ds=True, nb_path=None):
    """write modflow files and run the model.

    2 extra options:
        1. write the model dataset to cache
        2. copy the modelscript (typically a Jupyter Notebook) to the model
           workspace with a timestamp.


    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        groundwater flow model.
    ds : xarray.Dataset
        dataset with model data.
    write_ds : bool, optional
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
        dst = os.path.join(ds.model_ws, new_nb_fname)
        logger.info(f"write script {new_nb_fname} to model workspace")
        copyfile(nb_path, dst)

    if write_ds:
        logger.info("write model dataset to cache")
        ds.attrs["model_dataset_written_to_disk_on"] = dt.datetime.now().strftime(
            "%Y%m%d_%H:%M:%S"
        )
        ds.to_netcdf(os.path.join(ds.attrs["cachedir"], "full_ds.nc"))

    logger.info("write modflow files to model workspace")
    gwf.simulation.write_simulation()
    ds.attrs["model_data_written_to_disk_on"] = dt.datetime.now().strftime(
        "%Y%m%d_%H:%M:%S"
    )

    logger.info("run model")
    assert gwf.simulation.run_simulation()[0], "Modflow run not succeeded"
    ds.attrs["model_ran_on"] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")


def gwf(ds, sim, **kwargs):
    """create groundwater flow model from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
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
    model_nam_file = f"{ds.model_name}.nam"

    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=ds.model_name, model_nam_file=model_nam_file,
        **kwargs
    )

    return gwf


def ims(sim, complexity="MODERATE", pname='ims', **kwargs):
    """create IMS package


    Parameters
    ----------
    sim : flopy MFSimulation
        simulation object.
    complexity : str, optional
        solver complexity for default settings. The default is "MODERATE".
    pname : str, optional
        package name

    Returns
    -------
    ims : flopy ModflowIms
        ims object.

    """

    logger.info("creating modflow IMS")

    # Create the Flopy iterative model solver (ims) Package object
    ims = flopy.mf6.modflow.mfims.ModflowIms(
        sim, pname=pname, print_option="summary", complexity=complexity,
        **kwargs
    )

    return ims


def dis(ds, gwf, length_units="METERS",
        pname='dis', **kwargs):
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
        **kwargs
    )

    return dis


def disv(ds, gwf, length_units="METERS",
                       pname='disv', **kwargs):
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

    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        xorigin = ds.attrs["xorigin"]
        yorigin = ds.attrs["yorigin"]
        angrot = ds.attrs["angrot"]
    else:
        xorigin = 0.0
        yorigin = 0.0
        angrot = 0.0

    vertices = mdims.mgrid.get_vertices_from_ds(ds)
    cell2d = mdims.mgrid.get_cell2d_from_ds(ds)
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
        **kwargs
    )
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        gwf.modelgrid.set_coord_info(xoff=xorigin,
                                     yoff=yorigin,
                                     angrot=angrot)

    return disv


def npf(ds, gwf, icelltype=0, save_flows=False, 
                      pname='npf',
                      **kwargs):
    """get node property flow package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    icelltype : int, optional
        celltype. The default is 0.
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


def ghb(ds, gwf, da_name, pname='ghb', **kwargs):
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

    if ds.gridtype == "structured":
        ghb_rec = mdims.data_array_2d_to_rec_list(
            ds,
            ds[f"{da_name}_cond"] != 0,
            col1=f"{da_name}_peil",
            col2=f"{da_name}_cond",
            first_active_layer=True,
            only_active_cells=False,
            layer=0,
        )
    elif ds.gridtype == "vertex":
        ghb_rec = mdims.data_array_1d_vertex_to_rec_list(
            ds,
            ds[f"{da_name}_cond"] != 0,
            col1=f"{da_name}_peil",
            col2=f"{da_name}_cond",
            first_active_layer=True,
            only_active_cells=False,
            layer=0,
        )
    else:
        raise ValueError(f"did not recognise gridtype {ds.gridtype}")

    if len(ghb_rec) > 0:
        ghb = flopy.mf6.ModflowGwfghb(
            gwf,
            print_input=True,
            maxbound=len(ghb_rec),
            stress_period_data=ghb_rec,
            save_flows=True,
            pname=pname,
            **kwargs
        )
        return ghb

    else:
        print("no ghb cells added")

        return None


def ic(ds, gwf, starting_head="starting_head",
                     pname='ic', **kwargs):
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
    if isinstance(starting_head, str):
        pass
    elif isinstance(starting_head, numbers.Number):
        ds["starting_head"] = starting_head * xr.ones_like(ds["idomain"])
        ds["starting_head"].attrs["units"] = "mNAP"
        starting_head = "starting_head"

    ic = flopy.mf6.ModflowGwfic(gwf, pname=pname, strt=ds[starting_head].data,
                                **kwargs)

    return ic


def sto(ds, gwf, sy=0.2, ss=0.000001, iconvert=1, 
                      save_flows=False, pname="sto", **kwargs):
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

    if ds.time.steady_state:
        return None
    else:
        if ds.time.steady_start:
            sts_spd = {0: True}
            trn_spd = {1: True}
        else:
            sts_spd = None
            trn_spd = {0: True}

        sto = flopy.mf6.ModflowGwfsto(
            gwf,
            pname=pname,
            save_flows=save_flows,
            iconvert=iconvert,
            ss=ss,
            sy=sy,
            steady_state=sts_spd,
            transient=trn_spd,
            **kwargs
        )
        return sto


def chd(ds, gwf, chd="chd", head="starting_head", 
                      pname='chd', **kwargs):
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
    # get the stress_period_data
    if ds.gridtype == "structured":
        chd_rec = mdims.data_array_3d_to_rec_list(
            ds, ds[chd] != 0, col1=head
        )
    elif ds.gridtype == "vertex":
        cellids = np.where(ds[chd])
        chd_rec = list(zip(zip(cellids[0], cellids[1]), [1.0] * len(cellids[0])))

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname=pname,
        maxbound=len(chd_rec),
        stress_period_data=chd_rec,
        save_flows=True,
        **kwargs
    )

    return chd


def surface_drain_from_ds(ds, gwf, surface_drn_cond=1000,
                                pname='drn', **kwargs):
    """get surface level drain (maaivelddrainage in Dutch) from the model
    dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    surface_drn_cond : int or float, optional
        conductivity of the surface drain. The default is 1000.
    pname : str, optional
        package name

    Returns
    -------
    drn : flopy ModflowGwfdrn
        drn package
    """

    ds.attrs["surface_drn_cond"] = surface_drn_cond
    mask = ds["ahn"].notnull()
    if ds.gridtype == "structured":
        drn_rec = mdims.data_array_2d_to_rec_list(
            ds,
            mask,
            col1="ahn",
            first_active_layer=True,
            only_active_cells=False,
            col2=ds.surface_drn_cond,
        )
    elif ds.gridtype == "vertex":
        drn_rec = mdims.data_array_1d_vertex_to_rec_list(
            ds,
            mask,
            col1="ahn",
            col2=ds.surface_drn_cond,
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
        **kwargs
    )

    return drn


def rch(ds, gwf, pname='rch', **kwargs):
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

    # create recharge package
    rch = recharge.model_datasets_to_rch(gwf, ds,
                                         pname=pname, **kwargs)

    return rch


def oc(ds, gwf, save_head=False,
                     save_budget=True, print_head=True,
                     pname='oc', **kwargs):
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
    # Create the output control package
    headfile = f"{ds.model_name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{ds.model_name}.cbc"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "LAST")]
    if save_head:
        saverecord = [("HEAD", "ALL")]
    if save_budget:
        saverecord.append(("BUDGET", "ALL"))
    if print_head:
        printrecord = [("HEAD", "LAST")]
    else:
        printrecord = None

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname=pname,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
        **kwargs
    )

    return oc
