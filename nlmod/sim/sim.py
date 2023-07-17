# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 17:20:34 2021.

@author: oebbe
"""
import datetime as dt
import logging
import os
from shutil import copyfile

import flopy
import numpy as np
import pandas as pd

from .. import util

logger = logging.getLogger(__name__)


def write_and_run(sim, ds, write_ds=True, nb_path=None, silent=False):
    """write modflow files and run the model.

    2 extra options:
        1. write the model dataset to cache
        2. copy the modelscript (typically a Jupyter Notebook) to the model
           workspace with a timestamp.


    Parameters
    ----------
    sim : flopy.mf6.MFSimulation or flopy.mf6.ModflowGwf
        MF6 Simulation or MF6 Groundwater Flow object.
    ds : xarray.Dataset
        dataset with model data.
    write_ds : bool, optional
        if True the model dataset is cached to a NetCDF-file (.nc) with a name equal
        to its attribute called "model_name". The default is True.
    nb_path : str or None, optional
        full path of the Jupyter Notebook (.ipynb) with the modelscript. The
        default is None. Preferably this path does not have to be given
        manually but there is currently no good option to obtain the filename
        of a Jupyter Notebook from within the notebook itself.
    silent : bool, optional
        write and run model silently
    """
    if isinstance(sim, flopy.mf6.ModflowGwf):
        sim = sim.simulation

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
        ds.to_netcdf(os.path.join(ds.attrs["cachedir"], f"{ds.model_name}.nc"))

    logger.info("write modflow files to model workspace")
    sim.write_simulation(silent=silent)
    ds.attrs["model_data_written_to_disk_on"] = dt.datetime.now().strftime(
        "%Y%m%d_%H:%M:%S"
    )

    logger.info("run model")
    assert sim.run_simulation(silent=silent)[0], "Modflow run not succeeded"
    ds.attrs["model_ran_on"] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")


def get_tdis_perioddata(ds):
    """Get tdis_perioddata from ds.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with time variant model data

    Returns
    -------
    tdis_perioddata : [perlen, nstp, tsmult]
        - perlen (double) is the length of a stress period.
        - nstp (integer) is the number of time steps in a stress period.
        - tsmult (double) is the multiplier for the length of successive time
          steps. The length of a time step is calculated by multiplying the
          length of the previous time step by TSMULT. The length of the first
          time step, :math:`\\Delta t_1`, is related to PERLEN, NSTP, and
          TSMULT by the relation :math:`\\Delta t_1= perlen \frac{tsmult -
          1}{tsmult^{nstp}-1}`.
    """
    deltat = pd.to_timedelta(1, ds.time.time_units)
    perlen = [
        (pd.to_datetime(ds["time"].data[0]) - pd.to_datetime(ds.time.start)) / deltat
    ]

    if len(ds["time"]) > 1:
        perlen.extend(np.diff(ds["time"]) / deltat)

    if "nstp" in ds:
        nstp = ds["nstp"].values
    else:
        nstp = [ds.time.nstp] * len(perlen)

    if "tsmult" in ds:
        tsmult = ds["tsmult"].values
    else:
        tsmult = [ds.time.tsmult] * len(perlen)

    tdis_perioddata = list(zip(perlen, nstp, tsmult))

    return tdis_perioddata


def sim(ds, exe_name=None):
    """create sim from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start,
        perlen, nstp, tsmult
    exe_name: str, optional
        path to modflow executable, default is None, which assumes binaries
        are available in nlmod/bin directory. Binaries can be downloaded
        using `nlmod.util.download_mfbinaries()`.

    Returns
    -------
    sim : flopy MFSimulation
        simulation object.
    """

    # start creating model
    logger.info("creating mf6 SIM")

    if exe_name is None:
        exe_name = util.get_exe_path(ds.mfversion)

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=ds.model_name,
        exe_name=exe_name,
        version=ds.mfversion,
        sim_ws=ds.model_ws,
    )

    return sim


def tdis(ds, sim, pname="tdis"):
    """create tdis package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: time_units, start, perlen, nstp, tsmult
    sim : flopy MFSimulation
        simulation object.
    pname : str, optional
        package name

    Returns
    -------
    dis : flopy TDis
        tdis object.
    """

    # start creating model
    logger.info("creating mf6 TDIS")

    tdis_perioddata = get_tdis_perioddata(ds)

    # Create the Flopy temporal discretization object
    tdis = flopy.mf6.ModflowTdis(
        sim,
        pname=pname,
        time_units=ds.time.time_units,
        nper=len(ds.time),
        start_date_time=pd.Timestamp(ds.time.start).isoformat(),
        perioddata=tdis_perioddata,
    )

    return tdis


def ims(sim, complexity="MODERATE", pname="ims", **kwargs):
    """create IMS package.

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

    logger.info("creating mf6 IMS")

    # Create the Flopy iterative model solver (ims) Package object
    ims = flopy.mf6.ModflowIms(
        sim,
        pname=pname,
        print_option="summary",
        complexity=complexity,
        **kwargs,
    )

    return ims


def register_ims_package(sim, model, ims):
    sim.register_ims_package(ims, [model.name])
