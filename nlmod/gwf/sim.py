# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 17:20:34 2021.

@author: oebbe
"""
import logging

import flopy
import numpy as np
import pandas as pd

from .. import util

logger = logging.getLogger(__name__)


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
        (pd.to_datetime(ds["time"].data[0]) - pd.to_datetime(ds.time.start_time))
        / deltat
    ]
    
    if len(ds["time"]) > 1:
        perlen.extend(np.diff(ds["time"]) / deltat)
        
    if 'nstp' in ds:
        nstp = ds['nstp'].values
    else:
        nstp = [ds.time.nstp] * len(perlen)
    
    if 'tsmult' in ds:
        tsmult = ds['tsmult'].values
    else:
        tsmult = [ds.time.tsmult] * len(perlen)
    
    tdis_perioddata = [(p, n, t) for p, n, t in zip(perlen, nstp, tsmult)]

    return tdis_perioddata


def sim(ds, exe_name=None):
    """create sim from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start_time,
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
    logger.info("creating modflow SIM")

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
        attributes: time_units, start_time, perlen, nstp, tsmult
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
    logger.info("creating modflow TDIS")

    tdis_perioddata = get_tdis_perioddata(ds)

    # Create the Flopy temporal discretization object
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname=pname,
        time_units=ds.time.time_units,
        nper=len(ds.time),
        # start_date_time=ds.time.start_time, # disable until fix in modpath
        perioddata=tdis_perioddata,
    )

    return tdis
