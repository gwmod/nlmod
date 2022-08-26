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


def get_tdis_perioddata(model_ds):
    """Get tdis_perioddata from model_ds.

    Parameters
    ----------
    model_ds : xarray.Dataset
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
    deltat = pd.to_timedelta(1, model_ds.time.time_units)
    perlen = [
        (
            pd.to_datetime(model_ds["time"].data[0])
            - pd.to_datetime(model_ds.time.start_time)
        )
        / deltat
    ]
    if len(model_ds["time"]) > 1:
        perlen.extend(np.diff(model_ds["time"]) / deltat)
    tdis_perioddata = [(p, model_ds.time.nstp, model_ds.time.tsmult) for p in perlen]

    return tdis_perioddata


def sim_from_model_ds(model_ds, exe_name=None):
    """create sim from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
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
        exe_name = util.get_exe_path(model_ds.mfversion)

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=model_ds.model_name,
        exe_name=exe_name,
        version=model_ds.mfversion,
        sim_ws=model_ds.model_ws,
    )

    return sim


def tdis_from_model_ds(model_ds, sim):
    """create tdis package from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: time_units, start_time, perlen, nstp, tsmult
    sim : flopy MFSimulation
        simulation object.

    Returns
    -------
    dis : flopy TDis
        tdis object.
    """

    # start creating model
    logger.info("creating modflow SIM, TDIS, GWF and IMS")

    tdis_perioddata = get_tdis_perioddata(model_ds)

    # Create the Flopy temporal discretization object
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units=model_ds.time.time_units,
        nper=len(model_ds.time),
        start_date_time=model_ds.time.start_time,
        perioddata=tdis_perioddata,
    )

    return tdis
