# -*- coding: utf-8 -*-
"""Created on Fri Apr 17 13:50:48 2020.

@author: oebbe
"""

import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def set_model_ds_time(model_ds, time=None, steady_state=False,
                      steady_start=True, steady_start_perlen=3652.0,
                      time_units='DAYS', start_time=None,
                      transient_timesteps=0, perlen=1.0,
                      nstp=1, tsmult=1.0):
    """Set timing for a model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        Dataset to add time information to
    time : list, array or DatetimeIndex of pandas.Timestamps, optional
        an array with the start-time of the model and the ending times of the
        stress-periods. When steady_start is True, the first value of time is
        the start of the transient model period. The default is None.
    steady_state : bool, optional
        if True the model is steady state. The default is False.
    steady_start : bool, optional
        if True the model is transient with a steady state start time step. The
        default is True.
    steady_start_perlen : float, optional
        stress-period length of the first steady state stress period.
        Only used if steady_start is True. The period is used to determine the
        recharge in this steady state stress period. Default is 3652 days
        (approximately 10 years).
    time_units : str, optional
        time unit of the model. The default is 'DAYS', which is the only
        allowed value for now.
    start_time : str or datetime, optional
        start time of the model. When steady_start is True, this is the
        start_time of the transient model period. Input is ignored when time is
        assigned. The default is January 1, 2000.
    transient_timesteps : int, optional
        number of transient time steps. Only used when steady_state is False.
        Input is ignored when time is assigned or perlen is an iterable. The
        default is 0.
    perlen : float, int, list or np.array, optional
        length of each stress-period depending on the type:
        - float or int: this is the length of all the stress periods.
        - list or array: the items are the length of the stress-periods in 
          days. The length of perlen should match the number of stress-periods
          (including steady-state stress-periods).
        When steady_start is True, the length of the first steady state stress
        period is defined by steady_start_perlen. Input for perlen is ignored
        when time is assigned. The default is 1.0.
    nstp : int, optional
        number of steps. The default is 1.
    tsmult : float, optional
        timestep multiplier. The default is 1.0.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with time variant model data
    """
    # checks
    if len(model_ds.model_name) > 16 and model_ds.mfversion == 'mf6':
        raise ValueError('model_name can not have more than 16 characters')
    elif time_units.lower() != 'days':
        raise NotImplementedError()
    if time is not None:
        start_time = time[0]
        perlen = np.diff(time) / pd.to_timedelta(1, unit=time_units)
        if steady_start:
            perlen = np.insert(perlen, 0, steady_start_perlen)

    if start_time is None:
        start_time = '2000'
    start_time = pd.to_datetime(start_time)

    if steady_state:
        if isinstance(perlen, (float, int)):
            nper = 1
            perlen = [perlen] * nper
        else:
            nper = len(perlen)
    else:
        if isinstance(perlen, (float, int)):
            perlen = [perlen] * (transient_timesteps + steady_start)
        if steady_start:
            perlen[0] = steady_start_perlen
        nper = len(perlen)

        if hasattr(perlen, '__iter__'):
            transient_timesteps = len(perlen)
            if steady_start:
                transient_timesteps = transient_timesteps - 1
        nper = transient_timesteps
        if steady_start:
            start_time = start_time - dt.timedelta(days=perlen[0])
    time_dt = start_time + np.cumsum(pd.to_timedelta(perlen, unit=time_units))

    model_ds = model_ds.assign_coords(coords={'time': time_dt})

    model_ds.attrs['time_units'] = time_units
    model_ds.attrs['start_time'] = str(start_time)
    model_ds.attrs['nstp'] = nstp
    model_ds.attrs['tsmult'] = tsmult

    # netcdf files cannot handle booleans
    model_ds.attrs['steady_start'] = int(steady_start)
    model_ds.attrs['steady_state'] = int(steady_state)

    return model_ds
