# -*- coding: utf-8 -*-
"""Created on Fri Apr 17 13:50:48 2020.

@author: oebbe
"""

import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def set_ds_time(
    ds,
    time=None,
    steady_state=False,
    steady_start=True,
    steady_start_perlen=3652.0,
    time_units="DAYS",
    start_time=None,
    transient_timesteps=0,
    perlen=1.0,
    nstp=1,
    tsmult=1.0,
):
    """Set timing for a model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
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
    ds : xarray.Dataset
        dataset with time variant model data
    """
    # checks
    if time_units.lower() != "days":
        raise NotImplementedError()
    if time is not None:
        start_time = time[0]
        perlen = np.diff(time) / pd.to_timedelta(1, unit=time_units)
        if steady_start:
            perlen = np.insert(perlen, 0, steady_start_perlen)

    if start_time is None:
        start_time = "2000"
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

        if hasattr(perlen, "__iter__"):
            transient_timesteps = len(perlen)
            if steady_start:
                transient_timesteps = transient_timesteps - 1
        nper = transient_timesteps
        if steady_start:
            start_time = start_time - dt.timedelta(days=perlen[0])
    time_dt = start_time + np.cumsum(pd.to_timedelta(perlen, unit=time_units))

    ds = ds.assign_coords(coords={"time": time_dt})

    ds.time.attrs["time_units"] = time_units
    ds.time.attrs["start_time"] = str(start_time)
    ds.time.attrs["nstp"] = nstp
    ds.time.attrs["tsmult"] = tsmult

    # netcdf files cannot handle booleans
    ds.time.attrs["steady_start"] = int(steady_start)
    ds.time.attrs["steady_state"] = int(steady_state)

    return ds


def estimate_nstp(forcing, perlen=1, tsmult=1.1, nstp_min=1, nstp_max=25):
    """Scale the nstp's linearly between the min and max of the forcing.

    Ensures that the first time step of this stress period connects to the
    last time step of the previous stress period. The ratio between the
    two time-step durations can be at most tsmult.

    Parameters
    ----------
    forcing : array-like
        Array with a forcing value for each stress period. Forcing can be
        for example a pumping rate of a rainfall intensity.
    perlen : float or array of floats (nper)
        An array of the stress period lengths.
    nstp : int or array of ints (nper)
        Number of time steps in each stress period.
    nstp_min : int
        nstp value for the stress period with the smallest forcing.
    nstp_max : int
        nstp value for the stress period with the largest forcing.


    Returns
    -------
    nstp : np.ndarray
        Array with a nstp for each stress period.
    """

    nt = len(forcing)

    # Scaled linear between min and max. array nstp will be modified along the way
    nstp = np.round((forcing - forcing.min()) /
                    (forcing.max() - forcing.min()) * (nstp_max - nstp_min)
                    + nstp_min).astype(int)
    perlen = np.full(nt, fill_value=perlen)
    tsmult = np.full(nt, fill_value=tsmult)

    # Duration of the first time step of each stress period. Equation TM6A16 p.4-5 eq.1
    dt0_arr = perlen * (tsmult - 1) / (tsmult ** nstp - 1)
    # Add axis so that durations of subsequent time steps can be added for each stress period.
    dt_arr = [[i] for i in dt0_arr]

    for i, (perleni, tsmulti, nstpi, dt0) in enumerate(zip(perlen, tsmult, nstp, dt0_arr)):
        for _ in range(nstpi - 1):
            dt_i = dt_arr[i][-1] * tsmulti
            dt_arr[i].append(dt_i)

        # reduce the length of the first time step of the next period to match the last time step of this period.
        dt0_next = dt_arr[i][-1] * tsmulti
        if i <= nt - 2 and dt0_next < dt_arr[i + 1][0]:
            # Equation derived from TM6A16 p.4-5 eq.1
            nstp_next = np.round(np.log(perleni * (tsmulti - 1) / dt0_next + 1) / np.log(tsmulti)).astype(int)
            nstp[i + 1] = nstp_next

            # Equation TM6A16 p.4-5 eq.1
            dt_arr[i + 1] = [perlen[i + 1] * (tsmult[i + 1] - 1) / (tsmult[i + 1] ** nstp_next - 1)]

    nstp = [len(i) for i in dt_arr]
    return nstp
