import datetime as dt
import logging
import warnings

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime, OutOfBoundsTimedelta
from xarray import IndexVariable

from .attributes_encodings import dim_attrs

logger = logging.getLogger(__name__)


def set_ds_time_deprecated(
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
    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.time.set_ds_time() in the future.",
        DeprecationWarning,
    )

    # checks
    if time_units.lower() != "days":
        raise NotImplementedError()
    if time is not None:
        if isinstance(time, str):
            time = pd.to_datetime(time)
        if not hasattr(time, "__iter__"):
            time = [time]
        start_time = time[0]
        if len(time) > 1:
            perlen = np.diff(time) / pd.to_timedelta(1, unit=time_units)
        else:
            perlen = []
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
    ds.time.attrs["start"] = str(start_time)
    ds.time.attrs["nstp"] = nstp
    ds.time.attrs["tsmult"] = tsmult

    # netcdf files cannot handle booleans
    ds.time.attrs["steady_start"] = int(steady_start)
    ds.time.attrs["steady_state"] = int(steady_state)

    # add to ds (for new version nlmod)
    # add steady, nstp and tsmult to dataset
    steady = int(steady_state) * np.ones(len(time_dt), dtype=int)
    if steady_start:
        steady[0] = 1
    ds["steady"] = ("time",), steady

    if isinstance(nstp, (int, np.integer)):
        nstp = nstp * np.ones(len(time), dtype=int)
    ds["nstp"] = ("time",), nstp

    if isinstance(tsmult, float):
        tsmult = tsmult * np.ones(len(time))
    ds["tsmult"] = ("time",), tsmult

    return ds


def _pd_timestamp_to_cftime(time_pd):
    """Convert a pandas timestamp into a cftime stamp

    Parameters
    ----------
    time_pd : pd.Timestamp or list of pd.Timestamp
        datetimes

    Returns
    -------
    cftime.datetime or list of cftime.datetime
    """
    if hasattr(time_pd, "__iter__"):
        return [_pd_timestamp_to_cftime(tpd) for tpd in time_pd]
    else:
        return cftime.datetime(
            time_pd.year,
            time_pd.month,
            time_pd.day,
            time_pd.hour,
            time_pd.minute,
            time_pd.second,
        )


def set_ds_time(
    ds,
    start,
    time=None,
    perlen=None,
    steady=False,
    steady_start=True,
    time_units="DAYS",
    nstp=1,
    tsmult=1.0,
):
    """Set time discretisation for model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset
    start : int, float, str, pandas.Timestamp or cftime.datetime
        model start. When start is an integer or float it is interpreted as the number
        of days of the first stress-period. When start is a string, pandas Timestamp or
        cftime datetime it is the start datetime of the simulation. Use cftime datetime
        when you get an OutOfBounds error using pandas.
    time : float, int or array-like, optional
        float(s) (indicating elapsed time) or timestamp(s) corresponding to the end of
        each stress period in the model. When time is a single value, the model will
        have only one stress period. When time is None, the stress period lengths have
        to be supplied via perlen. The default is None.
    perlen : float, int or array-like, optional
        length of each stress-period. Only used when time is None. When perlen is a
        single value, the model will have only one stress period. The default is None.
    steady : arraylike or bool, optional
        arraylike indicating which stress periods are steady-state, by default False,
        which sets all stress periods to transient with the first period determined by
        value of `steady_start`.
    steady_start : bool, optional
        whether to set the first period to steady-state, default is True, only used
        when steady is passed as single boolean.
    time_units : str, optional
        time units, by default "DAYS"
    nstp : int or array-like, optional
        number of steps per stress period, stored in ds.attrs, default is 1
    tsmult : float, optional
        timestep multiplier within stress periods, stored in ds.attrs, default is 1.0

    Returns
    -------
    ds : xarray.Dataset
        model dataset with added time coordinate
    """
    if time is None and perlen is None:
        raise (ValueError("Please specify either time or perlen in set_ds_time"))
    elif perlen is not None:
        if time is not None:
            msg = f"Cannot use both time and perlen. Ignoring perlen: {perlen}"
            logger.warning(msg)
        else:
            if isinstance(perlen, (int, np.integer, float)):
                perlen = [perlen]
            time = np.cumsum(perlen)

    if isinstance(time, str) or not hasattr(time, "__iter__"):
        time = [time]

    try:
        # parse start
        if isinstance(start, (int, np.integer, float)):
            if isinstance(time[0], (int, np.integer, float, str)):
                raise TypeError(
                    "Make sure 'start' or 'time' argument is a valid TimeStamp"
                )
            start = time[0] - pd.to_timedelta(start, "D")
        elif isinstance(start, str):
            start = pd.Timestamp(start)
        elif isinstance(start, (pd.Timestamp, cftime.datetime)):
            pass
        elif isinstance(start, np.datetime64):
            start = pd.Timestamp(start)
        else:
            raise TypeError("Cannot parse start datetime.")

        # parse time make sure 'time' and 'start' are same type (pd.Timestamps or cftime.datetime)
        if isinstance(time[0], (int, np.integer, float)):
            if isinstance(start, cftime.datetime):
                time = [start + dt.timedelta(days=int(td)) for td in time]
            else:
                time = start + pd.to_timedelta(time, time_units)
        elif isinstance(time[0], str):
            time = pd.to_datetime(time)
            if isinstance(start, cftime.datetime):
                time = _pd_timestamp_to_cftime(time)
        elif isinstance(time[0], (pd.Timestamp)):
            if isinstance(start, cftime.datetime):
                time = _pd_timestamp_to_cftime(time)
        elif isinstance(time[0], (np.datetime64, xr.core.variable.Variable)):
            logger.info(
                "time arguments with types np.datetime64, xr.core.variable.Variable not tested!"
            )
        elif isinstance(time[0], cftime.datetime):
            start = _pd_timestamp_to_cftime(start)
        else:
            msg = f"Cannot process 'time' argument. Datatype -> {type(time)} not understood."
            raise TypeError(msg)
    except (OutOfBoundsDatetime, OutOfBoundsTimedelta) as e:
        msg = (
            "cannot convert 'start' and 'time' to pandas datetime, use cftime "
            "types for 'start' and 'time'"
        )
        raise type(e)(msg) from e

    if time[0] <= start:
        msg = (
            "The timestamp of the first stress period cannot be before or "
            "equal to the model start time! Please modify `time` or `start`!"
        )
        logger.error(msg)
        raise ValueError(msg)

    # create time coordinates
    ds = ds.assign_coords(coords={"time": time})
    ds.coords["time"].attrs = dim_attrs["time"]

    # add steady, nstp and tsmult to dataset
    ds = set_time_variables(
        ds, start, time, steady, steady_start, time_units, nstp, tsmult
    )

    return ds


def set_ds_time_numeric(
    ds,
    start,
    time=None,
    perlen=None,
    steady=False,
    steady_start=True,
    time_units="DAYS",
    nstp=1,
    tsmult=1.0,
):
    """Set a numerical time discretisation for a model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset
    start : int, float, str, pandas.Timestamp or cftime.datetime
        model start. When start is an integer or float it is interpreted as the number
        of days of the first stress-period. When start is a string, pandas Timestamp or
        cftime datetime it is the start datetime of the simulation. Use cftime datetime
        when you get an OutOfBounds error using pandas.
    time : float, int or array-like, optional
        float(s) (indicating elapsed time) corresponding to the end of each stress
        period in the model. When time is a single value, the model will have only one
        stress period. When time is None, the stress period lengths have to be supplied
        via perlen. The default is None.
    perlen : float, int or array-like, optional
        length of each stress-period. Only used when time is None. When perlen is a
        single value, the model will have only one stress period. The default is None.
    steady : arraylike or bool, optional
        arraylike indicating which stress periods are steady-state, by default False,
        which sets all stress periods to transient with the first period determined by
        value of `steady_start`.
    steady_start : bool, optional
        whether to set the first period to steady-state, default is True, only used
        when steady is passed as single boolean.
    time_units : str, optional
        time units, by default "DAYS"
    nstp : int or array-like, optional
        number of steps per stress period, stored in ds.attrs, default is 1
    tsmult : float, optional
        timestep multiplier within stress periods, stored in ds.attrs, default is 1.0

    Returns
    -------
    ds : xarray.Dataset
        model dataset with added time coordinate
    """
    if time is None and perlen is None:
        raise (ValueError("Please specify either time or perlen in set_ds_time"))
    elif perlen is not None:
        if time is not None:
            msg = f"Cannot use both time and perlen. Ignoring perlen: {perlen}"
            logger.warning(msg)
        else:
            if isinstance(perlen, (int, np.integer, float)):
                perlen = [perlen]
            time = np.cumsum(perlen)

    if isinstance(time, str) or not hasattr(time, "__iter__"):
        time = [time]

    # check time
    if isinstance(time[0], (str, pd.Timestamp, cftime.datetime)):
        raise TypeError("'time' argument should be of a numerical type")

    time = np.asarray(time, dtype=float)
    if (time <= 0.0).any():
        msg = "timesteps smaller or equal to 0 are not allowed"
        logger.error(msg)
        raise ValueError(msg)

    # create time coordinates
    ds = ds.assign_coords(coords={"time": time})
    ds.coords["time"].attrs = dim_attrs["time"]

    # add steady, nstp and tsmult to dataset
    ds = set_time_variables(
        ds, start, time, steady, steady_start, time_units, nstp, tsmult
    )

    return ds


def set_time_variables(ds, start, time, steady, steady_start, time_units, nstp, tsmult):
    """Add data variables: steady, nstp and tsmult, set attributes: start, time_units

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset
    start : int, float, str, pandas.Timestamp or cftime.datetime
        model start. When start is an integer or float it is interpreted as the number
        of days of the first stress-period. When start is a string, pandas Timestamp or
        cftime datetime it is the start datetime of the simulation. Use cftime datetime
        when you get an OutOfBounds error using pandas.
    time : array-like, optional
        numerical (indicating elapsed time) or timestamps corresponding to the end
        of each stress period in the model.
    steady : arraylike or bool, optional
        arraylike indicating which stress periods are steady-state, by default False,
        which sets all stress periods to transient with the first period determined by
        value of `steady_start`.
    steady_start : bool, optional
        whether to set the first period to steady-state, default is True, only used
        when steady is passed as single boolean.
    time_units : str, optional
        time units, by default "DAYS"
    nstp : int or array-like, optional
        number of steps per stress period, stored in ds.attrs, default is 1
    tsmult : float, optional
        timestep multiplier within stress periods, stored in ds.attrs, default is 1.0

    Returns
    -------
    ds : xarray.Dataset
        model dataset with attributes added to the time coordinates
    """
    # add steady, nstp and tsmult to dataset
    if isinstance(steady, bool):
        steady = int(steady) * np.ones(len(time), dtype=int)
        if steady_start:
            steady[0] = 1
    ds["steady"] = ("time",), steady

    if isinstance(nstp, (int, np.integer)):
        nstp = nstp * np.ones(len(time), dtype=int)
    ds["nstp"] = ("time",), nstp

    if isinstance(tsmult, float):
        tsmult = tsmult * np.ones(len(time))
    ds["tsmult"] = ("time",), tsmult

    if time_units == "D":
        time_units = "DAYS"
    ds.time.attrs["time_units"] = time_units
    ds.time.attrs["start"] = str(start)

    return ds


def ds_time_idx_from_tdis_settings(start, perlen, nstp=1, tsmult=1.0, time_units="D"):
    """Get time index from TDIS perioddata: perlen, nstp, tsmult.

    Parameters
    ----------
    start : str, pd.Timestamp
        start datetime
    perlen : array-like
        array of period lengths
    nstp : int, or array-like optional
        number of steps per period, by default 1
    tsmult : float or array-like, optional
        timestep multiplier per period, by default 1.0
    time_units : str, optional
        time units, by default "D"

    Returns
    -------
    IndexVariable
        time coordinate for xarray data-array or dataset
    """
    deltlist = []
    for kper, delt in enumerate(perlen):
        if not isinstance(nstp, int):
            kstpkper = nstp[kper]
        else:
            kstpkper = nstp

        if not isinstance(tsmult, float):
            tsm = tsmult[kper]
        else:
            tsm = tsmult

        if tsm > 1.0:
            delt0 = delt * (tsm - 1) / (tsm**kstpkper - 1)
            delt = delt0 * tsm ** np.arange(kstpkper)
        else:
            delt = np.ones(kstpkper) * delt / kstpkper
        deltlist.append(delt)

    dt_arr = np.cumsum(np.concatenate(deltlist))

    return ds_time_idx(dt_arr, start_datetime=start, time_units=time_units)


def estimate_nstp(
    forcing, perlen=1, tsmult=1.1, nstp_min=1, nstp_max=25, return_dt_arr=False
):
    """Scale the nstp's linearly between the min and max of the forcing.

    Ensures that the first time step of this stress period connects to the
    last time step of the previous stress period. The ratio between the
    two time-step durations can be at most tsmult.

    Parameters
    ----------
    forcing : array-like
        Array with a forcing value for each stress period. Forcing can be
        for example a pumping rate or a rainfall intensity.
    perlen : float or array of floats (nper)
        An array of the stress period lengths.
    tsmult : float or array of floats (nper)
        Time step multiplier.
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
    dt_arr : np.ndarray (optional)
        if `return_dt_arr` is `True` returns the durations of the timesteps
        corresponding with the returned nstp.
    """
    nt = len(forcing)

    # Scaled linear between min and max. array nstp will be modified along the way
    nstp = (forcing - np.min(forcing)) / (np.max(forcing) - np.min(forcing)) * (
        nstp_max - nstp_min
    ) + nstp_min
    perlen = np.full(nt, fill_value=perlen)
    tsmult = np.full(nt, fill_value=tsmult)

    # Duration of the first time step of each stress period. Equation TM6A16 p.4-5 eq.1
    dt0_arr = np.where(
        tsmult == 1.0,
        perlen / nstp,
        perlen * (tsmult - 1) / (tsmult**nstp - 1),
    )

    for i in range(nt - 1):
        dt_end = dt0_arr[i] * tsmult[i] ** nstp[i]
        dt0_next = dt_end * tsmult[i + 1]

        if dt0_next < dt0_arr[i + 1]:
            dt0_arr[i + 1] = dt0_next

            # Equation derived from TM6A16 p.4-5 eq.1
            if tsmult[i + 1] == 1.0:
                nstp[i + 1] = perlen[i + 1] / dt0_arr[i + 1]
            else:
                nstp[i + 1] = np.log(
                    perlen[i + 1] * (tsmult[i + 1] - 1) / dt0_next + 1
                ) / np.log(tsmult[i + 1])

    nstp_ceiled = np.ceil(nstp).astype(int)

    if return_dt_arr:
        dt0_ceiled = np.where(
            tsmult == 1.0,
            perlen / nstp_ceiled,
            perlen * (tsmult - 1) / (tsmult**nstp_ceiled - 1),
        )
        dt_lists = [
            [dt0i * tsmulti**nstpii for nstpii in range(nstpi)]
            for dt0i, nstpi, tsmulti in zip(dt0_ceiled, nstp_ceiled, tsmult)
        ]
        dt_arr = np.concatenate(dt_lists)

        return nstp_ceiled, dt_arr

    else:
        return nstp_ceiled


def get_perlen(ds):
    """Get perlen from ds.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with time variant model data

    Returns
    -------
    perlen : list of floats
        length of a stress period.
    """
    deltat = pd.to_timedelta(1, ds.time.time_units)
    if ds.time.dtype.kind == "M":
        # dtype is pandas timestamps
        perlen = [
            (pd.to_datetime(ds["time"].data[0]) - pd.to_datetime(ds.time.start))
            / deltat
        ]
        if len(ds["time"]) > 1:
            perlen.extend(np.diff(ds["time"]) / deltat)
    elif ds.time.dtype.kind == "O":
        perlen = [
            (ds["time"].data[0] - _pd_timestamp_to_cftime(pd.Timestamp(ds.time.start)))
            / deltat
        ]
        if len(ds["time"]) > 1:
            perlen.extend(np.diff(ds["time"]) / deltat)
    elif ds.time.dtype.kind in ["i", "f"]:
        perlen = [ds["time"][0]]
        perlen.extend(np.diff(ds["time"].values))
    return perlen


def get_time_step_length(perlen, nstp, tsmult):
    """Get the length of the timesteps within a single stress-period.

    Parameters
    ----------
    perlen : float
        The length of the stress period, in the time unit of the model (generally days).
    nstp : int
        The numer of timesteps within the stress period.
    tsmult : float
        THe time step multiplier, generally equal or lager than 1.

    Returns
    -------
    t : np.ndarray
        An array with the length of each of the timesteps within the stress period, in
        the same unit as perlen.
    """
    t = np.array([tsmult**x for x in range(nstp)])
    t = t * perlen / t.sum()
    return t


def ds_time_from_model(gwf):
    warnings.warn(
        "this function was renamed to `ds_time_idx_from_model`. "
        "Please use the new function name.",
        DeprecationWarning,
    )
    return ds_time_idx_from_model(gwf)


def ds_time_idx_from_model(gwf):
    """Get time index variable from model (gwf or gwt).

    Parameters
    ----------
    gwf : flopy MFModel object
        groundwater flow or groundwater transport model

    Returns
    -------
    IndexVariable
        time coordinate for xarray data-array or dataset
    """
    return ds_time_idx_from_modeltime(gwf.modeltime)


def ds_time_from_modeltime(modeltime):
    warnings.warn(
        "this function was renamed to `ds_time_idx_from_model`. "
        "Please use the new function name.",
        DeprecationWarning,
    )
    return ds_time_idx_from_modeltime(modeltime)


def ds_time_idx_from_modeltime(modeltime):
    """Get time index variable from modeltime object.

    Parameters
    ----------
    modeltime : flopy ModelTime object
        modeltime object (e.g. gwf.modeltime)

    Returns
    -------
    IndexVariable
        time coordinate for xarray data-array or dataset
    """
    return ds_time_idx(
        np.cumsum(modeltime.perlen),
        start_datetime=modeltime.start_datetime,
        time_units=modeltime.time_units,
    )


def ds_time_idx(t, start_datetime=None, time_units="D", dtype="datetime"):
    """Get time index variable from elapsed time array.

    Parameters
    ----------
    t : np.array
        array containing elapsed time, usually in days
    start_datetime : str, pd.Timestamp, optional
        starting datetime
    time_units : str, optional
        time units, default is days
    dtype : str, optional
        dtype of time index. Can be 'datetime' or 'float'. If 'datetime' try to create
        a pandas datetime index if that fails use cftime lib. Default is 'datetime'.

    Returns
    -------
    IndexVariable
        time coordinate for xarray data-array or dataset
    """
    if (start_datetime is None) or (dtype in ["int", "float"]):
        times = t
    else:
        try:
            dtarr = pd.to_timedelta(t, time_units)
            times = pd.Timestamp(start_datetime) + dtarr
        except (OutOfBoundsDatetime, OutOfBoundsTimedelta) as e:
            msg = f"using cftime time index because of {e}"
            logger.debug(msg)
            start = _pd_timestamp_to_cftime(pd.Timestamp(start_datetime))
            times = [start + dt.timedelta(days=int(td)) for td in t]

    time = IndexVariable(["time"], times)
    time.attrs["time_units"] = time_units
    if start_datetime is not None:
        time.attrs["start"] = str(start_datetime)

    return time


def dataframe_to_flopy_timeseries(
    df,
    ds=None,
    package=None,
    filename=None,
    time_series_namerecord=None,
    interpolation_methodrecord="stepwise",
    append=False,
    shift_label_from_right_to_left=False,
):
    """
    Convert a pandas DataFrame to a FloPy time series.

    This function converts a time-indexed DataFrame into a list of tuples
    formatted for use with FloPy's time series input. If a FloPy package is
    provided, it will either initialize or append a time series package.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series data. The index must be datetime-like
        if `ds` is provided.
    ds : xarray.Dataset, optional
        Model Dataset.
    package : flopy.mf6.ModflowGwfwel, flopy.mf6.ModflowGwfdrn or similar, optional
        The FloPy package to which the time series should be attached.
    filename : str, optional
        Filename for the time series package. If not provided, defaults to
        `{package.filename}_ts`.
    time_series_namerecord : list of str, optional
        List of time series names corresponding to DataFrame columns.
        If not provided, defaults to the DataFrame column names.
    interpolation_methodrecord : str or list of str, default "stepwise"
        Interpolation method(s) to use for the time series. Can be a single string
        or a list of methods (one for each column).
    append : bool, default False
        If True, the time series will be appended to an existing time series package.
        Otherwise, a new package will be initialized.
    shift_label_from_right_to_left : bool, default False
        If True, sets the index of df from the end of the timestep to the start of the
        timestep (which is required if interpolation_methodrecord="stepwise"). The
        default is False.

    Returns
    -------
    list of tuple or flopy.mf6.ModflowUtlts
        If `package` is None, returns a list of (time, value1, value2, ...) tuples.
        If `package` is provided, returns the created or updated time series package.

    Raises
    ------
    AssertionError
        If DataFrame contains NaNs or if `ds.time` is not datetime64.
    """

    assert not df.isna().any(axis=None)
    assert (
        ds.time.dtype.kind == "M"
    ), "get recharge requires a datetime64[ns] time index"

    if shift_label_from_right_to_left and ds is not None:
        df.loc[pd.to_datetime(ds.time.start)] = None
        df = df.sort_index().shift(-1).ffill()

    if ds is not None:
        # set index to days after the start of the simulation
        df = df.copy()
        df.index = (df.index - pd.to_datetime(ds.time.start)) / pd.Timedelta(1, "D")
    # generate a list of tuples with time as the first record, followed by the columns
    timeseries = [(i,) + tuple(v) for i, v in zip(df.index, df.values)]
    if package is None:
        return timeseries
    if filename is None:
        filename = f"{package.filename}_ts"
    if time_series_namerecord is None:
        time_series_namerecord = list(df.columns)

    if isinstance(interpolation_methodrecord, str):
        interpolation_methodrecord = [interpolation_methodrecord] * len(df.columns)

    # initialize or append a new package
    method = package.ts.append_package if append else package.ts.initialize
    return method(
        filename=filename,
        timeseries=timeseries,
        time_series_namerecord=time_series_namerecord,
        interpolation_methodrecord=interpolation_methodrecord,
    )


def ds_time_to_pandas_index(da_or_ds, include_start=True):
    """Convert xarray time index to pandas datetime index.

    Parameters
    ----------
    da_or_ds : xarray.Dataset or xarray.DataArray
        dataset with time index or time data array
    include_start : bool, optional
        include the start time in the index, by default True

    Returns
    -------
    pd.DatetimeIndex
        pandas datetime index
    """
    if isinstance(da_or_ds, xr.Dataset):
        da = da_or_ds["time"]
    elif isinstance(da_or_ds, xr.DataArray):
        da = da_or_ds
    else:
        raise ValueError(
            "Either pass model dataset or time dataarray (e.g. `ds.time`)."
        )

    if include_start:
        if da.dtype.kind == "M":  # "M" is a numpy datetime-type
            return da.to_index().insert(0, pd.Timestamp(da.start))
        elif da.dtype.kind == "O":
            start = _pd_timestamp_to_cftime(pd.Timestamp(da.start))
            return da.to_index().insert(0, start)
        elif da.dtype.kind in ["i", "f"]:
            return da.to_index().insert(0, 0)
        else:
            raise TypeError("Unknown time dtype")
    else:
        return da.to_index()
