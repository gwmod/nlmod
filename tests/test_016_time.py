import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd

import cftime
import nlmod
import pytest

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime, OutOfBoundsTimedelta


def test_estimate_nstp():
    forcing = [0.0, 10.0] + 100 * [0.0]
    nstp_min, nstp_max = 1, 25
    tsmult = 1.01
    nstp, dt_arr = nlmod.time.estimate_nstp(
        forcing=forcing,
        tsmult=tsmult,
        nstp_min=nstp_min,
        nstp_max=nstp_max,
        return_dt_arr=True,
    )

    assert nstp[0] == nstp_min
    assert nstp[-1] == nstp_min
    assert max(nstp) == nstp_max
    assert min(nstp) == nstp_min


def test_ds_time_from_tdis_settings():
    tidx = nlmod.time.ds_time_idx_from_tdis_settings(
        "2000", [100, 100, 100], nstp=[1, 2, 2], tsmult=[1.0, 1.0, 2.0]
    )

    elapsed = (tidx.to_numpy() - np.datetime64("2000")) / np.timedelta64(1, "D")
    assert np.allclose(elapsed, [100, 150, 200, 233.33333333, 300.0])


def test_get_time_step_length():
    assert (nlmod.time.get_time_step_length(100, 2, 1.5) == np.array([40, 60])).all()


def test_time_options():
    """Attempt to list all the variations of start, time and perlen
    caling the nlmod.dims.set_ds_time functions
    """

    ds = nlmod.get_ds([0, 1000, 2000, 3000])

    # start_time str and time int
    _ = nlmod.dims.set_ds_time(ds, start="2000-1-1", time=10)

    # start_time str and time list of int
    _ = nlmod.dims.set_ds_time(ds, start="2000-1-1", time=[10, 40, 50])

    # start_time str and time list of timestamps
    _ = nlmod.dims.set_ds_time(
        ds, start="2000-1-1", time=pd.to_datetime(["2000-2-1", "2000-3-1"])
    )

    # start_time str and time list of str
    _ = nlmod.dims.set_ds_time(ds, start="2000-1-1", time=["2000-2-1", "2000-3-1"])

    # start_time int and time timestamp
    _ = nlmod.dims.set_ds_time(ds, start=5, time=pd.Timestamp("2000-1-1"))

    # start_time timestamp and time timestamp
    _ = nlmod.dims.set_ds_time(
        ds, start=pd.Timestamp("2000-1-1"), time=pd.Timestamp("2000-2-1")
    )

    # start_time timestamp and time int
    _ = nlmod.dims.set_ds_time(ds, start=pd.Timestamp("2000-1-1"), time=10)

    # start_time timestamp and time str
    _ = nlmod.dims.set_ds_time(ds, start=pd.Timestamp("2000-1-1"), time="2000-2-1")

    # start_time timestamp and time list of timestamp
    _ = nlmod.dims.set_ds_time(
        ds,
        start=pd.Timestamp("2000-1-1"),
        time=pd.to_datetime(["2000-2-1", "2000-3-1"]),
    )

    # start_time str and perlen int
    _ = nlmod.dims.set_ds_time(ds, start="2000-1-1", perlen=10)

    # start_time str and perlen list of int
    _ = nlmod.dims.set_ds_time(ds, start="2000-1-1", perlen=[10, 30, 50])

    # start_time timestamp and perlen list of int
    _ = nlmod.dims.set_ds_time(ds, start=pd.Timestamp("2000-1-1"), perlen=[10, 30, 50])


def test_time_out_of_bounds():
    """related to this issue: https://github.com/gwmod/nlmod/issues/374

    pandas timestamps can only do computations with dates between the years 1678 and 2262.
    """

    ds = nlmod.get_ds([0, 1000, 2000, 3000])

    cftime_ind = xr.date_range("1000-01-02", "9999-01-01", freq="100YS")
    start_model = cftime.datetime(1000, 1, 1)

    # start cf.datetime and time CFTimeIndex
    _ = nlmod.dims.set_ds_time(ds, start=start_model, time=cftime_ind)

    # start cf.datetime and time list of int
    _ = nlmod.dims.set_ds_time(ds, start=start_model, time=[10, 20, 21, 55])

    ds = nlmod.dims.set_ds_time(ds, time="1000-1-1", start=1)

    # start cf.datetime and time list of int
    _ = nlmod.dims.set_ds_time(
        ds, start=start_model, time=pd.to_datetime(["2000-2-1", "2000-3-1"])
    )

    # start cf.datetime and time list of str (no general method to convert str to cftime)
    with pytest.raises(TypeError):
        nlmod.dims.set_ds_time(ds, start=start_model, time=["1000-01-02", "1000-01-03"])

    # start cf.datetime and perlen int
    _ = nlmod.dims.set_ds_time(ds, start=start_model, perlen=365)

    # start str and time CFTimeIndex
    _ = nlmod.dims.set_ds_time(ds, start="1000-01-01", time=cftime_ind)

    # start str and time int
    _ = nlmod.dims.set_ds_time(ds, start="1000-01-01", time=1)

    # start str and time list of int
    _ = nlmod.dims.set_ds_time(ds, start="1000-01-01", time=[10, 20, 21, 55])

    # start str and time list of timestamp
    _ = nlmod.dims.set_ds_time(
        ds, start="1000-01-01", time=pd.to_datetime(["2000-2-1", "2000-3-1"])
    )

    # start str and time list of str (no general method to convert str to cftime)
    with pytest.raises(TypeError):
        nlmod.dims.set_ds_time(ds, start="1000-01-01", time=["1000-2-1", "1000-3-1"])

    # start str and perlen int
    _ = nlmod.dims.set_ds_time(ds, start="1000-01-01", perlen=365000)

    # start numpy datetime and perlen list of int
    _ = nlmod.dims.set_ds_time(
        ds, start=np.datetime64("1000-01-01"), perlen=[10, 100, 24]
    )

    # start numpy datetime and time list of timestamps
    _ = nlmod.dims.set_ds_time(
        ds,
        start=np.datetime64("1000-01-01"),
        time=pd.to_datetime(["2000-2-1", "2000-3-1"]),
    )

    # start numpy datetime and time list of str
    with pytest.raises(TypeError):
        nlmod.dims.set_ds_time(
            ds, start=np.datetime64("1000-01-01"), time=["1000-2-1", "1000-3-1"]
        )

    # start timestamp and perlen list of int
    _ = nlmod.dims.set_ds_time(
        ds, start=pd.Timestamp("1000-01-01"), perlen=[10, 100, 24]
    )

    # start timestamp and time CFTimeIndex
    _ = nlmod.dims.set_ds_time(ds, start=pd.Timestamp("1000-01-01"), time=cftime_ind)

    # start int and time CFTimeIndex
    _ = nlmod.dims.set_ds_time(ds, start=96500, time=cftime_ind)

    # start int and time timestamp
    _ = nlmod.dims.set_ds_time(ds, start=96500, time=pd.Timestamp("1000-01-01"))

    # start int and time str
    _ = nlmod.dims.set_ds_time(ds, start=96500, time="1000-01-01")

    # start befor christ (BC), not yet working
    # start = '-0500-01-01'
    # end = '2000-01-01'
    # time = xr.date_range(start, end, freq='100YS')
    # start_model = time[0] - timedelta(days=10*365+2)

    # ds = nlmod.time.nlmod.dims.set_ds_time(ds, start=start_model, time=time)
