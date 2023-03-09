# -*- coding: utf-8 -*-
"""add knmi precipitation and evaporation to a modflow model."""


import logging

import flopy
import numpy as np
from tqdm import tqdm

from ..dims.grid import da_to_reclist
from ..sim.sim import get_tdis_perioddata

logger = logging.getLogger(__name__)


def model_datasets_to_rch(gwf, ds, mask=None, pname="rch", **kwargs):
    """Convert the recharge data in the model dataset to a rch package with
    time series.

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    ds : xr.DataSet
        dataset containing relevant model grid information
    mask : xr.DataArray
        data array containing mask, recharge is only added where mask is True
    pname : str, optional
        package name. The default is 'rch'.

    Returns
    -------
    rch : flopy.mf6.modflow.mfgwfrch.ModflowGwfrch
        recharge package
    """
    # check for nan values
    if ds["recharge"].isnull().any():
        raise ValueError("please remove nan values in recharge data array")

    # get stress period data
    rch_name_arr, rch_unique_dic = _get_unique_series(ds, "recharge", pname)
    ds["rch_name"] = ds["top"].dims, rch_name_arr
    if mask is not None:
        mask = (ds["rch_name"] != "") & mask
    else:
        mask = ds["rch_name"] != ""

    recharge = "rch_name"

    spd = da_to_reclist(
        ds,
        mask,
        col1=recharge,
        first_active_layer=True,
        only_active_cells=False,
    )

    # create rch package
    rch = flopy.mf6.ModflowGwfrch(
        gwf,
        filename=f"{gwf.name}.rch",
        pname=pname,
        fixed_cell=False,
        maxbound=len(spd),
        stress_period_data={0: spd},
        **kwargs,
    )

    # create timeseries packages
    _add_time_series(rch, rch_unique_dic, ds)

    return rch


def model_datasets_to_evt(
    gwf, ds, pname="evt", nseg=1, surface=None, depth=None, **kwargs
):
    """Convert the evaporation data in the model dataset to a evt package with
    time series.

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    ds : xr.DataSet
        dataset containing relevant model grid information
    pname : str, optional
        package name. The default is 'evt'.
    nseg : int, optional
        number of ET segments. Only 1 is supported for now. The default is 1.
    surface : str, float or xr.DataArray, optional
        The elevation of the ET surface. Set to 1 meter below top when None. The default
        is None.
    depth : str, float or xr.DataArray, optional
        The ET extinction depth. Set to 1 meter (below surface) when None. The default
        is None.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------

        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    evt : flopy.mf6.modflow.mfgwfevt.ModflowGwfevt
        evapotranspiiration package.
    """
    assert nseg == 1, "More than one evaporation segment not yet supported"
    if "surf_rate_specified" in kwargs:
        raise (Exception("surf_rate_specified not yet supported"))
    if surface is None:
        logger.info("Setting evaporation surface to 1 meter below top")
        surface = ds["top"] - 1.0
    if depth is None:
        logger.info("Setting extinction depth to 1 meter below surface")
        depth = 1.0

    if ds["evaporation"].isnull().any():
        raise ValueError("please remove nan values in evaporation data array")

    # get stress period data
    if ds.time.steady_state:
        mask = ds["evaporation"] != 0
        rate = "evaporation"
    else:
        evt_name_arr, evt_unique_dic = _get_unique_series(ds, "evaporation", pname)
        ds["evt_name"] = ds["top"].dims, evt_name_arr

        mask = ds["evt_name"] != ""
        rate = "evt_name"

    spd = da_to_reclist(
        ds,
        mask,
        col1=surface,
        col2=rate,
        col3=depth,
        first_active_layer=True,
        only_active_cells=False,
    )

    # create rch package
    evt = flopy.mf6.ModflowGwfevt(
        gwf,
        filename=f"{gwf.name}.evt",
        pname=pname,
        fixed_cell=False,
        maxbound=len(spd),
        nseg=nseg,
        stress_period_data={0: spd},
        **kwargs,
    )

    if ds.time.steady_state:
        return evt

    # create timeseries packages
    _add_time_series(evt, evt_unique_dic, ds)

    return evt


def _get_unique_series(ds, var, pname):
    """Get the location and values of unique time series from a variable var in
    ds.

    Parameters
    ----------
    ds : xr.Dataset
        The model Dataset.
    var : str
        The 3d (structured) or 2d (vertext) variable in ds that contains the timeseries.
    pname : str
        Package name, which is used for the name of the time series.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    rch_name : np.ndarray
        The name of the recharge series for each of the cells.
    rch_unique_dic : dict
        The values of each of the time series.
    """
    rch_name_arr = np.empty_like(ds["top"].values, dtype="U13")

    # transient
    if ds.gridtype == "structured":
        if len(ds[var].dims) != 3:
            raise ValueError(
                "expected dataarray with 3 dimensions"
                f"(time, y and x) or (y, x and time), not {ds[var].dims}"
            )
        recharge = ds[var].transpose("y", "x", "time").data
        shape = (ds.dims["y"] * ds.dims["x"], ds.dims["time"])
        rch_2d_arr = recharge.reshape(shape)

    elif ds.gridtype == "vertex":
        # dimension check
        if len(ds[var].dims) != 2:
            raise ValueError(
                "expected dataarray with 2 dimensions"
                f"(time, icell2d) or (icell2d, time), not {ds[var].dims}"
            )
        rch_2d_arr = ds[var].transpose("icell2d", "time").data

    rch_unique_arr = np.unique(rch_2d_arr, axis=0)
    rch_unique_dic = {}
    for i, unique_rch in enumerate(rch_unique_arr):
        mask = (rch_2d_arr == unique_rch).all(axis=1)
        if len(rch_name_arr.shape) > 1:
            mask = mask.reshape(rch_name_arr.shape)
        rch_name_arr[mask] = f"{pname}_{i}"
        rch_unique_dic[f"{pname}_{i}"] = unique_rch

    return rch_name_arr, rch_unique_dic


def _add_time_series(package, rch_unique_dic, ds):
    """Add time series to a package.

    Parameters
    ----------
    rch : mfpackage.MFPackage
        The Flopy package to which to add the timeseries.
    rch_unique_dic : dict
        A dictionary whch contains the time series values.
    ds : xr.Dataset
        The model Dataset. It is used to get the time of the time series.

    Returns
    -------
    None.
    """
    # get timesteps
    tdis_perioddata = get_tdis_perioddata(ds)
    perlen_arr = [t[0] for t in tdis_perioddata]
    time_steps_rch = [0.0] + np.array(perlen_arr).cumsum().tolist()

    for i, key in tqdm(
        enumerate(rch_unique_dic.keys()),
        total=len(rch_unique_dic.keys()),
        desc="Building ts packages rch",
    ):
        # add extra time step to the time series object (otherwise flopy fails)
        recharge_val = list(rch_unique_dic[key]) + [0.0]

        recharge = list(zip(time_steps_rch, recharge_val))
        if i == 0:
            package.ts.initialize(
                filename=f"{key}.ts",
                timeseries=recharge,
                time_series_namerecord=key,
                interpolation_methodrecord="stepwise",
            )
        else:
            package.ts.append_package(
                filename=f"{key}.ts",
                timeseries=recharge,
                time_series_namerecord=key,
                interpolation_methodrecord="stepwise",
            )
