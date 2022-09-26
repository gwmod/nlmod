# -*- coding: utf-8 -*-
"""add knmi precipitation and evaporation to a modflow model."""


import logging

import flopy
import numpy as np
import xarray as xr
from tqdm import tqdm

from .. import mdims
from .sim import get_tdis_perioddata

logger = logging.getLogger(__name__)


def model_datasets_to_rch(gwf, ds, print_input=False, pname="rch", **kwargs):
    """convert the recharge data in the model dataset to a recharge package
    with time series.

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    ds : xr.DataSet
        dataset containing relevant model grid information
    print_input : bool, optional
        value is passed to flopy.mf6.ModflowGwfrch() to determine if input
        should be printed to the lst file. Default is False
    pname : str, optional
        package name

    Returns
    -------
    rch : flopy.mf6.modflow.mfgwfrch.ModflowGwfrch
        recharge package
    """
    # check for nan values
    if ds["recharge"].isnull().any():
        raise ValueError("please remove nan values in recharge data array")

    # get stress period data
    if ds.time.steady_state:
        mask = ds["recharge"] != 0
        rch_spd_data = mdims.da_to_rec_list(
            ds, mask, col1="recharge", first_active_layer=True, only_active_cells=False
        )

        # create rch package
        rch = flopy.mf6.ModflowGwfrch(
            gwf,
            filename=f"{gwf.name}.rch",
            pname=pname,
            fixed_cell=False,
            maxbound=len(rch_spd_data),
            print_input=True,
            stress_period_data={0: rch_spd_data},
            **kwargs,
        )

        return rch

    # transient recharge
    if ds.gridtype == "structured":
        empty_str_array = np.zeros_like(ds["idomain"][0], dtype="S13")
        ds["rch_name"] = xr.DataArray(
            empty_str_array,
            dims=("y", "x"),
            coords={"y": ds.y, "x": ds.x},
        )
        ds["rch_name"] = ds["rch_name"].astype(str)
        # dimension check
        if ds["recharge"].dims == ("time", "y", "x"):
            axis = 0
            rch_2d_arr = (
                ds["recharge"]
                .data.reshape(
                    (
                        ds.dims["time"],
                        ds.dims["x"] * ds.dims["y"],
                    )
                )
                .T
            )

            # check if reshaping is correct
            if not (ds["recharge"].values[:, 0, 0] == rch_2d_arr[0]).all():
                raise ValueError(
                    "reshaping recharge to calculate unique time series did not work out as expected"
                )

        elif ds["recharge"].dims == ("y", "x", "time"):
            axis = 2
            rch_2d_arr = ds["recharge"].data.reshape(
                (
                    ds.dims["x"] * ds.dims["y"],
                    ds.dims["time"],
                )
            )

            # check if reshaping is correct
            if not (ds["recharge"].values[0, 0, :] == rch_2d_arr[0]).all():
                raise ValueError(
                    "reshaping recharge to calculate unique time series did not work out as expected"
                )

        else:
            raise ValueError(
                "expected dataarray with 3 dimensions"
                f'(time, y and x) or (y, x and time), not {ds["recharge"].dims}'
            )

        rch_unique_arr = np.unique(rch_2d_arr, axis=0)
        rch_unique_dic = {}
        for i, unique_rch in enumerate(rch_unique_arr):
            ds["rch_name"].data[
                np.isin(ds["recharge"].values, unique_rch).all(axis=axis)
            ] = f"rch_{i}"
            rch_unique_dic[f"rch_{i}"] = unique_rch

        mask = ds["rch_name"] != ""
        rch_spd_data = mdims.da_to_rec_list(
            ds,
            mask,
            col1="rch_name",
            first_active_layer=True,
            only_active_cells=False,
        )

    elif ds.gridtype == "vertex":
        empty_str_array = np.zeros_like(ds["idomain"][0], dtype="S13")
        ds["rch_name"] = xr.DataArray(empty_str_array, dims=("icell2d"))
        ds["rch_name"] = ds["rch_name"].astype(str)

        # dimension check
        if ds["recharge"].dims == ("icell2d", "time"):
            rch_2d_arr = ds["recharge"].values
        elif ds["recharge"].dims == ("time", "icell2d"):
            rch_2d_arr = ds["recharge"].values.T
        else:
            raise ValueError(
                "expected dataarray with 2 dimensions"
                f'(time, icell2d) or (icell2d, time), not {ds["recharge"].dims}'
            )

        rch_unique_arr = np.unique(rch_2d_arr, axis=0)
        rch_unique_dic = {}
        for i, unique_rch in enumerate(rch_unique_arr):
            ds["rch_name"][(rch_2d_arr == unique_rch).all(axis=1)] = f"rch_{i}"
            rch_unique_dic[f"rch_{i}"] = unique_rch

        mask = ds["rch_name"] != ""
        rch_spd_data = mdims.da_to_rec_list(
            ds,
            mask,
            col1="rch_name",
            first_active_layer=True,
            only_active_cells=False,
        )

    # create rch package
    rch = flopy.mf6.ModflowGwfrch(
        gwf,
        filename=f"{gwf.name}.rch",
        pname=pname,
        fixed_cell=False,
        maxbound=len(rch_spd_data),
        print_input=print_input,
        stress_period_data={0: rch_spd_data},
        **kwargs,
    )

    # get timesteps
    tdis_perioddata = get_tdis_perioddata(ds)
    perlen_arr = [t[0] for t in tdis_perioddata]
    time_steps_rch = [0.0] + np.array(perlen_arr).cumsum().tolist()

    # create timeseries packages
    for i, key in tqdm(
        enumerate(rch_unique_dic.keys()),
        total=len(rch_unique_dic.keys()),
        desc="Building ts packages rch",
    ):
        # add extra time step to the time series object (otherwise flopy fails)
        recharge_val = list(rch_unique_dic[key]) + [0.0]

        recharge = list(zip(time_steps_rch, recharge_val))
        if i == 0:
            rch.ts.initialize(
                filename=f"{key}.ts",
                timeseries=recharge,
                time_series_namerecord=key,
                interpolation_methodrecord="stepwise",
            )
        else:
            rch.ts.append_package(
                filename=f"{key}.ts",
                timeseries=recharge,
                time_series_namerecord=key,
                interpolation_methodrecord="stepwise",
            )

    return rch


def model_datasets_to_evt(gwf, ds, print_input=False, pname="evt", **kwargs):
    pass
