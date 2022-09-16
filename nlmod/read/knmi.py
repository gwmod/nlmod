import datetime as dt
import logging

import hydropandas as hpd
import numpy as np
import pandas as pd

from .. import cache, util
from ..mdims.resample import get_affine_mod_to_world

logger = logging.getLogger(__name__)


@cache.cache_netcdf
def get_recharge(ds, nodata=None):
    """add multiple recharge packages to the groundwater flow model with knmi
    data by following these steps:

       1. check for each cell (structured or vertex) which knmi measurement
          stations (prec and evap) are the closest.
       2. download precipitation and evaporation data for all knmi stations that
          were found at 1
       3. create a recharge package in which each cell has a reference to a
          timeseries. Timeseries are created for each unique combination of
          precipitation and evaporation. The following packages are created:
            a. the rch package itself in which cells with the same
               precipitation and evaporation stations are defined. This
               package also refers to all the time series package (see b).
            b. the time series packages in which the recharge flux is defined
               for the time steps of the model. Each package contains the
               time series for one or more cels (defined in a).

    Supports structured and unstructred datasets.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information
    nodata : int, optional
        if the first_active_layer data array in ds has this value,
        it means this cell is inactive in all layers. If nodata is None the
        nodata value in ds is used.
        the default is None.

    Returns
    -------
    ds : xr.DataSet
        dataset with spatial model data including the rch raster
    """
    if nodata is None:
        nodata = ds.nodata

    start = pd.Timestamp(ds.time.attrs["start_time"])
    end = pd.Timestamp(ds.time.data[-1])
    # include the end day in the time series.
    end = end + pd.Timedelta(1, "D")

    ds_out = util.get_ds_empty(ds)

    # get recharge data array
    if ds.gridtype == "structured":
        dims = ("y", "x")
    elif ds.gridtype == "vertex":
        dims = ("icell2d",)
    if not ds.time.steady_state:
        dims = dims + ("time",)

    shape = [len(ds_out[dim]) for dim in dims]
    ds_out["recharge"] = dims, np.zeros(shape)

    locations, oc_knmi_prec, oc_knmi_evap = get_knmi_at_locations(
        ds, start=start, end=end, nodata=nodata
    )

    # add closest precipitation and evaporation measurement station to each cell
    locations[["prec_point", "distance"]] = locations.geo.get_nearest_point(
        oc_knmi_prec
    )
    locations[["evap_point", "distance"]] = locations.geo.get_nearest_point(
        oc_knmi_evap
    )

    # find unique combination of precipitation and evaporation station
    unique_combinations = locations.drop_duplicates(["prec_point", "evap_point"])[
        ["prec_point", "evap_point"]
    ].values

    for prec_evap in unique_combinations:
        # get locations with the same prec and evap station
        loc_sel = locations.loc[
            (locations["prec_point"] == prec_evap[0])
            & (locations["evap_point"] == prec_evap[1])
        ]

        # calculate recharge time series
        prec = oc_knmi_prec.loc[prec_evap[0], "obs"]["RD"]
        evap = oc_knmi_evap.loc[prec_evap[1], "obs"]["EV24"]
        recharge_ts = (prec - evap).dropna()

        if recharge_ts.index[-1] < (end - pd.Timedelta(1, unit="D")):
            raise ValueError(
                f"no recharge available at precipitation stations {prec_evap[0]} and evaporation station {prec_evap[1]} for date {end}"
            )

        # fill recharge data array
        if ds.time.steady_state:
            rch_average = recharge_ts.mean()
            if ds.gridtype == "structured":
                # add data to ds_out
                for row, col in zip(loc_sel.row, loc_sel.col):
                    ds_out["recharge"].data[row, col] = rch_average
            elif ds.gridtype == "vertex":
                # add data to ds_out
                ds_out["recharge"].loc[loc_sel.index] = rch_average
        else:
            model_recharge = pd.Series(index=ds.time.data, dtype=float)
            for j, ts in enumerate(model_recharge.index):
                if j < (len(model_recharge) - 1):
                    model_recharge.loc[ts] = (
                        recharge_ts.loc[ts : model_recharge.index[j + 1]]
                        .iloc[:-1]
                        .mean()
                    )
                else:
                    model_recharge.loc[ts] = recharge_ts.loc[ts:end].iloc[:-1].mean()

            # add data to ds_out
            if ds.gridtype == "structured":
                for row, col in zip(loc_sel.row, loc_sel.col):
                    ds_out["recharge"].data[row, col, :] = model_recharge.values

            elif ds.gridtype == "vertex":
                ds_out["recharge"].loc[loc_sel.index, :] = model_recharge.values

    for datavar in ds_out:
        ds_out[datavar].attrs["source"] = "KNMI"
        ds_out[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
        ds_out[datavar].attrs["units"] = "m/day"

    return ds_out


def get_locations_vertex(ds, nodata=-999):
    """get dataframe with the locations of the grid cells of a vertex grid.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information
    nodata : int, optional
        if the first_active_layer data array in ds has this value,
        it means this cell is inactive in all layers. If nodata is None the
        nodata value in ds is used.
        the default is None

    Returns
    -------
    locations : pandas DataFrame
        DataFrame with the locations of all active grid cells.
        includes the columns: x, y and layer
    """
    # get active locations
    icell2d_active = np.where(ds["first_active_layer"] != nodata)[0]

    # create dataframe from active locations
    x = ds["x"].sel(icell2d=icell2d_active)
    y = ds["y"].sel(icell2d=icell2d_active)
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # transform coordinates into real-world coordinates
        affine = get_affine_mod_to_world(ds)
        x, y = affine * (x, y)
    layer = ds["first_active_layer"].sel(icell2d=icell2d_active)
    locations = pd.DataFrame(
        index=icell2d_active, data={"x": x, "y": y, "layer": layer}
    )
    locations = hpd.ObsCollection(locations)

    return locations


def get_locations_structured(ds, nodata=-999):
    """get dataframe with the locations of the grid cells of a structured grid.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information
    nodata : int, optional
        if the first_active_layer data array in ds has this value,
        it means this cell is inactive in all layers. If nodata is None the
        nodata value in ds is used.
        the default is None

    Returns
    -------
    locations : pandas DataFrame
        DataFrame with the locations of all active grid cells.
        includes the columns: x, y, row, col and layer
    """

    # store x and y mids in locations of active cells
    rows, columns = np.where(ds["first_active_layer"] != nodata)
    x = np.array([ds["x"].data[col] for col in columns])
    y = np.array([ds["y"].data[row] for row in rows])
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # transform coordinates into real-world coordinates
        affine = get_affine_mod_to_world(ds)
        x, y = affine * (x, y)
    layers = [
        ds["first_active_layer"].data[row, col] for row, col in zip(rows, columns)
    ]

    locations = hpd.ObsCollection(
        pd.DataFrame(
            data={"x": x, "y": y, "row": rows, "col": columns, "layer": layers}
        )
    )

    return locations


def get_knmi_at_locations(ds, start="2010", end=None, nodata=-999):
    """get knmi data at the locations of the active grid cells in ds.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information
    start : str or datetime, optional
        start date of measurements that you want, The default is '2010'.
    end :  str or datetime, optional
        end date of measurements that you want, The default is None.
    nodata : int, optional
        if the first_active_layer data array in ds has this value,
        it means this cell is inactive in all layers. If nodata is None the
        nodata value in ds is used.
        the default is None

    Raises
    ------
    ValueError
        wrong grid type specified.

    Returns
    -------
    locations : pandas DataFrame
        DataFrame with the locations of all active grid cells.
    oc_knmi_prec : hydropandas.ObsCollection
        ObsCollection with knmi data of the precipitation stations.
    oc_knmi_evap : hydropandas.ObsCollection
        ObsCollection with knmi data of the evaporation stations.
    """
    # get locations
    if ds.gridtype == "structured":
        locations = get_locations_structured(ds, nodata=nodata)
    elif ds.gridtype == "vertex":
        locations = get_locations_vertex(ds, nodata=nodata)
    else:
        raise ValueError("gridtype should be structured or vertex")

    # get knmi data stations closest to any grid cell
    oc_knmi_prec = hpd.ObsCollection.from_knmi(
        locations=locations, starts=[start], ends=[end], meteo_vars=["RD"]
    )

    oc_knmi_evap = hpd.ObsCollection.from_knmi(
        locations=locations, starts=[start], ends=[end], meteo_vars=["EV24"]
    )

    return locations, oc_knmi_prec, oc_knmi_evap
