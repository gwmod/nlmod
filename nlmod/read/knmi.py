import datetime as dt
import logging

import hydropandas as hpd
import numpy as np
import pandas as pd
from hydropandas.io import knmi as hpd_knmi

from .. import cache, util
from ..dims.grid import get_affine_mod_to_world, is_structured, is_vertex
from ..dims.layers import get_first_active_layer

logger = logging.getLogger(__name__)


@cache.cache_netcdf(coords_3d=True, coords_time=True)
def get_recharge(ds, oc_knmi=None, method="linear", most_common_station=False):
    """Add recharge to model dataset from KNMI data.

    Add recharge to the model dataset with knmi data by following these steps:
       1. check for each cell (structured or vertex) which knmi measurement
          stations (prec and evap) are the closest.
       2. download precipitation and evaporation data for all knmi stations that
          were found at 1
       3. create a recharge array in which each cell has a reference to a
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
    oc_knmi : hpd.ObsCollection
        ObsCollection with precipitation (RD) and evaporation (EV24) data from the knmi.
    method : str, optional
        If 'linear', calculate recharge by subtracting evaporation from precipitation.
        If 'separate', add precipitation as 'recharge' and evaporation as 'evaporation'.
        The default is 'linear'.
    most_common_station : bool, optional
        When True, only download data from the station that is most common in the model
        area. The default is False

    Returns
    -------
    ds : xr.DataSet
        dataset with spatial model data including the rch raster
    """
    if "time" not in ds:
        raise (
            AttributeError(
                "'Dataset' object has no 'time' dimension. "
                "Please run nlmod.time.set_ds_time()"
            )
        )
    if ds.time.dtype.kind != "M":
        raise TypeError("get recharge requires a datetime64[ns] time index")

    ds_out = util.get_ds_empty(ds, keep_coords=("time", "y", "x"))
    ds_out.attrs["gridtype"] = ds.gridtype

    # get recharge data array
    if is_structured(ds):
        dims = ("y", "x")
    elif is_vertex(ds):
        dims = ("icell2d",)
    else:
        raise ValueError("gridtype should be structured or vertex")
    dims = ("time",) + dims
    shape = [len(ds_out[dim]) for dim in dims]

    if oc_knmi is None:
        oc_knmi = get_knmi(ds,
                           most_common_station=most_common_station)

    locations = get_locations(ds,
                              oc_knmi=oc_knmi,
                              most_common_station=most_common_station)
    if method in ["linear"]:
        ds_out["recharge"] = dims, np.zeros(shape)
        

        # find unique combination of precipitation and evaporation station
        unique_combinations = locations.drop_duplicates(["stn_rd", "stn_ev24"])[
            ["stn_rd", "stn_ev24"]
        ].values
        if unique_combinations.shape[1] > 2:
            # bug fix for pandas 2.1 where three columns are returned
            unique_combinations = unique_combinations[:, :2]
        for stn_rd, stn_ev24 in unique_combinations:
            # get locations with the same prec and evap station
            mask = (locations["stn_rd"] == stn_rd) & (locations["stn_ev24"] == stn_ev24)
            loc_sel = locations.loc[mask]

            # calculate recharge time series
            prec = oc_knmi.loc[stn_rd, "obs"]["RD"].resample("D").nearest()
            evap = oc_knmi.loc[stn_ev24, "obs"]["EV24"].resample("D").nearest()
            ts = (prec - evap).dropna()
            ts.name = f"{prec.name}-{evap.name}"

            _add_ts_to_ds(ts, loc_sel, "recharge", ds_out)

    elif method == "separate":
        ds_out["recharge"] = dims, np.zeros(shape)
        for stn in locations["stn_rd"].unique():
            ts = oc_knmi.loc[stn, "obs"]["RD"].resample("D").nearest()
            loc_sel = locations.loc[(locations["stn_rd"] == stn)]
            _add_ts_to_ds(ts, loc_sel, "recharge", ds_out)

        ds_out["evaporation"] = dims, np.zeros(shape)
        for stn in locations["stn_ev24"].unique():
            ts = oc_knmi.loc[stn, "obs"]["EV24"].resample("D").nearest()
            loc_sel = locations.loc[(locations["stn_ev24"] == stn)]
            _add_ts_to_ds(ts, loc_sel, "evaporation", ds_out)
    else:
        raise (ValueError(f"Unknown method: {method}"))
    for datavar in ds_out:
        ds_out[datavar].attrs["source"] = "KNMI"
        ds_out[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
        ds_out[datavar].attrs["units"] = "m/day"

    return ds_out


def _add_ts_to_ds(timeseries, loc_sel, variable, ds):
    """Add a timeseries to a variable at location loc_sel in model DataSet."""
    end = pd.Timestamp(ds.time.data[-1])
    if timeseries.index[-1] < end:
        raise ValueError(
            f"no data available for time series'{timeseries.name}' on date {end}"
        )

    # fill recharge data array
    model_recharge = pd.Series(index=ds.time, dtype=float)
    for j, ts in enumerate(model_recharge.index):
        if j == 0:
            start = ds.time.start
        else:
            start = model_recharge.index[j - 1]
        mask = (timeseries.index > start) & (timeseries.index <= ts)
        model_recharge.loc[ts] = timeseries[mask].mean()
    if model_recharge.isna().any():
        # when the model frequency is higher than the timeseries-frequency,
        # there will be NaN's, which we fill by backfill
        model_recharge = model_recharge.fillna(method="bfill")
        if model_recharge.isna().any():
            raise (ValueError(f"There are NaN-values in {variable}."))

    # add data to ds
    values = np.repeat(model_recharge.values[:, np.newaxis], loc_sel.shape[0], 1)
    if is_structured(ds):
        ds[variable].data[:, loc_sel.row, loc_sel.col] = values
    elif is_vertex(ds):
        ds[variable].data[:, loc_sel.index] = values


def get_locations_vertex(ds):
    """Get dataframe with the locations of the grid cells of a vertex grid.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information

    Returns
    -------
    locations : pandas DataFrame
        DataFrame with the locations of all active grid cells.
        includes the columns: x, y and layer
    """
    # get active locations
    fal = get_first_active_layer(ds)
    icell2d_active = np.where(fal != fal.attrs["nodata"])[0]

    # create dataframe from active locations
    x = ds["x"].sel(icell2d=icell2d_active)
    y = ds["y"].sel(icell2d=icell2d_active)
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # transform coordinates into real-world coordinates
        affine = get_affine_mod_to_world(ds)
        x, y = affine * (x, y)
    layer = fal.sel(icell2d=icell2d_active)
    locations = pd.DataFrame(
        index=icell2d_active, data={"x": x, "y": y, "layer": layer}
    )
    locations = hpd.ObsCollection(locations)

    return locations


def get_locations_structured(ds):
    """Get dataframe with the locations of the grid cells of a structured grid.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information

    Returns
    -------
    locations : pandas DataFrame
        DataFrame with the locations of all active grid cells.
        includes the columns: x, y, row, col and layer
    """
    # store x and y mids in locations of active cells
    fal = get_first_active_layer(ds)
    rows, columns = np.where(fal != fal.attrs["nodata"])
    x = np.array([ds["x"].data[col] for col in columns])
    y = np.array([ds["y"].data[row] for row in rows])
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # transform coordinates into real-world coordinates
        affine = get_affine_mod_to_world(ds)
        x, y = affine * (x, y)
    layers = [fal.data[row, col] for row, col in zip(rows, columns)]

    locations = hpd.ObsCollection(
        pd.DataFrame(
            data={"x": x, "y": y, "row": rows, "col": columns, "layer": layers}
        )
    )

    return locations


def get_knmi(ds, most_common_station=False, start=None, end=None):
    """get precipitation (RD) and evaporation (EV24) data from the knmi at the grid
    cells.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information
    most_common_station : bool, optional
        When True, only download data from the station that is most common in the model
        area. The default is False
    start : str or datetime, optional
        start date of measurements that you want, The default is '2010'.
    end :  str or datetime, optional
        end date of measurements that you want, The default is None.

    Returns
    -------
    oc_knmi
        hpd.ObsCollection
    """

    locations = get_locations(ds, most_common_station=most_common_station)
    oc_knmi = get_knmi_at_locations(locations, ds=ds, start=start, end=end)

    return oc_knmi


def get_locations(ds, oc_knmi=None, most_common_station=False):
    """Get the locations of the active grid cells in ds and the nearest (or most common)
    precipitation and evaporation station.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model grid information
    oc_knmi : hpd.ObsCollection or None, optional
        ObsCollection with knmi station data. If None the nearest of all knmi stations
        is used.
    most_common_station : bool, optional
        When True, only download data from the station that is most common in the model
        area. The default is False

    Raises
    ------
    ValueError
        wrong grid type specified.

    Returns
    -------
    locations : pd.DataFrame
        each row contains a location (x and y) and the relevant precipitation (stn_rd)
        and evaporation (stn_ev24) stations.
    """
    # get locations
    if is_structured(ds):
        locations = get_locations_structured(ds)
    elif is_vertex(ds):
        locations = get_locations_vertex(ds)
    else:
        raise ValueError("gridtype should be structured or vertex")

    if oc_knmi is not None:
        locations["stn_rd"] = hpd_knmi.get_nearest_station_df(locations, stations=oc_knmi.loc[oc_knmi['meteo_var']=='RD'])
        locations["stn_ev24"] = hpd_knmi.get_nearest_station_df(locations, stations=oc_knmi.loc[oc_knmi['meteo_var']=='EV24'])
    else:
        locations["stn_rd"] = hpd_knmi.get_nearest_station_df(locations, meteo_var="RD")
        locations["stn_ev24"] = hpd_knmi.get_nearest_station_df(locations, meteo_var="EV24")

    if most_common_station:
        if is_structured(ds):
            # set the most common station to all locations
            locations["stn_rd"] = locations["stn_rd"].value_counts().idxmax()
            locations["stn_ev24"] = locations["stn_ev24"].value_counts().idxmax()
        else:
            # set the station with the largest area to all locations
            locations["area"] = ds["area"].loc[locations.index]
            locations["stn_rd"] = locations.groupby("stn_rd").sum()["area"].idxmax()
            locations["stn_ev24"] = locations.groupby("stn_ev24").sum()["area"].idxmax()

    return locations
    
def get_knmi_at_locations(locations, ds=None, start=None, end=None):
    """get precipitation (RD) and evaporation (EV24) data from the knmi at the locations

    Parameters
    ----------
    locations : pd.DataFrame
        each row contains a location (x and y) and the relevant precipitation (stn_rd)
        and evaporation (stn_ev24) stations.
    ds : xr.DataSet or None, optional
        dataset containing relevant time information. If None provide start and end.
    start : str or datetime, optional
        start date of measurements that you want, The default is '2010'.
    end :  str or datetime, optional
        end date of measurements that you want, The default is None.

    Returns
    -------
    oc_knmi
        hpd.ObsCollection
    """
    stns_rd = locations["stn_rd"].unique()
    stns_ev24 = locations["stn_ev24"].unique()

    # get start and end
    if start is None:
        start = pd.Timestamp(ds.time.attrs["start"])
    if end is None:
        end = pd.Timestamp(ds.time.data[-1])

    # get knmi data stations closest to any grid cell
    olist = []
    for stnrd in stns_rd:
        o = hpd.PrecipitationObs.from_knmi(
            meteo_var="RD", stn=stnrd, start=start, end=end, fill_missing_obs=True
        )

        olist.append(o)

    for stnev24 in stns_ev24:
        o = hpd.EvaporationObs.from_knmi(
            meteo_var="EV24", stn=stnev24, start=start, end=end, fill_missing_obs=True
        )

        olist.append(o)

    oc_knmi = hpd.ObsCollection(olist)

    return oc_knmi