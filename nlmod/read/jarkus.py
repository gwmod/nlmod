# -*- coding: utf-8 -*-
"""module with functions to deal with the northsea by:

    - identifying model cells with the north sea
    - add bathymetry of the northsea to the layer model
    - extrpolate the layer model below the northsea bed.


Note: if you like jazz please check this out: https://www.northseajazz.com
"""
import datetime as dt
import logging
import os

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .. import cache
from ..dims.resample import fillnan_da, get_extent, structured_da_to_ds
from ..util import get_da_from_da_ds, get_ds_empty

logger = logging.getLogger(__name__)


@cache.cache_netcdf
def get_bathymetry(ds, northsea, kind="jarkus", method="average"):
    """get bathymetry of the Northsea from the jarkus dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data where bathymetry is added to
    northsea : ??
        ??
    method : str, optional
        Method used to resample ahn to grid of ds. See the documentation of
        nlmod.resample.structured_da_to_ds for possible values. The default is
        'average'.

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with bathymetry

    Notes
    -----
    The nan values in the original bathymetry are filled and then the
    data is resampled to the modelgrid. Maybe we can speed up things by
    changing the order in which operations are executed.
    """
    ds_out = get_ds_empty(ds, keep_coords=("y", "x"))

    # no bathymetry if we don't have northsea
    if (northsea == 0).all():
        ds_out["bathymetry"] = get_da_from_da_ds(northsea, northsea.dims, data=np.nan)
        return ds_out

    # try to get bathymetry via opendap
    try:
        jarkus_ds = get_dataset_jarkus(get_extent(ds), kind=kind)
    except OSError:
        import gdown

        logger.warning(
            "cannot access Jarkus netCDF link, copy file from google drive instead"
        )
        fname_jarkus = os.path.join(ds.model_ws, "jarkus_nhflopy.nc")
        url = "https://drive.google.com/uc?id=1uNy4THL3FmNFrTDTfizDAl0lxOH-yCEo"
        gdown.download(url, fname_jarkus, quiet=False)
        jarkus_ds = xr.open_dataset(fname_jarkus)

    da_bathymetry_raw = jarkus_ds["z"]

    # fill nan values in bathymetry
    da_bathymetry_filled = fillnan_da(da_bathymetry_raw)

    # bathymetrie mag nooit groter zijn dan NAP 0.0
    da_bathymetry_filled = xr.where(da_bathymetry_filled > 0, 0, da_bathymetry_filled)

    # bathymetry projected on model grid
    da_bathymetry = structured_da_to_ds(da_bathymetry_filled, ds, method=method)

    ds_out["bathymetry"] = xr.where(northsea, da_bathymetry, np.nan)

    for datavar in ds_out:
        ds_out[datavar].attrs["source"] = kind
        ds_out[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
        if datavar == "bathymetry":
            ds_out[datavar].attrs["units"] = "mNAP"

    return ds_out


def get_dataset_jarkus(extent, kind="jarkus", return_tiles=False, time=-1):
    """Get bathymetry from Jarkus within a certain extent. If return_tiles is False, the
    following actions are performed:
    1. find Jarkus tiles within the extent
    2. download netcdf files of Jarkus tiles
    3. read Jarkus tiles and combine the 'z' parameter of the last time step of each
    tile (when time=1), to a dataarray.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid. Should be RD-new
        coordinates (EPSG:28992)
    kind : str, optional
        The kind of data. Can be "jarkus", "kusthoogte" or "vaklodingen". The default is
        "jarkus".
    return_tiles : bool, optional
        Return the individual tiles when True. The default is False.
    time : str, int or pd.TimeStamp, optional
        The time to return data for. When time="last_non_nan", this returns the last
        non-NaN-value for each pixel. This can take a while, as all tiles need to be
        checked. When time is an integer, it is used as the time index. When set to -1,
        this then downloads the last time available in each tile  (which can contain
        large areas with NaN-values). When time is a string (other than "last_non_nan")
        or a pandas Timestamp, only data on this exact time are downloaded. The default
        is -1.

    Returns
    -------
    z : xr.DataSet
        dataset containing bathymetry data

    """

    extent = [int(x) for x in extent]

    netcdf_tile_names = get_jarkus_tilenames(extent, kind)
    tiles = [xr.open_dataset(name.strip()) for name in netcdf_tile_names]
    if return_tiles:
        return tiles
    if time is not None:
        if time == "last_non_nan":
            tiles_last = []
            for tile in tiles:
                time = (~np.isnan(tile["z"])).cumsum("time").argmax("time")
                tiles_last.append(tile.isel(time=time))
            tiles = tiles_last
        elif isinstance(time, int):
            # only use the last timestep
            tiles = [tile.isel(time=time) for tile in tiles]
        else:
            time = pd.to_datetime(time)
            tiles_left = []
            for tile in tiles:
                if time in tile.time:
                    tiles_left.append(tile.sel(time=time))
                else:
                    extent_tile = list(
                        np.hstack(
                            (
                                tile.attrs["projectionCoverage_x"],
                                tile.attrs["projectionCoverage_y"],
                            )
                        )
                    )
                    logger.info(
                        f"no time={time} in {kind}-tile with extent {extent_tile}"
                    )
            tiles = tiles_left
    z_dataset = xr.combine_by_coords(tiles, combine_attrs="drop")
    return z_dataset


def get_jarkus_tilenames(extent, kind="jarkus"):
    """Find all Jarkus tilenames within a certain extent.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid. Should be RD-new
        coordinates (EPSG:28992)

    Returns
    -------
    netcdf_urls : list of str
        list of the urls of all netcdf files of the tiles with Jarkus data.
    """
    if kind == "jarkus":
        url = "http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/grids/catalog.nc"
    elif kind == "kusthoogte":
        url = "http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/kusthoogte/catalog.nc"
    elif kind == "vaklodingen":
        url = "http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/vaklodingen/catalog.nc"
    else:
        raise (Exception(f"Unsupported kind: {kind}"))

    ds_jarkus_catalog = xr.open_dataset(url)
    ew_x = ds_jarkus_catalog["projectionCoverage_x"].values
    sn_y = ds_jarkus_catalog["projectionCoverage_y"].values

    mask_ew = (ew_x[:, 1] > extent[0]) & (ew_x[:, 0] < extent[1])
    mask_sn = (sn_y[:, 1] > extent[2]) & (sn_y[:, 0] < extent[3])

    indices_tiles = np.where(mask_ew & mask_sn)[0]
    all_netcdf_tilenames = get_netcdf_tiles(kind)

    netcdf_tile_names = [all_netcdf_tilenames[i] for i in indices_tiles]

    return netcdf_tile_names


def get_netcdf_tiles(kind="jarkus"):
    """Find all Jarkus netcdf tile names.

    Returns
    -------
    netcdf_urls : list of str
        list of the urls of all netcdf files of the tiles with Jarkus data.

    Notes
    -----
    This function would be redundant if the jarkus catalog
    (http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/grids/catalog.nc)
    had a proper way of displaying the url's of each tile. It seems like an
    attempt was made to do this because there is a data variable
    named 'urlPath' in the catalog. However the dataarray of 'urlPath' has the
    same string for each tile.
    """
    if kind == "jarkus":
        url = "http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/grids/catalog.nc.ascii"
    elif kind == "kusthoogte":
        url = "http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/kusthoogte/catalog.nc.ascii"
    elif kind == "vaklodingen":
        url = "http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/vaklodingen/catalog.nc.ascii"
    else:
        raise (Exception(f"Unsupported kind: {kind}"))
    req = requests.get(url, timeout=5)
    s = req.content.decode("ascii")
    start = s.find("urlPath", s.find("urlPath") + 1)
    end = s.find("projectionCoverage_x", s.find("projectionCoverage_x") + 1)
    netcdf_urls = list(eval(s[start + 12 : end - 2]))
    return netcdf_urls


def add_bathymetry_to_top_bot_kh_kv(ds, bathymetry, fill_mask, kh_sea=10, kv_sea=10):
    """Add bathymetry to the top and bot of each layer for all cells with fill_mask.

    This method sets the top of the model at fill_mask to 0 m, and changes the first
    layer to sea, by setting the botm of this layer to bathymetry, kh to kh_sea and kv
    to kv_sea. If deeper layers are above bathymetry. the layer depth is set to
    bathymetry.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data, should
    bathymetry : xarray DataArray
        bathymetry data
    kh_sea : int or float, optional
        the horizontal conductance in sea s
    fill_mask : xr.DataArray
        cell value is 1 if you want to add bathymetry

    Returns
    -------
    ds : xarray.Dataset
        dataset with model data where the top, bot, kh and kv are changed
    """
    ds["top"].values = np.where(fill_mask, 0.0, ds["top"])

    lay = 0
    ds["botm"][lay] = xr.where(fill_mask, bathymetry, ds["botm"][lay])

    ds["kh"][lay] = xr.where(fill_mask, kh_sea, ds["kh"][lay])

    ds["kv"][lay] = xr.where(fill_mask, kv_sea, ds["kv"][lay])

    # reset bot for all layers based on bathymetrie
    for lay in range(1, ds.dims["layer"]):
        ds["botm"][lay] = np.where(
            ds["botm"][lay] > ds["botm"][lay - 1],
            ds["botm"][lay - 1],
            ds["botm"][lay],
        )
    return ds
