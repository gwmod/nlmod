"""Module with functions to deal with the northsea.

    - identify model cells with the north sea
    - add bathymetry of the northsea to the layer model
    - extrapolate the layer model below the northsea bed.


Note: if you like jazz please check this out: https://www.northseajazz.com
"""

import datetime as dt
import logging
import warnings

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .. import cache
from ..dims.grid import get_extent
from ..dims.resample import fillnan_da, structured_da_to_ds
from ..util import get_da_from_da_ds, get_ds_empty

logger = logging.getLogger(__name__)


@cache.cache_netcdf()
def download_bathymetry(extent, kind="jarkus"):
    """Download bathymetry data from the jarkus dataset.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent xmin, xmax, ymin, ymax.
    kind : str, optional
        The kind of data. Can be "jarkus", "kusthoogte" or "vaklodingen". The default is
        "jarkus".
    
    Returns
    -------
    da_bathymetry : xarray.DataArray
        bathymetry data
    """

    # try to get bathymetry via opendap
    jarkus_ds = get_dataset_jarkus(extent=extent, kind=kind)

    # disable try/except because because it only works for specific areas
    # try:
    #     jarkus_ds = get_dataset_jarkus(extent=extent, kind=kind)
    # except OSError:
    #     import gdown

    #     logger.warning(
    #         "cannot access Jarkus netCDF link, copy file from google drive instead"
    #     )
    #     fname_jarkus = os.path.join(ds.model_ws, "jarkus_nhflopy.nc")
    #     url = "https://drive.google.com/uc?id=1uNy4THL3FmNFrTDTfizDAl0lxOH-yCEo"
    #     gdown.download(url, fname_jarkus, quiet=False)
    #     jarkus_ds = xr.open_dataset(fname_jarkus)

    da_bathymetry = jarkus_ds["z"]

    da_bathymetry.attrs["source"] = kind
    da_bathymetry.attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
    da_bathymetry.attrs["units"] = "mNAP"

    return da_bathymetry


@cache.cache_netcdf()
def get_bathymetry(ds=None, extent=None, da_name="bathymetry",
                   datavar_sea='northsea',
                   kind="jarkus", method="average"):
    """Get bathymetry of the Northsea from the jarkus dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data where bathymetry is added to.
    extent : list, tuple or np.array, optional
        extent xmin, xmax, ymin, ymax. Only used if ds is None. The default is None.
    da_name : str, optional
        name of the datavar that is used to store the bathymetry data. The default is
        'bathymetry'.
    datavar_sea : str, optional
        datavariable in the dataset that is used to identify cells with sea in them.
        The default is 'northsea'.
    method : str, optional
        Method used to resample bathymetry data to the modelgrid. See the documentation
        of nlmod.resample.structured_da_to_ds for possible values. The default is
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
    if ds is None:
        warnings.warn(
        "calling 'get_bathymetry' with ds=None is deprecated and will raise an error "
        "in the future. Use 'download_bathymetry' to get the bathymetry data within an "
        "extent",
        DeprecationWarning,
    )

    if extent is None and ds is not None:
        extent = get_extent(ds)

    if ds is not None:
        ds_out = get_ds_empty(ds, keep_coords=("y", "x"))

        # no bathymetry if we don't have sea
        sea = ds[datavar_sea]
        if (sea == 0).all():
            ds_out[da_name] = get_da_from_da_ds(sea, sea.dims, data=np.nan)
            return ds_out

    # try to get bathymetry via opendap
    bathymetry_da = download_bathymetry(extent=extent, kind=kind)
    
    # bathymetry projected on model grid
    if ds is None:
        # fill nan values in bathymetry
        da_bathymetry_filled = fillnan_da(bathymetry_da)

        # bathymetry can never be larger than NAP 0.0
        da_bathymetry_filled = xr.where(da_bathymetry_filled > 0, 0, da_bathymetry_filled)
        return xr.Dataset({da_name:da_bathymetry_filled})
    else:
        discretize_bathymetry(ds, bathymetry_da=bathymetry_da)

    return ds_out


@cache.cache_netcdf()
def discretize_bathymetry(ds,
                          bathymetry_da,
                          da_name="bathymetry",
                          datavar_sea='northsea',
                          method="average"):
    """Discretize bathymetry data to model the model grid.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data where bathymetry is added to.
    bathymetry_da : xarray.DataArray
        bathymetry data
    da_name : str, optional
        name of the datavar that is used to store the bathymetry data. The default is
        'bathymetry'.
    datavar_sea : str, optional
        datavariable in the dataset that is used to identify cells with sea in them.
        The default is 'northsea'.
    method : str, optional
        Method used to resample bathymetry data to the modelgrid. See the documentation
        of nlmod.resample.structured_da_to_ds for possible values. The default is
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

    # fill nan values in bathymetry
    da_bathymetry_filled = fillnan_da(bathymetry_da)

    # bathymetry can never be larger than NAP 0.0
    da_bathymetry_filled = xr.where(da_bathymetry_filled > 0, 0, da_bathymetry_filled)

    # bathymetry projected on model grid
    da_bathymetry = structured_da_to_ds(da_bathymetry_filled, ds, method=method)

    ds_out = get_ds_empty(ds, keep_coords=("y", "x"))
    sea = ds[datavar_sea]

    ds_out[da_name] = xr.where(sea, da_bathymetry, np.nan)

    return ds_out


@cache.cache_netcdf()
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
    # drop 'lat' and 'lon' as these will create problems when resampling the data
    z_dataset = z_dataset.drop_vars(["lat", "lon"])
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
        raise (ValueError(f"Unsupported kind: {kind}"))

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