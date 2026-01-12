import datetime as dt
import logging
import os
import warnings
import tempfile
from typing import Literal

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
import rioxarray
import shapely
import xarray as xr
from rasterio.env import Env
from requests.exceptions import HTTPError
from rioxarray.merge import merge_arrays
from .. import NLMOD_DATADIR, cache
from ..dims.grid import get_extent
from ..dims.resample import structured_da_to_ds
from ..util import extent_to_polygon, get_ds_empty, tqdm
from .webservices import wcs

logger = logging.getLogger(__name__)


@cache.cache_netcdf(coords_2d=True)
def download_ahn(
    extent: list[float],
    identifier: str = None,
    merge_tiles=True,
    **kwargs,
) -> xr.DataArray:
    """Download ahn data within an extent.

    Parameters
    ----------
    extent : list, tuple or np.array
        The extent to be downloaded, consisting of 4 floats: xmin, xmax, ymin, ymax.
    version : str, optional
        The AHN_version, which can be 'AHN2', 'AHN3', 'AHN4', 'AHN5' and 'AHN6'.
        `version` is ignored if `identifier` is specified. The default is "AHN4", as
        this is available in the whole of the Netherlands.
    data_kind : str, optional
        The kind of data. This can be 'DTM' (terrain elevation) or "DSM" (surface
        elevation). `data_kind` is ignored if `identifier` is specified. The default is
        "DTM".
    tile_size : str, optional
        The size of the map tiles that the data is offered by the webservice. This can
        be '5x6.25km' or '1x1km'. `tile_size` is ignored if `identifier` is specified.
        The default is '5x6.25km'.
    resolution : float, optional
        The resolution of the AHN-data, which can be 0.5 and 5.0 m. `resolution` is
        ignored if `identifier` is specified. The default is 5.0.
    merge_tiles : bool, optional
        If True, the function returns a merged DataArray. If False, the function
        returns a list of DataArrays with the original tiles. The default is True.
    cut_extent : bool, optional
        If True, only keep the requested extent from the data. The defualts is True.
    config : dict, optional
        A dictionary with properties of the data sources of the different AHN-versions.
        When None, the configuration is retreived from the method get_configuration().
        The default is None.
    identifier : str, optional
        The identifier determines the AHN-version, the resolution and the type of height
        data. Possible values are (casing is important):
            AHN1: 'AHN1 maaiveldmodel (DTM) 5m'
            AHN2: 'AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd',
                  'AHN2 maaiveldmodel (DTM) ½m',
                  'AHN2 DSM ½m',
                  'AHN2 maaiveldmodel (DTM) 5m'
            AHN3: 'AHN3 maaiveldmodel (DTM) ½m',
                  'AHN3 DSM ½m',
                  'AHN3 maaiveldmodel (DTM) 5m',
                  'AHN3 DSM 5m'
            AHN4: 'AHN4 maaiveldmodel (DTM) ½m',
                  'AHN4 DSM ½m',
                  'AHN4 maaiveldmodel (DTM) 5m',
                  'AHN4 DSM 5m'
            AHN5: 'AHN5 maaiveldmodel (DTM) 5m',
                  'AHN5 DSM 5m',
                  'AHN5 maaiveldmodel (DTM) ½m',
                  'AHN5 DSM ½m'
        When no identifier is specified (the default), the url to download data from is
        taken from `config`, using the arguments `version`, `data_kind`, `tile_size` and
        `resolution` of this method. The default is None.

    Returns
    -------
    ahn_da : xr.DataArray
        DataArray with the ahn variable.
    """
    if identifier is not None:
        for key in ["version", "data_kind", "tile_size", "resolution"]:
            if key in kwargs:
                logger.warning("`{key}` is ignored when `identifier` is sepcified")
                kwargs.pop(key)
        ahn_da = _download_ahn_ellipsis(
            extent, identifier=identifier, merge_tiles=merge_tiles, **kwargs
        )
        if not merge_tiles:
            return ahn_da
        ahn_da.attrs["source"] = identifier
    else:
        ahn_da = _download_ahn_hwh(extent, merge_tiles=merge_tiles, **kwargs)
        if not merge_tiles:
            return ahn_da

    ahn_da.attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
    ahn_da.attrs["units"] = "mNAP"

    return ahn_da


@cache.cache_netcdf(coords_2d=True)
def discretize_ahn(
    ds: xr.Dataset, ahn_da: xr.DataArray, method: str = "average"
) -> xr.Dataset:
    """Discretize ahn data to model the model grid.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with the model information.
    ahn_da : xr.DataArray
        ahn data within model extent.
    method : str, optional
        Method used to resample ahn to grid of ds. See documentation of
        nlmod.resample.structured_da_to_ds for possible values. The default is
        'average'.

    Returns
    -------
    ds_out : xr.Dataset
        Dataset with the ahn variable.
    """
    if ds is not None:
        ahn_da = structured_da_to_ds(ahn_da, ds, method=method)

    ds_out = get_ds_empty(ds, keep_coords=("y", "x"))
    ds_out["ahn"] = ahn_da

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def get_ahn(
    ds: xr.Dataset | None = None,
    identifier: str = None,
    method: str = "average",
    extent: list[float] | None = None,
    merge_tiles: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Get a model dataset with ahn variable.

    Parameters
    ----------
    ds : xr.Dataset, optional
        dataset with the model information. Using ds=None is deprecated, instead use
        nlmod.read.ahn.download_ahn().
    method : str, optional
        Method used to resample ahn to grid of ds. See documentation of
        nlmod.resample.structured_da_to_ds for possible values. The default is
        'average'.
    extent : list, tuple or np.array
        The extent to be downloaded, consisting of 4 floats: xmin, xmax, ymin, ymax.
        Only used if ds is None. The default is None.
    version : str, optional
        The AHN_version, which can be 'AHN2', 'AHN3', 'AHN4', 'AHN5' and 'AHN6'.
        `version` is ignored if `identifier` is specified. The default is "AHN4", as
        this is available in the whole of the Netherlands.
    data_kind : str, optional
        The kind of data. This can be 'DTM' (terrain elevation) or "DSM" (surface
        elevation). `data_kind` is ignored if `identifier` is specified. The default is
        "DTM".
    tile_size : str, optional
        The size of the map tiles that the data is offered by the webservice. This can
        be '5x6.25km' or '1x1km'. `tile_size` is ignored if `identifier` is specified.
        The default is '5x6.25km'.
    resolution : float, optional
        The resolution of the AHN-data, which can be 0.5 and 5.0 m. `resolution` is
        ignored if `identifier` is specified. The default is 5.0.
    merge_tiles : bool, optional
        If True, the function returns a merged DataArray. If False, the function
        returns a list of DataArrays with the original tiles. The default is True.
    cut_extent : bool, optional
        If True, only keep the requested extent from the data. The defualts is True.
    config : dict, optional
        A dictionary with properties of the data sources of the different AHN-versions.
        When None, the configuration is retreived from the method get_configuration().
        The default is None.
    identifier : str, optional
        The identifier determines the AHN-version, the resolution and the type of height
        data. Possible values are (casing is important):
            AHN1: 'AHN1 maaiveldmodel (DTM) 5m'
            AHN2: 'AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd',
                  'AHN2 maaiveldmodel (DTM) ½m',
                  'AHN2 DSM ½m',
                  'AHN2 maaiveldmodel (DTM) 5m'
            AHN3: 'AHN3 maaiveldmodel (DTM) ½m',
                  'AHN3 DSM ½m',
                  'AHN3 maaiveldmodel (DTM) 5m',
                  'AHN3 DSM 5m'
            AHN4: 'AHN4 maaiveldmodel (DTM) ½m',
                  'AHN4 DSM ½m',
                  'AHN4 maaiveldmodel (DTM) 5m',
                  'AHN4 DSM 5m'
            AHN5: 'AHN5 maaiveldmodel (DTM) 5m',
                  'AHN5 DSM 5m',
                  'AHN5 maaiveldmodel (DTM) ½m',
                  'AHN5 DSM ½m'
        When no identifier is specified (the default), the url to download data from is
        taken from `config`, using the arguments `version`, `data_kind`, `tile_size` and
        `resolution` of this method. The default is None.


    Returns
    -------
    ds_out : xr.Dataset
        Dataset with the ahn variable.
    """
    if ds is None:
        warnings.warn(
            "calling 'get_ahn' with ds=None is deprecated and will raise an error in the "
            "future. Use 'nlmod.read.ahn.download_ahn' to get the ahn within an extent",
            DeprecationWarning,
        )

    if extent is None and ds is not None:
        extent = get_extent(ds)

    ahn_da = download_ahn(
        extent=extent, identifier=identifier, merge_tiles=merge_tiles, **kwargs
    )
    # this is probably redundant when we have the 'download_ahn' function
    if not merge_tiles:
        return ahn_da

    if ds is None:
        return ahn_da
    else:
        return discretize_ahn(ds, ahn_da, method=method)


def get_ahn_at_point(
    x: float,
    y: float,
    buffer: float = 0.75,
    return_da: bool = False,
    return_mean: bool = False,
    identifier: str = "dsm_05m",
    res: float = 0.5,
    **kwargs,
) -> float:
    """Get the height of the surface level at a certain point, defined by x and y.

    .. deprecated:: 0.10.0
        `get_ahn_at_point` will be removed in nlmod 1.0.0, it is replaced by
        `download_ahn_at_point` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    x : float
        The x-coordinate fo the point.
    y : float
        The y-coordinate fo the point..
    buffer : float, optional
        The buffer around x and y that is downloaded. The default is 0.75.
    return_da : bool, optional
        Return the downloaded DataArray when True. The default is False.
    return_mean : bool, optional
        Resturn the mean of all non-nan pixels within buffer. Return the center pixel
        when False. The default is False.
    identifier : str, optional
        The identifier passed onto download_latest_ahn_from_wcs. The default is "dsm_05m".
    res : float, optional
        The resolution that is passed onto download_latest_ahn_from_wcs. The default is 0.5.
    **kwargs : dict
        kwargs are passed onto the method download_latest_ahn_from_wcs.

    Returns
    -------
    float
        The surface level value at the requested point.
    """

    warnings.warn(
        "'get_ahn_at_point' is deprecated and will eventually be removed, "
        "please use nlmod.read.ahn.download_ahn_at_point() in the future.",
        DeprecationWarning,
    )

    return download_ahn_at_point(
        x, y, buffer, return_da, return_mean, identifier, res, **kwargs
    )


def download_ahn_at_point(
    x: float,
    y: float,
    buffer: float = 0.75,
    return_da: bool = False,
    return_mean: bool = False,
    identifier: str = "dsm_05m",
    res: float = 0.5,
    **kwargs,
) -> float:
    """Get the height of the surface level at a certain point, defined by x and y.

    Parameters
    ----------
    x : float
        The x-coordinate fo the point.
    y : float
        The y-coordinate fo the point..
    buffer : float, optional
        The buffer around x and y that is downloaded. The default is 0.75.
    return_da : bool, optional
        Return the downloaded DataArray when True. The default is False.
    return_mean : bool, optional
        Resturn the mean of all non-nan pixels within buffer. Return the center pixel
        when False. The default is False.
    identifier : str, optional
        The identifier passed onto download_latest_ahn_from_wcs. The default is "dsm_05m".
    res : float, optional
        The resolution that is passed onto download_latest_ahn_from_wcs. The default is 0.5.
    **kwargs : dict
        kwargs are passed onto the method download_latest_ahn_from_wcs.

    Returns
    -------
    float
        The surface level value at the requested point.
    """
    extent = [x - buffer, x + buffer, y - buffer, y + buffer]
    ahn = download_latest_ahn_from_wcs(extent, identifier=identifier, res=res, **kwargs)
    if return_da:
        # return a DataArray
        return ahn
    if return_mean:
        # return the mean (usefull when there are NaN's near the center)
        return float(ahn.mean())
    else:
        # return the center pixel
        return ahn.data[int((ahn.shape[0] - 1) / 2), int((ahn.shape[1] - 1) / 2)]


def get_ahn_along_line(
    line: shapely.LineString,
    ahn: xr.DataArray | None = None,
    dx: float | None = None,
    num: int | None = None,
    method: str = "linear",
    plot: bool = False,
) -> xr.DataArray:
    """Get the height of the surface level along a line.

    .. deprecated:: 0.10.0
        `get_ahn_along_line` will be removed in nlmod 1.0.0, it is replaced by
        `download_ahn_along_line` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    line : shapely.LineString
        The line along which the surface level is calculated.
    ahn : xr.DataArray, optional
        The 2d DataArray containing surface level values. If None, ahn4-values are
        downloaded from the web. The default is None.
    dx : float, optional
        The distance between the points along the line at which the surface level is
        calculated. Only used when num is None. When dx is None, it is set to the
        resolution of ahn. The default is None.
    num : int, optional
        If not None, the surface level is calculated at num equally spaced points along
        the line. The default is None.
    method : string, optional
        The method to interpolate the 2d surface level values to the points along the
        line. The default is "linear".
    plot : bool, optional
        if True, plot the 2d surface level, the line and the calculated heights. The
        default is False.

    Returns
    -------
    z : xr.DataArray
        A DataArray with dimension s, containing surface level values along the line.
    """
    warnings.warn(
        "'get_ahn_along_line' is deprecated and will eventually be removed, "
        "please use nlmod.read.ahn.download_ahn_along_line() in the future.",
        DeprecationWarning,
    )

    return download_ahn_along_line(line, ahn, dx, num, method, plot)


def download_ahn_along_line(
    line: shapely.LineString,
    ahn: xr.DataArray | None = None,
    dx: float | None = None,
    num: int | None = None,
    method: str = "linear",
    plot: bool = False,
) -> xr.DataArray:
    """Get the height of the surface level along a line.

    Parameters
    ----------
    line : shapely.LineString
        The line along which the surface level is calculated.
    ahn : xr.DataArray, optional
        The 2d DataArray containing surface level values. If None, ahn4-values are
        downloaded from the web. The default is None.
    dx : float, optional
        The distance between the points along the line at which the surface level is
        calculated. Only used when num is None. When dx is None, it is set to the
        resolution of ahn. The default is None.
    num : int, optional
        If not None, the surface level is calculated at num equally spaced points along
        the line. The default is None.
    method : string, optional
        The method to interpolate the 2d surface level values to the points along the
        line. The default is "linear".
    plot : bool, optional
        if True, plot the 2d surface level, the line and the calculated heights. The
        default is False.

    Returns
    -------
    z : xr.DataArray
        A DataArray with dimension s, containing surface level values along the line.
    """
    if ahn is None:
        bbox = line.bounds
        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        ahn = get_ahn4(extent)
    if num is not None:
        s = np.linspace(0.0, line.length, num)
    else:
        if dx is None:
            dx = float(ahn.x[1] - ahn.x[0])
        s = np.arange(0.0, line.length, dx)

    x, y = zip(*[p.xy for p in line.interpolate(s)])

    x = np.array(x)[:, 0]
    y = np.array(y)[:, 0]

    x = xr.DataArray(x, dims="s", coords={"s": s})
    y = xr.DataArray(y, dims="s", coords={"s": s})
    z = ahn.interp(x=x, y=y, method=method)

    if plot:
        _, ax = plt.subplots(figsize=(10, 10))
        ahn.plot(ax=ax)
        gpd.GeoDataFrame(geometry=[line]).plot(ax=ax)

        _, ax = plt.subplots(figsize=(10, 10))
        z.plot(ax=ax)
    return z


@cache.cache_netcdf()
def get_latest_ahn_from_wcs(
    extent: list[float] = None,
    identifier: Literal["dsm_05m", "dtm_05m"] = "dsm_05m",
    res: float | None = None,
    version: str = "1.0.0",
    fmt: str = "image/tiff",
    crs: str = "EPSG:28992",
    maxsize: int = 2000,
) -> xr.DataArray:
    """Get the latest AHN from the wcs service.

    .. deprecated:: 0.10.0
        `get_latest_ahn_from_wcs` will be removed in nlmod 1.0.0, it is replaced by
        `download_latest_ahn_from_wcs` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array, optional
        extent. The default is None.
    identifier : str, optional
        Possible values for identifier are:
            'dsm_05m'
            'dtm_05m'
        The default is 'dsm_05m'.
        the identifier contains resolution and type info:
        - 'dtm' is only surface level (maaiveld), 'dsm' has other surfaces
        such as buildings.
        - 5m or 05m is a resolution of 5x5 or 0.5x0.5 meter.
    res : float, optional
        resolution of ahn raster. If None the resolution is inferred from the
        identifier. The default is None.
    version : str, optional
        version of wcs service, options are '1.0.0' and '2.0.1'.
        The default is '1.0.0'.
    fmt : str, optional
        geotif format . The default is 'GEOTIFF_FLOAT32'.
    crs : str, optional
        coördinate reference system. The default is 'EPSG:28992'.
    maxsize : float, optional
        maximum number of cells in x or y direction. The default is
        2000.

    Returns
    -------
    xr.DataArray
    """
    warnings.warn(
        "'get_latest_ahn_from_wcs' is deprecated and will eventually be removed, "
        "please use 'nlmod.read.ahn.download_latest_ahn_from_wcs()' in the future.",
        DeprecationWarning,
    )

    return download_latest_ahn_from_wcs(
        extent, identifier, res, version, fmt, crs, maxsize
    )


@cache.cache_netcdf()
def download_latest_ahn_from_wcs(
    extent: list[float] = None,
    identifier: Literal["dsm_05m", "dtm_05m"] = "dsm_05m",
    res: float | None = None,
    version: str = "1.0.0",
    fmt: str = "image/tiff",
    crs: str = "EPSG:28992",
    maxsize: int = 2000,
) -> xr.DataArray:
    """Get the latest AHN from the wcs service.

    Parameters
    ----------
    extent : list, tuple or np.array, optional
        extent. The default is None.
    identifier : str, optional
        Possible values for identifier are:
            'dsm_05m'
            'dtm_05m'
        The default is 'dsm_05m'.
        the identifier contains resolution and type info:
        - 'dtm' is only surface level (maaiveld), 'dsm' has other surfaces
        such as buildings.
        - 5m or 05m is a resolution of 5x5 or 0.5x0.5 meter.
    res : float, optional
        resolution of ahn raster. If None the resolution is inferred from the
        identifier. The default is None.
    version : str, optional
        version of wcs service, options are '1.0.0' and '2.0.1'.
        The default is '1.0.0'.
    fmt : str, optional
        geotif format . The default is 'GEOTIFF_FLOAT32'.
    crs : str, optional
        coördinate reference system. The default is 'EPSG:28992'.
    maxsize : float, optional
        maximum number of cells in x or y direction. The default is
        2000.

    Returns
    -------
    xr.DataArray
    """
    url = "https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS&request=GetCapabilities"

    if isinstance(extent, xr.DataArray):
        extent = tuple(extent.values)

    # check resolution
    if res is None:
        if "05m" in identifier.split("_")[1]:
            res = 0.5
        elif "5m" in identifier.split("_")[1]:
            logger.warning(
                "5 meter resolution is no langer available via wcs, try "
                "nlmod.read.get_ahn4 to obtain ahn with a 5m resolution. For "
                "more info see: "
                "https://www.pdok.nl/-/nieuwe-versie-ahn-beschikbaar-via-pdok"
            )
            raise ValueError(
                "5 meter resolution no longer available via wcs use nlmod.read.get_ahn4"
            )
        else:
            raise ValueError("could not infer resolution from identifier")

    da = wcs(
        url,
        extent,
        res,
        identifier=identifier,
        version=version,
        fmt=fmt,
        crs=crs,
        maxsize=maxsize,
    )
    return da


def _download_tiles_ellipsis(
    extent: list[float] | None = None,
    crs: int = 28992,
    timeout: float = 120.0,
    base_url: str = "https://api.ellipsis-drive.com/v3",
    path_id: str = "a9d410ad-a2f6-404c-948a-fdf6b43e77a6",
    timestamp_id: str = "05931403-2510-43af-9cc3-f60a066d4482",
) -> gpd.GeoDataFrame:
    url = f"{base_url}/path/{path_id}/vector/timestamp/{timestamp_id}/listFeatures"

    r = requests.get(url, timeout=timeout)
    if not r.ok:
        raise (HTTPError(f"Request not successful: {r.url}"))
    gdf = gpd.GeoDataFrame.from_features(r.json()["result"]["features"], crs=4326)
    gdf = gdf.to_crs(crs)
    # remove small digits becuase of crs-transformation
    gdf.geometry = gdf.geometry.apply(_round_coordinates, ndigits=0)

    gdf = gdf.set_index("AHN")

    if extent is not None:
        gdf = gdf.loc[gdf.intersection(extent_to_polygon(extent)).area > 0]
    return gdf


def _get_tiles_from_file(
    fname: str,
    extent: list[float] | None = None,
    crs: int = 28992,
) -> gpd.GeoDataFrame:
    if crs != 28992:
        raise ValueError("Only crs 28992 is supported")

    gdf = gpd.read_file(fname)

    # remove small digits becuase of crs-transformation
    gdf = gdf.set_index("AHN")

    if extent is not None:
        gdf = gdf.loc[gdf.intersection(extent_to_polygon(extent)).area > 0]

    return gdf


def _round_coordinates(geom: shapely.Geometry, ndigits: int = 2) -> shapely.Geometry:
    def _round_coords(x, y, z=None):
        x = round(x, ndigits)
        y = round(y, ndigits)

        if z is not None:
            z = round(x, ndigits)
            return (x, y, z)
        else:
            return (x, y)

    return shapely.ops.transform(_round_coords, geom)


@cache.cache_netcdf()
def get_ahn1(
    extent: list[float],
    identifier: Literal["AHN1 maaiveldmodel (DTM) 5m"] = "AHN1 maaiveldmodel (DTM) 5m",
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN1.

    .. deprecated:: 0.10.0
          `get_ahn1` will be removed in nlmod 1.0.0, it is replaced by
          `download_ahn1` because of new naming convention
          https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. The only
        allowed value is 'AHN1 maaiveldmodel (DTM) 5m'. The default is
        "AHN1 maaiveldmodel (DTM) 5m".

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    warnings.warn(
        "'get_ahn1' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.ahn.download_ahn1' instead",
        DeprecationWarning,
    )
    return download_ahn1(extent, identifier, as_data_array, **kwargs)


@cache.cache_netcdf()
def download_ahn1(
    extent: list[float],
    identifier: Literal["AHN1 maaiveldmodel (DTM) 5m"] = "AHN1 maaiveldmodel (DTM) 5m",
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN1.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. The only
        allowed value is 'AHN1 maaiveldmodel (DTM) 5m'. The default is
        "AHN1 maaiveldmodel (DTM) 5m".

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    _assert_as_data_array_is_none(as_data_array)
    da = _download_ahn_ellipsis(extent, identifier, **kwargs)
    if "merge_tiles" in kwargs and kwargs["merge_tiles"]:
        return da
    # original data is in cm. Convert the data to m, which is the unit of other ahns
    da = da / 100
    return da


@cache.cache_netcdf()
def get_ahn2(
    extent: list[float],
    identifier: Literal[
        "AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd",
        "AHN2 maaiveldmodel (DTM) ½m",
        "AHN2 DSM ½m",
        "AHN2 maaiveldmodel (DTM) 5m",
    ] = "AHN2 maaiveldmodel (DTM) 5m",
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN2.

    .. deprecated:: 0.10.0
        `get_ahn2` will be removed in nlmod 1.0.0, it is replaced by
        `download_ahn2` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd',
        'AHN2 maaiveldmodel (DTM) ½m', 'AHN2 DSM ½m', and
        'AHN2 maaiveldmodel (DTM) 5m'. The default is
        "AHN2 maaiveldmodel (DTM) 5m".

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    warnings.warn(
        "'get_ahn2' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.ahn.download_ahn2' instead",
        DeprecationWarning,
    )
    return download_ahn2(extent, identifier, as_data_array, **kwargs)


@cache.cache_netcdf()
def download_ahn2(
    extent: list[float],
    identifier: Literal[
        "AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd",
        "AHN2 maaiveldmodel (DTM) ½m",
        "AHN2 DSM ½m",
        "AHN2 maaiveldmodel (DTM) 5m",
    ] = None,
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN2.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd',
        'AHN2 maaiveldmodel (DTM) ½m', 'AHN2 DSM ½m', and
        'AHN2 maaiveldmodel (DTM) 5m'. The default is None.
        When no identifier is specified (the default), the url to download data from is
        taken from `config`, using the keyword-arguments `data_kind`, `tile_size` and
        `resolution`, which all can be specified as kwargs.
    **kwargs : dict
        See docstring of `download_ahn` for a descption of extra arguments.

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    _assert_as_data_array_is_none(as_data_array)
    if identifier is not None:
        return _download_ahn_ellipsis(extent, identifier, **kwargs)
    return _download_ahn_hwh(extent=extent, version="AHN2", **kwargs)


@cache.cache_netcdf()
def get_ahn3(
    extent: list[float],
    identifier: Literal[
        "AHN3 maaiveldmodel (DTM) ½m",
        "AHN3 DSM ½m",
        "AHN3 maaiveldmodel (DTM) 5m",
        "AHN3 DSM 5m",
    ] = "AHN3 maaiveldmodel (DTM) 5m",
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN3.

    .. deprecated:: 0.10.0
        `get_ahn3` will be removed in nlmod 1.0.0, it is replaced by
        `download_ahn3` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN3 maaiveldmodel (DTM) ½m', 'AHN3 DSM ½m',
        'AHN3 maaiveldmodel (DTM) 5m', and 'AHN3 DSM 5m'. The default is
        "AHN3 maaiveldmodel (DTM) 5m".

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    warnings.warn(
        "'get_ahn3' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.ahn.download_ahn3' instead",
        DeprecationWarning,
    )
    return download_ahn3(extent, identifier, as_data_array, **kwargs)


@cache.cache_netcdf()
def download_ahn3(
    extent: list[float],
    identifier: Literal[
        "AHN3 maaiveldmodel (DTM) ½m",
        "AHN3 DSM ½m",
        "AHN3 maaiveldmodel (DTM) 5m",
        "AHN3 DSM 5m",
    ] = None,
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN3.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN3 maaiveldmodel (DTM) ½m', 'AHN3 DSM ½m',
        'AHN3 maaiveldmodel (DTM) 5m', and 'AHN3 DSM 5m'. The default is None.
        When no identifier is specified (the default), the url to download data from is
        taken from `config`, using the keyword-arguments `data_kind`, `tile_size` and
        `resolution`, which all can be specified as kwargs.
    **kwargs : dict
        See docstring of `download_ahn` for a descption of extra arguments.

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    _assert_as_data_array_is_none(as_data_array)
    if identifier is not None:
        return _download_ahn_ellipsis(extent, identifier, **kwargs)
    return _download_ahn_hwh(extent=extent, version="AHN3", **kwargs)


@cache.cache_netcdf()
def get_ahn4(
    extent: list[float],
    identifier: Literal[
        "AHN4 maaiveldmodel (DTM) ½m",
        "AHN4 DSM ½m",
        "AHN4 maaiveldmodel (DTM) 5m",
        "AHN4 DSM 5m",
    ] = "AHN4 maaiveldmodel (DTM) 5m",
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN4.

    .. deprecated:: 0.10.0
        `get_ahn4` will be removed in nlmod 1.0.0, it is replaced by
        `download_ahn4` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN4 maaiveldmodel (DTM) ½m', 'AHN4 DSM ½m',
        'AHN4 maaiveldmodel (DTM) 5m', and 'AHN4 DSM 5m'. The default is
        "AHN4 maaiveldmodel (DTM) 5m".

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    warnings.warn(
        "'get_ahn4' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.ahn.download_ahn4' instead",
        DeprecationWarning,
    )
    return download_ahn4(extent, identifier, as_data_array, **kwargs)


@cache.cache_netcdf()
def download_ahn4(
    extent: list[float],
    identifier: Literal[
        "AHN4 maaiveldmodel (DTM) ½m",
        "AHN4 DSM ½m",
        "AHN4 maaiveldmodel (DTM) 5m",
        "AHN4 DSM 5m",
    ] = None,
    as_data_array: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN4.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN4 maaiveldmodel (DTM) ½m', 'AHN4 DSM ½m',
        'AHN4 maaiveldmodel (DTM) 5m', and 'AHN4 DSM 5m'. The default is None.
        When no identifier is specified (the default), the url to download data from is
        taken from `config`, using the keyword-arguments `data_kind`, `tile_size` and
        `resolution`, which all can be specified as kwargs.
    **kwargs : dict
        See docstring of `download_ahn` for a descption of extra arguments.

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    _assert_as_data_array_is_none(as_data_array)
    if identifier is not None:
        return _download_ahn_ellipsis(extent, identifier, **kwargs)
    return _download_ahn_hwh(extent=extent, version="AHN4", **kwargs)


@cache.cache_netcdf()
def get_ahn5(
    extent: list[float],
    identifier: Literal[
        "AHN5 maaiveldmodel (DTM) 5m",
        "AHN5 DSM 5m",
        "AHN5 maaiveldmodel (DTM) ½m",
        "AHN5 DSM ½m",
    ] = "AHN5 maaiveldmodel (DTM) 5m",
    **kwargs,
) -> xr.DataArray:
    """Download AHN5.

    .. deprecated:: 0.10.0
        `get_ahn5` will be removed in nlmod 1.0.0, it is replaced by
        `download_ahn5` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN5 maaiveldmodel (DTM) 5m', 'AHN5 DSM 5m',
        'AHN5 maaiveldmodel (DTM) ½m', and 'AHN5 DSM ½m'. The default is
        "AHN5 maaiveldmodel (DTM) 5m".

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    warnings.warn(
        "'get_ahn5' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.ahn.download_ahn5' instead",
        DeprecationWarning,
    )
    return download_ahn5(extent, identifier, **kwargs)


@cache.cache_netcdf()
def download_ahn5(
    extent: list[float],
    identifier: Literal[
        "AHN5 maaiveldmodel (DTM) 5m",
        "AHN5 DSM 5m",
        "AHN5 maaiveldmodel (DTM) ½m",
        "AHN5 DSM ½m",
    ] = None,
    **kwargs,
) -> xr.DataArray:
    """Download AHN5.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str, optional
        The identifier determines the resolution and the type of height data. Possible
        values are 'AHN5 maaiveldmodel (DTM) 5m', 'AHN5 DSM 5m',
        'AHN5 maaiveldmodel (DTM) ½m', and 'AHN5 DSM ½m'. The default is None.
        When no identifier is specified (the default), the url to download data from is
        taken from `config`, using the keyword-arguments `data_kind`, `tile_size` and
        `resolution`, which all can be specified as kwargs.
    **kwargs : dict
        See docstring of `download_ahn` for a descption of extra arguments.

    Returns
    -------
    xr.DataArray
        DataArray of the AHN
    """
    if identifier is not None:
        return _download_ahn_ellipsis(extent, identifier, **kwargs)
    return _download_ahn_hwh(extent=extent, version="AHN5", **kwargs)


def download_ahn6(extent: list[float], tile_size: str = "1x1km", **kwargs):
    """Download AHN6.


    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    tile_size : str, optional
        The size of the map tiles that the data is offered by the webservice. For AHN6,
        this can only be '1x1km'. The default is "1x1km".
    **kwargs : dict
        See docstring of `download_ahn` for a descption of extra arguments.

    Returns
    -------
    xr.DataArray
        DataArray of the AHN

    """
    if tile_size != "1x1km":
        raise (ValueError("AHN6 is only available in map sheets of 1 x 1 km."))
    return _download_ahn_hwh(
        extent=extent, version="AHN6", tile_size=tile_size, **kwargs
    )


def _update_ellipsis_tiles_in_data() -> None:
    tiles = _download_tiles_ellipsis()
    pathname = os.path.join(NLMOD_DATADIR, "ahn")
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    fname = os.path.join(pathname, "ellipsis_tiles.geojson")
    tiles.to_file(fname)


def _save_tiles_from_config_in_data(config: dict = None):
    """Saves the location of the map tiles of AHN to the data-directory of nlmod.

    For each available combination of version, data_kind, tile_size and resolution, a
    json-file is downloaded and saved to the folder `ahn` in the data-directory of
    nlmod. When requesting ahn-data after running this method, the location of the
    map-tiles is then not downloaded anymore, but taken from the previously downloaded
    json-files.

    Parameters
    ----------
    config : dict, optional
        A dictionary with properties of the data sources of the different AHN-versions.
        When None, the configuration is retreived from the method get_configuration().
        The default is None.

    Returns
    -------
    None.

    """
    if config is None:
        config = get_configuration()

    pathname = os.path.join(NLMOD_DATADIR, "ahn")
    for version in config:
        for data_kind in config[version]:
            for tile_size in config[version][data_kind]:
                url = config[version][data_kind][tile_size]
                if isinstance(url, dict):
                    for resolution in url:
                        fname = os.path.join(pathname, url[resolution].split("/")[-1])
                        _download_and_save_json_file(url[resolution], fname)
                else:
                    fname = os.path.join(pathname, url.split("/")[-1])
                    _download_and_save_json_file(url, fname)


def _delete_tiles_from_config_in_data(config: dict = None):
    """Delete the location of the map tiles of AHN from the data-directory of nlmod.

    This method deletes the files downloaded with `_save_tiles_from_config_in_data` from
    the data directory again.

    Parameters
    ----------
    config : dict, optional
        A dictionary with properties of the data sources of the different AHN-versions.
        When None, the configuration is retreived from the method get_configuration().
        The default is None.

    Returns
    -------
    None.

    """
    if config is None:
        config = get_configuration()

    pathname = os.path.join(NLMOD_DATADIR, "ahn")
    for version in config:
        for data_kind in config[version]:
            for tile_size in config[version][data_kind]:
                url = config[version][data_kind][tile_size]
                if isinstance(url, dict):
                    for resolution in url:
                        fname = os.path.join(pathname, url[resolution].split("/")[-1])
                        if os.path.isfile(fname):
                            os.remove(fname)
                else:
                    fname = os.path.join(pathname, url.split("/")[-1])
                    if os.path.isfile(fname):
                        os.remove(fname)


def _download_and_save_json_file(url, fname, timeout=120.0):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with open(fname, "wb") as f:
        f.write(response.content)


def get_configuration():
    """Get the location of json-files with positions of map tiles with AHN-data

    Returns
    -------
    dict
        A dictionary with the locations of the json-files that contain the position of
        the map-tiles of AHN-data, for several combinations of `version`, `data_kind`,
        `tile_size` and `resolution`.

    """
    pathname = os.path.join(NLMOD_DATADIR, "ahn")
    base_url = "https://basisdata.nl/hwh-portal/20230609_tmp/links/nationaal/Nederland"
    config = {}

    def get_fname(file):
        fname = os.path.join(pathname, file)
        if os.path.isfile(fname):
            return fname
        else:
            return f"{base_url}/{file}"

    config["AHN2"] = {
        "DTM": {
            "5x6.25km": {
                0.5: get_fname("AHN2_DTM05.json"),
                5.0: get_fname("AHN2_DTM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN2_KM_DTM05.json"),
                5.0: get_fname("AHN2_KM_DTM5.json"),
            },
        },
        "DSM": {
            "5x6.25km": {
                0.5: get_fname("AHN2_DSM05.json"),
                5.0: get_fname("AHN2_DSM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN2_KM_DSM05.json"),
                5.0: get_fname("AHN2_KM_DSM5.json"),
            },
        },
        "PC": {
            "1x1km": get_fname("AHN2_KM_PC.json"),
        },
        "LAZ_g": {
            "5x6.25km": get_fname("AHN2_LAZ_g.json"),
        },
        "LAZ_u": {
            "5x6.25km": get_fname("AHN2_LAZ_u.json"),
        },
    }

    config["AHN3"] = {
        "DTM": {
            "5x6.25km": {
                0.5: get_fname("AHN3_DTM05.json"),
                5.0: get_fname("AHN3_DTM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN3_KM_DTM05.json"),
                5.0: get_fname("AHN3_KM_DTM5.json"),
            },
        },
        "DSM": {
            "5x6.25km": {
                0.5: get_fname("AHN3_DSM05.json"),
                5.0: get_fname("AHN3_DSM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN3_KM_DSM05.json"),
                5.0: get_fname("AHN3_KM_DSM5.json"),
            },
        },
        "PC": {
            "5x6.25km": get_fname("AHN3_PC.json"),
            "1x1km": get_fname("AHN3_KM_PC.json"),
        },
    }

    config["AHN4"] = {
        "DTM": {
            "5x6.25km": {
                0.5: get_fname("AHN4_DTM05.json"),
                5.0: get_fname("AHN4_DTM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN4_KM_DTM05.json"),
                5.0: get_fname("AHN4_KM_DTM5.json"),
            },
        },
        "DSM": {
            "5x6.25km": {
                0.5: get_fname("AHN4_DSM05.json"),
                5.0: get_fname("AHN4_DSM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN4_KM_DSM05.json"),
                5.0: get_fname("AHN4_KM_DSM5.json"),
            },
        },
        "PC": {
            "5x6.25km": get_fname("AHN4_PC.json"),
            "1x1km": get_fname("AHN4_KM_PC.json"),
        },
    }

    config["AHN5"] = {
        "DTM": {
            "5x6.25km": {
                0.5: get_fname("AHN5_DTM05.json"),
                5.0: get_fname("AHN5_DTM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN5_KM_DTM05.json"),
                5.0: get_fname("AHN5_KM_DTM5.json"),
            },
        },
        "DSM": {
            "5x6.25km": {
                0.5: get_fname("AHN5_DSM05.json"),
                5.0: get_fname("AHN5_DSM5.json"),
            },
            "1x1km": {
                0.5: get_fname("AHN5_KM_DSM05.json"),
                5.0: get_fname("AHN5_KM_DSM5.json"),
            },
        },
        "PC": {
            "5x6.25km": get_fname("AHN5_PC.json"),
            "1x1km": get_fname("AHN5_KM_PC.json"),
        },
    }

    config["AHN6"] = {
        "DTM": {
            "1x1km": {
                0.5: get_fname("AHN6_KM_DTM05.json"),
                5.0: get_fname("AHN6_KM_DTM5.json"),
            }
        },
        "DSM": {
            "1x1km": {
                0.5: get_fname("AHN6_KM_DSM05.json"),
                5.0: get_fname("AHN6_KM_DSM5.json"),
            }
        },
        "PC": {
            "1x1km": get_fname("AHN6_KM_PC.json"),
        },
        "COPC": {
            "1x1km": get_fname("AHN6_KM_PC_COPC.json"),
        },
    }

    return config


@cache.cache_netcdf()
def _download_tiles_hwh(
    extent: list[float],
    version: str = "AHN4",
    data_kind: str = "DTM",
    tile_size: str = "5x6.25km",
    resolution: float = 5.0,
    config: dict = None,
    timeout: float = 120.0,
):
    """
    Download tiles that contain AHN-data.

    Parameters
    ----------
    extent : list, tuple or np.array
        The extent to be downloaded, consisting of 4 floats: xmin, xmax, ymin, ymax.
    version : str, optional
        The AHN_version, which can be 'AHN2', 'AHN3', 'AHN4', 'AHN5' and 'AHN6'. The
        default is "AHN4", as this is available in the whole of the Netherlands.
    data_kind : str, optional
        The kind of data. This can be 'DTM' (terrain elevation) or "DSM" (surface
        elevation). The default is "DTM".
    tile_size : str, optional
        The size of the map tiles that the data is offered by the webservice. This can
        be '5x6.25km' or '1x1km'. The default is '5x6.25km'.
    resolution : float, optional
        The resolution of the AHN-data, which can be 0.5 and 5.0 m. The default is 5.0.
    config : dict, optional
        A dictionary with properties of the data sources of the different AHN-versions.
        When None, the configuration is retreived from the method get_configuration().
        The default is None.
    timeout : float, optional
        The amount of seconds to wait for the server to send data before giving up. The
        default is 120.

    Raises
    ------
    AssertionError
        If the specified value for `resolution` or `data_kind` is not supported
    KeyError
        When there is no data-source that matches the requested parameters.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the location of the map tiles, and url's with the
        location of the AHN-data.

    """
    assert resolution in [0.5, 5.0], "`resolution` should be 0.5 or 5.0 m"
    assert data_kind in ["DTM", "DSM"], "`data_kind` should be 'DTM' or 'DSM'"
    assert tile_size in [
        "5x6.25km",
        "1x1km",
    ], "`tile_size` should be '5x6.25km' or '1x1km'"
    if config is None:
        config = get_configuration()
    try:
        url = config[version][data_kind][tile_size]
        if isinstance(url, dict):
            url = url[resolution]
    except KeyError as e:
        raise KeyError(f"Data not available for {version}: {e}") from e
    if url.startswith("http"):
        # write the tiles to a TemporaryDirectory, as direct read from url fails for large files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = os.path.join(tmpdir, "data.json")
            _download_and_save_json_file(url, tmp_file, timeout=timeout)
            tiles = gpd.read_file(tmp_file)
    else:
        tiles = gpd.read_file(url)

    tiles = tiles.set_crs(28992, allow_override=True)
    tiles = tiles.loc[tiles.intersection(extent_to_polygon(extent)).area > 0]
    if len(tiles) == 0:
        raise (ValueError("No data found within extent"))
    return tiles


@cache.cache_netcdf()
def _download_ahn_hwh(
    extent: list[float],
    version: str = "AHN4",
    data_kind: str = "DTM",
    tile_size: str = "5x6.25km",
    resolution: float = 5.0,
    merge_tiles: bool = True,
    cut_extent: bool = True,
    config: dict = None,
):
    """
    Download AHN-data.

    Parameters
    ----------
    extent : list, tuple or np.array
        The extent to be downloaded, consisting of 4 floats: xmin, xmax, ymin, ymax.
    version : str, optional
        The AHN_version, which can be 'AHN2', 'AHN3', 'AHN4', 'AHN5' and 'AHN6'. The
        default is "AHN4", as this is available in the whole of the Netherlands.
    data_kind : str, optional
        The kind of data. This can be 'DTM' (terrain elevation) or "DSM" (surface
        elevation). The default is "DTM".
    tile_size : str, optional
        The size of the map tiles that the data is offered by the webservice. This can
        be '5x6.25km' or '1x1km'. The default is '5x6.25km'.
    resolution : float, optional
        The resolution of the AHN-data, which can be 0.5 and 5.0 m. The default is 5.0.
    merge_tiles : bool, optional
        If True, the function returns a merged DataArray. If False, the function
        returns a list of DataArrays with the original tiles. The default is True.
    cut_extent : bool, optional
        If True, only keep the requested extent from the data. The defualts is True.
    config : dict, optional
        A dictionary with properties of the data sources of the different AHN-versions.
        When None, the configuration is retreived from the method get_configuration().
        The default is None.

    Raises
    ------
    AssertionError
        If the specified value for `resolution` or `data_kind` is not supported
    KeyError
        When there is no data-source that matches the requested parameters.

    Returns
    -------
    xr.DataArray or list of xr.DataArray
        A DataArray with the AHN-data, or a list of DataArrays with AHN-data if
        merge_tiles=False.

    """
    tiles = _download_tiles_hwh(
        extent=extent,
        version=version,
        resolution=resolution,
        data_kind=data_kind,
        tile_size=tile_size,
        config=config,
    )

    rasterio_env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "YES",
        # "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        # "GDAL_TIFF_INTERNAL_MASK": "YES",
        "GDAL_IGNORE_MISSING_MASK": "YES",
    }

    das = []
    for index in tqdm(tiles.index, desc=f"Downloading tiles of {version}"):
        file = tiles.at[index, "file"]
        if file.endswith(".zip"):
            path = file.split("/")[-1].replace(".zip", ".TIF")
            if path.lower().endswith(".tif.tif"):
                path = path[:-4]
            with Env(**rasterio_env):
                da = rioxarray.open_rasterio(f"zip+{file}!/{path}", mask_and_scale=True)
        else:
            with Env(**rasterio_env):
                da = rioxarray.open_rasterio(file, mask_and_scale=True)
        if cut_extent:
            da = da.sel(x=slice(extent[0], extent[1]), y=slice(extent[3], extent[2]))
        das.append(da)
    if merge_tiles:
        da = merge_arrays(das)
        if da.dims[0] == "band":
            da = da[0].drop_vars("band")
        if "_FillValue" in da.attrs:
            del da.attrs["_FillValue"]
        da.attrs["version"] = version
        da.attrs["data_kind"] = data_kind
        da.attrs["tile_size"] = tile_size
        da.attrs["resolution"] = resolution
        return da
    return das


@cache.cache_netcdf()
def _download_ahn_ellipsis(
    extent: list[float],
    identifier: str,
    merge_tiles: bool = True,
    cut_extent: bool = True,
    **kwargs,
) -> xr.DataArray | list[xr.DataArray]:
    """Download and merge AHN tiles from the ellipsis.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : str
        The identifier determines the AHN-version, the resolution and the type of height
        data.
    merge_tiles : bool, optional
        If True, the function returns a merged DataArray. If False, the function
        returns a list of DataArrays with the original tiles. The default is True.
    cut_extent : bool, optional
        If True, only keep the requested extent from the data. The defualts is True.

    Returns
    -------
    xr.DataArray
        DataArray of the AHN

    """
    fname = os.path.join(NLMOD_DATADIR, "ahn", "ellipsis_tiles.geojson")
    tiles = _get_tiles_from_file(fname, extent=extent, **kwargs)

    identifier = _rename_identifier(identifier)
    if identifier not in tiles.columns:
        raise (ValueError(f"Unknown ahn-identifier: {identifier}"))
    tiles = tiles[~tiles[identifier].isna()]
    das = []
    for tile in tqdm(tiles.index, desc=f"Downloading tiles of {identifier}"):
        url = tiles.at[tile, identifier]
        if url == "None":
            continue
        if url.endswith(".zip"):
            path = url.split("/")[-1].replace(".zip", ".TIF")
            if path.lower().endswith(".tif.tif"):
                path = path[:-4]
            da = rioxarray.open_rasterio(f"zip+{url}!/{path}", mask_and_scale=True)
        else:
            da = rioxarray.open_rasterio(url, mask_and_scale=True)
        if cut_extent:
            da = da.sel(x=slice(extent[0], extent[1]), y=slice(extent[3], extent[2]))
        das.append(da)

    if len(das) == 0:
        raise (ValueError("No data found within extent"))

    if merge_tiles:
        da = merge_arrays(das)
        if da.dims[0] == "band":
            da = da[0].drop_vars("band")
        if "_FillValue" in da.attrs:
            del da.attrs["_FillValue"]
        return da
    return das


def _rename_identifier(identifier: str) -> str:
    rename = {
        "ahn1_5m": "AHN1 maaiveldmodel (DTM) 5m",
        "AHN1_5M": "AHN1 maaiveldmodel (DTM) 5m",
        "ahn2_05m_i": "AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd",
        "AHN2_05M_I": "AHN2 maaiveldmodel (DTM) ½m, geïnterpoleerd",
        "ahn2_05m_n": "AHN2 maaiveldmodel (DTM) ½m",
        "AHN2_05M_N": "AHN2 maaiveldmodel (DTM) ½m",
        "ahn2_05m_r": "AHN2 DSM ½m",
        "AHN2_05M_R": "AHN2 DSM ½m",
        "ahn2_5m": "AHN2 maaiveldmodel (DTM) 5m",
        "AHN2_5M_M": "AHN2 maaiveldmodel (DTM) 5m",
        "AHN3_05m_DTM": "AHN3 maaiveldmodel (DTM) ½m",
        "AHN3_05M_M": "AHN3 maaiveldmodel (DTM) ½m",
        "AHN3_05m_DSM": "AHN3 DSM ½m",
        "AHN3_05M_R": "AHN3 DSM ½m",
        "AHN3_5m_DTM": "AHN3 maaiveldmodel (DTM) 5m",
        "AHN3_5M_M": "AHN3 maaiveldmodel (DTM) 5m",
        "AHN3_5m_DSM": "AHN3 DSM 5m",
        "AHN3_5M_R": "AHN3 DSM 5m",
        "AHN4_DTM_05m": "AHN4 maaiveldmodel (DTM) ½m",
        "AHN4_05M_M": "AHN4 maaiveldmodel (DTM) ½m",
        "AHN4_DSM_05m": "AHN4 DSM ½m",
        "AHN4_05M_R": "AHN4 DSM ½m",
        "AHN4_DTM_5m": "AHN4 maaiveldmodel (DTM) 5m",
        "AHN4_5M_M": "AHN4 maaiveldmodel (DTM) 5m",
        "AHN4_DSM_5m": "AHN4 DSM 5m",
        "AHN4_5M_R": "AHN4 DSM 5m",
        "AHN5_5M_M": "AHN5 maaiveldmodel (DTM) 5m",
        "AHN5_5M_R": "AHN5 DSM 5m",
        "AHN5_05M_M": "AHN5 maaiveldmodel (DTM) ½m",
        "AHN5_05M_R": "AHN5 DSM ½m",
    }
    if identifier in rename:
        id_new = rename[identifier]
        logger.warning(f"The identifier {identifier} is deprecated. Rename to {id_new}")
        identifier = id_new
    return identifier


def _assert_as_data_array_is_none(as_data_array: bool | None) -> None:
    if as_data_array is not None:
        raise (
            DeprecationWarning(
                "The as_data_array-argument has been removed from the ahn-"
                "methods, and these methods now allways return a DataArray. "
                "Remove the as_data_array-argument."
            )
        )
