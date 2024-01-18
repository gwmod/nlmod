import datetime as dt
import logging

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
import rioxarray
import xarray as xr
from rasterio import merge
from rasterio.io import MemoryFile
from tqdm import tqdm

from .. import cache
from ..dims.resample import get_extent, structured_da_to_ds
from ..util import get_ds_empty
from .webservices import arcrest, wcs

logger = logging.getLogger(__name__)


@cache.cache_netcdf
def get_ahn(ds=None, identifier="AHN4_DTM_5m", method="average", extent=None):
    """Get a model dataset with ahn variable.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with the model information.
    identifier : str, optional
        Possible values for the different AHN-versions are (casing is important):
            AHN1: 'ahn1_5m'
            AHN2: 'ahn2_05m_i', 'ahn2_05m_n', 'ahn2_05m_r' or 'ahn2_5m'
            AHN3: 'AHN3_05m_DSM', 'AHN3_05m_DTM', 'AHN3_5m_DSM' or 'AHN3_5m_DTM'
            AHN4: 'AHN4_DTM_05m', 'AHN4_DTM_5m', 'AHN4_DSM_05m' or 'AHN4_DSM_5m'
        The default is 'AHN4_DTM_5m'.
    method : str, optional
        Method used to resample ahn to grid of ds. See documentation of
        nlmod.resample.structured_da_to_ds for possible values. The default is
        'average'.
    extent : list, tuple or np.array, optional
        extent. The default is None.

    Returns
    -------
    ds_out : xr.Dataset
        Dataset with the ahn variable.
    """
    if extent is None and ds is not None:
        extent = get_extent(ds)
    version = int(identifier[3])
    if version == 1:
        ahn_ds_raw = get_ahn1(extent, identifier=identifier)
    elif version == 2:
        ahn_ds_raw = get_ahn2(extent, identifier=identifier)
    elif version == 3:
        ahn_ds_raw = get_ahn3(extent, identifier=identifier)
    elif version == 4:
        ahn_ds_raw = get_ahn4(extent, identifier=identifier)
    else:
        raise (ValueError(f"Unknown ahn-version: {version}"))

    ahn_ds_raw = ahn_ds_raw.drop_vars("band")

    if ds is None:
        ahn_da = ahn_ds_raw
    else:
        ahn_da = structured_da_to_ds(ahn_ds_raw, ds, method=method)
    ahn_da.attrs["source"] = identifier
    ahn_da.attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
    ahn_da.attrs["units"] = "mNAP"

    if ds is None:
        return ahn_da

    ds_out = get_ds_empty(ds, keep_coords=("y", "x"))
    ds_out["ahn"] = ahn_da

    return ds_out


def get_ahn_at_point(
    x,
    y,
    buffer=0.75,
    return_da=False,
    return_mean=False,
    identifier="dsm_05m",
    res=0.5,
    **kwargs,
):
    """
    Get the height of the surface level at a certain point, defined by x and y.

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
        The identifier passed onto get_latest_ahn_from_wcs. The default is "dsm_05m".
    res : float, optional
        The resolution that is passed onto get_latest_ahn_from_wcs. The default is 0.5.
    **kwargs : dict
        kwargs are passed onto the method get_latest_ahn_from_wcs.

    Returns
    -------
    float
        The surface level value at the requested point.

    """
    extent = [x - buffer, x + buffer, y - buffer, y + buffer]
    ahn = get_latest_ahn_from_wcs(extent, identifier=identifier, res=res, **kwargs)
    if return_da:
        # return a DataArray
        return ahn
    if return_mean:
        # return the mean (usefull when there are NaN's near the center)
        return float(ahn.mean())
    else:
        # return the center pixel
        return ahn.data[int((ahn.shape[0] - 1) / 2), int((ahn.shape[1] - 1) / 2)]


def get_ahn_along_line(line, ahn=None, dx=None, num=None, method="linear", plot=False):
    """
    Get the height of the surface level along a line.

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


@cache.cache_netcdf
def get_latest_ahn_from_wcs(
    extent=None,
    identifier="dsm_05m",
    res=None,
    version="1.0.0",
    fmt="image/tiff",
    crs="EPSG:28992",
    maxsize=2000,
):
    """
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
    xr.DataArray or MemoryFile
        DataArray (if as_data_array is True) or Rasterio MemoryFile of the AHN
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


def get_ahn2_tiles(extent=None):
    """Get the tiles (kaartbladen) of AHN3 as a GeoDataFrame.

    The links in the tiles are cuurently incorrect. Thereore
    get_ahn3_tiles is used in get_ahn2 and get_ahn1, as the tiles from
    get_ahn3_tiles also contain information about the tiles of ahn1 and
    ahn2
    """
    url = "https://services.arcgis.com/nSZVuSZjHpEZZbRo/arcgis/rest/services/Kaartbladen_AHN2/FeatureServer"
    layer = 0
    gdf = arcrest(url, layer, extent)
    if not gdf.empty:
        gdf = gdf.set_index("Kaartblad")
    return gdf


def get_ahn3_tiles(extent=None):
    """Get the tiles (kaartbladen) of AHN3 as a GeoDataFrame."""
    url = "https://services.arcgis.com/nSZVuSZjHpEZZbRo/arcgis/rest/services/Kaartbladen_AHN3/FeatureServer"
    layer = 0
    gdf = arcrest(url, layer, extent)
    if not gdf.empty:
        gdf = gdf.set_index("Kaartblad")
    return gdf


def get_ahn4_tiles(extent=None):
    """Get the tiles (kaartbladen) of AHN4 as a GeoDataFrame with download
    links."""
    url = "https://services.arcgis.com/nSZVuSZjHpEZZbRo/arcgis/rest/services/Kaartbladen_AHN4/FeatureServer"
    layer = 0
    gdf = arcrest(url, layer, extent)
    if not gdf.empty:
        gdf = gdf.set_index("Name")
    return gdf


@cache.cache_netcdf
def get_ahn1(extent, identifier="ahn1_5m", as_data_array=True):
    """Download AHN1.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : TYPE, optional
        Only allowed value is 'ahn1_5m'. The default is "ahn1_5m".
    as_data_array : bool, optional
        return the data as as xarray DataArray if true. The default is True.

    Returns
    -------
    xr.DataArray or MemoryFile
        DataArray (if as_data_array is True) or Rasterio MemoryFile of the AHN
    """
    # tiles are equal to that of ahn3
    tiles = get_ahn3_tiles(extent)
    da = _download_and_combine_tiles(tiles, identifier, extent, as_data_array)
    if as_data_array:
        # original data is in cm. Convert the data to m, which is the unit of other ahns
        da = da / 100
    return da


@cache.cache_netcdf
def get_ahn2(extent, identifier="ahn2_5m", as_data_array=True):
    """Download AHN2.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : TYPE, optional
        Possible values are 'ahn2_05m_i', 'ahn2_05m_n', 'ahn2_05m_r' and 'ahn2_5m'. The
        default is "ahn2_5m".
    as_data_array : bool, optional
        return the data as as xarray DataArray if true. The default is True.

    Returns
    -------
    xr.DataArray or MemoryFile
        DataArray (if as_data_array is True) or Rasterio MemoryFile of the AHN
    """
    # tiles are equal to that of ahn3
    tiles = get_ahn3_tiles(extent)
    return _download_and_combine_tiles(tiles, identifier, extent, as_data_array)


@cache.cache_netcdf
def get_ahn3(extent, identifier="AHN3_5m_DTM", as_data_array=True):
    """Download AHN3.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : TYPE, optional
        Possible values are 'AHN3_05m_DSM', 'AHN3_05m_DTM', 'AHN3_5m_DSM' and
        'AHN3_5m_DTM'. The default is "AHN3_5m_DTM".
    as_data_array : bool, optional
        return the data as as xarray DataArray if true. The default is True.

    Returns
    -------
    xr.DataArray or MemoryFile
        DataArray (if as_data_array is True) or Rasterio MemoryFile of the AHN
    """
    tiles = get_ahn3_tiles(extent)
    return _download_and_combine_tiles(tiles, identifier, extent, as_data_array)


@cache.cache_netcdf
def get_ahn4(extent, identifier="AHN4_DTM_5m", as_data_array=True):
    """Download AHN4.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : TYPE, optional
        Possible values are 'AHN4_DTM_05m', 'AHN4_DTM_5m', 'AHN4_DSM_05m' and
        'AHN4_DSM_5m'. The default is "AHN4_DTM_5m".
    as_data_array : bool, optional
        return the data as as xarray DataArray if true. The default is True.

    Returns
    -------
    xr.DataArray or MemoryFile
        DataArray (if as_data_array is True) or Rasterio MemoryFile of the AHN
    """
    tiles = get_ahn4_tiles(extent)
    return _download_and_combine_tiles(tiles, identifier, extent, as_data_array)


def _download_and_combine_tiles(tiles, identifier, extent, as_data_array):
    """Internal method to download and combine ahn-data."""
    if tiles.empty:
        raise (Exception(f"{identifier} has no data for requested extent"))
    datasets = []
    for name in tqdm(tiles.index, desc=f"Downloading tiles of {identifier}"):
        url = tiles.at[name, identifier]
        path = url.split("/")[-1].replace(".zip", ".TIF")
        if path.lower().endswith(".tif.tif"):
            path = path[:-4]
        datasets.append(rasterio.open(f"zip+{url}!/{path}"))
    memfile = MemoryFile()
    merge.merge(datasets, dst_path=memfile)
    if as_data_array:
        da = rioxarray.open_rasterio(memfile.open(), mask_and_scale=True)[0]
        da = da.sel(x=slice(extent[0], extent[1]), y=slice(extent[3], extent[2]))
        return da
    return memfile
