# -*- coding: utf-8 -*-
"""Created on Fri Jun 12 15:33:03 2020.

@author: ruben
"""

import datetime as dt
import logging

import rasterio
import rioxarray
import xarray as xr
from rasterio import merge
from rasterio.io import MemoryFile
from tqdm import tqdm

from .. import cache
from ..dims.resample import get_extent, structured_da_to_ds
from ..util import get_ds_empty
from .webservices import arcrest, wcs, wfs

logger = logging.getLogger(__name__)


@cache.cache_netcdf
def get_ahn(ds, identifier="ahn3_5m_dtm", method="average"):
    """Get a model dataset with ahn variable.
    Parameters
    ----------
    ds : xr.Dataset
        dataset with the model information.
    identifier : str, optional
        Possible values for identifier are:
            'ahn2_05m_int'
            'ahn2_05m_non'
            'ahn2_05m_ruw'
            'ahn2_5m'
            'ahn3_05m_dsm'
            'ahn3_05m_dtm'
            'ahn3_5m_dsm'
            'ahn3_5m_dtm'
        The default is 'ahn3_5m_dtm'.
    method : str, optional
        Method used to resample ahn to grid of ds. See documentation of
        nlmod.resample.structured_da_to_ds for possible values. The default is
        'average'.

    Returns
    -------
    ds_out : xr.Dataset
        Dataset with the ahn variable.
    """

    url = _infer_url(identifier)
    extent = get_extent(ds)
    ahn_ds_raw = get_ahn_from_wcs(extent=extent, url=url, identifier=identifier)

    ahn_ds_raw = ahn_ds_raw.drop_vars("band")

    ahn_da = structured_da_to_ds(ahn_ds_raw, ds, method=method)
    ahn_da.attrs["source"] = identifier
    ahn_da.attrs["url"] = url
    ahn_da.attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
    ahn_da.attrs["units"] = "mNAP"

    ds_out = get_ds_empty(ds)
    ds_out["ahn"] = ahn_da

    return ds_out


def _infer_url(identifier=None):
    """infer the url from the identifier.

    Parameters
    ----------
    identifier : str, optional
        identifier of the ahn type. The default is None.

    Raises
    ------
    ValueError
        unknown identifier.

    Returns
    -------
    url : str
        ahn url corresponding to identifier.
    """

    # infer url from identifier
    if "ahn2" in identifier:
        url = "https://geodata.nationaalgeoregister.nl/ahn2/wcs?service=WCS"
    elif "ahn3" in identifier:
        url = "https://geodata.nationaalgeoregister.nl/ahn3/wcs?service=WCS"
    else:
        ValueError(f"unknown identifier -> {identifier}")

    return url


def get_ahn_from_wcs(
    extent=None,
    identifier="ahn3_5m_dtm",
    url=None,
    res=None,
    version="1.0.0",
    fmt="GEOTIFF_FLOAT32",
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
            'ahn2_05m_int'
            'ahn2_05m_non'
            'ahn2_05m_ruw'
            'ahn2_5m'
            'ahn3_05m_dsm'
            'ahn3_05m_dtm'
            'ahn3_5m_dsm'
            'ahn3_5m_dtm'
        The default is 'ahn3_5m_dtm'.
        the identifier also contains resolution and type info:
        - 5m or 05m is a resolution of 5x5 or 0.5x0.5 meter.
        - 'dtm' is only surface level (maaiveld), 'dsm' has other surfaces
        such as building.
    url : str or None, optional
        possible values None, 'ahn2' and 'ahn3'. If None the url is inferred
        from the identifier. The default is None.
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

    if isinstance(extent, xr.DataArray):
        extent = tuple(extent.values)

    # get url
    if url is None:
        url = _infer_url(identifier)
    elif url == "ahn2":
        url = "https://geodata.nationaalgeoregister.nl/ahn2/wcs?service=WCS"
    elif url == "ahn3":
        url = "https://geodata.nationaalgeoregister.nl/ahn3/wcs?service=WCS"
    elif not url.startswith("https://geodata.nationaalgeoregister.nl"):
        raise ValueError(f"unknown url -> {url}")

    # check resolution
    if res is None:
        if "05m" in identifier.split("_")[1]:
            res = 0.5
        elif "5m" in identifier.split("_")[1]:
            res = 5.0
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


def get_ahn3_tiles(extent=None, **kwargs):
    """Get the tiles (kaartbladen) of AHN3 as a GeoDataFrame."""
    url = "https://service.pdok.nl/rws/ahn3/wfs/v1_0?service=wfs"
    layer = "ahn3_bladindex"
    gdf = wfs(url, layer, extent=extent, **kwargs)
    if not gdf.empty:
        gdf = gdf.set_index("bladnr")
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


def get_ahn3(extent, identifier="DTM_5m", as_data_array=True):
    """Download AHN3.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    identifier : TYPE, optional
        Possible values are 'DSM_50cm', 'DTM_50cm', 'DSM_5m' and 'DTM_5m'. The default
        is "DTM_5m".
    as_data_array : bool, optional
        return the data as as xarray DataArray if true. The default is True.

    Returns
    -------
    xr.DataArray or MemoryFile
        DataArray (if as_data_array is True) or Rasterio MemoryFile of the AHN
    """
    tiles = get_ahn3_tiles(extent)
    if tiles.empty:
        raise (Exception("AHN3 has no data for requested extent"))
    datasets = []
    for bladnr in tqdm(tiles.index, desc=f"Downloading tiles of {identifier}"):
        url = "https://ns_hwh.fundaments.nl/hwh-ahn/AHN3/"
        if identifier == "DSM_50cm":
            url = f"{url}DSM_50cm/R_{bladnr.upper()}.zip"
        elif identifier == "DTM_50cm":
            url = f"{url}DTM_50cm/M_{bladnr.upper()}.zip"
        elif identifier == "DSM_5m":
            url = f"{url}DSM_5m/R5_{bladnr.upper()}.zip"
        elif identifier == "DTM_5m":
            url = f"{url}DTM_5m/M5_{bladnr.upper()}.zip"
        else:
            raise (Exception(f"Unknown identifier: {identifier}"))
        path = url.split("/")[-1].replace(".zip", ".TIF")
        datasets.append(rasterio.open(f"zip+{url}!/{path}"))
    memfile = MemoryFile()
    merge.merge(datasets, dst_path=memfile)
    if as_data_array:
        da = rioxarray.open_rasterio(memfile.open(), mask_and_scale=True)[0]
        da = da.sel(x=slice(extent[0], extent[1]), y=slice(extent[3], extent[2]))
        return da
    return memfile


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
    if tiles.empty:
        raise (Exception("AHN4 has no data for requested extent"))
    datasets = []
    for name in tqdm(tiles.index, desc=f"Downloading tiles of {identifier}"):
        url = tiles.at[name, identifier]
        path = url.split("/")[-1].replace(".zip", ".TIF")
        datasets.append(rasterio.open(f"zip+{url}!/{path}"))
    memfile = MemoryFile()
    merge.merge(datasets, dst_path=memfile)
    if as_data_array:
        da = rioxarray.open_rasterio(memfile.open(), mask_and_scale=True)[0]
        da = da.sel(x=slice(extent[0], extent[1]), y=slice(extent[3], extent[2]))
        return da
    return memfile
