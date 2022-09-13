# -*- coding: utf-8 -*-
"""Created on Fri Jun 12 15:33:03 2020.
@author: ruben
"""

import datetime as dt
import logging

import numpy as np
import xarray as xr
from owslib.wcs import WebCoverageService
import rasterio
from rasterio import merge
from rasterio.io import MemoryFile
import rioxarray
from tqdm import tqdm

from .. import cache, mdims, util
from .webservices import arcrest

logger = logging.getLogger(__name__)


@cache.cache_netcdf
def get_ahn(model_ds, identifier="ahn3_5m_dtm", method="average"):
    """Get a model dataset with ahn variable.
    Parameters
    ----------
    model_ds : xr.Dataset
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
        Method used to resample ahn to grid of model_ds. See
        mdims.resample.structured_da_to_ds for possible values. The default is
        'average'.

    Returns
    -------
    model_ds_out : xr.Dataset
        Dataset with the ahn variable.
    """

    url = _infer_url(identifier)
    extent = mdims.resample.get_extent(model_ds)
    ahn_ds_raw = get_ahn_within_extent(extent=extent, url=url, identifier=identifier)

    ahn_ds_raw = ahn_ds_raw.drop_vars('band')
        
    ahn_da = mdims.resample.structured_da_to_ds(ahn_ds_raw, model_ds, method=method)
    ahn_da.attrs["source"] = identifier
    ahn_da.attrs["url"] = url
    ahn_da.attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
    ahn_da.attrs["units"] = "mNAP"

    model_ds_out = util.get_model_ds_empty(model_ds)
    model_ds_out["ahn"] = ahn_da

    return model_ds_out


def split_ahn_extent(
    extent, x_segments, y_segments, maxsize, res, url, identifier, version, fmt, crs
):
    """There is a max height and width limit for the wcs server. This function
    splits your extent in chunks smaller than the limit. It returns a list of
    Memory files.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    res : float
        The resolution of the requested output-data
    x_segments : int
        number of tiles on the x axis
    y_segments : int
        number of tiles on the y axis
    maxsize : int or float
        maximum widht or height of ahn tile
    as_data_array : bool, optional
        return the data as as xarray DataArray if true. The default is True.
    **kwargs :
        keyword arguments of the get_ahn_extent function.

    Returns
    -------
    xr.DataArray or MemoryFile
        DataArray (if as_data_array is True) or Rasterio MemoryFile of the
        merged AHN
    Notes
    -----
    1. The resolution is used to obtain the ahn from the wcs server. Not sure
    what kind of interpolation is used to resample the original grid.
    """

    # write tiles
    datasets = []
    start_x = extent[0]
    pbar = tqdm(total=x_segments * y_segments)
    for tx in range(x_segments):
        if (tx + 1) == x_segments:
            end_x = extent[1]
        else:
            end_x = start_x + maxsize * res
        start_y = extent[2]
        for ty in range(y_segments):
            if (ty + 1) == y_segments:
                end_y = extent[3]
            else:
                end_y = start_y + maxsize * res
            subextent = [start_x, end_x, start_y, end_y]
            logger.debug(
                f"segment x {tx+1} of {x_segments}, segment y {ty+1} of {y_segments}"
            )

            memfile = _download_ahn(subextent, res, url, identifier, version, fmt, crs)

            datasets.append(memfile)
            start_y = end_y
            pbar.update(1)

        start_x = end_x

    pbar.close()
    memfile = MemoryFile()
    merge.merge([b.open() for b in datasets], dst_path=memfile)

    return memfile


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
        url = (
            "https://geodata.nationaalgeoregister.nl/ahn2/wcs?"
            "request=GetCapabilities&service=WCS"
        )
    elif "ahn3" in identifier:
        url = (
            "https://geodata.nationaalgeoregister.nl/ahn3/wcs?"
            "request=GetCapabilities&service=WCS"
        )
    else:
        ValueError(f"unknown identifier -> {identifier}")

    return url


def get_ahn_within_extent(
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
        url = (
            "https://geodata.nationaalgeoregister.nl/ahn2/wcs?"
            "request=GetCapabilities&service=WCS"
        )
    elif url == "ahn3":
        url = (
            "https://geodata.nationaalgeoregister.nl/ahn3/wcs?"
            "request=GetCapabilities&service=WCS"
        )
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

    # check if ahn is within limits
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]

    # check if size exceeds maxsize
    if (dx / res) > maxsize:
        x_segments = int(np.ceil((dx / res) / maxsize))
    else:
        x_segments = 1

    if (dy / res) > maxsize:
        y_segments = int(np.ceil((dy / res) / maxsize))
    else:
        y_segments = 1

    if (x_segments * y_segments) > 1:
        st = f"""requested ahn raster width or height bigger than {maxsize*res}
            -> splitting extent into {x_segments} * {y_segments} tiles"""
        logger.info(st)
        memfile = split_ahn_extent(
            extent,
            x_segments,
            y_segments,
            maxsize,
            res,
            url,
            identifier,
            version,
            fmt,
            crs,
        )
        da = rioxarray.open_rasterio(memfile.open(), mask_and_scale=True)[0]
    else:
        memfile = _download_ahn(extent, res, url, identifier, version, fmt, crs)
        da = rioxarray.open_rasterio(memfile.open(), mask_and_scale=True)[0]
        # load the data from the memfile otherwise lazy loading of xarray causes problems
        da.load()

    return da


def _download_ahn(extent, res, url, identifier, version, fmt, crs):
    """Download the ahn using a webservice, return a MemoryFile


    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    res : float, optional
        resolution of ahn raster
    url : str
        webservice url.
    identifier : str
        identifier.
    version : str
        version of wcs service, options are '1.0.0' and '2.0.1'.
    fmt : str, optional
        geotif format
    crs : str, optional
        coördinate reference system

    Raises
    ------
    Exception
        wrong version

    Returns
    -------
    memfile : rasterio.io.MemoryFile
        MemoryFile.

    """
    # download file
    logger.debug(
        f"- download ahn between: x ({str(extent[0])}, {str(extent[1])}); "
        f"y ({str(extent[2])}, {str(extent[3])})"
    )
    wcs = WebCoverageService(url, version=version)
    if version == "1.0.0":
        bbox = (extent[0], extent[2], extent[1], extent[3])
        output = wcs.getCoverage(
            identifier=identifier,
            bbox=bbox,
            format=fmt,
            crs=crs,
            resx=res,
            resy=res,
        )
    elif version == "2.0.1":
        # bbox, resx and resy do nothing in version 2.0.1
        subsets = [("x", extent[0], extent[1]), ("y", extent[2], extent[3])]
        output = wcs.getCoverage(
            identifier=[identifier], subsets=subsets, format=fmt, crs=crs
        )
    else:
        raise Exception(f"Version {version} not yet supported")

    memfile = MemoryFile(output.read())
    return memfile


def get_ahn4_tiles(extent=None):
    """Get the tiles (kaartbladen) of AHN4 as a GeoDataFrame with download links"""
    url = "https://services.arcgis.com/nSZVuSZjHpEZZbRo/arcgis/rest/services/Kaartbladen_AHN4/FeatureServer"
    layer = 0
    gdf = arcrest(url, layer, extent).set_index("Name")
    return gdf


def get_ahn4(extent, identifier="AHN4_DTM_5m", as_data_array=True):
    """
    Download AHN4

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
