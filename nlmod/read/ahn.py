# -*- coding: utf-8 -*-
"""Created on Fri Jun 12 15:33:03 2020.

@author: ruben
"""

import logging
import os
import tempfile

import numpy as np
import rasterio
import xarray as xr
from owslib.wcs import WebCoverageService
from rasterio import merge

from .. import mdims, cache, util

logger = logging.getLogger(__name__)


@cache.cache_netcdf
def get_ahn_at_grid(model_ds, identifier='ahn3_5m_dtm', gridprops=None):
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
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'

    Returns
    -------
    model_ds_out : xr.Dataset
        dataset with the ahn variable.
    """
    if (model_ds.gridtype == 'unstructured') and (gridprops is None):
        raise ValueError(
            'gridprops should be specified when gridtype is unstructured')

    cachedir = os.path.join(model_ds.model_ws, 'cache')

    fname_ahn = get_ahn_within_extent(extent=model_ds.extent,
                                      identifier=identifier,
                                      cache=True,
                                      cache_dir=cachedir)

    ahn_ds_raw = xr.open_rasterio(fname_ahn)
    ahn_ds_raw = ahn_ds_raw.rename({'band': 'layer'})
    nodata = ahn_ds_raw.attrs['nodatavals'][0]
    ahn_ds_raw = ahn_ds_raw.where(ahn_ds_raw != nodata)

    if model_ds.gridtype == 'structured':
        ahn_ds = mdims.resample_dataarray_to_structured_grid(ahn_ds_raw,
                                                             extent=model_ds.extent,
                                                             delr=model_ds.delr,
                                                             delc=model_ds.delc,
                                                             xmid=model_ds.x.data,
                                                             ymid=model_ds.y.data)
    elif model_ds.gridtype == 'unstructured':
        xyi, cid = mdims.get_xyi_cid(gridprops)
        ahn_ds = mdims.resample_dataarray3d_to_unstructured_grid(ahn_ds_raw,
                                                                 gridprops,
                                                                 xyi, cid)

    model_ds_out = util.get_model_ds_empty(model_ds)
    model_ds_out['ahn'] = ahn_ds[0]

    return model_ds_out


def split_ahn_extent(extent, res, x_segments, y_segments, maxsize,
                     fname=None, **kwargs):
    """There is a max height and width limit of 800 * res for the wcs server.
    This function splits your extent in chunks smaller than the limit. It
    returns a list of gdal Datasets.

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
    fname : str, optional
        path name of the ahn tif output file
    **kwargs :
        keyword arguments of the get_ahn_extent function.

    Returns
    -------
    fname : str, optional
        path name of the ahn tif output file

    Notes
    -----
    1. The resolution is used to obtain the ahn from the wcs server. Not sure
    what kind of interpolation is used to resample the original grid.
    """

    # write tiles
    dataset = []
    start_x = extent[0]
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
            logger.info(f'downloading subextent {subextent}')
            logger.info(f'x_segment-{tx}, y_segment-{ty}')

            fname_chunk = get_ahn_within_extent(subextent, res=res,
                                                **kwargs)
            dataset.append(rasterio.open(fname_chunk))
            start_y = end_y
        start_x = end_x

    # read tiles and merge
    dest, output_transform = merge.merge(dataset)

    # write merged raster
    out_meta = dataset[0].meta.copy()
    out_meta.update({'height': dest.shape[1],
                     'width': dest.shape[2],
                     'transform': output_transform})
    if fname is None:
        fname = 'ahn_{:.0f}_{:.0f}_{:.0f}_{:.0f}_{:.0f}.tiff'
        fname = fname.format(*extent, res)
        fname = os.path.join(os.path.split(fname_chunk)[0], fname)
    with rasterio.open(fname, "w", **out_meta) as dest1:
        dest1.write(dest)

    return fname


def get_ahn_within_extent(extent=None, identifier='ahn3_5m_dtm', url=None,
                          res=None, version='1.0.0', format='GEOTIFF_FLOAT32',
                          crs='EPSG:28992', cache=True, cache_dir=None,
                          maxsize=800, fname=None):
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
    format : str, optional
        geotif format . The default is 'GEOTIFF_FLOAT32'.
    crs : str, optional
        coördinate reference system. The default is 'EPSG:28992'.
    cache : boolean, optional
        used cached data if available. The default is True.
    cache_dir : str or None, optional

    maxsize : float, optional
        maximum number of cells in x or y direction. The default is
        800.

    Returns
    -------
    fname : str
        file of the geotiff

    """

    if isinstance(extent, xr.DataArray):
        extent = tuple(extent.values)

    # check or infer url
    if url is None:
        # infer url from identifier
        if 'ahn2' in identifier:
            url = ('https://geodata.nationaalgeoregister.nl/ahn2/wcs?'
                   'request=GetCapabilities&service=WCS')
        elif 'ahn3' in identifier:
            url = ('https://geodata.nationaalgeoregister.nl/ahn3/wcs?'
                   'request=GetCapabilities&service=WCS')
        else:
            ValueError(f'unknown identifier -> {identifier}')
    elif url == 'ahn2':
        url = ('https://geodata.nationaalgeoregister.nl/ahn2/wcs?'
               'request=GetCapabilities&service=WCS')
    elif url == 'ahn3':
        url = ('https://geodata.nationaalgeoregister.nl/ahn3/wcs?'
               'request=GetCapabilities&service=WCS')
    else:
        raise ValueError(f'unknown url -> {url}')

    # check resolution
    if res is None:
        if '05m' in identifier.split('_')[1]:
            res = 0.5
        elif '5m' in identifier.split('_')[1]:
            res = 5.0
        else:
            raise ValueError('could not infer resolution from identifier')

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
        st = f'''requested ahn raster width or height bigger than {maxsize*res}
            -> splitting extent into {x_segments} * {y_segments} tiles'''
        logger.info(st)
        return split_ahn_extent(extent, res, x_segments, y_segments, maxsize,
                                identifier=identifier,
                                version=version, format=format, crs=crs,
                                cache=cache, cache_dir=cache_dir,
                                fname=fname)

    # get filename
    if fname is None:
        fname = 'ahn_{:.0f}_{:.0f}_{:.0f}_{:.0f}_{:.0f}.tiff'

        fname = fname.format(*extent, res * 1000)
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'ahn', identifier)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        fname = os.path.join(cache_dir, fname)
    else:
        cache = False

    if not cache or not os.path.exists(fname):
        # download file
        wcs = WebCoverageService(url, version=version)
        if version == '1.0.0':
            bbox = (extent[0], extent[2], extent[1], extent[3])
            output = wcs.getCoverage(identifier=identifier, bbox=bbox,
                                     format=format, crs=crs, resx=res,
                                     resy=res)
        elif version == '2.0.1':
            # bbox, resx and resy do nothing in version 2.0.1
            subsets = [('x', extent[0], extent[1]),
                       ('y', extent[2], extent[3])]
            output = wcs.getCoverage(identifier=[identifier], subsets=subsets,
                                     format=format, crs=crs)
        else:
            raise (Exception('Version {} not yet supported'.format(version)))
        # write file to disk
        f = open(fname, 'wb')
        f.write(output.read())
        f.close()
        logger.info(f"- download {fname}")
    else:
        logger.info(f"- from cache {fname}")

    return fname
