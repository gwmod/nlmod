# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:33:03 2020

@author: ruben
"""

import os
import tempfile

import gdal
import numpy as np
import rasterio
from rasterio import merge
import xarray as xr
from owslib.wcs import WebCoverageService

from .mgrid import (get_xyi_cid, resample_dataarray_to_structured_grid,
                    resample_dataarray_to_unstructured_grid)
from .util import get_cache_netcdf, get_model_ds_empty


def get_ahn_dataset(model_ds, gridprops=None, use_cache=True,
                    cachedir=None,
                    fname_netcdf='ahn_model_ds.nc', verbose=False):
    """ get an xarray dataset from the ahn values within an extent.

    Parameters
    ----------
    model_ds : xr.Dataset
        dataset with the model information.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'
    use_cache : bool, optional
        if True the cached resampled regis dataset is used.
        The default is False.

    Returns
    -------
    ahn_ds : xarray.Dataset
        dataset with ahn data (not yet corresponding to the modelgrid)

    Notes
    -----

    1. The ahn raster is now cached in a tempdir. Should be changed to the
    cachedir of the model I think.

    """
    ahn_ds = get_cache_netcdf(use_cache, cachedir, fname_netcdf,
                              get_ahn_at_grid, model_ds, check_time=False,
                              verbose=verbose, gridprops=gridprops)

    return ahn_ds


def get_ahn_at_grid(model_ds, gridprops=None):
    """ Get a model dataset with ahn variable.


    Parameters
    ----------
    model_ds : xr.Dataset
        dataset with the model information.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'

    Returns
    -------
    model_ds_out : xr.Dataset
        dataset with the ahn variable.

    """
    
    if model_ds.gridtype=='structured':
        resolution = min(model_ds.delr, model_ds.delc)
    elif model_ds.gridtype=='unstructured':
        resolution = min(model_ds.delr, model_ds.delc)/model_ds.levels
    
    
    fname_ahn = get_ahn_within_extent(extent=model_ds.extent, 
                                      res=resolution,
                                      return_fname=True,
                                      cache=True)

    ahn_ds_raw = xr.open_rasterio(fname_ahn)
    ahn_ds_raw = ahn_ds_raw.rename({'band': 'layer'})
    nodata = ahn_ds_raw.attrs['nodatavals'][0]
    ahn_ds_raw = ahn_ds_raw.where(ahn_ds_raw != nodata)

    if model_ds.gridtype == 'structured':
        ymid = model_ds.y.data[::-1]
        ahn_ds = resample_dataarray_to_structured_grid(ahn_ds_raw,
                                                       extent=model_ds.extent,
                                                       delr=model_ds.delr,
                                                       delc=model_ds.delc,
                                                       xmid=model_ds.x.data,
                                                       ymid=ymid)
    elif model_ds.gridtype == 'unstructured':
        xyi, cid = get_xyi_cid(gridprops)
        ahn_ds = resample_dataarray_to_unstructured_grid(ahn_ds_raw,
                                                         gridprops,
                                                         xyi, cid)

    model_ds_out = get_model_ds_empty(model_ds)
    model_ds_out['ahn'] = ahn_ds[0]

    return model_ds_out


def split_ahn_extent(extent, res, x_segments, y_segments, maxsize,
                     return_fname=False, fname=None, **kwargs):
    """There is a max height and width limit of 4000 meters for the wcs server.
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
    **kwargs :
        keyword arguments of the get_ahn_extent function.

    Returns
    -------
    ds : osgeo.gdal.dataset
        merged ahn dataset
        
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
            fname_chunk = get_ahn_within_extent(subextent, res=res,
                                                return_fname=True, **kwargs)
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

    if return_fname:
        return fname

    # load new tif
    ds = load_ahn_tif(fname)

    return ds


def get_ahn_within_extent(extent=None, url='ahn3', identifier='ahn3_5m_dtm',
                          res=5., version='1.0.0', format='GEOTIFF_FLOAT32',
                          crs='EPSG:28992', cache=True, cache_dir=None,
                          return_fname=False, maxsize=4000,
                          verbose=True, fname=None):
    """

    Parameters
    ----------
    extent : list, tuple or np.array, optional
        extent. The default is None.
    url : str, optional
        possible values 'ahn3' and 'ahn2'. The default is 'ahn3'.
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
    res : float, optional
        resolution of requested ahn raster. The default is 5..
    version : str, optional
        version of wcs service, options are '1.0.0' and '2.0.1'.
        The default is '1.0.0'.
    format : str, optional
        geotif format . The default is 'GEOTIFF_FLOAT32'.
    crs : str, optional
        coördinate reference system. The default is 'EPSG:28992'.
    cache : boolean, optional
        used cached data if available. The default is True.
    return_fname : boolean, optional
        return path instead of gdal dataset. The default is False.
    maxsize : float, optional
        max width and height of the result of the wcs service. The default is
        4000.
    verbose : boolean, optional
        additional information is printed to the terminal. The default is True.

    Returns
    -------
    osgeo.gdal.dataset or str
        gdal dataset or filename if return_fname is True

    """

    if extent is None:
        extent = [253000, 265000, 481000, 488000]
    if url == 'ahn3':
        url = ('https://geodata.nationaalgeoregister.nl/ahn3/wcs?'
               'request=GetCapabilities&service=WCS')
    elif url == 'ahn2':
        url = ('https://geodata.nationaalgeoregister.nl/ahn2/wcs?'
               'request=GetCapabilities&service=WCS')

    # check if ahn is within limits
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]

    if dx > maxsize:
        x_segments = int(np.ceil((dx / res) / maxsize))
    else:
        x_segments = 1

    if dy > maxsize:
        y_segments = int(np.ceil((dy / res) / maxsize))
    else:
        y_segments = 1

    if (x_segments * y_segments) > 1:
        if verbose:
            st = f'''requested ahn raster width or height bigger than {maxsize}
            -> splitting extent into {x_segments * y_segments} tiles'''
            print(st)
        return split_ahn_extent(extent, res, x_segments, y_segments, maxsize,
                                url=url, identifier=identifier,
                                version=version, format=format, crs=crs,
                                cache=cache, cache_dir=cache_dir,
                                return_fname=return_fname, fname=fname)
    if fname is None:
        fname = 'ahn_{:.0f}_{:.0f}_{:.0f}_{:.0f}_{:.0f}.tiff'
        fname = fname.format(*extent, res)
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'ahn', identifier)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        fname = os.path.join(cache_dir, fname)
    else:
        cache = False
    if not cache or not os.path.exists(fname):
        # url='https://geodata.nationaalgeoregister.nl/ahn3/wcs?request=GetCapabilities'
        # identifier='ahn3:ahn3_5m_dsm'

        wcs = WebCoverageService(url, version=version)
        # wcs.contents
        # cvg = wcs.contents[identifier]
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
        f = open(fname, 'wb')
        f.write(output.read())
        f.close()
        if verbose:
            print(f"- downloaded {fname}")
    else:
        if verbose:
            print(f"- from cache {fname}")

    if return_fname:
        return fname
    else:
        # load ahn
        ds = load_ahn_tif(fname)

        return ds


def load_ahn_tif(filename):
    """check if a rasterband can be obtained and set nodata value for data
    smaller than -100.

    Parameters
    ----------
    ds : osgeo.gdal.dataset


    Returns
    -------
    ds : osgeo.gdal.dataset
        set nodata value for data smaller than -100.
    """
    ds = gdal.Open(filename)

    try:
        band = ds.GetRasterBand(1)
    except AttributeError as e:
        print(e)
        raise (Exception("there's probably something wrong with your ahn "
                         "download. Please check if your area is within the "
                         "Netherlands"))

    if band.GetNoDataValue() is None:
        # change NoDataValue, as this is incorrect
        H = ds.ReadAsArray()
        Hmin = float(H.min())
        if Hmin < -100:
            band.SetNoDataValue(Hmin)
        else:
            band.SetNoDataValue(0.0)
    return ds
