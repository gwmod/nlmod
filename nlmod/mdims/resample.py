# -*- coding: utf-8 -*-
"""Created on Fri Apr  2 15:08:50 2021.

@author: oebbe
"""
import logging

import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.interpolate import griddata

from .. import util
from . import mgrid

logger = logging.getLogger(__name__)


def resample_dataarray2d_to_unstructured_grid(da_in, gridprops=None,
                                              xyi=None, cid=None,
                                              method='nearest',
                                              **kwargs):
    """resample a 2d dataarray (xarray) from a structured grid to a new
    dataaraay of an unstructured grid.

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions (y, x). y and x are from the original
        grid
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen.
    xyi : numpy.ndarray, optional
        array with x and y coördinates of cell centers, shape(len(cid), 2). If
        xyi is None xyi is calculated from the gridproperties.
    cid : list or numpy.ndarray, optional
        list with cellids. If  cid is None cid is calculated from the
        gridproperties.
    method : str, optional
        type of interpolation used to resample. The default is 'nearest'.

    Returns
    -------
    da_out : xarray.DataArray
        data array with dimension (cid).
    """
    if (xyi is None) or (cid is None):
        xyi, cid = mgrid.get_xyi_cid(gridprops)

    # get x and y values of all cells in dataarray
    mg = np.meshgrid(da_in.x.data, da_in.y.data)
    points = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    # regrid
    arr_out = griddata(points, da_in.data.flatten(), xyi, method=method,
                       **kwargs)

    # new dataset
    da_out = xr.DataArray(arr_out, dims=('cid'),
                          coords={'cid': cid})

    return da_out


def resample_dataarray3d_to_unstructured_grid(da_in, gridprops=None,
                                              xyi=None, cid=None,
                                              method='nearest'):
    """resample a dataarray (xarray) from a structured grid to a new dataaraay
    of an unstructured grid.

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions (layer, y, x). y and x are from the original
        grid
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen.
    xyi : numpy.ndarray, optional
        array with x and y coördinates of cell centers, shape(len(cid), 2). If
        xyi is None xyi is calculated from the gridproperties.
    cid : list or numpy.ndarray, optional
        list with cellids. If  cid is None cid is calculated from the
        gridproperties.
    method : str, optional
        type of interpolation used to resample. The default is 'nearest'.

    Returns
    -------
    da_out : xarray.DataArray
        data array with dimensions (layer,cid).
    """
    if (xyi is None) or (cid is None):
        xyi, cid = mgrid.get_xyi_cid(gridprops=gridprops)

    # get x and y values of all cells in dataarray
    mg = np.meshgrid(da_in.x.data, da_in.y.data)
    points = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    layers = da_in.layer.data
    arr_out = np.zeros((len(layers), len(xyi)))
    for i, lay in enumerate(layers):

        ds_lay = da_in.sel(layer=lay)

        # regrid
        arr_out[i] = griddata(
            points, ds_lay.data.flatten(), xyi, method=method)

    # new dataset
    da_out = xr.DataArray(arr_out, dims=('layer', 'cid'),
                          coords={'cid': cid,
                                  'layer': layers})

    return da_out


def resample_dataset_to_unstructured_grid(ds_in, gridprops,
                                          method='nearest'):
    """resample a dataset (xarray) from an structured grid to a new dataset
    from an unstructured grid.

    Parameters
    ----------
    ds_in : xarray.Dataset
        dataset with dimensions (layer, y, x). y and x are from the original
        structured grid
    gridprops : dictionary
        dictionary with grid properties output from gridgen.
    method : str, optional
        type of interpolation used to resample. The default is 'nearest'.

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with dimensions (layer, cid), cid are cell id's from the new
        grid.
    """

    assert isinstance(ds_in, xr.core.dataset.Dataset)

    xyi, cid = mgrid.get_xyi_cid(gridprops)

    ds_out = xr.Dataset(coords={'cid': cid,
                                'layer': ds_in.layer.data})

    # add x and y coordinates
    ds_out['x'] = xr.DataArray(xyi[:, 0], dims=('cid'),
                               coords={'cid': cid})
    ds_out['y'] = xr.DataArray(xyi[:, 0], dims=('cid'),
                               coords={'cid': cid})

    # add other variables
    for data_var in ds_in.data_vars:
        if ds_in[data_var].dims == ('layer', 'y', 'x'):
            data_arr = resample_dataarray3d_to_unstructured_grid(ds_in[data_var],
                                                                 xyi=xyi, cid=cid,
                                                                 method=method)
        elif ds_in[data_var].dims == ('y', 'x'):
            data_arr = resample_dataarray2d_to_unstructured_grid(ds_in[data_var],
                                                                 xyi=xyi, cid=cid,
                                                                 method=method)

        elif ds_in[data_var].dims == ('layer'):
            data_arr = ds_in[data_var]

        else:
            logger.warning(f'did not resample data array {data_var} because conversion with dimensions {ds_in[data_var].dims} is not (yet) supported')
            continue

        ds_out[data_var] = data_arr

    return ds_out


def resample_dataarray_to_structured_grid(da_in, extent=None, 
                                          delr=None, delc=None,
                                          xmid=None, ymid=None,
                                          kind='linear', nan_factor=0.01,
                                          **kwargs):
    """resample a dataarray (xarray) from a structured grid to a new dataaraay
    from a different structured grid.

    Also flips the y-coordinates to make them descending instead of ascending.
    This makes it easier to export array to flopy. In other words, make sure
    that both lines of code create the same plot::

        da_in['top'].sel(layer=b'Hlc').plot()
        plt.imshow(da_in['top'].sel(layer=b'Hlc').data)

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions (layer, y, x). y and x are from the original
        grid
    extent : list, tuple or np.array, optional
        extent (xmin, xmax, ymin, ymax) of the desired grid, if not defined
        xmid and ymid are used
    delr : int or float, optional
        cell size along rows of the desired grid, if not defined xmid and
        ymid are used
    delc : int or float, optional
        cell size along columns of the desired grid, if not defined xmid and
        ymid are used
    xmid : np.array, optional
        x coördinates of the cell centers of the desired grid shape(ncol), if
        not defined xmid and ymid are calculated from the extent, delr and delc.
    ymid : np.array, optional
        y coördinates of the cell centers of the desired grid shape(nrow), if
        not defined xmid and ymid are calculated from the extent, delr and delc.
    kind : str, optional
        type of interpolation used to resample. The default is 'linear'.
    nan_factor : float, optional
        the nan values in the original raster are filled with zeros before
        interpolation because the interp2d function cannot handle nan values
        very well. Therefore an extra interpolation is done to determine how
        much these nan values have influenced the new raster values. If the
        the interpolated value is influenced more than this factor by a nan
        value. The value in the interpolated raster is set to nan.
        See also: https://stackoverflow.com/questions/51474792/2d-interpolation-with-nan-values-in-python

    Returns
    -------
    ds_out : xarray.DataArray
        data array with dimensions (layer, y, x). y and x are from the new
        grid.
    """

    assert isinstance(da_in, xr.core.dataarray.DataArray), f'expected type xr.core.dataarray.DataArray got {type(da_in)} instead'
    
    # check if ymid is in descending order
    assert np.array_equal(ymid,np.sort(ymid)[::-1]), 'ymid should be in descending order'

    if xmid is None:
        xmid, ymid = mgrid.get_xy_mid_structured(extent, delr, delc)

    layers = da_in.layer.data
    arr_out = np.zeros((len(layers), len(ymid), len(xmid)))
    for i, lay in enumerate(layers):

        ds_lay = da_in.sel(layer=lay)
        # check for nan values
        if (ds_lay.isnull().sum() > 0) and (kind == "linear"):
            arr_out[i] = resample_2d_struc_da_nan_linear(ds_lay, xmid, ymid,
                                                         nan_factor, **kwargs)
        # faster for linear
        elif kind == "linear" or kind=='cubic':
            # no need to fill nan values
            f = interpolate.interp2d(ds_lay.x.data, ds_lay.y.data,
                                     ds_lay.data, kind='linear', **kwargs)
            # for some reason interp2d flips the y-values
            arr_out[i] = f(xmid, ymid)[::-1]
        elif kind == 'nearest':
            xydata = np.vstack([v.ravel() for v in
                                np.meshgrid(ds_lay.x.data, ds_lay.y.data)]).T
            xyi = np.vstack([v.ravel() for v in np.meshgrid(xmid, ymid)]).T
            fi = griddata(xydata, ds_lay.data.ravel(), xyi, method=kind,
                          **kwargs)
            arr_out[i] = fi.reshape(ymid.shape[0], xmid.shape[0])
        else:
            raise ValueError(f'unexpected value for "kind": {kind}')

    # new dataset
    da_out = xr.DataArray(arr_out, dims=('layer', 'y', 'x'),
                          coords={'x': xmid,
                                  'y': ymid,
                                  'layer': layers})

    return da_out


def resample_2d_struc_da_nan_linear(da_in, new_x, new_y,
                                    nan_factor=0.01, **kwargs):
    """resample a structured, 2d data-array with nan values onto a new grid.

    Parameters
    ----------
    da_in : xarray DataArray
        dataset you want to project on a new grid
    new_x : numpy array
        x coördinates of the new grid
    new_y : numpy array
        y coördaintes of the new grid
    nan_factor : float, optional
        the nan values in the original raster are filled with zeros before
        interpolation because the interp2d function cannot handle nan values
        very well. Therefore an extra interpolation is done to determine how
        much these nan values have influenced the new raster values. If the
        the interpolated value is influenced more than this factor by a nan
        value. The value in the interpolated raster is set to nan.
        See also: https://stackoverflow.com/questions/51474792/2d-interpolation-with-nan-values-in-python

    Returns
    -------
    arr_out : numpy array
        resampled array
    """
    nan_map = np.where(da_in.isnull().data, 1, 0)
    fill_map = np.where(da_in.isnull().data, 0, da_in.data)
    f = interpolate.interp2d(da_in.x.data, da_in.y.data,
                             fill_map, kind='linear', **kwargs)
    f_nan = interpolate.interp2d(da_in.x.data, da_in.y.data,
                                 nan_map, kind='linear')
    arr_out_raw = f(new_x, new_y)
    nan_new = f_nan(new_x, new_y)
    arr_out_raw[nan_new > nan_factor] = np.nan
    
    # for some reason interp2d flips the y-values
    arr_out = arr_out_raw[::-1]

    return arr_out


def resample_dataset_to_structured_grid(ds_in, extent, delr, delc, kind='linear'):
    """Resample a dataset (xarray) from a structured grid to a new dataset from
    a different structured grid.

    Parameters
    ----------
    ds_in : xarray.Dataset
        dataset with dimensions (layer, y, x). y and x are from the original
        grid
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    delr : int or float
        cell size along rows of the desired grid (dx).
    delc : int or float
        cell size along columns of the desired grid (dy).
    kind : str, optional
        type of interpolation used to resample. The default is 'linear'.

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with dimensions (layer, y, x). y and x are from the new
        grid.
    """

    assert isinstance(ds_in, xr.core.dataset.Dataset)
    

    xmid, ymid = mgrid.get_xy_mid_structured(extent, delr, delc)

    ds_out = xr.Dataset(coords={'y': ymid,
                                'x': xmid,
                                'layer': ds_in.layer.data})
    for data_var in ds_in.data_vars:
        data_arr = resample_dataarray_to_structured_grid(ds_in[data_var],
                                                         xmid=xmid,
                                                         ymid=ymid,
                                                         kind=kind)
        ds_out[data_var] = data_arr

    return ds_out


def get_resampled_ml_layer_ds_unstruc(raw_ds=None,
                                      extent=None,
                                      gridprops=None):
    """Project model layer dataset on an unstructured model grid.

    Parameters
    ----------
    raw_ds : xr.Dataset, optional
        raw model layer dataset. The default is None.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Used as the
        definition of the unstructured grid.

    Returns
    -------
    ml_layer_ds : xr.dataset
        model layer dataset projected onto the modelgrid.
    """

    logger.info('resample model layer data to unstructured modelgrid')
    ml_layer_ds = resample_dataset_to_unstructured_grid(
        raw_ds, gridprops)
    ml_layer_ds['x'] = xr.DataArray([r[1] for r in gridprops['cell2d']],
                                    dims=('cid'),
                                    coords={'cid': ml_layer_ds.cid.data})

    ml_layer_ds['y'] = xr.DataArray([r[2] for r in gridprops['cell2d']],
                                    dims=('cid'),
                                    coords={'cid': ml_layer_ds.cid.data})
    ml_layer_ds.attrs['gridtype'] = 'unstructured'
    ml_layer_ds.attrs['delr'] = raw_ds.delr
    ml_layer_ds.attrs['delc'] = raw_ds.delc
    ml_layer_ds.attrs['extent'] = extent

    return ml_layer_ds



def fillnan_dataarray_structured_grid(xar_in):
    """fill not-a-number values in a structured grid, DataArray.

    The fill values are determined using the 'nearest' method of the
    scipy.interpolate.griddata function


    Parameters
    ----------
    xar_in : xarray DataArray
        DataArray with nan values. DataArray should have 2 dimensions
        (y and x).

    Returns
    -------
    xar_out : xarray DataArray
        DataArray without nan values. DataArray has 3 dimensions
        (layer, y and x)

    Notes
    -----
    can be slow if the xar_in is a large raster
    """
    # check dimensions
    if xar_in.dims != ('y', 'x'):
        raise ValueError(
            f"expected dataarray with dimensions ('y' and 'x'), got dimensions -> {xar_in.dims}")

    # get list of coordinates from all points in raster
    mg = np.meshgrid(xar_in.x.data, xar_in.y.data)
    points_all = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    # fill nan values in bathymetry
    values_all = xar_in.data.flatten()

    # get 1d arrays with only values where bathymetry is not nan
    mask1 = ~np.isnan(values_all)
    points_in = points_all[np.where(mask1)[0]]
    values_in = values_all[np.where(mask1)[0]]

    # get nearest value for all nan values
    values_out = griddata(points_in, values_in, points_all, method='nearest')
    arr_out = values_out.reshape(xar_in.shape)

    # bathymetry without nan values
    xar_out = xr.DataArray([arr_out], dims=('layer', 'y', 'x'),
                           coords={'x': xar_in.x.data,
                                   'y': xar_in.y.data,
                                   'layer': [0]})

    return xar_out


def fillnan_dataarray_unstructured_grid(xar_in, gridprops=None,
                                        xyi=None, cid=None):
    """can be slow if the xar_in is a large raster.

    Parameters
    ----------
    xar_in : xr.DataArray
        data array with nan values. Shape is (cid)
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen.
    xyi : numpy.ndarray
        array with x and y coördinates of cell centers, shape(len(cid), 2).
    cid : list
        list with cellids.

    Returns
    -------
    xar_out : xr.DataArray
        data array with nan values. Shape is (cid)
    """

    # get list of coordinates from all points in raster
    if (xyi is None) or (cid is None):
        xyi, cid = mgrid.get_xyi_cid(gridprops)

    # fill nan values in bathymetry
    values_all = xar_in.data

    # get 1d arrays with only values where bathymetry is not nan
    mask1 = ~np.isnan(values_all)
    xyi_in = xyi[mask1]
    values_in = values_all[mask1]

    # get nearest value for all nan values
    values_out = griddata(xyi_in, values_in, xyi, method='nearest')

    # bathymetry without nan values
    xar_out = xr.DataArray([values_out], dims=('layer', 'cid'),
                           coords={'cid': xar_in.cid.data,
                                   'layer': [0]})

    return xar_out


def resample_unstr_2d_da_to_struc_2d_da(da_in, model_ds=None,
                                        xmid=None, ymid=None,
                                        cellsize=25,
                                        method='nearest'):
    """resample a 2d dataarray (xarray) from an unstructured grid to a new 
    dataaraay from a structured grid.   

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions ('cid'). 
    model_ds : xarray.DataArray
        model dataset with 'x' and 'y' data variables.
    cellsize : int or float, optional
        required cell size of structured grid. The default is 25.
    method : str, optional
        method used for resampling. The default is 'nearest'.

    Returns
    -------
    da_out : xarray.DataArray
        data array with dimensions ('y', 'x').

    """
    if xmid is None or ymid is None:
        xmid = model_ds.x.values
        ymid = model_ds.y.values
    
    points_unstr = np.array([xmid, ymid]).T
    modelgrid_x = np.arange(xmid.min(),
                            xmid.max(),
                            cellsize)
    modelgrid_y = np.arange(ymid.max(),
                            ymid.min()-cellsize,
                            -cellsize)
    mg = np.meshgrid(modelgrid_x, modelgrid_y)
    points = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    arr_out_1d = griddata(points_unstr, da_in.values, points, method=method)
    arr_out2d = arr_out_1d.reshape(len(modelgrid_y),
                                   len(modelgrid_x))

    da_out = xr.DataArray(arr_out2d,
                          dims=('y', 'x'),
                          coords={'y': modelgrid_y,
                                  'x': modelgrid_x})

    return da_out