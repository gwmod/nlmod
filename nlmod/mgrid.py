# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:41:57 2020

@author: oebbe
"""
import copy
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from scipy.interpolate import griddata
from shapely.prepared import prep
import rasterio
from rasterio.warp import reproject

from . import util, regis, geotop
import flopy
from flopy.utils.gridintersect import GridIntersect
from flopy.utils.gridgen import Gridgen


def update_model_ds_from_ml_layer_ds(model_ds, ml_layer_ds, keep_vars=None,
                                     gridtype='structured',
                                     anisotropy=10,
                                     fill_value_kh=1.,
                                     fill_value_kv=0.1,
                                     verbose=False):
    """ Update a model dataset with a model layer dataset.


    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data, preferably without a grid definition.
    ml_layer_ds : xarray.Dataset
        dataset with model layer data corresponding to the modelgrid
    keep_vars : list of str
        variables in ml_layer_ds that will be used in model_ds
    gridtype : str, optional
        type of grid, options are 'structured' and 'unstructured'. 
        The default is 'structured'.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data 
    """

    if keep_vars is None:
        raise ValueError(
            'please select variables that will be used to update model_ds')
    else:
        # update variables
        model_ds.update(ml_layer_ds[keep_vars])
        # update attributes
        _ = [model_ds.attrs.update({key: item})
             for key, item in ml_layer_ds.attrs.items()]
    
    model_ds = add_idomain_from_bottom_to_dataset(ml_layer_ds['bottom'],
                                                  model_ds)

    model_ds = add_top_bot_to_model_ds(ml_layer_ds, model_ds,
                                       gridtype=gridtype)
    
    model_ds = add_kh_kv_from_ml_layer_to_dataset(ml_layer_ds,
                                                model_ds,
                                                anisotropy,
                                                fill_value_kh,
                                                fill_value_kv,
                                                verbose=verbose)
    
    

    return model_ds


def get_model_ds_from_regis_ds(regis_ds, keep_vars=None, verbose=False):
    """ Get a model dataset with the same coordinates and dimensions as the
    input regis dataset.


    Parameters
    ----------
    regis_ds : xarray.Dataset
        dataset with regis data corresponding to the modelgrid
    keep_vars : list of str
        variables in regis_ds that will be used in model_ds

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data 
    """

    if keep_vars is None:
        # lelijke manier om een dataset te maken van een andere dataset zonder
        # de variabelen maar wel de dimensies en coordinates
        key = list(regis_ds.keys())[0]
        model_ds = regis_ds[[key]].copy()
        model_ds = model_ds.drop(key)
    else:
        model_ds = regis_ds[keep_vars]

    return model_ds


def get_first_active_layer_from_idomain(idomain, nodata=-999, verbose=False):
    """ get the first active layer in each cell from the idomain

    Parameters
    ----------
    idomain : xr.DataArray
        idomain. Shape can be (layer, y, x) or (layer, cid)
    nodata : int, optional
        nodata value. used for cells that are inactive in all layers. 
        The default is -999.

    Returns
    -------
    first_active_layer : xr.DataArray
        raster in which each cell has the zero based number of the first
        active layer. Shape can be (y, x) or (cid)
    """
    if verbose:
        print('get first active modellayer for each cell in idomain')

    first_active_layer = xr.where(idomain[0] == 1, 0, nodata)
    for i in range(1, idomain.shape[0]):
        first_active_layer = xr.where((first_active_layer == nodata) & (idomain[i] == 1),
                                      i,
                                      first_active_layer)

    return first_active_layer


def add_idomain_from_bottom_to_dataset(bottom, model_ds, nodata=-999,
                                       verbose=False):
    """ add idomain and first_active_layer to model_ds
    The active layers are defined as the layers where the bottom is not nan

    Parameters
    ----------
    bottom : xarray.DataArray
        DataArray with bottom values of each layer. Nan values indicate 
        inactive cells.
    model_ds : xarray.Dataset
        dataset with model data where idomain and first_active_layer
        are added to.
    nodata : int, optional
        nodata value used in integer arrays. For float arrays np.nan is use as
        nodata value. The default is -999.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including idomain and first_active_layer
    """
    if verbose:
        print('get active cells (idomain) from bottom DataArray')

    idomain = xr.where(bottom.isnull(), -1, 1)

    # if the top cell is inactive idomain is 0, otherwise it is -1
    idomain[0] = xr.where(idomain[0] == -1, 0, idomain[0])
    for i in range(1, bottom.shape[0]):
        idomain[i] = xr.where((idomain[i - 1] == 0) &
                              (idomain[i] == -1), 0, idomain[i])

    model_ds['idomain'] = idomain
    model_ds['first_active_layer'] = get_first_active_layer_from_idomain(idomain,
                                                                         nodata=nodata,
                                                                         verbose=verbose)

    model_ds.attrs['nodata'] = nodata

    return model_ds


def get_lay_from_ml_layers(raw_layer_mod, verbose=False):
    """ get number and name of layers based on the number of non-nan layers

    Parameters
    ----------
    raw_layer_mod : xarray.Dataset
        dataset with raw layer model from regis or geotop.

    Returns
    -------
    nlay : int
        number of active layers within regis_ds_raw.
    lay_sel : list of str
        names of the active layers.
    """
    if verbose:
        print('find active layers in raw layer model')

    bot_raw_all = raw_layer_mod['bottom']
    lay_sel = []
    for lay in bot_raw_all.layer.data:
        if not bot_raw_all.sel(layer=lay).isnull().all():
            lay_sel.append(lay)
    nlay = len(lay_sel)

    if verbose:
        print(f'there are {nlay} active regis layers within the extent')

    return nlay, lay_sel


def fit_extent_to_regis(extent, delr, delc, cs_regis=100.,
                        verbose=False):
    """
    redifine extent and calculate the number of rows and columns.

    The extent will be redefined so that the borders os the grid (xmin, xmax, 
    ymin, ymax) correspond with the borders of the regis grid.

    Parameters
    ----------
    extent : list, tuple or np.array
        original extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    cs_regis : int or float, optional
        cell size of regis grid. The default is 100..

    Returns
    -------
    extent : list, tuple or np.array
        adjusted extent
    nrow : int
        number of rows.
    ncol : int
        number of columns.

    """
    if verbose:
        print(f'redefining current extent: {extent}, fit to regis raster')

    for d in [delr, delc]:
        if float(d) not in [10., 20., 25., 50., 100., 200., 400., 500., 800.]:
            print(f'you probably cannot run the model with this '
                  f'cellsize -> {delc, delr}')

    # if extents ends with 50 do nothing, otherwise rescale extent to fit regis
    if extent[0] % cs_regis == 0 or not extent[0] % (0.5 * cs_regis) == 0:
        extent[0] -= extent[0] % 100
        extent[0] = extent[0] - 0.5 * cs_regis
    # get number of columns
    ncol = int(np.ceil((extent[1] - extent[0]) / delr))
    extent[1] = extent[0] + (ncol * delr)  # round x1 up to close grid

    # round y0 down to next 50 necessary for regis
    if extent[2] % cs_regis == 0 or not extent[2] % (0.5 * cs_regis) == 0:
        extent[2] -= extent[2] % 100
        extent[2] = extent[2] - 0.5 * cs_regis
    nrow = int(np.ceil((extent[3] - extent[2]) / delc))  # get number of rows
    extent[3] = extent[2] + (nrow * delc)  # round y1 up to close grid

    if verbose:
        print(
            f'new extent is {extent} model has {nrow} rows and {ncol} columns')

    return extent, nrow, ncol


def get_xy_mid_structured(extent, delr, delc):
    """
    calculates the x and y coordinates of the cell centers of a structured
    grid.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy

    Returns
    -------
    xmid : np.array
        x coördinates of the cell centers shape(ncol)
    ymid : np.array
        y coördinates of the cell centers shape(nrow)

    """
    # get cell mids
    x_mid_start = extent[0] + 0.5 * delr
    x_mid_end = extent[1] - 0.5 * delr
    y_mid_start = extent[2] + 0.5 * delc
    y_mid_end = extent[3] - 0.5 * delc

    ncol = int((extent[1] - extent[0]) / delr)
    nrow = int((extent[3] - extent[2]) / delc)

    xmid = np.linspace(x_mid_start, x_mid_end, ncol)
    ymid = np.linspace(y_mid_start, y_mid_end, nrow)

    return xmid, ymid


def fillnan_dataarray_structured_grid(xar_in):
    """ can be slow if the xar_in is a large raster

    Parameters
    ----------
    xar_in : TYPE
        DESCRIPTION.

    Returns
    -------
    xar_out : TYPE
        DESCRIPTION.

    """

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
    """ can be slow if the xar_in is a large raster

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
        xyi, cid = get_xyi_cid(gridprops)

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


def resample_dataarray_to_structured_grid(da_in, extent=None, delr=None, delc=None,
                                          xmid=None, ymid=None,
                                          kind='linear', nan_factor=0.01):
    """ resample a dataarray (xarray) from a structured grid to a new dataaraay 
    from a different structured grid.

    Also flips the y-coordinates to make them descending instead of ascending.
    This makes it easier to export array to flopy. In other words, make sure 
    that both lines of code create the same plot:
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

    Raises
    ------
    NotImplementedError
        Not many interpolation methods are available (yet).

    Returns
    -------
    ds_out : xarray.DataArray
        data array with dimensions (layer, y, x). y and x are from the new
        grid.

    """

    assert isinstance(da_in, xr.core.dataarray.DataArray)

    if xmid is None:
        xmid, ymid = get_xy_mid_structured(extent, delr, delc)

    layers = da_in.layer.data
    arr_out = np.zeros((len(layers), len(ymid), len(xmid)))
    for i, lay in enumerate(layers):

        ds_lay = da_in.sel(layer=lay)
        # check for nan values
        if (ds_lay.isnull().sum() > 0) and (kind == "linear"):
            arr_out[i] = resample_2d_struc_da_nan_linear(ds_lay, xmid, ymid,
                                                         nan_factor)
        elif kind == "linear":
            # no need to fill nan values
            f = interpolate.interp2d(ds_lay.x.data, ds_lay.y.data,
                                     ds_lay.data, kind='linear')
            arr_out[i] = f(xmid, ymid)[::-1]
        elif kind == 'nearest':
            xydata = np.vstack([v.ravel() for v in
                                np.meshgrid(ds_lay.x.data, ds_lay.y.data)]).T
            xyi = np.vstack([v.ravel() for v in np.meshgrid(xmid, ymid)]).T
            fi = griddata(xydata, ds_lay.data.ravel(), xyi, method=kind)
            arr_out[i] = fi.reshape(ymid.shape[0], xmid.shape[0])[::-1]
        else:
            raise ValueError(f'unexpected value for "kind": {kind}')

    # new dataset
    da_out = xr.DataArray(arr_out, dims=('layer', 'y', 'x'),
                          coords={'x': xmid,
                                  'y': ymid[::-1],
                                  'layer': layers})

    return da_out


def resample_2d_struc_da_nan_linear(da_in, new_x, new_y,
                                    nan_factor=0.01):
    """ resample a structured, 2d data-array with nan values onto a new grid

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
                             fill_map, kind='linear')
    f_nan = interpolate.interp2d(da_in.x.data, da_in.y.data,
                                 nan_map, kind='linear')
    arr_out_raw = f(new_x, new_y)
    nan_new = f_nan(new_x, new_y)
    arr_out_raw[nan_new > nan_factor] = np.nan
    arr_out = arr_out_raw[::-1]

    return arr_out


def resample_dataset_to_structured_grid(ds_in, extent, delr, delc, kind='linear'):
    """ resample a dataset (xarray) from a structured grid to a new dataset 
    from a different structured grid.

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

    Raises
    ------
    NotImplementedError
        Not many interpolation methods are available (yet).

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with dimensions (layer, y, x). y and x are from the new
        grid.
    """

    assert isinstance(ds_in, xr.core.dataset.Dataset)

    xmid, ymid = get_xy_mid_structured(extent, delr, delc)

    ds_out = xr.Dataset(coords={'y': ymid[::-1],
                                'x': xmid,
                                'layer': ds_in.layer.data})
    for data_var in ds_in.data_vars:
        data_arr = resample_dataarray_to_structured_grid(ds_in[data_var],
                                                         xmid=xmid, ymid=ymid,
                                                         kind=kind)
        ds_out[data_var] = data_arr

    return ds_out


def create_unstructured_grid(gridgen_ws, model_name, gwf,
                             shp_fname, level, extent,
                             nlay, nrow, ncol, delr,
                             delc, cachedir=None,
                             use_cache=False,
                             verbose=False):
    """ created unstructured grid. Refine grid using a shapefile and
    refinement levels.     

    Parameters
    ----------
    gridgen_ws : str
        directory to save gridgen files.
    model_name : str
        name of the model.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    shp_fname : str
        path to shapefiles that is used to refine grid.
    level : int
        DESCRIPTION.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    nlay : int
        number of model layers.
    nrow : int
        number of model rows.
    ncol : int
        number of model columns
    delr : int or float
        cell size along rows of the desired grid (dx).
    delc : int or float
        cell size along columns of the desired grid (dy).
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached resampled regis dataset is used. 
        The default is False.

    Returns
    -------
    g : flopy.utils.gridgen.Gridgen
        gridgen object used to generate an unstructured grid.

    """

    if not os.path.isdir(gridgen_ws):
        os.makedirs(gridgen_ws)

    if cachedir is None:
        cachedir = tempfile.gettempdir()

    fname_g_pickle = os.path.join(cachedir, 'grid.pklz')
    if os.path.isfile(fname_g_pickle) and use_cache:
        if verbose:
            print(f'using cached griddata from file {fname_g_pickle}')

        with open(fname_g_pickle, 'rb') as fo:
            g = pickle.load(fo)

        return g

    if verbose:
        print(f'create unstructured grid using gridgen')

    # create temporary groundwaterflow model with dis package
    _gwf_temp = copy.deepcopy(gwf)
    _dis_temp = flopy.mf6.ModflowGwfdis(_gwf_temp, pname='dis',
                                        nlay=nlay, nrow=nrow,
                                        ncol=ncol,
                                        xorigin=extent[0],
                                        yorigin=extent[2],
                                        delr=delr, delc=delc,
                                        filename='{}.dis'.format(model_name))

    exe_name = os.path.join(os.path.dirname(__file__),
                            '..', 'tools', 'gridgen.exe')
    g = Gridgen(_dis_temp, model_ws=gridgen_ws, exe_name=exe_name)

    g.add_refinement_features(shp_fname, 'line', level, range(nlay))
    g.build()

    if verbose:
        print(f'write cache for griddata data to {fname_g_pickle}')

    with open(fname_g_pickle, 'wb') as fo:
        pickle.dump(g, fo)

    return g


def get_xyi_cid(gridprops):
    """ Get x and y coördinates of the cell mids from the cellids in the grid
    properties.


    Parameters
    ----------
    gridprops : dictionary
        dictionary with grid properties output from gridgen.

    Returns
    -------
    xyi : numpy.ndarray
        array with x and y coördinates of cell centers, shape(len(cid), 2).
    cid : list
        list with cellids.
    """

    xc_gwf = [cell2d[1] for cell2d in gridprops['cell2d']]
    yc_gwf = [cell2d[2] for cell2d in gridprops['cell2d']]
    xyi = np.vstack((xc_gwf, yc_gwf)).T
    cid = [c[0] for c in gridprops['cell2d']]

    return xyi, cid


def resample_dataarray_to_unstructured_grid(da_in, gridprops=None,
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

    Raises
    ------
    NotImplementedError
        Not many interpolation methods are available (yet).

    Returns
    -------
    da_out : xarray.DataArray
        data array with dimensions (layer, y, x). y and x are from the new
        grid.

    """
    if (xyi is None) or (cid is None):
        xyi, cid = get_xyi_cid(gridprops)

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
    """ resample a dataset (xarray) from an structured grid to a new dataset 
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

    Raises
    ------
    NotImplementedError
        Not many interpolation methods are available (yet).

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with dimensions (layer, cid), cid are cell id's from the new
        grid.
    """

    assert isinstance(ds_in, xr.core.dataset.Dataset)

    xyi, cid = get_xyi_cid(gridprops)

    ds_out = xr.Dataset(coords={'cid': cid,
                                'layer': ds_in.layer.data})

    # add x and y coordinates
    ds_out['x'] = xr.DataArray(xyi[:, 0], dims=('cid'),
                               coords={'cid': cid})
    ds_out['y'] = xr.DataArray(xyi[:, 0], dims=('cid'),
                               coords={'cid': cid})

    # add other variables
    for data_var in ds_in.data_vars:
        data_arr = resample_dataarray_to_unstructured_grid(ds_in[data_var],
                                                           xyi=xyi, cid=cid,
                                                           method=method)
        ds_out[data_var] = data_arr

    return ds_out


def col_to_list(col_in, model_ds, cellids):
    """ convert array data in model_ds to a list of values for specific cells.
    This function is typically used to create a rec_array with stress period
    data for the modflow packages.

    Can be used for structured and unstructured grids.

    Parameters
    ----------
    col_in : str, int or float
        if col_in is a str type it is the name of the column in model_ds.
        if col_in is an int or a float it is a value that will be used for all
        cells in cellids.
    model_ds : xarray.Dataset
        dataset with model data. Can have dimension (layer, y, x) or 
        (layer, cid).
    cellids : tuple of numpy arrays
        tuple with indices of the cells that will be used to create the list
        with values. There are 3 options:
            1. cellids contains (layers, rows, columns)
            2. cellids contains (rows, columns) or (layers, cids)
            3. cellids contains (cids)

    Raises
    ------
    ValueError
        raised if the cellids are in the wrong format.

    Returns
    -------
    col_lst : list
        raster values from model_ds presented in a list per cell.

    """

    if isinstance(col_in, str):
        if len(cellids) == 3:
            # 3d grid
            col_lst = [model_ds[col_in].data[lay, row, col]
                       for lay, row, col in zip(cellids[0], cellids[1], cellids[2])]
        elif len(cellids) == 2:
            # 2d grid or unstructured 3d grid
            col_lst = [model_ds[col_in].data[row, col]
                       for row, col in zip(cellids[0], cellids[1])]
        elif len(cellids) == 1:
            # 2d unstructured grid
            col_lst = model_ds[col_in].data[cellids[0]]
        else:
            raise ValueError(
                f'could not create a column list for col_in={col_in}')
    else:
        col_lst = [col_in] * len(cellids[0])

    return col_lst


def lrc_to_rec_list(layers, rows, columns, cellids, model_ds,
                    col1=None, col2=None, col3=None):
    """ Create a rec list for stress period data from a set of cellids.

    Used for structured grids.


    Parameters
    ----------
    layers : list or numpy.ndarray
        list with the layer for each cell in the rec_list.
    rows : list or numpy.ndarray
        list with the rows for each cell in the rec_list.
    columns : list or numpy.ndarray
        list with the columns for each cell in the rec_list.
    cellids : tuple of numpy arrays
        tuple with indices of the cells that will be used to create the list
        with values.
    model_ds : xarray.Dataset
        dataset with model data. Can have dimension (layer, y, x) or 
        (layer, cid).
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1, col2) for each row.

        col3 should be the following value for each package (can also be the
            name of a timeseries):

    Raises
    ------
    ValueError
        Question: will this error ever occur?.

    Returns
    -------
    rec_list : list of tuples
        every row consist of ((layer,row,column), col1, col2, col3).

    """

    if col1 is None:
        rec_list = list(zip(zip(layers, rows, columns)))
    elif (col1 is not None) and col2 is None:
        col1_lst = col_to_list(col1, model_ds, cellids)
        rec_list = list(zip(zip(layers, rows, columns),
                            col1_lst))
    elif (col2 is not None) and col3 is None:
        col1_lst = col_to_list(col1, model_ds, cellids)
        col2_lst = col_to_list(col2, model_ds, cellids)
        rec_list = list(zip(zip(layers, rows, columns),
                            col1_lst, col2_lst))
    elif (col3 is not None):
        col1_lst = col_to_list(col1, model_ds, cellids)
        col2_lst = col_to_list(col2, model_ds, cellids)
        col3_lst = col_to_list(col3, model_ds, cellids)
        rec_list = list(zip(zip(layers, rows, columns),
                            col1_lst, col2_lst, col3_lst))
    else:
        raise ValueError(
            'invalid combination of values for col1, col2 and col3')

    return rec_list


def data_array_3d_to_rec_list(model_ds, mask,
                              col1=None, col2=None, col3=None,
                              only_active_cells=True):
    """ Create a rec list for stress period data from a model dataset.

    Used for structured grids.


    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data and dimensions (layer, y, x)
    mask : xarray.DataArray for booleans
        True for the cells that will be used in the rec list.
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]
            riv: stage [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]
            riv: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1, col2) for each row.

        col3 should be the following value for each package (can also be the
            name of a timeseries):
            riv: river bottom [L]

    only_active_cells : bool, optional
        If True an extra mask is used to only include cells with an idomain 
        of 1. The default is True.

    Returns
    -------
    rec_list : list of tuples
        every row consist of ((layer,row,column), col1, col2, col3).

    """
    if only_active_cells:
        cellids = np.where((mask) & (model_ds['idomain'] == 1))
    else:
        cellids = np.where(mask)

    layers = cellids[0]
    rows = cellids[1]
    columns = cellids[2]

    rec_list = lrc_to_rec_list(layers, rows, columns, cellids, model_ds,
                               col1, col2, col3)

    return rec_list


def data_array_2d_to_rec_list(model_ds, mask,
                              col1=None, col2=None, col3=None,
                              layer=0,
                              first_active_layer=False,
                              only_active_cells=True):
    """ Create a rec list for stress period data from a model dataset.

    Used for structured grids.


    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data and dimensions (layer, y, x)
    mask : xarray.DataArray for booleans
        True for the cells that will be used in the rec list.
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1, col2) for each row.

        col3 should be the following value for each package (can also be the
            name of a timeseries):
    layer : int, optional
        layer used in the rec_list. Not used if first_active_layer is True.
        default is 0
    first_active_layer : bool, optional
        If True an extra mask is applied to use the first active layer of each 
        cell in the grid. The default is False.
    only_active_cells : bool, optional
        If True an extra mask is used to only include cells with an idomain 
        of 1. The default is True.

    Returns
    -------
    rec_list : list of tuples
        every row consist of ((layer,row,column), col1, col2, col3).
    """

    if first_active_layer:
        cellids = np.where(
            (mask) & (model_ds['first_active_layer'] != model_ds.nodata))
        layers = col_to_list('first_active_layer', model_ds, cellids)
    elif only_active_cells:
        cellids = np.where((mask) & (model_ds['idomain'][layer] == 1))
        layers = col_to_list(layer, model_ds, cellids)
    else:
        cellids = np.where(mask)
        layers = col_to_list(layer, model_ds, cellids)

    rows = cellids[0]
    columns = cellids[1]

    rec_list = lrc_to_rec_list(layers, rows, columns, cellids, model_ds,
                               col1, col2, col3)

    return rec_list


def lcid_to_rec_list(layers, cellids, model_ds,
                     col1=None, col2=None, col3=None):
    """ Create a rec list for stress period data from a set of cellids.

    Used for unstructured grids.


    Parameters
    ----------
    layers : list or numpy.ndarray
        list with the layer for each cell in the rec_list.
    cellids : tuple of numpy arrays
        tuple with cell ids of the cells that will be used to create the list
        with values.
    model_ds : xarray.Dataset
        dataset with model data. Can have dimension (layer, y, x) or 
        (layer, cid).
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]
            riv: stage [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]
            riv: conductacnt [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        ((layer,row,column), col1, col2) for each row.

        col3 should be the following value for each package (can also be the
            name of a timeseries):
            riv: bottom [L]

    Raises
    ------
    ValueError
        Question: will this error ever occur?.

    Returns
    -------
    rec_list : list of tuples
        every row consist of ((layer, cid), col1, col2, col3)
        grids.

    """
    if col1 is None:
        rec_list = list(zip(zip(layers, cellids[0])))
    elif (col1 is not None) and col2 is None:
        col1_lst = col_to_list(col1, model_ds, cellids)
        rec_list = list(zip(zip(layers, cellids[0]),
                            col1_lst))
    elif (col2 is not None) and col3 is None:
        col1_lst = col_to_list(col1, model_ds, cellids)
        col2_lst = col_to_list(col2, model_ds, cellids)
        rec_list = list(zip(zip(layers, cellids[0]),
                            col1_lst, col2_lst))
    elif (col3 is not None):
        col1_lst = col_to_list(col1, model_ds, cellids)
        col2_lst = col_to_list(col2, model_ds, cellids)
        col3_lst = col_to_list(col3, model_ds, cellids)
        rec_list = list(zip(zip(layers, cellids[0]),
                            col1_lst, col2_lst, col3_lst))
    else:
        raise ValueError(
            'invalid combination of values for col1, col2 and col3')

    return rec_list


def data_array_unstructured_to_rec_list(model_ds, mask,
                                        col1=None, col2=None, col3=None,
                                        layer=0,
                                        first_active_layer=False,
                                        only_active_cells=True):
    """ Create a rec list for stress period data from a model dataset.

    Used for unstructured grids.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data and dimensions (layer, cid)
    mask : xarray.DataArray for booleans
        True for the cells that will be used in the rec list.
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,cid)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        (((layer,cid), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        (((layer,cid), col1, col2) for each row.

        col3 should be the following value for each package (can also be the
            name of a timeseries):
    layer : int, optional
        layer used in the rec_list. Not used if first_active_layer is True.
        default is 0
    first_active_layer : bool, optional
        If True an extra mask is applied to use the first active layer of each 
        cell in the grid. The default is False.
    only_active_cells : bool, optional
        If True an extra mask is used to only include cells with an idomain 
        of 1. The default is True.

    Returns
    -------
    rec_list : list of tuples
        every row consist of ((layer,cid), col1, col2, col3).

    """
    if first_active_layer:
        cellids = np.where(
            (mask) & (model_ds['first_active_layer'] != model_ds.nodata))
        layers = col_to_list('first_active_layer', model_ds, cellids)
    elif only_active_cells:
        cellids = np.where((mask) & (model_ds['idomain'][layer] == 1))
        layers = col_to_list(layer, model_ds, cellids)
    else:
        cellids = np.where(mask)
        layers = col_to_list(layer, model_ds, cellids)

    rec_list = lcid_to_rec_list(layers, cellids, model_ds, col1, col2, col3)

    return rec_list


def polygon_to_area(modelgrid, polygon, da,
                    gridtype='structured'):
    """ create a grid with the surface area in each cell based on a 
    polygon value.


    Parameters
    ----------
    gwf : flopy.discretization.structuredgrid.StructuredGrid
        grid.
    polygon : shapely.geometry.polygon.Polygon
        polygon feature.
    da : xarray.DataArray
        data array that is use to fill output 

    Returns
    -------
    area_array : xarray.DataArray
        area of polygon within each modelgrid cell

    """

    ix = GridIntersect(modelgrid)
    opp_cells = ix.intersect(polygon)

    area_array = xr.zeros_like(da)

    if gridtype == 'structured':
        for opp_row in opp_cells:
            area = opp_row[-2]
            area_array[opp_row[0][0], opp_row[0][1]] = area
    elif gridtype == 'unstructured':
        cids = opp_cells.cellids
        area = opp_cells.areas
        area_array[cids.astype(int)] = area

    return area_array


def gdf_to_bool_data_array(gdf, mfgrid, model_ds):
    """ convert a GeoDataFrame with polygon geometries into a data array
    corresponding to the modelgrid in which each cell is 1 (True) if one or
    more geometries are (partly) in that cell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        polygon shapes with surface water.
    mfgrid : flopy grid
        model grid.
    model_ds : xr.DataSet
        xarray with model data

    Returns
    -------
    da : xr.DataArray
        1 if polygon is in cell, 0 otherwise. Grid dimensions according to
        model_ds and mfgrid.

    """

    # build list of gridcells
    ix = GridIntersect(mfgrid, method="vertex")

    da = xr.zeros_like(model_ds['top'])
    for geom in gdf.geometry.values:
        # prepare shape for efficient batch intersection check
        prepshp = prep(geom)

        # get only gridcells that intersect
        filtered = filter(prepshp.intersects, ix._get_gridshapes())

        # cell ids for intersecting cells
        cids = [c.name for c in filtered]

        if model_ds.gridtype == 'structured':
            for cid in cids:
                da[cid[0], cid[1]] = 1
        elif model_ds.gridtype == 'unstructured':
            da[cids] = 1
        else:
            raise ValueError(
                'function only support structured or unstructured gridtypes')

    return da


def gdf_to_bool_dataset(model_ds, gdf, mfgrid, da_name):
    """ convert a GeoDataFrame with polygon geometries into a model dataset
    with a data_array named 'da_name' in which each cell is 1 (True) if one or
    more geometries are (partly) in that cell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        polygon shapes with surface water.
    mfgrid : flopy grid
        model grid.
    model_ds : xr.DataSet
        xarray with model data

    Returns
    -------
    model_ds_out : xr.Dataset
        Dataset with a single DataArray, this DataArray is 1 if polygon is in
        cell, 0 otherwise. Grid dimensions according to model_ds and mfgrid.

    """
    model_ds_out = util.get_model_ds_empty(model_ds)
    model_ds_out[da_name] = gdf_to_bool_data_array(gdf, mfgrid, model_ds)

    return model_ds_out


def get_thickness_from_topbot(top, bot):
    """get thickness from data arrays with top and bots

    Parameters
    ----------
    top : xr.DataArray
        raster with top of each cell. dimensions should be (y,x) or (cid).
    bot : xr.DataArray
        raster with bottom of each cell. dimensions should be (layer, y,x) or 
        (layer, cid).

    Returns
    -------
    thickness : xr.DataArray
        raster with thickness of each cell. dimensions should be (layer, y,x) 
        or (layer, cid).

    """
    if np.ndim(top) > 2:
        raise NotImplementedError('function works only for 2d top')

    # get thickness
    thickness = xr.zeros_like(bot)
    for lay in range(len(bot)):
        if lay == 0:
            thickness[lay] = top - bot[lay]
        else:
            thickness[lay] = bot[lay - 1] - bot[lay]

    return thickness


def update_idomain_from_thickness(idomain, thickness, mask):
    """
    get new idomain from thickness in the cells where mask is 1 (or True).
    Idomain becomes:
    1: if cell thickness is bigger than 0
    0: if cell thickness is 0 and it is the top layer
    -1: if cell thickness is 0 and the layer is in between active cells

    Parameters
    ----------
    idomain : xr.DataArray
        raster with idomain of each cell. dimensions should be (layer, y,x) or 
        (layer, cid).
    thickness : xr.DataArray
        raster with thickness of each cell. dimensions should be (layer, y,x) or 
        (layer, cid).
    mask : xr.DataArray
        raster with ones in cell where the ibound is adjusted. dimensions 
        should be (y,x) or (cid).

    Returns
    -------
    idomain : xr.DataArray
        raster with adjusted idomain of each cell. dimensions should be 
        (layer, y,x) or (layer, cid).

    """

    for lay in range(len(thickness)):
        if lay == 0:
            mask1 = (thickness[lay] == 0) * mask
            idomain[lay] = xr.where(mask1, 0, idomain[lay])
            mask2 = (thickness[lay] > 0) * mask
            idomain[lay] = xr.where(mask2, 1, idomain[lay])
        else:
            mask1 = (thickness[lay] == 0) * mask * (idomain[lay - 1] == 0)
            idomain[lay] = xr.where(mask1, 0, idomain[lay])

            mask2 = (thickness[lay] == 0) * mask * (idomain[lay - 1] != 0)
            idomain[lay] = xr.where(mask2, -1, idomain[lay])

            mask3 = (thickness[lay] != 0) * mask
            idomain[lay] = xr.where(mask3, 1, idomain[lay])

    return idomain


def get_surface_area(gwf):
    if isinstance(gwf.modelgrid, flopy.discretization.StructuredGrid):
        delr, delc = np.meshgrid(gwf.modelgrid.delr, gwf.modelgrid.delc)
        return delr * delc
    else:
        msg = 'Calculation of area for {} not yet supported'
        raise(Exception(msg.format(type(gwf.modelgrid))))


def get_rasterio_src(fname):
    if isinstance(fname, str):
        src = rasterio.open(fname)
    elif isinstance(fname, rasterio.DatasetReader):
        src = fname
    else:
        raise(Exception('Unknown file: {}'.format(fname)))
    return src


def get_xy_mid(fname, transform=None):
    if isinstance(fname, str) or isinstance(fname, rasterio.DatasetReader):
        src = get_rasterio_src(fname)
        extent = rasterio.plot.plotting_extent(src)
        transform = src.transform
    else:
        if transform is None:
            raise('When supplying extent, also supply transform')
        if len(fname) == 2:
            # first argument is the shape
            shape = fname
            extent = [transform[2], transform[2] + transform[0] * shape[1],
                      transform[5] + transform[4] * shape[0], transform[5]]
        else:
            # first argument is the extent
            extent = fname
    x = np.arange(extent[0] + transform[0] / 2, extent[1], transform[0])
    y = np.arange(extent[3] + transform[4] / 2, extent[2], transform[4])
    return x, y


def raster_to_quadtree_grid(fname, model_ds, dst_crs=None,
                            resampling=rasterio.enums.Resampling.average,
                            return_data_array=True):
    """Resample a raster-file to a quadtree-grid, using different advanced
    resample algoritms"""
    area = model_ds['area'].values
    x = model_ds.x.values
    y = model_ds.y.values
    z = np.full(area.shape, np.NaN)

    for ar in np.unique(area):
        mask = area == ar
        extent = model_ds.attrs['extent']
        dx = dy = np.sqrt(ar)
        dst_transform = rasterio.transform.from_origin(extent[0], extent[3],
                                                       dx, dy)
        width = int((extent[1] - extent[0]) / dx)
        height = int((extent[3] - extent[2]) / dy)
        dst_shape = (height, width)
        zt = np.zeros(dst_shape)
        with rasterio.open(fname) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=zt,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs if dst_crs is None else dst_crs,
                resampling=resampling,
                dst_nodata=np.NaN)
        # use an xarray to get the right values using .sel()
        xt, yt = get_xy_mid(extent, dst_transform)
        da = xr.DataArray(zt, coords=(yt, xt), dims=['y', 'x'])
        if len(mask.shape) == 2:
            x, y = np.meshgrid(x, y)
        z[mask] = da.sel(y=xr.DataArray(y[mask]),
                         x=xr.DataArray(x[mask])).values
    if return_data_array:
        z_da = xr.full_like(model_ds['top'], np.NaN)
        z_da.data = z
        return z_da

    return z











def add_kh_kv_from_ml_layer_to_dataset(ml_layer_ds, model_ds, anisotropy,
                                       fill_value_kh, fill_value_kv,
                                       verbose=False):
    """ add kh and kv from a model layer dataset to THE model dataset.

    Supports structured and unstructured grids.

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with kh and kv
    model_ds : xarray.Dataset
        dataset with model data where kh and kv are added to
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 1.0.
    verbose : bool, optional
        print additional information. default is False

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data with new kh and kv

    Notes
    -----
    some model dataset, such as regis, also have 'c' and 'kd' values. These 
    are ignored at the moment
    """
    model_ds.attrs['anisotropy'] = anisotropy
    model_ds.attrs['fill_value_kh'] = fill_value_kh
    model_ds.attrs['fill_value_kv'] = fill_value_kv
    kh_arr = ml_layer_ds['kh'].data
    kv_arr = ml_layer_ds['kv'].data

    if verbose:
        print('add kh and kv from model layer dataset to modflow model')

    kh, kv = get_kh_kv(kh_arr, kv_arr, anisotropy,
                       fill_value_kh=fill_value_kh,
                       fill_value_kv=fill_value_kv,
                       verbose=verbose)

    model_ds['kh'] = xr.ones_like(model_ds['idomain']) * kh

    model_ds['kv'] = xr.ones_like(model_ds['idomain']) * kv

    return model_ds


def get_kh_kv(kh_in, kv_in, anisotropy,
              fill_value_kh=1.0, fill_value_kv=1.0,
              verbose=False):
    """maak kh en kv rasters voor flopy vanuit een regis raster met nan
    waardes.

    vul kh raster door:
    1. pak kh uit regis, tenzij nan dan:
    2. pak kv uit regis vermenigvuldig met anisotropy, tenzij nan dan:
    3. pak fill_value_kh

    vul kv raster door:
    1. pak kv uit regis, tenzij nan dan:
    2. pak kh uit regis deel door anisotropy, tenzij nan dan:
    3. pak fill_value_kv

    Supports structured and unstructured grids.

    Parameters
    ----------
    kh_in : np.ndarray
        kh from regis with nan values shape(nlay, nrow, ncol) or 
        shape(nlay, len(cid))
    kv_in : np.ndarray
        kv from regis with nan values shape(nlay, nrow, ncol) or 
        shape(nlay, len(cid))
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 1.0.
    verbose : bool, optional
        print additional information. default is False

    Returns
    -------
    kh_out : np.ndarray
        kh without nan values (nlay, nrow, ncol) or shape(nlay, len(cid))
    kv_out : np.ndarray
        kv without nan values (nlay, nrow, ncol) or shape(nlay, len(cid))
    """
    kh_out = np.zeros_like(kh_in)
    for i, kh_lay in enumerate(kh_in):
        kh_new = kh_lay.copy()
        kv_new = kv_in[i].copy()
        if ~np.all(np.isnan(kh_new)):
            if verbose:
                print(f'layer {i} has a kh')
            kh_out[i] = np.where(np.isnan(kh_new), kv_new * anisotropy, kh_new)
            kh_out[i] = np.where(np.isnan(kh_out[i]), fill_value_kh, kh_out[i])
        elif ~np.all(np.isnan(kv_new)):
            if verbose:
                print(f'layer {i} has a kv')
            kh_out[i] = np.where(
                np.isnan(kv_new), fill_value_kh, kv_new * anisotropy)
        else:
            if verbose:
                print(f'kv and kh both undefined in layer {i}')
            kh_out[i] = fill_value_kh

    kv_out = np.zeros_like(kv_in)
    for i, kv_lay in enumerate(kv_in):
        kv_new = kv_lay.copy()
        kh_new = kh_in[i].copy()
        if ~np.all(np.isnan(kv_new)):
            if verbose:
                print(f'layer {i} has a kv')
            kv_out[i] = np.where(np.isnan(kv_new), kh_new / anisotropy, kv_new)
            kv_out[i] = np.where(np.isnan(kv_out[i]), fill_value_kv, kv_out[i])
        elif ~np.all(np.isnan(kh_new)):
            if verbose:
                print(f'layer {i} has a kh')
            kv_out[i] = np.where(
                np.isnan(kh_new), fill_value_kv, kh_new / anisotropy)
        else:
            if verbose:
                print(f'kv and kh both undefined in layer {i}')
            kv_out[i] = fill_value_kv

    return kh_out, kv_out


def add_top_bot_to_model_ds(ml_layer_ds, model_ds,
                            nodata=None,
                            max_per_nan_bot=50,
                            gridtype='structured',
                            verbose=False):
    """ add top and bot from a model layer dataset to THE model dataset.

    Supports structured and unstructured grids.

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with a top and bottom
    model_ds : xarray.Dataset
        dataset with model data where top and bot are added to
    nodata : int, optional
        if the first_active_layer data array in model_ds has this value,
        it means this cell is inactive in all layers. If nodata is None the
        nodata value in model_ds is used.
        the default is None
    max_per_nan_bot : int or float, optional
        if the percentage of cells that have nan values in all layers is
        higher than this an error is raised. The default is 50.
    gridtype : str, optional
        type of grid, options are 'structured' and 'unstructured'. 
        The default is 'structured'.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including top and bottom

    """
    if nodata is None:
        nodata = model_ds.attrs['nodata']

    if verbose:
        print('using top and bottom from model layers dataset for modflow model')
        print('replace nan values for inactive layers with dummy value')

    if gridtype == 'structured':

        model_ds = add_top_bot_structured(ml_layer_ds, model_ds,
                                          nodata=nodata,
                                          max_per_nan_bot=max_per_nan_bot)

    elif gridtype == 'unstructured':
        model_ds = add_top_bot_unstructured(ml_layer_ds, model_ds,
                                            nodata=nodata,
                                            max_per_nan_bot=max_per_nan_bot)

    return model_ds


def add_top_bot_unstructured(ml_layer_ds, model_ds, nodata=-999,
                             max_per_nan_bot=50):
    """ voeg top en bottom vanuit model layer dataset toe aan de model dataset

    Deze functie is bedoeld voor unstructured arrays in modflow 6

    stappen:
    1. Zorg dat de onderste laag altijd een bodemhoogte heeft, als de bodem
    van alle bovenliggende lagen nan is, pak dan 0.
    2. Zorg dat de top van de bovenste laag altijd een waarde heeft, als de
    top van alle onderligende lagen nan is, pak dan 0.
    3. Vul de nan waarden in alle andere lagen door:
        a. pak bodem uit regis, tenzij nan dan:
        b. gebruik bodem van de laag erboven (of de top voor de bovenste laag)

    Supports only unstructured grids.

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with a top and bottom
    model_ds : xarray.Dataset
        dataset with model data where top and bottom are added to
    nodata : int, optional
        if the first_active_layer data array in model_ds has this value,
        it means this cell is inactive in all layers 
    max_per_nan_bot : int or float, optional
        if the percentage of cells that have nan values in all layers. 
        The default is 50.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including top and bottom
    """
    # step 1:
    # set nan-value in bottom array
    # set to zero if value is nan in all layers
    # set to minimum value of all layers if there is any value in any layer
    active_domain = model_ds['first_active_layer'].data != nodata

    lowest_bottom = ml_layer_ds['bottom'].data[-1].copy()
    if np.any(active_domain == False):
        percentage = 100 * (active_domain == False).sum() / \
            (active_domain.shape[0])
        if percentage > max_per_nan_bot:
            print(f"{percentage:0.1f}% cells with nan values in every layer"
                  " check your extent")
            raise MemoryError('if you want to'
                              ' ignore this error set max_per_nan_bot higher')

        # set bottom to zero if bottom in a cell is nan in all layers
        lowest_bottom = np.where(active_domain, lowest_bottom, 0)

    if np.any(np.isnan(lowest_bottom)):
        # set bottom in a cell to lowest bottom of all layers
        i_nan = np.where(np.isnan(lowest_bottom))
        for i in i_nan:
            val = np.nanmin(ml_layer_ds['bottom'].data[:, i])
            lowest_bottom[i] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Artesia')

    # step 2: get highest top values of all layers without nan values
    highest_top = ml_layer_ds['top'].data[0].copy()
    if np.any(np.isnan(highest_top)):
        highest_top = np.where(active_domain, highest_top, 0)

    if np.any(np.isnan(highest_top)):
        i_nan = np.where(np.isnan(highest_top))
        for i in i_nan:
            val = np.nanmax(ml_layer_ds['top'].data[:, i])
            highest_top[i] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Artesia')

    # step 3: fill nans in all layers
    nlay = model_ds.dims['layer']
    top_bot_raw = np.ones((nlay + 1, model_ds.dims['cid']))
    top_bot_raw[0] = highest_top
    top_bot_raw[1:-1] = ml_layer_ds['bottom'].data[:-1].copy()
    top_bot_raw[-1] = lowest_bottom
    top_bot = np.ones_like(top_bot_raw)
    for i_from_bot, blay in enumerate(top_bot_raw[::-1]):
        i_from_top = nlay - i_from_bot
        new_lay = blay.copy()
        if np.any(np.isnan(new_lay)):
            lay_from_bot = i_from_bot
            lay_from_top = nlay - lay_from_bot
            while np.any(np.isnan(new_lay)):
                new_lay = np.where(np.isnan(new_lay),
                                   top_bot_raw[lay_from_top],
                                   new_lay)
                lay_from_bot += 1
                lay_from_top = nlay - lay_from_bot

        top_bot[i_from_top] = new_lay

    model_ds['bot'] = xr.DataArray(top_bot[1:], dims=('layer', 'cid'),
                                   coords={'cid': model_ds.cid.data,
                                           'layer': model_ds.layer.data})
    model_ds['top'] = xr.DataArray(top_bot[0], dims=('cid'),
                                   coords={'cid': model_ds.cid.data})

    return model_ds


def add_top_bot_structured(ml_layer_ds, model_ds, nodata=-999,
                           max_per_nan_bot=50):
    """ voeg top en bottom vanuit een model layer dataset toe aan DE model 
    dataset

    Deze functie is bedoeld voor structured arrays in modflow 6

    stappen:
    1. Zorg dat de onderste laag altijd een bodemhoogte heeft, als de bodem
    van alle bovenliggende lagen nan is, pak dan 0.
    2. Zorg dat de top van de bovenste laag altijd een waarde heeft, als de
    top van alle onderligende lagen nan is, pak dan 0.
    3. Vul de nan waarden in alle andere lagen door:
        a. pak bodem uit de model layer dataset, tenzij nan dan:
        b. gebruik bodem van de laag erboven (of de top voor de bovenste laag)

    Supports only structured grids.

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with a top and bottom
    model_ds : xarray.Dataset
        dataset with model data where top and bottom are added to
    nodata : int, optional
        if the first_active_layer data array in model_ds has this value,
        it means this cell is inactive in all layers 
    max_per_nan_bot : int or float, optional
        if the percentage of cells that have nan values in all layers. 
        The default is 50.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including top and bottom

    """

    active_domain = model_ds['first_active_layer'].data != nodata

    # step 1:
    # set nan-value in bottom array
    # set to zero if value is nan in all layers
    # set to minimum value of all layers if there is any value in any layer
    lowest_bottom = ml_layer_ds['bottom'].data[-1].copy()
    if np.any(active_domain == False):
        percentage = 100 * (active_domain == False).sum() / \
            (active_domain.shape[0] * active_domain.shape[1])
        if percentage > max_per_nan_bot:
            print(f"{percentage:0.1f}% cells with nan values in every layer"
                  " check your extent")
            raise MemoryError('if you want to'
                              ' ignore this error set max_per_nan_bot higher')

        # set bottom to zero if bottom in a cell is nan in all layers
        lowest_bottom = np.where(active_domain, lowest_bottom, 0)

    if np.any(np.isnan(lowest_bottom)):
        # set bottom in a cell to lowest bottom of all layers
        rc_nan = np.where(np.isnan(lowest_bottom))
        for row, col in zip(rc_nan[0], rc_nan[1]):
            val = np.nanmin(ml_layer_ds['bottom'].data[:, row, col])
            lowest_bottom[row, col] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Onno')

    # step 2: get highest top values of all layers without nan values
    highest_top = ml_layer_ds['top'].data[0].copy()
    if np.any(np.isnan(highest_top)):
        # set top to zero if top in a cell is nan in all layers
        highest_top = np.where(active_domain, highest_top, 0)

    if np.any(np.isnan(highest_top)):
        # set top in a cell to highest top of all layers
        rc_nan = np.where(np.isnan(highest_top))
        for row, col in zip(rc_nan[0], rc_nan[1]):
            val = np.nanmax(ml_layer_ds['top'].data[:, row, col])
            highest_top[row, col] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Onno')

    # step 3: fill nans in all layers
    nlay = model_ds.dims['layer']
    nrow = model_ds.dims['y']
    ncol = model_ds.dims['x']
    top_bot_raw = np.ones((nlay + 1, nrow, ncol))
    top_bot_raw[0] = highest_top
    top_bot_raw[1:-1] = ml_layer_ds['bottom'].data[:-1].copy()
    top_bot_raw[-1] = lowest_bottom
    top_bot = np.ones_like(top_bot_raw)
    for i_from_bot, blay in enumerate(top_bot_raw[::-1]):
        i_from_top = nlay - i_from_bot
        new_lay = blay.copy()
        if np.any(np.isnan(new_lay)):
            lay_from_bot = i_from_bot
            lay_from_top = nlay - lay_from_bot
            while np.any(np.isnan(new_lay)):
                new_lay = np.where(np.isnan(new_lay),
                                   top_bot_raw[lay_from_top],
                                   new_lay)
                lay_from_bot += 1
                lay_from_top = nlay - lay_from_bot

        top_bot[i_from_top] = new_lay

    model_ds['bot'] = xr.DataArray(top_bot[1:], dims=('layer', 'y', 'x'),
                                   coords={'x': model_ds.x.data,
                                           'y': model_ds.y.data,
                                           'layer': model_ds.layer.data})

    model_ds['top'] = xr.DataArray(top_bot[0], dims=('y', 'x'),
                                   coords={'x': model_ds.x.data,
                                           'y': model_ds.y.data})

    return model_ds


def fill_top_bot_kh_kv_at_mask(model_ds, fill_mask,
                               gridtype='structured',
                               gridprops=None):
    """ fill values in top, bot, kh and kv where:
        1. the cell is True in mask
        2. the cell thickness is greater than 0

    fill values:
        top: 0
        bot: minimum of bottom_filled or top
        kh: kh_filled if thickness is greater than 0
        kv: kv_filled if thickness is greater than 0

    Parameters
    ----------
    model_ds : xr.DataSet
        model dataset, should contain 'first_active_layer'
    fill_mask : xr.DataArray
        1 where a cell should be replaced by masked value.
    gridtype : str, optional
        type of grid.        
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Default is None

    Returns
    -------
    model_ds : xr.DataSet
        model dataset with adjusted data variables:
            'top', 'bot', 'kh', 'kv'
    """

    # zee cellen hebben altijd een top gelijk aan 0
    model_ds['top'] = xr.where(fill_mask, 0, model_ds['top'])

    if gridtype == 'structured':
        fill_function = fillnan_dataarray_structured_grid
        fill_function_kwargs = {}
    elif gridtype == 'unstructured':
        fill_function = fillnan_dataarray_unstructured_grid
        fill_function_kwargs = {'gridprops': gridprops}

    for lay in range(model_ds.dims['layer']):
        bottom_nan = xr.where(fill_mask, np.nan, model_ds['bot'][lay])
        bottom_filled = fill_function(bottom_nan, **fill_function_kwargs)[0]

        kh_nan = xr.where(fill_mask, np.nan, model_ds['kh'][lay])
        kh_filled = fill_function(kh_nan, **fill_function_kwargs)[0]

        kv_nan = xr.where(fill_mask, np.nan, model_ds['kv'][lay])
        kv_filled = fill_function(kv_nan, **fill_function_kwargs)[0]

        if lay == 0:
            # top ligt onder bottom_filled -> laagdikte wordt 0
            # top ligt boven bottom_filled -> laagdikte o.b.v. bottom_filled
            mask_top = model_ds['top'] < bottom_filled
            model_ds['bot'][lay] = xr.where(fill_mask * mask_top,
                                            model_ds['top'],
                                            bottom_filled)
            model_ds['kh'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kh'][lay],
                                           kh_filled)
            model_ds['kv'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kv'][lay],
                                           kv_filled)

        else:
            # top ligt onder bottom_filled -> laagdikte wordt 0
            # top ligt boven bottom_filled -> laagdikte o.b.v. bottom_filled
            mask_top = model_ds['bot'][lay - 1] < bottom_filled
            model_ds['bot'][lay] = xr.where(fill_mask * mask_top,
                                            model_ds['bot'][lay - 1],
                                            bottom_filled)
            model_ds['kh'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kh'][lay],
                                           kh_filled)
            model_ds['kv'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kv'][lay],
                                           kv_filled)

    return model_ds


def add_bathymetry_to_top_bot_kh_kv(model_ds, bathymetry,
                                    fill_mask,
                                    kh_sea=10,
                                    kv_sea=10):
    """ add bathymetry to the top and bot of each layer for all cells with
    fill_mask.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data, should 
    fill_mask : xr.DataArray
        cell value is 1 if you want to add bathymetry

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data where the top, bot, kh and kv are changed

    """
    model_ds['top'] = xr.where(fill_mask,
                               0.0,
                               model_ds['top'])

    lay = 0
    model_ds['bot'][lay] = xr.where(fill_mask,
                                    bathymetry,
                                    model_ds['bot'][lay])

    model_ds['kh'][lay] = xr.where(fill_mask,
                                   kh_sea,
                                   model_ds['kh'][lay])

    model_ds['kv'][lay] = xr.where(fill_mask,
                                   kv_sea,
                                   model_ds['kv'][lay])

    # reset bot for all layers based on bathymetrie
    for lay in range(1, model_ds.dims['layer']):
        model_ds['bot'][lay] = np.where(model_ds['bot'][lay] > model_ds['bot'][lay - 1],
                                        model_ds['bot'][lay - 1],
                                        model_ds['bot'][lay])

    return model_ds


def get_ml_layer_dataset_struc(raw_ds=None,
                               extent=None,
                               delr=None,
                               delc=None,
                               gridtype='structured',
                               gridprops=None,
                               interp_method="linear",
                               cachedir=None,
                               fname_netcdf='regis.nc',
                               use_cache=False,
                               verbose=False):
    """ Get a model layer dataset that is resampled to the modelgrid

    Parameters
    ----------
    raw_ds : xarray.Dataset, optional
        raw model layer dataset. If the gridtype is structured this should be 
        the original regis netcdf dataset. If gridtype is unstructured this 
        should be a regis dataset that is already projected on a structured 
        modelgrid. The default is None.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    delr : int or float
        cell size along rows of the desired grid (dx).
    delc : int or float
        cell size along columns of the desired grid (dy).
    gridtype : str, optional
        type of grid, options are 'structured' and 'unstructured'.
        The default is 'structured'.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'
    interp_method : str, optional
        interpolation method, default is 'linear'
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    fname_netcdf : str
        name of the netcdf file that is stored in the cachedir.
    use_cache : bool, optional
        if True the cached resampled regis dataset is used.
        The default is False.

    Returns
    -------
    regis_ds : xarray.Dataset
        dataset with regis data corresponding to the modelgrid

    """

    ml_layer_ds = util.get_cache_netcdf(use_cache, cachedir, fname_netcdf,
                                       get_resampled_ml_layer_ds_struc,
                                       verbose=verbose,
                                       raw_ds=raw_ds, extent=extent,
                                       gridprops=gridprops, check_time=False,
                                       delr=delr, delc=delc,
                                       kind=interp_method)

    return ml_layer_ds


def get_resampled_ml_layer_ds_struc(raw_ds=None,
                                    extent=None, delr=None, delc=None,
                                    gridprops=None,
                                    verbose=False,
                                    kind='linear'):
    """ project regis data on to the modelgrid.


    Parameters
    ----------
    raw_ds : xr.Dataset, optional
        regis dataset. If the gridtype is structured this should be the
        original regis netcdf dataset. If gridtype is unstructured this should
        be a regis dataset that is already projected on a structured modelgrid. 
        The default is None.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    delr : int or float
        cell size along rows of the desired grid (dx).
    delc : int or float
        cell size along columns of the desired grid (dy).
    gridtype : str, optional
        type of grid, options are 'structured' and 'unstructured'. 
        The default is 'structured'.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'
    verbose : bool, optional
        print additional information. default is False
    kind : str, optional
        kind of interpolation to use. default is 'linear'

    Returns
    -------
    ml_layer_ds : xr.dataset
        model layer dataset projected onto the modelgrid.

    """

    if verbose:
        print('resample regis data to structured modelgrid')
    ml_layer_ds = resample_dataset_to_structured_grid(raw_ds, extent,
                                                            delr, delc,
                                                            kind=kind)
    ml_layer_ds.attrs['extent'] = extent
    ml_layer_ds.attrs['delr'] = delr
    ml_layer_ds.attrs['delc'] = delc
    ml_layer_ds.attrs['gridtype'] = 'structured'

    return ml_layer_ds


def get_ml_layer_dataset_unstruc(raw_ds=None,
                                 extent=None,
                                 gridtype='structured',
                                 gridprops=None,
                                 interp_method="linear",
                                 cachedir=None,
                                 fname_netcdf='regis.nc',
                                 use_cache=False,
                                 verbose=False):
    """ Get a model layer dataset that is resampled to the modelgrid

    Parameters
    ----------
    raw_ds : xarray.Dataset, optional
        raw model layer dataset. If the gridtype is structured this should be 
        the original regis netcdf dataset. If gridtype is unstructured this 
        should be a regis dataset that is already projected on a structured 
        modelgrid. The default is None.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    gridtype : str, optional
        type of grid, options are 'structured' and 'unstructured'.
        The default is 'structured'.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'
    interp_method : str, optional
        interpolation method, default is 'linear'
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    fname_netcdf : str
        name of the netcdf file that is stored in the cachedir.
    use_cache : bool, optional
        if True the cached resampled regis dataset is used.
        The default is False.

    Returns
    -------
    regis_ds : xarray.Dataset
        dataset with regis data corresponding to the modelgrid

    """

    ml_layer_ds = util.get_cache_netcdf(use_cache, cachedir, fname_netcdf,
                                       get_resampled_ml_layer_ds_unstruc,
                                       verbose=verbose, check_time=False,
                                       raw_ds=raw_ds, extent=extent,
                                       gridprops=gridprops)

    return ml_layer_ds


def get_resampled_ml_layer_ds_unstruc(raw_ds=None,
                                      extent=None,
                                      gridprops=None,
                                      verbose=False):
    """ project model layer dataset on an unstructured model grid.


    Parameters
    ----------
    raw_ds : xr.Dataset, optional
        regis dataset. If the gridtype is structured this should be the
        original regis netcdf dataset. If gridtype is unstructured this should
        be a regis dataset that is already projected on a structured modelgrid. 
        The default is None.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'
    verbose : bool, optional
        print additional information. default is False

    Returns
    -------
    ml_layer_ds : xr.dataset
        model layer dataset projected onto the modelgrid.

    """

    if verbose:
        print('resample regis data to unstructured modelgrid')
    ml_layer_ds = resample_dataset_to_unstructured_grid(
        raw_ds, gridprops)
    ml_layer_ds['x'] = xr.DataArray([r[1] for r in gridprops['cell2d']],
                                    dims=('cid'),
                                    coords={'cid': ml_layer_ds.cid.data})

    ml_layer_ds['y'] = xr.DataArray([r[2] for r in gridprops['cell2d']],
                                    dims=('cid'),
                                    coords={'cid': ml_layer_ds.cid.data})
    ml_layer_ds.attrs['gridtype'] = 'unstructured'
    ml_layer_ds.attrs['extent'] = extent

    return ml_layer_ds


def get_regis_ds_from_raw_ds(regis_ds_raw=None,
                             extent=None, delr=None, delc=None,
                             gridtype='structured',
                             gridprops=None,
                             verbose=False,
                             kind='linear'):
    """ project regis data on to the modelgrid.


    Parameters
    ----------
    regis_ds_raw : xr.Dataset, optional
        regis dataset. If the gridtype is structured this should be the
        original regis netcdf dataset. If gridtype is unstructured this should
        be a regis dataset that is already projected on a structured modelgrid. 
        The default is None.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    delr : int or float
        cell size along rows of the desired grid (dx).
    delc : int or float
        cell size along columns of the desired grid (dy).
    gridtype : str, optional
        type of grid, options are 'structured' and 'unstructured'. 
        The default is 'structured'.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Only used if
        gridtype = 'unstructured'
    verbose : bool, optional
        print additional information. default is False
    kind : str, optional
        kind of interpolation to use. default is 'linear'

    Returns
    -------
    regis_ds : xr.dataset
        regis dataset projected onto the modelgrid.

    """

    if gridtype == 'structured':
        if verbose:
            print('resample regis data to structured modelgrid')
        regis_ds = resample_dataset_to_structured_grid(regis_ds_raw, extent,
                                                             delr, delc, kind=kind)
        regis_ds.attrs['extent'] = extent
        regis_ds.attrs['delr'] = delr
        regis_ds.attrs['delc'] = delc
        regis_ds.attrs['gridtype'] = gridtype
    elif gridtype == 'unstructured':
        if verbose:
            print('resample regis data to unstructured modelgrid')
        regis_ds = resample_dataset_to_unstructured_grid(
            regis_ds_raw, gridprops)
        regis_ds['x'] = xr.DataArray([r[1] for r in gridprops['cell2d']],
                                     dims=('cid'),
                                     coords={'cid': regis_ds.cid.data})

        regis_ds['y'] = xr.DataArray([r[2] for r in gridprops['cell2d']],
                                     dims=('cid'),
                                     coords={'cid': regis_ds.cid.data})
        regis_ds.attrs['gridtype'] = gridtype
        regis_ds.attrs['extent'] = extent

    return regis_ds