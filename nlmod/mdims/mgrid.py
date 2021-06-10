# -*- coding: utf-8 -*-
"""
This module contains functions to:
    - project data on different grid forms
    - obtain various types of rec_lists from a grid that can be used as input for a mf
    package
    - fill, interpolate and resample grid data
    - 
"""
import copy
import os
import pickle
import sys
import tempfile

import flopy
import numpy as np
import geopandas as gpd
import xarray as xr
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.utils.gridgen import Gridgen
from flopy.utils.gridintersect import GridIntersect

import shapely
from shapely.prepared import prep

from . import resample
from .. import mfpackages, util
from ..read import jarkus


def modelgrid_from_model_ds(model_ds, gridprops=None):
    """ obtain the flopy modelgrid from model_ds


    Parameters
    ----------
    model_ds : xarray DataSet
        model dataset.
    gridprops : dic, optional
        extra model properties when using unstructured grids. The default is None.

    Returns
    -------
    modelgrid : flopy StructuredGrid or flopy VertexGrid 
        grid information.

    """

    if model_ds.gridtype == 'structured':
        if not isinstance(model_ds.extent, (tuple, list, np.ndarray)):
            raise TypeError(
                f'extent should be a list, tuple or numpy array, not {type(model_ds.extent)}')

        modelgrid = StructuredGrid(delc=np.array([model_ds.delc] * model_ds.dims['y']),
                                   delr=np.array([model_ds.delc] *
                                                 model_ds.dims['x']),
                                   xoff=model_ds.extent[0], yoff=model_ds.extent[2])
    elif model_ds.gridtype == 'unstructured':
        _, gwf = mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds,
                                                           verbose=False)
        flopy.mf6.ModflowGwfdisv(gwf, idomain=model_ds['idomain'].data,
                                 **gridprops)
        modelgrid = gwf.modelgrid

    return modelgrid


def update_model_ds_from_ml_layer_ds(model_ds, ml_layer_ds,
                                     gridtype='structured',
                                     gridprops=None,
                                     keep_vars=None,
                                     add_northsea=True,
                                     anisotropy=10,
                                     fill_value_kh=1.,
                                     fill_value_kv=0.1,
                                     cachedir=None,
                                     use_cache=False,
                                     verbose=False):
    """ Update a model dataset with a model layer dataset. Follow these steps:
    1. add the data variables in 'keep_vars' from the model layer dataset
    to the model dataset
    2. add the attributes of the model layer dataset to the model dataset
    3. compute idomain from the bot values in the model layer dataset, add
    to model dataset
    4. compute top and bots from model layer dataset, add to model dataset
    5. compute kh, kv from model layer dataset, add to model dataset
    6. if gridtype is unstructured add top, bot and area to gridprops
    7. if add_northsea is True:
        a. get cells from modelgrid that are within the northsea, add data
        variable 'northsea' to model_ds
        b. fill top, bot, kh and kv add northsea cell by extrapolation
        c. get bathymetry (northsea depth) from jarkus. Add datavariable 
        bathymetry to model dataset


    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data, preferably without a grid definition.
    ml_layer_ds : xarray.Dataset
        dataset with model layer data corresponding to the modelgrid
    gridtype : str, optional
        type of grid, default is 'structured'
    gridprops : dictionary
        dictionary with grid properties output from gridgen.
    keep_vars : list of str
        variables in ml_layer_ds that will be used in model_ds
    add_northsea : bool, optional
        if True the nan values at the northsea are filled using the 
        bathymetry from jarkus
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 1.0.
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached resampled regis dataset is used. 
        The default is False.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data 
    """
    model_ds.attrs['gridtype'] = gridtype

    if keep_vars is None:
        raise ValueError(
            'please select variables that will be used to update model_ds')
    else:
        # update variables
        model_ds.update(ml_layer_ds[keep_vars])
        # update attributes
        _ = [model_ds.attrs.update({key: item})
             for key, item in ml_layer_ds.attrs.items()]

    model_ds = add_idomain_from_bottom_to_dataset(ml_layer_ds['bot'],
                                                  model_ds)

    model_ds = add_top_bot_to_model_ds(ml_layer_ds, model_ds,
                                       gridtype=gridtype)

    model_ds = add_kh_kv_from_ml_layer_to_dataset(ml_layer_ds,
                                                  model_ds,
                                                  anisotropy,
                                                  fill_value_kh,
                                                  fill_value_kv,
                                                  verbose=verbose)

    if gridtype == 'unstructured':
        gridprops['top'] = model_ds['top'].data
        gridprops['botm'] = model_ds['bot'].data

        # add surface area of each cell
        model_ds['area'] = ('cid', gridprops.pop('area')
                            [:len(model_ds['cid'])])

    else:
        gridprops = None

    if add_northsea:
        if verbose:
            print(
                'nan values at the northsea are filled using the bathymetry from jarkus')

        modelgrid = modelgrid_from_model_ds(model_ds, gridprops=gridprops)

        # find grid cells with northsea
        model_ds = jarkus.get_modelgrid_sea(model_ds,
                                            modelgrid=modelgrid,
                                            cachedir=cachedir,
                                            use_cache=use_cache,
                                            verbose=verbose)

        # fill top, bot, kh, kv at sea cells
        fill_mask = (model_ds['first_active_layer']
                     == model_ds.nodata) * model_ds['northsea']
        model_ds = fill_top_bot_kh_kv_at_mask(model_ds, fill_mask,
                                              gridtype=gridtype,
                                              gridprops=gridprops)

        # add bathymetry noordzee
        model_ds = jarkus.get_modelgrid_bathymetry(model_ds,
                                                   gridprops=gridprops,
                                                   cachedir=cachedir,
                                                   use_cache=use_cache,
                                                   verbose=verbose)

        model_ds = jarkus.add_bathymetry_to_top_bot_kh_kv(model_ds,
                                                          model_ds['bathymetry'],
                                                          fill_mask)

        # update idomain on adjusted tops and bots
        model_ds['thickness'] = get_thickness_from_topbot(model_ds['top'],
                                                          model_ds['bot'])
        model_ds['idomain'] = update_idomain_from_thickness(model_ds['idomain'],
                                                            model_ds['thickness'],
                                                            model_ds['northsea'])
        model_ds['first_active_layer'] = get_first_active_layer_from_idomain(
            model_ds['idomain'])

        if gridtype == 'unstructured':
            gridprops['top'] = model_ds['top'].data
            gridprops['botm'] = model_ds['bot'].data

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


def get_xy_mid_structured(extent, delr, delc):
    """Calculates the x and y coordinates of the cell centers of a 
    structured grid.

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
        x-coordinates of the cell centers shape(ncol)
    ymid : np.array
        y-coordinates of the cell centers shape(nrow)

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


def create_unstructured_grid(gridgen_ws, model_name, gwf=None,
                             refine_features=None, extent=None,
                             nlay=None, nrow=None, ncol=None,
                             delr=None, delc=None,
                             cachedir=None,
                             use_cache=False,
                             verbose=False):
    """Create unstructured grid. Refine grid using refinement features.     

    Parameters
    ----------
    gridgen_ws : str
        directory to save gridgen files.
    model_name : str
        name of the model.
    gwf : flopy.mf6.ModflowGwf
        groundwater flow model, if structured grid is already defined
        parameters defining the grid are taken from modelgrid if not 
        explicitly passed.
    refine_features : list of tuples, optional
        list of tuples containing refinement features, tuples must each 
        contain [(geometry, shape_type, level)]. Geometry can be a path 
        pointing to a shapefile or an object defining the geometry. 
        For accepted types for each entry, see 
        `flopy.utils.gridgen.Gridgen.add_refinement_features()`
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    nlay : int, optional
        number of model layers. If not passed, 
    nrow : int, optional
        number of model rows.
    ncol : int, optional
        number of model columns
    delr : int or float, optional
        cell size along rows of the desired grid (dx).
    delc : int or float, optional
        cell size along columns of the desired grid (dy).
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached resampled regis dataset is used. 
        The default is False.

    Returns
    -------
    gridprops : dictionary
        gridprops with the unstructured grid information.

    """

    if not os.path.isdir(gridgen_ws):
        os.makedirs(gridgen_ws)

    if cachedir is None:
        cachedir = tempfile.gettempdir()

    fname_gridprops_pickle = os.path.join(cachedir, 'gridprops.pklz')
    if os.path.isfile(fname_gridprops_pickle) and use_cache:
        if verbose:
            print(f'using cached griddata from file {fname_gridprops_pickle}')

        with open(fname_gridprops_pickle, 'rb') as fo:
            gridprops = pickle.load(fo)

        return gridprops

    if verbose:
        print('create unstructured grid using gridgen')

    # if existing structured grid, take parameters from grid if not
    # explicitly passed
    if gwf is not None:
        if gwf.modelgrid.grid_type == "structured":
            nlay = gwf.modelgrid.nlay if nlay is None else nlay
            nrow = gwf.modelgrid.nrow if nrow is None else nrow
            ncol = gwf.modelgrid.ncol if ncol is None else ncol
            delr = gwf.modelgrid.delr if delr is None else delr
            delc = gwf.modelgrid.delc if delc is None else delc
            extent = gwf.modelgrid.extent if extent is None else extent

    # create temporary groundwaterflow model with dis package
    if gwf is not None:
        _gwf_temp = copy.deepcopy(gwf)
    else:
        _sim_temp = flopy.mf6.MFSimulation()
        _gwf_temp = flopy.mf6.MFModel(_sim_temp)
    _dis_temp = flopy.mf6.ModflowGwfdis(_gwf_temp, pname='dis',
                                        nlay=nlay, nrow=nrow, ncol=ncol,
                                        delr=delr, delc=delc,
                                        xorigin=extent[0],
                                        yorigin=extent[2],
                                        filename='{}.dis'.format(model_name))

    exe_name = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'bin', 'gridgen')
    if sys.platform.startswith('win'):
        exe_name += ".exe"

    g = Gridgen(_dis_temp, model_ws=gridgen_ws, exe_name=exe_name)

    if refine_features is not None:
        for shp_fname, shp_type, lvl in refine_features:
            if isinstance(shp_fname, str):
                shp_fname = os.path.relpath(shp_fname, gridgen_ws)
                if shp_fname.endswith('.shp'):
                    shp_fname = shp_fname[:-4]
            g.add_refinement_features(shp_fname, shp_type, lvl, range(nlay))

    g.build()

    gridprops = g.get_gridprops_disv()
    gridprops['area'] = g.get_area()

    if verbose:
        print(f'write cache for griddata data to {fname_gridprops_pickle}')

    with open(fname_gridprops_pickle, 'wb') as fo:
        pickle.dump(gridprops, fo)

    return gridprops


def get_xyi_cid(gridprops=None, model_ds=None):
    """ Get x and y coördinates of the cell mids from the cellids in the grid
    properties.


    Parameters
    ----------
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. If gridprops is
        None xyi and cid will be obtained from model_ds.
    model_ds : xarray.Dataset
        dataset with model data. Should have dimension (layer, cid).

    Returns
    -------
    xyi : numpy.ndarray
        array with x and y coördinates of cell centers, shape(len(cid), 2).
    cid : numpy.ndarray
        array with cellids, shape(len(cid))
    """
    if not gridprops is None:
        xc_gwf = [cell2d[1] for cell2d in gridprops['cell2d']]
        yc_gwf = [cell2d[2] for cell2d in gridprops['cell2d']]
        xyi = np.vstack((xc_gwf, yc_gwf)).T
        cid = np.array([c[0] for c in gridprops['cell2d']])
    elif not model_ds is None:
        xyi = np.array(list(zip(model_ds.x.values, model_ds.y.values)))
        cid = model_ds.cid.values
    else:
        raise ValueError('either gridprops or model_ds should be specified')

    return xyi, cid


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

    rows = cellids[-2]
    columns = cellids[-1]

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
        tuple with indices of the cells that will be used to create the list
        with values for a column. There are 2 options:
            1. cellids contains (layers, cids)
            2. cellids contains (cids)
    model_ds : xarray.Dataset
        dataset with model data. Should have dimensions (layer, cid).
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,cid)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]
            riv: stage [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        ((layer,cid), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]
            riv: conductacnt [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        ((layer,cid), col1, col2) for each row.

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
        rec_list = list(zip(zip(layers, cellids[-1])))
    elif (col1 is not None) and col2 is None:
        col1_lst = col_to_list(col1, model_ds, cellids)
        rec_list = list(zip(zip(layers, cellids[-1]),
                            col1_lst))
    elif (col2 is not None) and col3 is None:
        col1_lst = col_to_list(col1, model_ds, cellids)
        col2_lst = col_to_list(col2, model_ds, cellids)
        rec_list = list(zip(zip(layers, cellids[-1]),
                            col1_lst, col2_lst))
    elif (col3 is not None):
        col1_lst = col_to_list(col1, model_ds, cellids)
        col2_lst = col_to_list(col2, model_ds, cellids)
        col3_lst = col_to_list(col3, model_ds, cellids)
        rec_list = list(zip(zip(layers, cellids[-1]),
                            col1_lst, col2_lst, col3_lst))
    else:
        raise ValueError(
            'invalid combination of values for col1, col2 and col3')

    return rec_list


def data_array_2d_unstr_to_rec_list(model_ds, mask,
                                    col1=None, col2=None, col3=None,
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
            riv: bottom [L]
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

    rec_list = lcid_to_rec_list(layers, cellids, model_ds,
                                col1, col2, col3)

    return rec_list


def data_array_1d_unstr_to_rec_list(model_ds, mask,
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
            riv: bottom [L]
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
        data array that is filled with polygon data

    Returns
    -------
    area_array : xarray.DataArray
        area of polygon within each modelgrid cell

    """
    if polygon.type != 'Polygon':
        raise TypeError(
            f'input geometry should by of type "Polygon" not {polygon.type}')

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
    gdf : geopandas.GeoDataFrame or shapely.geometry
        shapes that will be rasterised.
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

    if isinstance(gdf, gpd.GeoDataFrame):
        geoms = gdf.geometry.values
    elif isinstance(gdf, shapely.geometry.base.BaseGeometry):
        geoms = [gdf]

    for geom in geoms:
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

    lowest_bottom = ml_layer_ds['bot'].data[-1].copy()
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
            val = np.nanmin(ml_layer_ds['bot'].data[:, i])
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
    top_bot_raw[1:-1] = ml_layer_ds['bot'].data[:-1].copy()
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
    lowest_bottom = ml_layer_ds['bot'].data[-1].copy()
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
            val = np.nanmin(ml_layer_ds['bot'].data[:, row, col])
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
    top_bot_raw[1:-1] = ml_layer_ds['bot'].data[:-1].copy()
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
        1. the cell is True in fill_mask
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
        fill_function = resample.fillnan_dataarray_structured_grid
        fill_function_kwargs = {}
    elif gridtype == 'unstructured':
        fill_function = resample.fillnan_dataarray_unstructured_grid
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
