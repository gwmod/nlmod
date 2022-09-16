# -*- coding: utf-8 -*-
"""Module containing model grid functions.

-   project data on different grid types
-   obtain various types of rec_lists from a grid that
    can be used as input for a MODFLOW package
-   fill, interpolate and resample grid data
"""
import logging
import flopy
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.gridgen import Gridgen
from flopy.utils.gridintersect import GridIntersect
from shapely.prepared import prep
from tqdm import tqdm
from scipy.interpolate import griddata

from shapely.geometry import Point
from .. import util
from .mlayers import set_idomain, get_first_active_layer_from_idomain
from .resample import (
    get_resampled_ml_layer_ds_vertex,
    affine_transform_gdf,
    get_affine_world_to_mod,
)
from .rdp import rdp

logger = logging.getLogger(__name__)


def xy_to_icell2d(xy, ds):
    """get the icell2d value of a point defined by its x and y coordinates.

    Parameters
    ----------
    xy : list, tuple
        coordinates of ta point.
    ds : xarary dataset
        model dataset.

    Returns
    -------
    icell2d : int
        number of the icell2d value of a cell containing the xy point.

    """

    icell2d = (np.abs(ds.x - xy[0]) + np.abs(ds.y - xy[1])).argmin().item()

    return icell2d


def modelgrid_from_ds(ds, rotated=True):
    """Get flopy modelgrid from ds.

    Parameters
    ----------
    ds : xarray DataSet
        model dataset.

    Returns
    -------
    modelgrid : StructuredGrid, VertexGrid
        grid information.
    """
    if rotated and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        xoff = ds.attrs["xorigin"]
        yoff = ds.attrs["yorigin"]
        angrot = ds.attrs["angrot"]
    else:
        if ds.gridtype == "structured":
            xoff = ds.extent[0]
            yoff = ds.extent[2]
        else:
            xoff = 0.0
            yoff = 0.0
        angrot = 0.0

    if ds.gridtype == "structured":
        if not isinstance(ds.extent, (tuple, list, np.ndarray)):
            raise TypeError(
                f"extent should be a list, tuple or numpy array, not {type(ds.extent)}"
            )
        delc = np.array([ds.delc] * ds.dims["y"])
        delr = np.array([ds.delr] * ds.dims["x"])
        modelgrid = StructuredGrid(
            delc=delc,
            delr=delr,
            xoff=xoff,
            yoff=yoff,
            angrot=angrot,
        )
    elif ds.gridtype == "vertex":
        vertices = get_vertices_from_ds(ds)
        cell2d = get_cell2d_from_ds(ds)
        modelgrid = VertexGrid(
            vertices=vertices,
            cell2d=cell2d,
            xoff=xoff,
            yoff=yoff,
            angrot=angrot,
        )
    return modelgrid


def modelgrid_to_vertex_ds(mg, ds, nodata=-1):
    """Add information about the calculation-grid to a model dataset"""
    # add modelgrid to ds
    ds["xv"] = ("iv", mg.verts[:, 0])
    ds["yv"] = ("iv", mg.verts[:, 1])

    cell2d = mg.cell2d
    ncvert_max = np.max([x[3] for x in cell2d])
    icvert = np.full((mg.ncpl, ncvert_max), nodata)
    for i in range(mg.ncpl):
        icvert[i, : cell2d[i][3]] = cell2d[i][4:]
    ds["icvert"] = ("icell2d", "icv"), icvert
    ds["icvert"].attrs["_FillValue"] = nodata
    return ds


def gridprops_to_vertex_ds(gridprops, ds, nodata=-1):
    """Gridprops is a dictionairy containing keyword arguments needed to generate
    a flopy modelgrid instance"""
    ds["xv"] = ("iv", [i[1] for i in gridprops["vertices"]])
    ds["yv"] = ("iv", [i[2] for i in gridprops["vertices"]])

    cell2d = gridprops["cell2d"]
    ncvert_max = np.max([x[3] for x in cell2d])
    icvert = np.full((gridprops["ncpl"], ncvert_max), nodata)
    for i in range(gridprops["ncpl"]):
        icvert[i, : cell2d[i][3]] = cell2d[i][4:]
    ds["icvert"] = ("icell2d", "icv"), icvert
    ds["icvert"].attrs["_FillValue"] = nodata
    return ds


def get_vertices_from_ds(ds):
    """Get the vertices-list from a model dataset. Flopy needs needs this list
    to build a disv-package"""
    vertices = list(zip(ds["iv"].data, ds["xv"].data, ds["yv"].data))
    return vertices


def get_cell2d_from_ds(ds):
    """Get the cell2d-list from a model dataset. Flopy needs this list to build
    a disv-package"""
    icell2d = ds["icell2d"].data
    x = ds["x"].data
    y = ds["y"].data
    icvert = ds["icvert"].data
    cell2d = []
    nodata = ds["icvert"].attrs["_FillValue"]
    for i, cid in enumerate(icell2d):
        mask = ds["icvert"].data[i] != nodata
        cell2d.append((cid, x[i], y[i], mask.sum(), *icvert[i][mask]))
    return cell2d


def refine(
    ds,
    model_ws=None,
    refinement_features=None,
    exe_name=None,
    remove_nan_layers=True,
    model_coordinates=False,
):
    """
    Refine the grid (discretization by vertices, disv), using Gridgen

    Parameters
    ----------
    ds : xarray.Datset
        A structured model datset.
    model_ws : str, optional
        The working directory fpr GridGen. Get from ds when model_ws is None.
        The default is None.
    refinement_features : list of tuple of length 2, optional
        List of tuples containing refinement features. Each tuple must be of
        the form (GeoDataFrame, level) or (geometry, shape_type, level). The
        default is None.
    exe_name : str, optional
        Filepath to the gridgen executable. The file path within nlmod is chose
        if exe_name is None. The default is None.
    remove_nan_layers : bool, optional
        if True layers that are inactive everywhere are removed from the model.
        If False nan layers are kept which might be usefull if you want
        to keep some layers that exist in other models. The default is True.
    model_coordinates : bool, optional
        When model_coordinates is True, the features supplied in refinement features are
        allready in model-coordinates. Only used when a grid is rotated. The default is
        False.

    Returns
    -------
    xarray.Dataset
        The refined model dataset.

    """
    assert "icell2d" not in ds.dims
    logger.info("create vertex grid using gridgen")
    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.MFModel(sim)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nrow=len(ds.y),
        ncol=len(ds.x),
        delr=ds.delr,
        delc=ds.delc,
        xorigin=ds.extent[0],
        yorigin=ds.extent[2],
    )
    if exe_name is None:
        exe_name = util.get_exe_path("gridgen")

    if model_ws is None:
        model_ws = ds.model_ws
    g = Gridgen(dis, model_ws=model_ws, exe_name=exe_name)

    ds_has_rotation = "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0
    if model_coordinates:
        if not ds_has_rotation:
            raise (Exception("The supplied shapes need to be in realworld coordinates"))
    elif ds_has_rotation:
        affine_matrix = get_affine_world_to_mod(ds).to_shapely()

    if refinement_features is not None:
        for refinement_feature in refinement_features:
            if len(refinement_feature) == 3:
                # the feature is a file or a list of geometries
                fname, geom_type, level = refinement_feature
                if not model_coordinates and ds_has_rotation:
                    raise (
                        Exception("Converting files to model coordinates not supported")
                    )
                g.add_refinement_features(fname, geom_type, level, layers=[0])
            elif len(refinement_feature) == 2:
                # the feature is a geodataframe
                gdf, level = refinement_feature
                if not model_coordinates and ds_has_rotation:
                    gdf = affine_transform_gdf(gdf, affine_matrix)
                geom_types = gdf.geom_type.str.replace("Multi", "")
                geom_types = geom_types.str.replace("String", "")
                geom_types = geom_types.str.lower()
                for geom_type in geom_types.unique():
                    if flopy.__version__ == "3.3.5" and geom_type == "line":
                        # a bug in flopy that is fixed in the dev branch
                        raise (
                            Exception(
                                "geom_type line is buggy in flopy 3.3.5. "
                                "See https://github.com/modflowpy/flopy/issues/1405"
                            )
                        )
                    mask = geom_types == geom_type
                    features = [gdf[mask].unary_union]
                    g.add_refinement_features(features, geom_type, level, layers=[0])
    g.build()
    gridprops = g.get_gridprops_disv()
    gridprops["area"] = g.get_area()
    ds = get_resampled_ml_layer_ds_vertex(ds, gridprops=gridprops)
    # recalculate idomain, as the interpolation changes idomain to floats
    ds = set_idomain(ds, remove_nan_layers=remove_nan_layers)
    return ds


def col_to_list(col_in, ds, cellids):
    """Convert array data in ds to a list of values for specific cells.

    This function is typically used to create a rec_array with stress period
    data for the modflow packages. Can be used for structured and
    vertex grids.

    Parameters
    ----------
    col_in : str, int or float
        if col_in is a str type it is the name of the column in ds.
        if col_in is an int or a float it is a value that will be used for all
        cells in cellids.
    ds : xarray.Dataset
        dataset with model data. Can have dimension (layer, y, x) or
        (layer, icell2d).
    cellids : tuple of numpy arrays
        tuple with indices of the cells that will be used to create the list
        with values. There are 3 options:
            1.   cellids contains (layers, rows, columns)
            2.   cellids contains (rows, columns) or (layers, icell2ds)
            3.   cellids contains (icell2ds)

    Raises
    ------
    ValueError
        raised if the cellids are in the wrong format.

    Returns
    -------
    col_lst : list
        raster values from ds presented in a list per cell.
    """

    if isinstance(col_in, str):
        if len(cellids) == 3:
            # 3d grid
            col_lst = [
                ds[col_in].data[lay, row, col]
                for lay, row, col in zip(cellids[0], cellids[1], cellids[2])
            ]
        elif len(cellids) == 2:
            # 2d grid or vertex 3d grid
            col_lst = [
                ds[col_in].data[row, col]
                for row, col in zip(cellids[0], cellids[1])
            ]
        elif len(cellids) == 1:
            # 2d vertex grid
            col_lst = ds[col_in].data[cellids[0]]
        else:
            raise ValueError(f"could not create a column list for col_in={col_in}")
    else:
        col_lst = [col_in] * len(cellids[0])

    return col_lst


def lrc_to_rec_list(
    layers, rows, columns, cellids, ds, col1=None, col2=None, col3=None
):
    """Create a rec list for stress period data from a set of cellids.

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
    ds : xarray.Dataset
        dataset with model data. Can have dimension (layer, y, x) or
        (layer, icell2d).
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
        col1_lst = col_to_list(col1, ds, cellids)
        rec_list = list(zip(zip(layers, rows, columns), col1_lst))
    elif (col2 is not None) and col3 is None:
        col1_lst = col_to_list(col1, ds, cellids)
        col2_lst = col_to_list(col2, ds, cellids)
        rec_list = list(zip(zip(layers, rows, columns), col1_lst, col2_lst))
    elif col3 is not None:
        col1_lst = col_to_list(col1, ds, cellids)
        col2_lst = col_to_list(col2, ds, cellids)
        col3_lst = col_to_list(col3, ds, cellids)
        rec_list = list(zip(zip(layers, rows, columns), col1_lst, col2_lst, col3_lst))
    else:
        raise ValueError("invalid combination of values for col1, col2 and col3")

    return rec_list


def data_array_3d_to_rec_list(
    ds, mask, col1=None, col2=None, col3=None, only_active_cells=True
):
    """Create a rec list for stress period data from a model dataset.

    Used for structured grids.


    Parameters
    ----------
    ds : xarray.Dataset
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
        cellids = np.where((mask) & (ds["idomain"] == 1))
    else:
        cellids = np.where(mask)

    layers = cellids[0]
    rows = cellids[1]
    columns = cellids[2]

    rec_list = lrc_to_rec_list(
        layers, rows, columns, cellids, ds, col1, col2, col3
    )

    return rec_list


def data_array_2d_to_rec_list(
    ds,
    mask,
    col1=None,
    col2=None,
    col3=None,
    layer=0,
    first_active_layer=False,
    only_active_cells=True,
):
    """Create a rec list for stress period data from a model dataset.

    Used for structured grids.


    Parameters
    ----------
    ds : xarray.Dataset
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
        if "first_active_layer" not in ds:
            ds["first_active_layer"] = get_first_active_layer_from_idomain(
                ds["idomain"]
            )

        cellids = np.where((mask) & (ds["first_active_layer"] != ds.nodata))
        layers = col_to_list("first_active_layer", ds, cellids)
    elif only_active_cells:
        cellids = np.where((mask) & (ds["idomain"][layer] == 1))
        layers = col_to_list(layer, ds, cellids)
    else:
        cellids = np.where(mask)
        layers = col_to_list(layer, ds, cellids)

    rows = cellids[-2]
    columns = cellids[-1]

    rec_list = lrc_to_rec_list(
        layers, rows, columns, cellids, ds, col1, col2, col3
    )

    return rec_list


def lcid_to_rec_list(layers, cellids, ds, col1=None, col2=None, col3=None):
    """Create a rec list for stress period data from a set of cellids.

    Used for vertex grids.


    Parameters
    ----------
    layers : list or numpy.ndarray
        list with the layer for each cell in the rec_list.
    cellids : tuple of numpy arrays
        tuple with indices of the cells that will be used to create the list
        with values for a column. There are 2 options:
            1. cellids contains (layers, cids)
            2. cellids contains (cids)
    ds : xarray.Dataset
        dataset with model data. Should have dimensions (layer, icell2d).
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,icell2d)) for each row. col1 should be the following value for
        each package (can also be the name of a timeseries):
        -   rch: recharge [L/T]
        -   ghb: head [L]
        -   drn: drain level [L]
        -   chd: head [L]
        -   riv: stage [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        ((layer,icell2d), col1) for each row. col2 should be the following
        value for each package (can also be the name of a timeseries):
        -   ghb: conductance [L^2/T]
        -   drn: conductance [L^2/T]
        -   riv: conductacnt [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        ((layer,icell2d), col1, col2) for each row. col3 should be the following
        value for each package (can also be the name of a timeseries):
        -   riv: bottom [L]

    Raises
    ------
    ValueError
        Question: will this error ever occur?.

    Returns
    -------
    rec_list : list of tuples
        every row consist of ((layer, icell2d), col1, col2, col3)
        grids.
    """
    if col1 is None:
        rec_list = list(zip(zip(layers, cellids[-1])))
    elif (col1 is not None) and col2 is None:
        col1_lst = col_to_list(col1, ds, cellids)
        rec_list = list(zip(zip(layers, cellids[-1]), col1_lst))
    elif (col2 is not None) and col3 is None:
        col1_lst = col_to_list(col1, ds, cellids)
        col2_lst = col_to_list(col2, ds, cellids)
        rec_list = list(zip(zip(layers, cellids[-1]), col1_lst, col2_lst))
    elif col3 is not None:
        col1_lst = col_to_list(col1, ds, cellids)
        col2_lst = col_to_list(col2, ds, cellids)
        col3_lst = col_to_list(col3, ds, cellids)
        rec_list = list(zip(zip(layers, cellids[-1]), col1_lst, col2_lst, col3_lst))
    else:
        raise ValueError("invalid combination of values for col1, col2 and col3")

    return rec_list


def data_array_2d_vertex_to_rec_list(
    ds, mask, col1=None, col2=None, col3=None, only_active_cells=True
):
    """Create a rec list for stress period data from a model dataset.

    Used for vertex grids.


    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data and dimensions (layer, icell2d)
    mask : xarray.DataArray for booleans
        True for the cells that will be used in the rec list.
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,icell2d)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        (((layer,icell2d), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        (((layer,icell2d), col1, col2) for each row.

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
        cellids = np.where((mask) & (ds["idomain"] == 1))
    else:
        cellids = np.where(mask)

    layers = cellids[0]

    rec_list = lcid_to_rec_list(layers, cellids, ds, col1, col2, col3)

    return rec_list


def data_array_1d_vertex_to_rec_list(
    ds,
    mask,
    col1=None,
    col2=None,
    col3=None,
    layer=0,
    first_active_layer=False,
    only_active_cells=True,
):
    """Create a rec list for stress period data from a model dataset.

    Used for vertex grids.


    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data and dimensions (layer, icell2d)
    mask : xarray.DataArray for booleans
        True for the cells that will be used in the rec list.
    col1 : str, int or float, optional
        1st column of the rec_list, if None the rec_list will be a list with
        ((layer,icell2d)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]

    col2 : str, int or float, optional
        2nd column of the rec_list, if None the rec_list will be a list with
        (((layer,icell2d), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the rec_list, if None the rec_list will be a list with
        (((layer,icell2d), col1, col2) for each row.

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
        every row consist of ((layer,icell2d), col1, col2, col3).
    """
    if first_active_layer:
        cellids = np.where((mask) & (ds["first_active_layer"] != ds.nodata))
        layers = col_to_list("first_active_layer", ds, cellids)
    elif only_active_cells:
        cellids = np.where((mask) & (ds["idomain"][layer] == 1))
        layers = col_to_list(layer, ds, cellids)
    else:
        cellids = np.where(mask)
        layers = col_to_list(layer, ds, cellids)

    rec_list = lcid_to_rec_list(layers, cellids, ds, col1, col2, col3)

    return rec_list


def polygon_to_area(modelgrid, polygon, da, gridtype="structured"):
    """create a grid with the surface area in each cell based on a polygon
    value.

    Parameters
    ----------
    modelgrid : flopy.discretization.structuredgrid.StructuredGrid
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
    if polygon.type == "Polygon":
        pass
    elif polygon.type == "MultiPolygon":
        Warning(
            "function not tested for MultiPolygon type, can have unexpected results"
        )
    else:
        raise TypeError(
            f'input geometry should by of type "Polygon" not {polygon.type}'
        )

    ix = GridIntersect(modelgrid, method="vertex")
    opp_cells = ix.intersect(polygon)

    if gridtype == "structured":
        area_array = util.get_da_from_da_ds(da, dims=("y", "x"), data=0)
        for opp_row in opp_cells:
            area = opp_row[-2]
            area_array[opp_row[0][0], opp_row[0][1]] = area
    elif gridtype == "vertex":
        area_array = util.get_da_from_da_ds(da, dims=("icell2d",), data=0)
        cids = opp_cells.cellids
        area = opp_cells.areas
        area_array[cids.astype(int)] = area

    return area_array


def gdf2data_array_struc(gdf, gwf, field="VALUE", agg_method=None, interp_method=None):
    """Project vector data on a structured grid. Aggregate data if multiple
    geometries are in a single cell

    Parameters
    ----------
    gdf : geopandas.GeoDataframe
        vector data can only contain a single geometry type.
    gwf : flopy groundwater flow model
        model with a structured grid.
    field : str, optional
        column name in the geodataframe. The default is 'VALUE'.
    interp_method : str or None, optional
        method to obtain values in cells without geometry by interpolating
        between cells with values. Options are 'nearest' and 'linear'.
    agg_method : str, optional
        aggregation method to handle multiple geometries in one cell, options
        are:
        - max, min, mean,
        - length_weighted (lines), max_length (lines),
        - area_weighted (polygon), area_max (polygon).
        The default is 'max'.

    Returns
    -------
    da : xarray DataArray
        DESCRIPTION.

    """
    x = gwf.modelgrid.get_xcellcenters_for_layer(0)[0]
    y = gwf.modelgrid.get_ycellcenters_for_layer(0)[:, 0]
    da = xr.DataArray(np.nan, dims=("y", "x"), coords={"y": y, "x": x})

    # interpolate data
    if interp_method is not None:
        arr = interpolate_gdf_to_array(gdf, gwf, field=field, method=interp_method)
        da.values = arr

        return da

    gdf_cellid = gdf2grid(gdf, gwf, "vertex")

    if gdf_cellid.cellid.duplicated().any():
        # aggregate data
        if agg_method is None:
            raise ValueError(
                "multiple geometries in one cell please define aggregation method"
            )
        gdf_agg = aggregate_vector_per_cell(gdf_cellid, {field: agg_method}, gwf)
    else:
        # aggregation not neccesary
        gdf_agg = gdf_cellid[[field]]
        gdf_agg.set_index(
            pd.MultiIndex.from_tuples(gdf_cellid.cellid.values), inplace=True
        )

    for ind, row in gdf_agg.iterrows():
        da.values[ind[0], ind[1]] = row[field]

    return da


def interpolate_gdf_to_array(gdf, gwf, field="values", method="nearest"):
    """interpolate data from a point gdf


    Parameters
    ----------
    gdf : geopandas.GeoDataframe
        vector data can only contain a single geometry type.
    gwf : flopy groundwater flow model
        model with a structured grid.
    field : str, optional
        column name in the geodataframe. The default is 'values'.
    method : str or None, optional
        method to obtain values in cells without geometry by interpolating
        between cells with values. Options are 'nearest' and 'linear'.

    Returns
    -------
    arr : np.array
        numpy array with interpolated data.

    """
    # check geometry
    geom_types = gdf.geometry.type.unique()
    if geom_types[0] != "Point":
        raise NotImplementedError("can only use interpolation with point geometries")

    # check field
    if field not in gdf.columns:
        raise ValueError(f"Missing column in DataFrame: {field}")

    points = np.array([[g.x, g.y] for g in gdf.geometry])
    values = gdf[field].values
    xi = np.vstack(
        (
            gwf.modelgrid.xcellcenters.flatten(),
            gwf.modelgrid.ycellcenters.flatten(),
        )
    ).T
    vals = griddata(points, values, xi, method=method)
    arr = np.reshape(vals, (gwf.modelgrid.nrow, gwf.modelgrid.ncol))

    return arr


def _agg_max_area(gdf, col):
    return gdf.loc[gdf.area.idxmax(), col]


def _agg_area_weighted(gdf, col):
    nanmask = gdf[col].isna()
    aw = (gdf.area * gdf[col]).sum(skipna=True) / gdf.loc[~nanmask].area.sum()
    return aw


def _agg_max_length(gdf, col):
    return gdf.loc[gdf.length.idxmax(), col]


def _agg_length_weighted(gdf, col):
    nanmask = gdf[col].isna()
    aw = (gdf.length * gdf[col]).sum(skipna=True) / gdf.loc[~nanmask].length.sum()
    return aw


def _agg_nearest(gdf, col, gwf):
    cid = gdf["cellid"].values[0]
    cellcenter = Point(
        gwf.modelgrid.xcellcenters[0][cid[1]],
        gwf.modelgrid.ycellcenters[:, 0][cid[0]],
    )
    val = gdf.iloc[gdf.distance(cellcenter).argmin()].loc[col]
    return val


def _get_aggregates_values(group, fields_methods, gwf=None):

    agg_dic = {}
    for field, method in fields_methods.items():
        # aggregation is only necesary if group shape is greater than 1
        if group.shape[0] == 1:
            agg_dic[field] = group[field].values[0]
        if method == "max":
            agg_dic[field] = group[field].max()
        elif method == "min":
            agg_dic[field] = group[field].min()
        elif method == "mean":
            agg_dic[field] = group[field].mean()
        elif method == "nearest":
            agg_dic[field] = _agg_nearest(group, field, gwf)
        elif method == "length_weighted":  # only for lines
            agg_dic[field] = _agg_length_weighted(group, field)
        elif method == "max_length":  # only for lines
            agg_dic[field] = _agg_max_length(group, field)
        elif method == "area_weighted":  # only for polygons
            agg_dic[field] = _agg_area_weighted(group, field)
        elif method == "max_area":  # only for polygons
            agg_dic[field] = _agg_max_area(group, field)
        elif method == "center_grid":  # only for polygons
            raise NotImplementedError
        else:
            raise ValueError(f"Method '{method}' not recognized!")

    return agg_dic


def aggregate_vector_per_cell(gdf, fields_methods, gwf=None):
    """Aggregate vector features per cell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing points, lines or polygons per grid cell.
    fields_methods: dict
        fields (keys) in the Geodataframe with their aggregation method (items)
        aggregation methods can be:
        max, min, mean, length_weighted (lines), max_length (lines),
        area_weighted (polygon), area_max (polygon).
    gwf : flopy Groundwater flow model
        only necesary if one of the field methods is 'nearest'

    Returns
    -------
    celldata : pd.DataFrame
        DataFrame with aggregated surface water parameters per grid cell
    """
    # check geometry types
    geom_types = gdf.geometry.type.unique()
    if len(geom_types) > 1:
        if (
            len(geom_types) == 2
            and ("Polygon" in geom_types)
            and ("MultiPolygon" in geom_types)
        ):
            pass
        else:
            raise TypeError("cannot aggregate geometries of different types")
    if bool({"length_weighted", "max_length"} & set(fields_methods.values())):
        assert (
            geom_types[0] == "LineString"
        ), "can only use length methods with line geometries"
    if bool({"area_weighted", "max_area"} & set(fields_methods.values())):
        if ("Polygon" in geom_types) or ("MultiPolygon" in geom_types):
            pass
        else:
            raise TypeError("can only use area methods with polygon geometries")

    # check fields
    missing_cols = set(fields_methods.keys()).difference(gdf.columns)
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # aggregate data
    gr = gdf.groupby(by="cellid")
    celldata = pd.DataFrame(index=gr.groups.keys())
    for cid, group in tqdm(gr, desc="Aggregate vector data"):
        agg_dic = _get_aggregates_values(group, fields_methods, gwf)
        for key, item in agg_dic.items():
            celldata.loc[cid, key] = item

    return celldata


def gdf_to_bool_data_array(gdf, mfgrid, ds):
    """convert a GeoDataFrame with polygon geometries into a data array
    corresponding to the modelgrid in which each cell is 1 (True) if one or
    more geometries are (partly) in that cell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or shapely.geometry
        shapes that will be rasterised.
    mfgrid : flopy grid
        model grid.
    ds : xr.DataSet
        xarray with model data

    Returns
    -------
    da : xr.DataArray
        1 if polygon is in cell, 0 otherwise. Grid dimensions according to
        ds and mfgrid.
    """

    # build list of gridcells
    ix = GridIntersect(mfgrid, method="vertex")

    if ds.gridtype == "structured":
        da = util.get_da_from_da_ds(ds, dims=("y", "x"), data=0)
    elif ds.gridtype == "vertex":
        da = util.get_da_from_da_ds(ds, dims=("icell2d",), data=0)
    else:
        raise ValueError("function only support structured or vertex gridtypes")

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

        if ds.gridtype == "structured":
            for cid in cids:
                da[cid[0], cid[1]] = 1
        elif ds.gridtype == "vertex":
            da[cids] = 1

    return da


def gdf_to_bool_dataset(ds, gdf, mfgrid, da_name):
    """convert a GeoDataFrame with polygon geometries into a model dataset with
    a data_array named 'da_name' in which each cell is 1 (True) if one or more
    geometries are (partly) in that cell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        polygon shapes with surface water.
    mfgrid : flopy grid
        model grid.
    ds : xr.DataSet
        xarray with model data

    Returns
    -------
    ds_out : xr.Dataset
        Dataset with a single DataArray, this DataArray is 1 if polygon is in
        cell, 0 otherwise. Grid dimensions according to ds and mfgrid.
    """
    ds_out = util.get_ds_empty(ds)
    ds_out[da_name] = gdf_to_bool_data_array(gdf, mfgrid, ds)

    return ds_out


def gdf2grid(
    gdf,
    ml=None,
    method="vertex",
    ix=None,
    desc="Intersecting with grid",
    **kwargs,
):
    """Cut a geodataframe gdf by the grid of a flopy modflow model ml. This
    method is just a wrapper around the GridIntersect method from flopy.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame that needs to be cut by the grid. The GeoDataFrame can
        consist of multiple types (Point, LineString, Polygon and the Multi-
        variants).
    ml : flopy.modflow.Modflow or flopy.mf6.ModflowGwf or xarray.Dataset, optional
        The flopy model or xarray dataset that defines the grid. When a Dataset is
        supplied, and the grid is rotated, the geodataframe is transformed in model
        coordinates. The default is None.
    method : string, optional
        Method passed to the GridIntersect-class. The default is 'vertex'.
    ix : flopy.utils.GridIntersect, optional
        GridIntersect, if not provided the modelgrid in ml is used.
    **kwargs : keyword arguments
        keyword arguments are passed to the intersect_*-methods.

    Returns
    -------
    geopandas.GeoDataFrame
        The GeoDataFrame with the geometries per grid-cell.
    """
    if ml is None and ix is None:
        raise (Exception("Either specify ml or ix"))
        
    if ml is not None:
        if isinstance(ml, xr.Dataset):
            ds = ml
            modelgrid = modelgrid_from_ds(ds, rotated=False)
            if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
                # transform gdf into model coordinates
                affine = get_affine_world_to_mod(ds)
                gdf = affine_transform_gdf(gdf, affine)
        else:
            modelgrid = ml.modelgrid
        
    if ix is None:
        ix = flopy.utils.GridIntersect(modelgrid, method=method)
    shps = []
    geometry = gdf._geometry_column_name
    for _, shp in tqdm(gdf.iterrows(), total=gdf.shape[0], desc=desc):
        r = ix.intersect(shp[geometry], **kwargs)
        for i in range(r.shape[0]):
            shpn = shp.copy()
            shpn["cellid"] = r["cellids"][i]
            shpn[geometry] = r["ixshapes"][i]
            shps.append(shpn)
    return gpd.GeoDataFrame(shps, geometry=geometry)


def get_thickness_from_topbot(top, bot):
    """get thickness from data arrays with top and bots.

    Parameters
    ----------
    top : xr.DataArray
        raster with top of each cell. dimensions should be (y,x) or (icell2d).
    bot : xr.DataArray
        raster with bottom of each cell. dimensions should be (layer, y,x) or
        (layer, icell2d).

    Returns
    -------
    thickness : xr.DataArray
        raster with thickness of each cell. dimensions should be (layer, y,x)
        or (layer, icell2d).
    """
    DeprecationWarning(
        "function is deprecated please use calculate_thickness function instead"
    )

    if np.ndim(top) > 2:
        raise NotImplementedError("function works only for 2d top")

    # get thickness
    if bot.ndim == 3:
        thickness = util.get_da_from_da_ds(bot, dims=("layer", "y", "x"))
    elif bot.ndim == 2:
        thickness = util.get_da_from_da_ds(bot, dims=("layer", "icell2d"))
    else:
        raise ValueError("function only support structured or vertex gridtypes")

    for lay in range(len(bot)):
        if lay == 0:
            thickness[lay] = top - bot[lay]
        else:
            thickness[lay] = bot[lay - 1] - bot[lay]

    return thickness


def get_vertices(ds, modelgrid=None, vert_per_cid=4, rotated=False):
    """get vertices of a vertex modelgrid from a ds or the modelgrid.
    Only return the 4 corners of each cell and not the corners of
    adjacent cells thus limiting the vertices per cell to 4 points.

    This method uses the xvertices and yvertices attributes of the modelgrid.
    When no modelgrid is supplied, a modelgrid-object is created from ds.

    Parameters
    ----------
    ds : xr.DataSet
        model dataset, attribute grid_type should be 'vertex'
    modelgrid : flopy.discretization.vertexgrid.VertexGrid
        vertex grid with attributes xvertices and yvertices.
    vert_per_cid : int or None:
        number of vertices per cell:
        - 4 return the 4 vertices of each cell
        - 5 return the 4 vertices of each cell + one duplicate vertex
        (sometimes useful if you want to create polygons)
        - anything else, the maximum number of vertices. For locally refined
        cells this includes all the vertices adjacent to the cell.

        if vert_per_cid is 4 or 5 vertices are removed using the
        Ramer-Douglas-Peucker Algorithm -> https://github.com/fhirschmann/rdp.

    Returns
    -------
    vertices_da : xarray DataArray
         Vertex coördinates per cell with dimensions(cid, no_vert, 2).
    """

    # obtain

    if modelgrid is None:
        modelgrid = modelgrid_from_ds(ds, rotated=rotated)
    xvert = modelgrid.xvertices
    yvert = modelgrid.yvertices
    if vert_per_cid == 4:
        vertices_arr = np.array(
            [rdp(list(zip(xvert[i], yvert[i])))[:-1] for i in range(len(xvert))]
        )
    elif vert_per_cid == 5:
        vertices_arr = np.array(
            [rdp(list(zip(xvert[i], yvert[i]))) for i in range(len(xvert))]
        )
    else:
        raise NotImplementedError()

    vertices_da = xr.DataArray(
        vertices_arr,
        dims=("icell2d", "vert_per_cid", "xy"),
        coords={"xy": ["x", "y"]},
    )

    return vertices_da
