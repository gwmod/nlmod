# -*- coding: utf-8 -*-
"""Created on Fri Apr  2 15:08:50 2021.

@author: oebbe
"""
import logging

import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.interpolate import griddata
import rasterio
from rasterio.warp import reproject
from affine import Affine

logger = logging.getLogger(__name__)


def resample_dataarray2d_to_vertex_grid(
    da_in, model_ds=None, x=None, y=None, method="nearest", **kwargs
):
    """resample a 2d dataarray (xarray) from a structured grid to a new
    dataaraay of a vertex grid.

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions (y, x). y and x are from the original
        grid
    model_ds : xarray.Dataset
        The model dataset to which the datarray needs to be resampled.
    x : numpy.ndarray
        array with x coördinate of cell centers, len(icell2d). If x is None x
        is retreived from model_ds.
    y : numpy.ndarray
        array with x coördinate of cell centers, len(icell2d). If y is None y
        is retreived from model_ds.
    method : str, optional
        type of interpolation used to resample. The default is 'nearest'.

    Returns
    -------
    da_out : xarray.DataArray
        data array with dimension (icell2d).
    """
    if x is None:
        x = model_ds["x"].data
    if y is None:
        y = model_ds["y"].data

    # get x and y values of all cells in dataarray
    mg = np.meshgrid(da_in.x.data, da_in.y.data)
    points = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    # regrid
    xyi = np.column_stack((x, y))
    arr_out = griddata(
        points, da_in.data.flatten(), xyi, method=method, **kwargs
    )

    # new dataset
    da_out = xr.DataArray(arr_out, dims=("icell2d"))

    return da_out


def resample_dataarray3d_to_vertex_grid(
    da_in, model_ds=None, x=None, y=None, method="nearest"
):
    """resample a dataarray (xarray) from a structured grid to a new dataaraay
    of a vertex grid.

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions (layer, y, x). y and x are from the original
        grid
    model_ds : xarray.Dataset
        The model dataset to which the datarray needs to be resampled.
    x : numpy.ndarray
        array with x coördinate of cell centers, len(icell2d). If x is None x
        is retreived from model_ds.
    y : numpy.ndarray
        array with x coördinate of cell centers, len(icell2d). If y is None y
        is retreived from model_ds.
    method : str, optional
        type of interpolation used to resample. The default is 'nearest'.

    Returns
    -------
    da_out : xarray.DataArray
        data array with dimensions (layer,icell2d).
    """
    if x is None:
        x = model_ds["x"].data
    if y is None:
        y = model_ds["y"].data

    # get x and y values of all cells in dataarray
    mg = np.meshgrid(da_in.x.data, da_in.y.data)
    points = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    layers = da_in.layer.data
    xyi = np.column_stack((x, y))
    arr_out = np.zeros((len(layers), len(xyi)))
    for i, lay in enumerate(layers):

        ds_lay = da_in.sel(layer=lay)

        # regrid
        arr_out[i] = griddata(
            points, ds_lay.data.flatten(), xyi, method=method
        )

    # new dataset
    da_out = xr.DataArray(
        arr_out, dims=("layer", "icell2d"), coords={"layer": layers}
    )

    return da_out


def resample_dataset_to_vertex_grid(ds_in, gridprops, method="nearest"):
    """resample a dataset (xarray) from an structured grid to a new dataset
    from a vertex grid.

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
        dataset with dimensions (layer, icell2d), icell2d are cell id's from the new
        grid.
    """

    assert isinstance(ds_in, xr.core.dataset.Dataset)

    xyi, _ = get_xyi_icell2d(gridprops)
    x = xr.DataArray(xyi[:, 0], dims=("icell2d"))
    y = xr.DataArray(xyi[:, 1], dims=("icell2d"))
    if method in ["nearest", "linear"]:
        # resample the entire dataset in one line
        return ds_in.interp(
            x=x, y=y, method=method, kwargs={"fill_value": None}
        )

    ds_out = xr.Dataset(coords={"layer": ds_in.layer.data})

    # add x and y coordinates
    ds_out["x"] = x
    ds_out["y"] = y

    # add other variables
    for data_var in ds_in.data_vars:
        if ds_in[data_var].dims == ("layer", "y", "x"):
            data_arr = resample_dataarray3d_to_vertex_grid(
                ds_in[data_var], x=x, y=y, method=method
            )
        elif ds_in[data_var].dims == ("y", "x"):
            data_arr = resample_dataarray2d_to_vertex_grid(
                ds_in[data_var], x=x, y=y, method=method
            )

        elif ds_in[data_var].dims in ("layer", ("layer",)):
            data_arr = ds_in[data_var]

        else:
            logger.warning(
                f"did not resample data array {data_var} because conversion with dimensions {ds_in[data_var].dims} is not (yet) supported"
            )
            continue

        ds_out[data_var] = data_arr

    return ds_out


def get_xyi_icell2d(gridprops=None, model_ds=None):
    """Get x and y coördinates of the cell mids from the cellids in the grid
    properties.

    Parameters
    ----------
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. If gridprops is
        None xyi and icell2d will be obtained from model_ds.
    model_ds : xarray.Dataset
        dataset with model data. Should have dimension (layer, icell2d).

    Returns
    -------
    xyi : numpy.ndarray
        array with x and y coördinates of cell centers, shape(len(icell2d), 2).
    icell2d : numpy.ndarray
        array with cellids, shape(len(icell2d))
    """
    if gridprops is not None:
        xc_gwf = [cell2d[1] for cell2d in gridprops["cell2d"]]
        yc_gwf = [cell2d[2] for cell2d in gridprops["cell2d"]]
        xyi = np.vstack((xc_gwf, yc_gwf)).T
        icell2d = np.array([c[0] for c in gridprops["cell2d"]])
    elif model_ds is not None:
        xyi = np.array(list(zip(model_ds.x.values, model_ds.y.values)))
        icell2d = model_ds.icell2d.values
    else:
        raise ValueError("either gridprops or model_ds should be specified")

    return xyi, icell2d


def get_xy_mid_structured(extent, delr, delc, descending_y=True):
    """Calculates the x and y coordinates of the cell centers of a structured
    grid.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    descending_y : bool, optional
        if True the resulting ymid array is in descending order. This is the
        default for MODFLOW models. default is True.

    Returns
    -------
    x : np.array
        x-coordinates of the cell centers shape(ncol)
    y : np.array
        y-coordinates of the cell centers shape(nrow)
    """
    # check if extent is valid
    if (extent[1] - extent[0]) % delr != 0.0:
        raise ValueError(
            "invalid extent, the extent should contain an integer"
            " number of cells in the x-direction"
        )
    if (extent[3] - extent[2]) % delc != 0.0:
        raise ValueError(
            "invalid extent, the extent should contain an integer"
            " number of cells in the y-direction"
        )

    # get cell mids
    x_mid_start = extent[0] + 0.5 * delr
    x_mid_end = extent[1] - 0.5 * delr
    y_mid_start = extent[2] + 0.5 * delc
    y_mid_end = extent[3] - 0.5 * delc

    ncol = int((extent[1] - extent[0]) / delr)
    nrow = int((extent[3] - extent[2]) / delc)

    x = np.linspace(x_mid_start, x_mid_end, ncol)
    if descending_y:
        y = np.linspace(y_mid_end, y_mid_start, nrow)
    else:
        y = np.linspace(y_mid_start, y_mid_end, nrow)

    return x, y


def resample_dataarray2d_to_structured_grid(
    da_in,
    extent=None,
    delr=None,
    delc=None,
    x=None,
    y=None,
    kind="linear",
    nan_factor=0.01,
    **kwargs,
):
    """resample a dataarray (xarray) from a structured grid to a new dataaraay
    from a different structured grid.

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions (y, x). y and x are from the original
        grid
    extent : list, tuple or np.array, optional
        extent (xmin, xmax, ymin, ymax) of the desired grid, if not defined
        x and y are used
    delr : int or float, optional
        cell size along rows of the desired grid, if not defined xmid and
        ymid are used
    delc : int or float, optional
        cell size along columns of the desired grid, if not defined xmid and
        ymid are used
    x : np.array, optional
        x coördinates of the cell centers of the desired grid shape(ncol), if
        not defined x and y are calculated from the extent, delr and delc.
    y : np.array, optional
        y coördinates of the cell centers of the desired grid shape(nrow), if
        not defined x and y are calculated from the extent, delr and delc.
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
        data array with dimensions (y, x). y and x are from the new grid.
    """

    msg = (
        f"expected type xr.core.dataarray.DataArray got {type(da_in)} instead"
    )
    assert isinstance(da_in, xr.core.dataarray.DataArray), msg

    if x is None or y is None:
        x, y = get_xy_mid_structured(extent, delr, delc)

    # check if ymid is in descending order
    msg = "ymid should be in descending order"
    assert np.array_equal(y, np.sort(y)[::-1]), msg

    # check for nan values
    if (da_in.isnull().sum() > 0) and (kind == "linear"):
        arr_out = resample_2d_struc_da_nan_linear(
            da_in, x, y, nan_factor, **kwargs
        )
    # faster for linear
    elif kind in ["linear", "cubic"]:
        # no need to fill nan values
        f = interpolate.interp2d(
            da_in.x.data, da_in.y.data, da_in.data, kind="linear", **kwargs
        )
        # for some reason interp2d flips the y-values
        arr_out = f(x, y)[::-1]
    elif kind == "nearest":
        xydata = np.vstack(
            [v.ravel() for v in np.meshgrid(da_in.x.data, da_in.y.data)]
        ).T
        xyi = np.vstack([v.ravel() for v in np.meshgrid(x, y)]).T
        fi = griddata(xydata, da_in.data.ravel(), xyi, method=kind, **kwargs)
        arr_out = fi.reshape(y.shape[0], x.shape[0])
    else:
        raise ValueError(f'unexpected value for "kind": {kind}')

    # new dataset
    da_out = xr.DataArray(arr_out, dims=("y", "x"), coords={"x": x, "y": y})

    return da_out


def resample_dataarray3d_to_structured_grid(
    da_in,
    extent=None,
    delr=None,
    delc=None,
    x=None,
    y=None,
    kind="linear",
    nan_factor=0.01,
    **kwargs,
):
    """resample a dataarray (xarray) from a structured grid to a new dataaraay
    from a different structured grid.

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions (layer, y, x). y and x are from the original
        grid
    extent : list, tuple or np.array, optional
        extent (xmin, xmax, ymin, ymax) of the desired grid, if not defined
        xmid and ymid are used
    delr : int or float, optional
        cell size along rows of the desired grid, if not defined x and
        y are used
    delc : int or float, optional
        cell size along columns of the desired grid, if not defined x and
        y are used
    x : np.array, optional
        x coördinates of the cell centers of the desired grid shape(ncol), if
        not defined x and y are calculated from the extent, delr and delc.
    y : np.array, optional
        y coördinates of the cell centers of the desired grid shape(nrow), if
        not defined x and y are calculated from the extent, delr and delc.
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

    assert isinstance(
        da_in, xr.core.dataarray.DataArray
    ), f"expected type xr.core.dataarray.DataArray got {type(da_in)} instead"

    # check if ymid is in descending order
    assert np.array_equal(
        y, np.sort(y)[::-1]
    ), "ymid should be in descending order"

    if (x is None) or (y is None):
        x, y = get_xy_mid_structured(extent, delr, delc)

    layers = da_in.layer.data
    arr_out = np.zeros((len(layers), len(y), len(x)))
    for i, lay in enumerate(layers):

        ds_lay = da_in.sel(layer=lay)
        # check for nan values
        if (ds_lay.isnull().sum() > 0) and (kind == "linear"):
            arr_out[i] = resample_2d_struc_da_nan_linear(
                ds_lay, x, y, nan_factor, **kwargs
            )
        # faster for linear
        elif kind in ["linear", "cubic"]:
            # no need to fill nan values
            f = interpolate.interp2d(
                ds_lay.x.data,
                ds_lay.y.data,
                ds_lay.data,
                kind="linear",
                **kwargs,
            )
            # for some reason interp2d flips the y-values
            arr_out[i] = f(x, y)[::-1]
        elif kind == "nearest":
            xydata = np.vstack(
                [v.ravel() for v in np.meshgrid(ds_lay.x.data, ds_lay.y.data)]
            ).T
            xyi = np.vstack([v.ravel() for v in np.meshgrid(x, y)]).T
            fi = griddata(
                xydata, ds_lay.data.ravel(), xyi, method=kind, **kwargs
            )
            arr_out[i] = fi.reshape(y.shape[0], x.shape[0])
        else:
            raise ValueError(f'unexpected value for "kind": {kind}')

    # new dataset
    da_out = xr.DataArray(
        arr_out,
        dims=("layer", "y", "x"),
        coords={"x": x, "y": y, "layer": layers},
    )

    return da_out


def resample_2d_struc_da_nan_linear(
    da_in, new_x, new_y, nan_factor=0.01, **kwargs
):
    """resample a structured, 2d data-array with nan values onto a new grid.

    Parameters
    ----------
    da_in : xarray DataArray
        dataset you want to project on a new grid
    new_x : numpy array
        x coördinates of the new grid
    new_y : numpy array
        y coördinates of the new grid
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
    f = interpolate.interp2d(
        da_in.x.data, da_in.y.data, fill_map, kind="linear", **kwargs
    )
    f_nan = interpolate.interp2d(
        da_in.x.data, da_in.y.data, nan_map, kind="linear"
    )
    arr_out_raw = f(new_x, new_y)
    nan_new = f_nan(new_x, new_y)
    arr_out_raw[nan_new > nan_factor] = np.nan

    # for some reason interp2d flips the y-values
    arr_out = arr_out_raw[::-1]

    return arr_out


def resample_dataset_to_structured_grid(
    ds_in, extent, delr, delc, kind="nearest"
):
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

    x, y = get_xy_mid_structured(extent, delr, delc)
    if kind in ["nearest", "linear"]:
        return ds_in.interp(x=x, y=y, method=kind)

    ds_out = xr.Dataset(coords={"y": y, "x": x, "layer": ds_in.layer.data})
    for data_var in ds_in.data_vars:
        data_arr = resample_dataarray3d_to_structured_grid(
            ds_in[data_var], x=x, y=y, kind=kind
        )
        ds_out[data_var] = data_arr

    return ds_out


def get_resampled_ml_layer_ds_vertex(
    raw_ds=None, extent=None, gridprops=None, nodata=-1
):
    """Project model layer dataset on a vertex model grid.

    Parameters
    ----------
    raw_ds : xr.Dataset, optional
        raw model layer dataset. The default is None.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. Used as the
        definition of the vertex grid.
    nodata : int, optional
        integer to represent nodata-values. Defaults to -1.


    Returns
    -------
    ml_layer_ds : xr.dataset
        model layer dataset projected onto the modelgrid.
    """

    logger.info("resample model layer data to vertex modelgrid")
    ml_layer_ds = resample_dataset_to_vertex_grid(raw_ds, gridprops)
    if "area" in gridprops:
        # only keep the first layer of area
        area = gridprops["area"][: len(ml_layer_ds["icell2d"])]
        ml_layer_ds["area"] = ("icell2d", area)
    # add information about the vertices
    _, xv, yv = zip(*gridprops["vertices"])
    ml_layer_ds["xv"] = ("iv", np.array(xv))
    ml_layer_ds["yv"] = ("iv", np.array(yv))
    # and set which nodes use which vertices
    ncvert_max = np.max([x[3] for x in gridprops["cell2d"]])
    icvert = np.full((gridprops["ncpl"], ncvert_max), nodata)
    for i in range(gridprops["ncpl"]):
        icvert[i, : gridprops["cell2d"][i][3]] = gridprops["cell2d"][i][4:]

    ml_layer_ds["icvert"] = ("icell2d", "nvert"), icvert
    ml_layer_ds["icvert"].attrs["_FillValue"] = nodata

    ml_layer_ds.attrs["gridtype"] = "vertex"
    ml_layer_ds.attrs["delr"] = raw_ds.delr
    ml_layer_ds.attrs["delc"] = raw_ds.delc
    ml_layer_ds.attrs["extent"] = extent

    return ml_layer_ds


def fillnan_dataarray_structured_grid(xar_in, method="nearest"):
    """fill not-a-number values in a structured grid, DataArray.

    The fill values are determined using the 'nearest' method of the
    scipy.interpolate.griddata function


    Parameters
    ----------
    xar_in : xarray DataArray
        DataArray with nan values. DataArray should have 2 dimensions
        (y and x).
    method : str, optional
        method used in scipy.interpolate.griddata to resample, default is
        nearest.

    Returns
    -------
    xar_out : xarray DataArray
        DataArray without nan values. DataArray has 2 dimensions
        (y and x)

    Notes
    -----
    can be slow if the xar_in is a large raster
    """
    # check dimensions
    if xar_in.dims != ("y", "x"):
        raise ValueError(
            f"expected dataarray with dimensions ('y' and 'x'), got dimensions -> {xar_in.dims}"
        )

    # get list of coordinates from all points in raster
    mg = np.meshgrid(xar_in.x.data, xar_in.y.data)
    points_all = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    # get all values in DataArray
    values_all = xar_in.data.flatten()

    # get 1d arrays with only values where DataArray is not nan
    mask1 = ~np.isnan(values_all)
    points_in = points_all[np.where(mask1)[0]]
    values_in = values_all[np.where(mask1)[0]]

    # get value for all nan values
    values_out = griddata(points_in, values_in, points_all, method=method)
    arr_out = values_out.reshape(xar_in.shape)

    # create DataArray without nan values
    xar_out = xr.DataArray(
        arr_out,
        dims=("y", "x"),
        coords={"x": xar_in.x.data, "y": xar_in.y.data},
    )

    return xar_out


def fillnan_dataarray_vertex_grid(
    xar_in, model_ds=None, x=None, y=None, method="nearest"
):
    """fill not-a-number values in a vertex grid, DataArray.

    The fill values are determined using the 'nearest' method of the
    scipy.interpolate.griddata function

    Parameters
    ----------
    xar_in : xr.DataArray
        data array with nan values. Shape is (icell2d)
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen.
    x : np.array, optional
        x coördinates of the cell centers shape(icell2d), if not defined use x
        from model_ds.
    y : np.array, optional
        y coördinates of the cell centers shape(icell2d), if not defined use y
        from model_ds.
    method : str, optional
        method used in scipy.interpolate.griddata to resample, default is
        nearest.

    Returns
    -------
    xar_out : xr.DataArray
        data array with nan values. Shape is (icell2d)

    Notes
    -----
    can be slow if the xar_in is a large raster
    """

    # get list of coordinates from all points in raster
    if x is None:
        x = model_ds["x"].data
    if y is None:
        y = model_ds["y"].data

    xyi = np.column_stack((x, y))

    # fill nan values in DataArray
    values_all = xar_in.data

    # get 1d arrays with only values where DataArray is not nan
    mask1 = ~np.isnan(values_all)
    xyi_in = xyi[mask1]
    values_in = values_all[mask1]

    # get value for all nan values
    values_out = griddata(xyi_in, values_in, xyi, method=method)

    # create DataArray without nan values
    xar_out = xr.DataArray(values_out, dims=("icell2d"))

    return xar_out


def resample_vertex_2d_da_to_struc_2d_da(
    da_in, model_ds=None, x=None, y=None, cellsize=25, method="nearest"
):
    """resample a 2d dataarray (xarray) from a vertex grid to a new dataaraay
    from a structured grid.

    Parameters
    ----------
    da_in : xarray.DataArray
        data array with dimensions ('icell2d').
    model_ds : xarray.DataArray
        model dataset with 'x' and 'y' data variables.
    x : np.array, optional
        x coördinates of the cell centers of the desired grid shape(icell2d),
        if not defined use x from model_ds.
    y : np.array, optional
        y coördinates of the cell centers of the desired grid shape(icell2d),
        if not defined use y from model_ds.
    cellsize : int or float, optional
        required cell size of structured grid. The default is 25.
    method : str, optional
        method used for resampling. The default is 'nearest'.

    Returns
    -------
    da_out : xarray.DataArray
        data array with dimensions ('y', 'x').
    """
    if x is None:
        x = model_ds.x.values
    if y is None:
        y = model_ds.y.values

    points_vertex = np.array([x, y]).T
    modelgrid_x = np.arange(x.min(), x.max(), cellsize)
    modelgrid_y = np.arange(y.max(), y.min() - cellsize, -cellsize)
    mg = np.meshgrid(modelgrid_x, modelgrid_y)
    points = np.vstack((mg[0].ravel(), mg[1].ravel())).T

    arr_out_1d = griddata(points_vertex, da_in.values, points, method=method)
    arr_out2d = arr_out_1d.reshape(len(modelgrid_y), len(modelgrid_x))

    da_out = xr.DataArray(
        arr_out2d, dims=("y", "x"), coords={"y": modelgrid_y, "x": modelgrid_x}
    )

    return da_out


def raster_to_quadtree_grid(
    fname,
    model_ds,
    dst_crs=None,
    resampling=rasterio.enums.Resampling.average,
    return_data_array=True,
    x0=None,
    y0=None,
    width=None,
    height=None,
    extent=None,
    src_nodata=None,
    src_crs=None,
    src_transform=None,
):
    """Resample a raster-file to a quadtree-grid, using different advanced
    resample algoritms"""
    if not isinstance(resampling, rasterio.enums.Resampling):
        if hasattr(rasterio.enums.Resampling, resampling):
            resampling = getattr(rasterio.enums.Resampling, resampling)
        else:
            raise (Exception(f"Unknown resample algoritm: {resampling}"))

    if x0 is None and "x0" in model_ds.attrs:
        x0 = model_ds.attrs["x0"]
    if y0 is None and "y0" in model_ds.attrs:
        y0 = model_ds.attrs["y0"]
    if width is None and "width" in model_ds.attrs:
        width = model_ds.attrs["width"]
    if height is None and "height" in model_ds.attrs:
        height = model_ds.attrs["height"]
    if extent is None and "extent" in model_ds.attrs:
        extent = model_ds.attrs["extent"]
    if extent is not None:
        x0 = extent[0]
        y0 = extent[2]
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
    if x0 is None or y0 is None or width is None or height is None:
        raise (Exception("Cannot determine dst_transform"))

    area = model_ds["area"]
    x = model_ds.x.values
    y = model_ds.y.values
    z = np.full(area.shape, np.NaN)

    for ar in np.unique(area):
        mask = area == ar
        dx = dy = np.sqrt(ar)
        dst_transform = Affine.translation(x0, y0) * Affine.scale(dx, dy)
        dst_shape = (int((height) / dy), int((width) / dx))
        zt = np.zeros(dst_shape)

        if isinstance(fname, xr.DataArray):
            da = fname
            if src_transform is None:
                src_transform = get_dataset_transform(da)
            if src_crs is None:
                src_crs = 28992
            if dst_crs is None:
                dst_crs = 28992
            reproject(
                da.data,
                destination=zt,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                dst_nodata=np.NaN,
                src_nodata=src_nodata,
            )
        else:
            with rasterio.open(fname) as src:
                if dst_crs is None:
                    dst_crs = src.crs
                reproject(
                    source=rasterio.band(src, 1),
                    destination=zt,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    dst_nodata=np.NaN,
                    src_nodata=src_nodata,
                )
        # use an xarray to get the right values using .sel()
        xt = np.arange(
            extent[0] + dst_transform[0] / 2, extent[1], dst_transform[0]
        )
        yt = np.arange(
            extent[3] + dst_transform[4] / 2, extent[2], dst_transform[4]
        )

        da = xr.DataArray(zt, coords=(yt, xt), dims=["y", "x"])
        if len(mask.shape) == 2:
            x, y = np.meshgrid(x, y)
        z[mask] = da.sel(
            y=xr.DataArray(y[mask]), x=xr.DataArray(x[mask])
        ).values

    if return_data_array:
        z_da = xr.full_like(model_ds["area"], np.NaN)
        z_da.data = z
        return z_da
    return z


def get_dataset_transform(ds):
    """
    Get an Affine Transform object from a model Dataset

    Parameters
    ----------
    ds : xr.dataset
        The model dataset for which the transform needs to be calculated.

    Returns
    -------
    transform : affine.Affine
        An affine transformation object.

    """
    xsize = ds.x.values[1] - ds.x.values[0]
    ysize = ds.y.values[1] - ds.y.values[0]
    dx = np.unique(np.diff(ds.x.values))
    assert len(dx) == 1
    xsize = dx[0]
    dy = np.unique(np.diff(ds.y.values))
    assert len(dy) == 1
    ysize = dy[0]
    west = ds.x.values[0] - xsize / 2
    north = ds.y.values[0] - ysize / 2
    transform = rasterio.transform.from_origin(west, north, xsize, -ysize)
    return transform
