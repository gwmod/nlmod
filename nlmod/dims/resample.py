# -*- coding: utf-8 -*-
"""Created on Fri Apr  2 15:08:50 2021.

@author: oebbe
"""
import logging

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from ..util import get_da_from_da_ds

logger = logging.getLogger(__name__)


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


def ds_to_structured_grid(
    ds_in,
    extent,
    delr,
    delc=None,
    xorigin=0.0,
    yorigin=0.0,
    angrot=0.0,
    method="nearest",
):
    """Resample a dataset (xarray) from a structured grid to a new dataset from
    a different structured grid.

    Parameters
    ----------
    ds_in : xarray.Dataset
        dataset with dimensions (layer, y, x). y and x are from the original
        grid
    extent : list, tuple or np.array of length 4
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    delr : int or float
        cell size along rows of the desired grid (dx).
    delc : int or float
        cell size along columns of the desired grid (dy).
    xorigin : int or float, optional
        lower left x coordinate of the model grid only used if angrot != 0.
        Default is 0.0.
    yorigin : int or float, optional
        lower left y coordinate of the model grid only used if angrot != 0.
        Default is 0.0.
    angrot : int or float, optinal
        the rotation of the grid in counter clockwise degrees, default is 0.0
    method : str, optional
        type of interpolation used to resample. Sea structured_da_to_ds for
        possible values of method. The default is 'nearest'.

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with dimensions (layer, y, x). y and x are from the new
        grid.
    """

    assert isinstance(ds_in, xr.core.dataset.Dataset)
    if delc is None:
        delc = delr

    x, y = get_xy_mid_structured(extent, delr, delc)

    attrs = ds_in.attrs.copy()
    _set_angrot_attributes(extent, xorigin, yorigin, angrot, attrs)

    # add new attributes
    attrs["gridtype"] = "structured"
    attrs["delr"] = delr
    attrs["delc"] = delc

    if method in ["nearest", "linear"] and angrot == 0.0:
        ds_out = ds_in.interp(
            x=x, y=y, method=method, kwargs={"fill_value": "extrapolate"}
        )
        ds_out.attrs = attrs
        return ds_out

    ds_out = xr.Dataset(coords={"y": y, "x": x, "layer": ds_in.layer.data}, attrs=attrs)
    for var in ds_in.data_vars:
        ds_out[var] = structured_da_to_ds(ds_in[var], ds_out, method=method)
    return ds_out


def _set_angrot_attributes(extent, xorigin, yorigin, angrot, attrs):
    """Internal method to set the properties of the grid in an attribute
    dictionary.

    Parameters
    ----------
    extent : list, tuple or np.array of length 4
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    xorigin : float
        x-position of the lower-left corner of the model grid. Only used when angrot is
        not 0.
    yorigin : float
        y-position of the lower-left corner of the model grid. Only used when angrot is
        not 0.
    angrot : float
        counter-clockwise rotation angle (in degrees) of the lower-left corner of the
        model grid.
    attrs : dict
        Attributes of a model dataset.

    Returns
    -------
    None.
    """
    # make sure extent is a list and the original extent is not changed
    extent = list(extent)
    if angrot == 0.0:
        if xorigin != 0.0:
            extent[0] = extent[0] + xorigin
            extent[1] = extent[1] + xorigin
        if yorigin != 0.0:
            extent[2] = extent[2] + yorigin
            extent[3] = extent[3] + yorigin
        attrs["extent"] = extent
    else:
        if xorigin == 0.0:
            xorigin = extent[0]
            extent[0] = 0.0
            extent[1] = extent[1] - xorigin
        elif extent[0] != 0.0:
            raise (Exception("Either extent[0] or xorigin needs to be 0.0"))
        if yorigin == 0.0:
            yorigin = extent[2]
            extent[2] = 0.0
            extent[3] = extent[3] - yorigin
        elif extent[2] != 0.0:
            raise (Exception("Either extent[2] or yorigin needs to be 0.0"))
        attrs["extent"] = extent
        attrs["xorigin"] = xorigin
        attrs["yorigin"] = yorigin
        attrs["angrot"] = angrot


def fillnan_da_structured_grid(xar_in, method="nearest"):
    """fill not-a-number values in a structured grid, DataArray.

    The fill values are determined using the 'nearest' method of the
    scipy.interpolate.griddata function


    Parameters
    ----------
    xar_in : xarray DataArray
        DataArray with nan values. DataArray should at least have dimensions x
        and y.
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
    # if "x" not in xar_in.dims or "y" not in xar_in.dims:
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
    # xar_out = xar_in.rio.interpolate_na(method=method)

    return xar_out


def fillnan_da_vertex_grid(xar_in, ds=None, x=None, y=None, method="nearest"):
    """fill not-a-number values in a vertex grid, DataArray.

    The fill values are determined using the 'nearest' method of the
    scipy.interpolate.griddata function

    Parameters
    ----------
    xar_in : xr.DataArray
        data array with nan values. Shape is (icell2d)
    ds : xr.Dataset
        Dataset containing grid-properties
    x : np.array, optional
        x coördinates of the cell centers shape(icell2d), if not defined use x
        from ds.
    y : np.array, optional
        y coördinates of the cell centers shape(icell2d), if not defined use y
        from ds.
    method : str, optional
        method used in scipy.interpolate.griddata to resample, default is
        nearest.

    Returns
    -------
    xar_out : xr.DataArray
        data array without nan values. Shape is (icell2d)

    Notes
    -----
    can be slow if the xar_in is a large raster
    """

    # get list of coordinates from all points in raster
    if x is None:
        x = ds["x"].data
    if y is None:
        y = ds["y"].data

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


def fillnan_da(da, ds=None, method="nearest"):
    """fill not-a-number values in a DataArray.

    The fill values are determined using the 'nearest' method of the
    scipy.interpolate.griddata function

    Parameters
    ----------
    da : xr.DataArray
        data array with nan values.
    ds : xr.Dataset, optional
        Dataset containing grid-properties. Needed when a Vertex grid is used.
    method : str, optional
        method used in scipy.interpolate.griddata to resample. The default is nearest.

    Returns
    -------
    xar_out : xr.DataArray
        data array without nan values.

    Notes
    -----
    can be slow if the xar_in is a large raster
    """
    if len(da.shape) > 1 and len(da.y) == da.shape[-2] and len(da.x) == da.shape[-1]:
        # the dataraary is structured
        return fillnan_da_structured_grid(da, method=method)
    else:
        return fillnan_da_vertex_grid(da, ds, method=method)


def vertex_da_to_ds(da, ds, method="nearest"):
    """Resample a vertex DataArray to a structured model dataset.

    Parameters
    ----------
    da : xaray.DataArray
        A vertex DataArray. When the DataArray does not have 'icell2d' as a
        dimension, the original DataArray is retured. The DataArray da can
        contain other dimensions as well (for example 'layer' or time'' ).
    ds : xarray.Dataset
        The structured model dataset with coordinates x and y.
    method : str, optional
        The interpolation method, see griddata. The default is "nearest".

    Returns
    -------
    xarray.DataArray
        THe structured DataArray, with coordinates 'x' and 'y'
    """
    if hasattr(ds.attrs, "gridtype") and ds.gridtype == "vertex":
        raise (Exception("Resampling from vertex da to vertex ds not supported"))
    if "icell2d" not in da.dims:
        return da
    points = np.array((da.x.data, da.y.data)).T
    xg, yg = np.meshgrid(ds.x, ds.y)
    xi = np.stack((xg, yg), axis=2)

    if len(da.dims) > 1:
        # when there are more dimensions than cell2d
        z = []
        if method == "nearest":
            # geneterate the tree only once, to increase speed
            tree = cKDTree(points)
            _, i = tree.query(xi)
        dims = np.array(da.dims)
        dims = dims[dims != "icell2d"]

        def dim_to_regular_dim(da, dims, z):
            for dimval in da[dims[0]]:
                dat = da.loc[{dims[0]: dimval}]
                if len(dims) > 1:
                    zl = []
                    dim_to_regular_dim(dat, dims[dims != dims[0]], zl)
                else:
                    if method == "nearest":
                        zl = dat.data[i]
                    else:
                        zl = griddata(points, dat.data, xi, method=method)
                z.append(zl)

        dim_to_regular_dim(da, dims, z)
        dims = list(dims) + ["y", "x"]
        coords = dict(da.coords)
        coords["x"] = ds.x
        coords["y"] = ds.y
        coords.pop("icell2d")
    else:
        # just use griddata
        z = griddata(points, da.data, xi, method=method)
        dims = ["y", "x"]
        coords = dict(x=ds.x, y=ds.y)
    return xr.DataArray(z, dims=dims, coords=coords)


def structured_da_to_ds(da, ds, method="average", nodata=np.NaN):
    """Resample a DataArray to the coordinates of a model dataset.

    Parameters
    ----------
    da : xarray.DataArray
        THe data-array to be resampled, with dimensions x and y.
    ds : xarray.Dataset
        The model dataset.
    method : string or rasterio.enums.Resampling, optional
        The method to resample the DataArray. Possible values are "linear",
        "nearest" and all the values in rasterio.enums.Resampling. These values
        can be provided as a string ('average') or as an attribute of
        rasterio.enums.Resampling (rasterio.enums.Resampling.average). When
        method is 'linear' or 'nearest' da.interp() is used. Otherwise
        da.rio.reproject_match() is used. The default is "average".

    Returns
    -------
    da_out : xarray.DataArray
        The resampled DataArray
    """
    has_rotation = "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0
    if method in ["linear", "nearest"] and not has_rotation:
        kwargs = {}
        if ds.gridtype == "structured":
            kwargs["fill_value"] = "extrapolate"
        da_out = da.interp(x=ds.x, y=ds.y, method=method, kwargs=kwargs)
        return da_out
    if isinstance(method, rasterio.enums.Resampling):
        resampling = method
    else:
        if hasattr(rasterio.enums.Resampling, method):
            resampling = getattr(rasterio.enums.Resampling, method)
        else:
            raise (Exception(f"Unknown resample method: {method}"))
    # fill crs if it is None for da or ds
    if ds.rio.crs is None and da.rio.crs is None:
        ds = ds.rio.write_crs(28992)
        da = da.rio.write_crs(28992)
    elif ds.rio.crs is None:
        ds = ds.rio.write_crs(da.rio.crs)
    elif da.rio.crs is None:
        da = da.rio.write_crs(ds.rio.crs)
    if ds.gridtype == "structured":
        if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
            affine = get_affine(ds)
            # save crs as it is deleted by write_transform...
            crs = ds.rio.crs
            ds = ds.rio.write_transform(affine)
            ds = ds.rio.write_crs(crs)
        da_out = da.rio.reproject_match(ds, resampling, nodata=nodata)

    elif ds.gridtype == "vertex":
        # assume the grid is a quadtree grid, where cells are refined by splitting them
        # in 4
        dims = list(da.dims)
        dims.remove("y")
        dims.remove("x")
        dims.append("icell2d")
        da_out = get_da_from_da_ds(ds, dims=tuple(dims), data=nodata)
        for area in np.unique(ds["area"]):
            dx = dy = np.sqrt(area)
            x, y = get_xy_mid_structured(ds.extent, dx, dy)
            da_temp = xr.DataArray(nodata, dims=["y", "x"], coords=dict(x=x, y=y))
            if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
                affine = get_affine(ds)
                da_temp = da_temp.rio.write_transform(affine, inplace=True)
            # make sure da_temp has a crs if da has a crs
            da_temp = da_temp.rio.write_crs(da.rio.crs)
            da_temp = da.rio.reproject_match(da_temp, resampling, nodata=nodata)
            mask = ds["area"] == area
            da_out.loc[dict(icell2d=mask)] = da_temp.sel(
                y=ds["y"][mask], x=ds["x"][mask]
            )
    else:
        raise (Exception(f"Gridtype {ds.gridtype} not supported"))

    # somehow the spatial_ref (jarkus) and band (ahn) coordinates are added by the reproject_match function
    if "spatial_ref" in da_out.coords:
        da_out = da_out.drop_vars("spatial_ref")
        if "grid_mapping" in da_out.encoding:
            del da_out.encoding["grid_mapping"]

    return da_out


def extent_to_polygon(extent):
    """Generate a shapely Polygon from an extent ([xmin, xmax, ymin, ymax])"""
    nw = (extent[0], extent[2])
    no = (extent[1], extent[2])
    zo = (extent[1], extent[3])
    zw = (extent[0], extent[3])
    return Polygon([nw, no, zo, zw])


def _get_attrs(ds):
    if isinstance(ds, dict):
        return ds
    else:
        return ds.attrs


def get_extent_polygon(ds, rotated=True):
    """Get the model extent, as a shapely Polygon."""
    attrs = _get_attrs(ds)
    polygon = extent_to_polygon(attrs["extent"])
    if rotated and "angrot" in ds.attrs and attrs["angrot"] != 0.0:
        affine = get_affine_mod_to_world(ds)
        polygon = affine_transform(polygon, affine.to_shapely())
    return polygon


def affine_transform_gdf(gdf, affine):
    """Apply an affine transformation to a geopandas GeoDataFrame."""
    if isinstance(affine, Affine):
        affine = affine.to_shapely()
    gdfm = gdf.copy()
    gdfm.geometry = gdf.affine_transform(affine)
    return gdfm


def get_extent(ds, rotated=True):
    """Get the model extent, corrected for angrot if necessary."""
    attrs = _get_attrs(ds)
    extent = attrs["extent"]
    if rotated and "angrot" in attrs and attrs["angrot"] != 0.0:
        affine = get_affine_mod_to_world(ds)
        xc = np.array([extent[0], extent[1], extent[1], extent[0]])
        yc = np.array([extent[2], extent[2], extent[3], extent[3]])
        xc, yc = affine * (xc, yc)
        extent = [xc.min(), xc.max(), yc.min(), yc.max()]
    return extent


def get_affine_mod_to_world(ds):
    """Get the affine-transformation from model to real-world coordinates."""
    attrs = _get_attrs(ds)
    xorigin = attrs["xorigin"]
    yorigin = attrs["yorigin"]
    angrot = attrs["angrot"]
    return Affine.translation(xorigin, yorigin) * Affine.rotation(angrot)


def get_affine_world_to_mod(ds):
    """Get the affine-transformation from real-world to model coordinates."""
    attrs = _get_attrs(ds)
    xorigin = attrs["xorigin"]
    yorigin = attrs["yorigin"]
    angrot = attrs["angrot"]
    return Affine.rotation(-angrot) * Affine.translation(-xorigin, -yorigin)


def get_affine(ds, sx=None, sy=None):
    """Get the affine-transformation, from pixel to real-world coordinates."""
    attrs = _get_attrs(ds)
    xorigin = attrs["xorigin"]
    yorigin = attrs["yorigin"]
    angrot = -attrs["angrot"]
    # xorigin and yorigin represent the lower left corner, while for the transform we
    # need the upper left
    dy = attrs["extent"][3] - attrs["extent"][2]
    xoff = xorigin + dy * np.sin(angrot * np.pi / 180)
    yoff = yorigin + dy * np.cos(angrot * np.pi / 180)

    if sx is None:
        sx = attrs["delr"]
    if sy is None:
        sy = -attrs["delc"]
    return (
        Affine.translation(xoff, yoff) * Affine.scale(sx, sy) * Affine.rotation(angrot)
    )
