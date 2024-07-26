import logging
import numbers

import numpy as np
import rasterio
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from ..util import get_da_from_da_ds

logger = logging.getLogger(__name__)


def get_xy_mid_structured(extent, delr, delc, descending_y=True):
    """Calculates the x and y coordinates of the cell centers of a structured grid.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax)
    delr : int, float, list, tuple or array, optional
        The gridsize along columns (dx). The default is 100. meter.
    delc : None, int, float, list, tuple or array, optional
        The gridsize along rows (dy). Set to delr when None. If None delc=delr
    descending_y : bool, optional
        if True the resulting ymid array is in descending order. This is the
        default for MODFLOW models. default is True.

    Returns
    -------
    x : np.array
        x-coordinates of the cell centers shape (ncol)
    y : np.array
        y-coordinates of the cell centers shape (nrow)
    """
    if isinstance(delr, (numbers.Number)):
        if not isinstance(delc, (numbers.Number)):
            raise TypeError("if delr is a number delc should be a number as well")

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

    elif isinstance(delr, np.ndarray) and isinstance(delc, np.ndarray):
        delr = np.asarray(delr)
        delc = np.asarray(delc)
        if (delr.ndim != 1) or (delc.ndim != 1):
            raise ValueError("expected 1d array")

        x = []
        for i, dx in enumerate(delr):
            x.append(extent[0] + dx / 2 + sum(delr[:i]))

        # you always want descending y in this case, so not using
        # the keyword argument
        y = []
        for i, dy in enumerate(delc):
            y.append(extent[3] - dy / 2 - sum(delc[:i]))

        return x, y

    else:
        raise TypeError("unexpected type for delr and/or delc")


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
    """Resample a dataset (xarray) from a structured grid to a new dataset from a
    different structured grid.

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
        lower left x coordinate of the model grid. When angrot == 0, xorigin is added to
        the first two values of extent. Otherwise it is the x-coordinate of the point
        the grid is rotated around, and xorigin is added to the Dataset-attributes.
        The default is 0.0.
    yorigin : int or float, optional
        lower left y coordinate of the model grid. When angrot == 0, yorigin is added to
        the last two values of extent. Otherwise it is the y-coordinate of the point
        the grid is rotated around, and yorigin is added to the Dataset-attributes.
        The default is 0.0.
    angrot : int or float, optinal
        the rotation of the grid in counter clockwise degrees. When angrot != 0 the grid
        is rotated, and all coordinates of the model are in model coordinates. See
        https://nlmod.readthedocs.io/en/stable/examples/11_grid_rotation.html for more
        infomation. The default is 0.0.
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
    if hasattr(ds_in, "gridtype"):
        assert ds_in.attrs["gridtype"] == "structured"
    if delc is None:
        delc = delr

    attrs = ds_in.attrs.copy()
    _set_angrot_attributes(extent, xorigin, yorigin, angrot, attrs)

    x, y = get_xy_mid_structured(attrs["extent"], delr, delc)

    # add new attributes
    attrs["gridtype"] = "structured"

    if isinstance(delr, numbers.Number) and isinstance(delc, numbers.Number):
        delr = np.full_like(x, delr)
        delc = np.full_like(y, delc)

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
    """Internal method to set the properties of the grid in an attribute dictionary.

    Parameters
    ----------
    extent : list, tuple or np.array of length 4
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    xorigin : int or float, optional
        lower left x coordinate of the model grid. When angrot == 0, xorigin is added to
        the first two values of extent. Otherwise it is the x-coordinate of the point
        the grid is rotated around, and xorigin is added to the Dataset-attributes.
        The default is 0.0.
    yorigin : int or float, optional
        lower left y coordinate of the model grid. When angrot == 0, yorigin is added to
        the last two values of extent. Otherwise it is the y-coordinate of the point
        the grid is rotated around, and yorigin is added to the Dataset-attributes.
        The default is 0.0.
    angrot : int or float, optinal
        the rotation of the grid in counter clockwise degrees. When angrot != 0 the grid
        is rotated, and all coordinates of the model are in model coordinates. See
        https://nlmod.readthedocs.io/en/stable/examples/11_grid_rotation.html for more
        infomation. The default is 0.0.
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
            raise (ValueError("Either extent[0] or xorigin needs to be 0.0"))
        if yorigin == 0.0:
            yorigin = extent[2]
            extent[2] = 0.0
            extent[3] = extent[3] - yorigin
        elif extent[2] != 0.0:
            raise (ValueError("Either extent[2] or yorigin needs to be 0.0"))
        attrs["extent"] = extent
        attrs["xorigin"] = xorigin
        attrs["yorigin"] = yorigin
        attrs["angrot"] = angrot


def fillnan_da_structured_grid(xar_in, method="nearest"):
    """Fill not-a-number values in a structured grid, DataArray.

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
    """Fill not-a-number values in a vertex grid, DataArray.

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
    if xar_in.dims != ("icell2d",):
        raise ValueError(
            f"expected dataarray with dimensions ('icell2d'), got dimensions -> {xar_in.dims}"
        )

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
    """Fill not-a-number values in a DataArray.

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
    """Resample a vertex DataArray to a model dataset.

    Parameters
    ----------
    da : xaray.DataArray
        A vertex DataArray. When the DataArray does not have 'icell2d' as a dimension,
        the original DataArray is retured. The DataArray da can contain other dimensions
        as well (for example 'layer' or time'' ).
    ds : xarray.Dataset
        The model dataset to which the DataArray needs to be resampled, with coordinates
        x and y.
    method : str, optional
        The interpolation method, see griddata. The default is "nearest".

    Returns
    -------
    xarray.DataArray
        A DataArray, with the same gridtype as ds.
    """
    if "icell2d" not in da.dims:
        return structured_da_to_ds(da, ds, method=method)
    points = np.array((da.x.data, da.y.data)).T

    if "gridtype" in ds.attrs and ds.gridtype == "vertex":
        if len(da.dims) == 1:
            xi = list(zip(ds.x.values, ds.y.values))
            z = griddata(points, da.values, xi, method=method)
            coords = {"icell2d": ds.icell2d}
            return xr.DataArray(z, dims="icell2d", coords=coords)
        else:
            raise NotImplementedError(
                "Resampling from multidmensional vertex da to vertex ds not yet supported"
            )

    xg, yg = np.meshgrid(ds.x, ds.y)
    xi = np.stack((xg, yg), axis=2)

    if len(da.dims) > 1:
        # when there are more dimensions than icell2d
        z = []
        if method == "nearest":
            # generate the tree only once, to increase speed
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
        for key in list(coords):
            if "icell2d" in coords[key].dims:
                coords.pop(key)
    else:
        # just use griddata
        z = griddata(points, da.data, xi, method=method)
        dims = ["y", "x"]
        coords = {"x": ds.x, "y": ds.y}
    return xr.DataArray(z, dims=dims, coords=coords)


def structured_da_to_ds(da, ds, method="average", nodata=np.nan):
    """Resample a DataArray to the coordinates of a model dataset.

    Parameters
    ----------
    da : xarray.DataArray
        The data-array to be resampled, with dimensions x and y.
    ds : xarray.Dataset
        The model dataset.
    method : string or rasterio.enums.Resampling, optional
        The method to resample the DataArray. Possible values are "linear",
        "nearest" and all the values in rasterio.enums.Resampling. These values
        can be provided as a string ('average') or as an attribute of
        rasterio.enums.Resampling (rasterio.enums.Resampling.average). When
        method is 'linear' or 'nearest' da.interp() is used. Otherwise
        da.rio.reproject_match() is used. The default is "average".
    nodata : float, optional
        The nodata value in input and output. The default is np.NaN.

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

        # some stuff is added by the interp function that should not be there
        added_coords = set(da_out.coords) - set(ds.coords)
        return da_out.drop_vars(added_coords)
    if isinstance(method, rasterio.enums.Resampling):
        resampling = method
    else:
        if hasattr(rasterio.enums.Resampling, method):
            resampling = getattr(rasterio.enums.Resampling, method)
        else:
            raise ValueError(f"Unknown resample method: {method}")
    # fill crs if it is None for da or ds
    if ds.rio.crs is None and da.rio.crs is None:
        logger.info("No crs in da and ds. Assuming ds and da are both in EPSG:28992")
        ds = ds.rio.write_crs(28992)
        da = da.rio.write_crs(28992)
    elif ds.rio.crs is None:
        logger.info(f"No crs in ds. Setting crs equal to da: {da.rio.crs}")
        ds = ds.rio.write_crs(da.rio.crs)
    elif da.rio.crs is None:
        logger.info(f"No crs in da. Setting crs equal to ds: {ds.rio.crs}")
        da = da.rio.write_crs(ds.rio.crs)
    if ds.gridtype == "structured":
        # assume an average delr and delc to calculate the affine
        if "extent" in ds.attrs:
            # xmin, xmax, ymin, ymax
            dx = (ds.attrs["extent"][1] - ds.attrs["extent"][0]) / len(ds.x)
            dy = (ds.attrs["extent"][3] - ds.attrs["extent"][2]) / len(ds.y)
        else:
            raise ValueError(
                "No extent or delr and delc in ds. Cannot determine affine."
            )
        from .grid import get_affine

        da_out = da.rio.reproject(
            dst_crs=ds.rio.crs,
            shape=(len(ds.y), len(ds.x)),
            transform=get_affine(ds, sx=dx, sy=-dy),
            resampling=resampling,
            nodata=nodata,
        )
        if "x" not in da_out.coords or "y" not in da_out.coords:
            # when grid-rotation is used, there are no x and y in coords
            da_out = da_out.assign_coords(x=ds.x, y=ds.y)
    elif ds.gridtype == "vertex":
        # assume the grid is a quadtree grid, where cells are refined by splitting them
        # in 4
        # We perform a reproject-match for every refinement-level
        dims = list(da.dims)
        dims.remove("y")
        dims.remove("x")
        dims.append("icell2d")
        da_out = get_da_from_da_ds(ds, dims=tuple(dims), data=nodata)
        from .grid import get_affine

        for area in np.unique(ds["area"]):
            dx = dy = np.sqrt(area)
            x, y = get_xy_mid_structured(ds.extent, dx, dy)
            da_temp = da.rio.reproject(
                dst_crs=ds.rio.crs,
                shape=(len(y), len(x)),
                transform=get_affine(ds, sx=dx, sy=-dy),
                resampling=resampling,
                nodata=nodata,
            )
            if "x" not in da_temp.coords or "y" not in da_temp.coords:
                # when grid-rotation is used, there are no x and y in coords
                da_temp = da_temp.assign_coords(x=x, y=y)

            mask = ds["area"] == area
            da_out.loc[{"icell2d": mask}] = da_temp.sel(
                y=ds["y"][mask], x=ds["x"][mask]
            )
    else:
        raise (NotImplementedError(f"Gridtype {ds.gridtype} not supported"))

    # some stuff is added by the reproject_match function that should not be there
    added_coords = set(da_out.coords) - set(ds.coords)
    da_out = da_out.drop_vars(added_coords)

    if "grid_mapping" in da_out.encoding:
        del da_out.encoding["grid_mapping"]

    # remove the long_name, standard_name and units attributes of the x and y coordinates
    for coord in ["x", "y"]:
        if coord not in da_out.coords:
            continue
        for name in ["long_name", "standard_name", "units", "axis"]:
            if name in da_out[coord].attrs.keys():
                del da_out[coord].attrs[name]

    return da_out


def extent_to_polygon(extent):
    logger.warning(
        "nlmod.resample.extent_to_polygon is deprecated. "
        "Use nlmod.util.extent_to_polygon instead."
    )
    from ..util import extent_to_polygon

    return extent_to_polygon(extent)


def get_extent_polygon(ds, rotated=True):
    """Get the model extent, as a shapely Polygon."""
    logger.warning(
        "nlmod.resample.get_extent_polygon is deprecated. "
        "Use nlmod.grid.get_extent_polygon instead."
    )
    from .grid import get_extent_polygon

    return get_extent_polygon(ds, rotated=rotated)


def affine_transform_gdf(gdf, affine):
    """Apply an affine transformation to a geopandas GeoDataFrame."""
    logger.warning(
        "nlmod.resample.affine_transform_gdf is deprecated. "
        "Use nlmod.grid.affine_transform_gdf instead."
    )
    from .grid import affine_transform_gdf

    return affine_transform_gdf(gdf, affine)


def get_extent(ds, rotated=True):
    """Get the model extent, corrected for angrot if necessary."""
    logger.warning(
        "nlmod.resample.get_extent is deprecated. Use nlmod.grid.get_extent instead."
    )
    from .grid import get_extent

    return get_extent(ds, rotated=rotated)


def get_affine_mod_to_world(ds):
    """Get the affine-transformation from model to real-world coordinates."""
    logger.warning(
        "nlmod.resample.get_affine_mod_to_world is deprecated. "
        "Use nlmod.grid.get_affine_mod_to_world instead."
    )
    from .grid import get_affine_mod_to_world

    return get_affine_mod_to_world(ds)


def get_affine_world_to_mod(ds):
    """Get the affine-transformation from real-world to model coordinates."""
    logger.warning(
        "nlmod.resample.get_affine_world_to_mod is deprecated. "
        "Use nlmod.grid.get_affine_world_to_mod instead."
    )
    from .grid import get_affine_world_to_mod

    return get_affine_world_to_mod(ds)


def get_affine(ds, sx=None, sy=None):
    """Get the affine-transformation, from pixel to real-world coordinates."""
    logger.warning(
        "nlmod.resample.get_affine is deprecated. Use nlmod.grid.get_affine instead."
    )
    from .grid import get_affine

    return get_affine(ds, sx=sx, sy=sy)
