import numpy as np
import xarray as xr
from scipy.spatial import Delaunay

from nlmod.dims import grid


def _compute_interpolation_weights(xy, uv, d=2):
    """Calculate interpolation weights for barycentric interpolation [1]_.

    Parameters
    ----------
    xy : np.array
        array containing x-coordinates in first column and y-coordinates
        in second column
    uv : np.array
        array containing coordinates at which interpolation weights should
        be calculated, x-coordinates in first column and y-coordinates in second column
    d : int, optional
        dimension of data (the default is 2, which works for 2D data)

    Returns
    -------
    vertices: np.array
        array containing interpolation vertices (len(xy), 3)
    weights: np.array
        array containing interpolation weights per point (len(xy), 3)


    References
    ----------
    .. [1] https://stackoverflow.com/questions/20915502/
    speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    """
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def _interpolate_vertices_weights(values, vertices, weights, fill_value=np.nan):
    """Interpolate values at locations given computed vertices and weights.

    As calculated by compute_barycentric_interpolation_weights function [2]_.

    Parameters
    ----------
    values : np.array
        array containing values to interpolate
    vtx : np.array
        array containing interpolation vertices, see
        compute_barycentric_interpolation_weights()
    wts : np.array
        array containing interpolation weights, see
        compute_barycentric_interpolation_weights()
    fill_value : float
        fill value for points that have to be extrapolated (e.g. at or
        beyond edges of the known points)

    Returns
    -------
    arr: np.array
        array containing interpolated values at locations as given by
        vtx and wts

    References
    ----------
    .. [2] https://stackoverflow.com/questions/20915502/
    speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    """
    ret = np.einsum("nj,nj->n", np.take(values, vertices), weights)
    ret[np.any(weights < 0, axis=1)] = fill_value
    return ret


def interpolate_to_points_2d(da, xi, yi, x="x", y="y"):
    """Interpolate values at locations given by x and y coordinates.

    This function only works for 2D (planar) data, both for structured
    and vertex grid types.

    Parameters
    ----------
    da : np.array or xr.DataArray
        array containing values to interpolate
    xi : np.array
        x-coordinates of the points to interpolate
    yi : np.array
        y-coordinates of the points to interpolate
    x : str or np.array, optional
        x-coordinates of the array, default is "x", which assumes da is
        an xr.DataArray with an x-coordinate named "x"
    y : str or np.array, optional
        y-coordinates of the array, default is "y", which assumes da is
        an xr.DataArray with an y-coordinate named "y"

    Returns
    -------
    np.array
        array containing interpolated values at locations defined by x and y
    """
    # interpolation points
    xyi = np.vstack([xi, yi]).T
    # data coordinates
    if isinstance(x, str):
        x = da[x].values
    if isinstance(y, str):
        y = da[y].values

    if grid.is_structured(da):
        assert da.ndim == 2, "DataArray must be 2D!"
        xy = np.vstack([arr.ravel() for arr in np.meshgrid(x, y)]).T
    elif grid.is_vertex(da):
        assert da.ndim == 1, "DataArray must be 1D!"
        xy = np.vstack([x, y]).T
    else:
        raise ValueError("da must be structured or vertex grid")
    vertices, weights = _compute_interpolation_weights(xy, xyi, d=2)
    return _interpolate_vertices_weights(da.values, vertices, weights)


def interpolate_to_points(
    da, pts, xi="x", yi="y", x="x", y="y", layer="layer", full_output=False
):
    """Linear interpolation of point values from dataarray.

    This function works for all types of data arrays, e.g. 2D (planar), 3D (layered or
    time, planar) or 4D (time, layered) dataarrays for both structured and
    vertex grid types.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing values to interpolate
    pts : xr.DataArray
        Dataset containing x, y-coordinates and optionally the layer of the
        points to interpolate.
    xi : str, optional
        x-coordinates of the points to interpolate, default is "x"
    yi : str, optional
        y-coordinates of the points to interpolate, default is "y"
    x : str, optional
        x-coordinates of the array, default is "x", which assumes da is
        an xr.DataArray with an x-coordinate named "x"
    y : str, optional
        y-coordinates of the array, default is "y", which assumes da is
        an xr.DataArray with an y-coordinate named "y"
    layer : str, optional
        name of the layer dimension in the dataarray, default is "layer"
    full_output : bool, optional
        if True, return the interpolated values and the vertices and
        weights used for interpolation. If False, only return the
        interpolated values. The default is False.

    Returns
    -------
    interp_da : xr.DataArray
        DataArray containing interpolated values at locations defined by
        x and y coordinates in pts. If full_output is True, also
        contains the vertices and weights used for interpolation.

    See Also
    --------
    nlmod.layers.get_modellayers_screens
        get model layers for observation wells
    nlmod.layers.get_modellayers_indexer
        get an xarray dataset with model layers from a dataframe containing location
        information for observation wells, useful as input for this function.
    """
    if grid.is_structured(da):
        input_dims = [y, x]
    elif grid.is_vertex(da):
        input_dims = ["icell2d"]
    else:
        raise ValueError("da must be structured or vertex grid")
    if np.isin(["vertices", "weights"], pts.data_vars).all():
        # no need to recompute vertices and weights
        interp_da = pts.copy()
        dim = pts["vertices"].dims[0]
    else:
        # compute vertices and weightss
        xi = pts[xi]
        yi = pts[yi]
        interp_da = xr.Dataset(data_vars={"x": xi, "y": yi}, coords=pts.coords)
        xyi = np.vstack([xi, yi]).T
        if grid.is_structured(da):
            xy = np.vstack(
                [arr.ravel() for arr in np.meshgrid(da[x].values, da[y].values)]
            ).T
        else:
            # only other allowed gridtype at the start of this method is vertex
            xy = np.vstack([da[x].values, da[y].values]).T
        vertices, weights = _compute_interpolation_weights(xy, xyi, d=2)
        dim = xi.dims[0]
        interp_da["vertices"] = (dim, "iv"), vertices
        interp_da["weights"] = (dim, "iv"), weights

    # apply interpolation over layer and time dimensions, then select
    # appropriate model layer for each location. This means more
    # interpolations are carried out than strictly necessary, but maybe
    # this is still faster?
    interpolated = xr.apply_ufunc(
        _interpolate_vertices_weights,
        da,  # DataArray with dimensions  e.g. (time, layer, y, x)
        interp_da["vertices"],  # DataArray with dimensions (dim, iv)
        interp_da["weights"],  # DataArray with dimensions (dim, iv)
        input_core_dims=[input_dims, [dim, "iv"], [dim, "iv"]],
        output_core_dims=[[dim]],
        exclude_dims=set(input_dims),  # Exclude spatial dimensions
        vectorize=True,  # Enable vectorization over non-core dimensions
    )
    # layer in pts and layer in dataarray (spatial 3D), select layer per point
    if layer in pts and layer in da.dims:
        interp_da["layer"] = pts[layer]
        interp_da["interpolated"] = interpolated.sel(
            layer=interp_da["layer"]
        ).drop_vars("layer")
    # layer in pts but not in dataarray (spatial 2D), ignores layer in pts
    elif layer in pts and layer not in da.dims:
        interp_da["interpolated"] = interpolated.drop_vars("layer")
    # layer not in pts and not in dataarray (spatial 2D)
    elif layer not in pts and layer not in da.dims:
        interp_da["interpolated"] = interpolated
    # layer not in pts, but layer is in dataarray (spatial 3D), return result all layers
    else:
        # NOTE: do I need to deal with added layer dim here?
        interp_da["interpolated"] = interpolated

    if full_output:
        return interp_da  # also returns vertices and weights
    else:
        return interp_da["interpolated"]
