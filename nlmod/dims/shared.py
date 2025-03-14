from enum import Enum

import numpy as np
import xarray as xr


class GridTypeDims(Enum):
    """Enum for grid dimensions."""

    STRUCTURED_LAYERED = ("layer", "y", "x")
    VERTEX_LAYERED = ("layer", "icell2d")
    STRUCTURED = ("y", "x")
    VERTEX = ("icell2d",)

    @classmethod
    def parse_dims(cls, ds):
        """Get GridTypeDim from dataset or dataarray.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataArray
            Dataset or DataArray to parse.
        """
        for gridtype in GridTypeDims:
            if set(gridtype.value).issubset(ds.dims):
                return gridtype


def get_delr(ds):
    """
    Get the distance along rows (delr) from the x-coordinate of a structured model ds.

    Parameters
    ----------
    ds : xr.Dataset
        A model dataset containing an x-coordinate and an attribute 'extent'.

    Returns
    -------
    delr : np.ndarray
        The cell-size along rows (of length ncol).

    """
    assert ds.gridtype == "structured"
    x = (ds.x - ds.extent[0]).values
    delr = _get_delta_along_axis(x)
    return delr


def get_delc(ds):
    """
    Get the distance along columns (delc) from the y-coordinate of a structured model
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        A model dataset containing an y-coordinate and an attribute 'extent'.

    Returns
    -------
    delc : np.ndarray
        The cell-size along columns (of length nrow).

    """
    assert ds.gridtype == "structured"
    y = (ds.extent[3] - ds.y).values
    delc = _get_delta_along_axis(y)
    return delc


def _get_delta_along_axis(x):
    """Internal method to determine delr or delc from x or y relative to xmin or ymax"""
    delr = [x[0] * 2]
    for xi in x[1:]:
        delr.append((xi - np.sum(delr)) * 2)
    return np.array(delr)


def _shoelace_formula(x, y):
    """Calculate the area of a polygon using the shoelace formula.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates of the polygon.
    y : np.ndarray
        y-coordinates of the polygon.

    Returns
    -------
    area : float
        area of the polygon.
    """
    x = x - np.min(x)
    y = y - np.min(y)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_area(ds):
    """Calculate the area of each cell in the model grid.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset.

    Returns
    -------
    ds : xr.Dataset
        model dataset with an area variable
    """
    if ds.gridtype == "structured":
        area = xr.DataArray(
            np.outer(get_delc(ds), get_delr(ds)),
            dims=("y", "x"),
            coords={"y": ds["y"], "x": ds["x"]},
        )
    elif ds.gridtype == "vertex":
        area = np.zeros(ds["icell2d"].size)
        for icell2d in ds["icell2d"]:
            area[icell2d] = _shoelace_formula(
                ds["xv"][ds["icvert"].isel(icell2d=icell2d)],
                ds["yv"][ds["icvert"].isel(icell2d=icell2d)],
            )
        area = xr.DataArray(
            area,
            dims=("icell2d"),
            coords={"icell2d": ds["icell2d"]},
        )
    else:
        raise ValueError("function only support structured or vertex gridtypes")
    return area
