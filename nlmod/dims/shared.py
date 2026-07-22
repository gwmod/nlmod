from enum import Enum

try:
    import numba

    _NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    _NUMBA_AVAILABLE = False
import numpy as np
import xarray as xr


class GridTypeDims(Enum):
    """Enum for grid dimensions."""

    VERTEX_LAYERED = ("layer", "icell2d")
    VERTEX = ("icell2d",)
    STRUCTURED_LAYERED = ("layer", "y", "x")
    STRUCTURED = ("y", "x")

    @classmethod
    def parse_dims(cls, ds):
        """Get GridTypeDim from dataset or dataarray.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataArray
            Dataset or DataArray to parse.

        Returns
        -------
        gridtype : GridTypeDims
            type of grid

        Raises
        ------
        ValueError
            If no partially matching gridtype is found.
        """
        layer_is_dim = "layer" in ds.dims
        if "x" in ds and "y" in ds:
            x_dims = set(ds["x"].dims)
            y_dims = set(ds["y"].dims)
            if "icell2d" in x_dims or "icell2d" in y_dims:
                return cls.VERTEX_LAYERED if layer_is_dim else cls.VERTEX
            if x_dims == {"x"} and y_dims == {"y"}:
                return cls.STRUCTURED_LAYERED if layer_is_dim else cls.STRUCTURED
        for gridtype in GridTypeDims:
            if set(gridtype.value).issubset(ds.dims):
                return gridtype
        # raises ValueError if no gridtype is found
        return cls(ds.dims)


def is_structured(ds):
    """Check if a dataset is structured.

    Parameters
    ----------
    ds : xr.Dataset or xr.Dataarray
        dataset or dataarray

    Returns
    -------
    bool
        True if the dataset is structured.
    """
    return GridTypeDims.parse_dims(ds) in (
        GridTypeDims.STRUCTURED,
        GridTypeDims.STRUCTURED_LAYERED,
    )


def is_vertex(ds):
    """Check if a dataset is vertex.

    Parameters
    ----------
    ds : xr.Dataset or xr.Dataarray
        dataset or dataarray

    Returns
    -------
    bool
        True if the dataset is structured.
    """
    return GridTypeDims.parse_dims(ds) in (
        GridTypeDims.VERTEX,
        GridTypeDims.VERTEX_LAYERED,
    )


def is_layered(ds):
    """Check if a dataset is layered.

    Parameters
    ----------
    ds : xr.Dataset or xr.Dataarray
        dataset or dataarray

    Returns
    -------
    bool
        True if the dataset is layered.
    """
    return "layer" in ds.dims


def is_rotated(ds):
    """Check if a dataset is rotated.

    Parameters
    ----------
    ds : xr.Dataset or xr.Dataarray
        dataset or dataarray

    Returns
    -------
    bool
        True if the dataset is rotated.
    """
    return "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0


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
    assert is_structured(ds)
    if "extent" in ds.attrs:
        west_model = ds.extent[0]
    else:
        west_model = float(ds.x[0] - (ds.x[1] - ds.x[0]) / 2)
    x = (ds.x - west_model).values
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
    assert is_structured(ds)
    if "extent" in ds.attrs:
        north_model = ds.extent[3]
    else:
        north_model = float(ds.y[0] + (ds.y[0] - ds.y[1]) / 2)
    y = (north_model - ds.y).values
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


if _NUMBA_AVAILABLE:

    @numba.njit(parallel=True)
    def _compute_vertex_areas_numba(xv, yv, icvert, fill_value=-1):
        ncells, max_vert = icvert.shape
        areas = np.empty(ncells, dtype=np.float64)

        # parallel=True parallelizes this outer loop across your CPU cores
        for i in numba.prange(ncells):
            # Determine actual number of vertices for this cell (ignoring padding)
            nv = 0
            for j in range(max_vert):
                if icvert[i, j] == fill_value:
                    break
                nv += 1

            if nv < 3:
                areas[i] = 0.0
                continue

            # Inline Shoelace formula (avoids array allocations like np.roll)
            area_sum = 0.0
            for j in range(nv):
                # Current vertex index and next vertex index (wrapped around)
                v1 = icvert[i, j]
                v2 = icvert[i, (j + 1) % nv]

                area_sum += xv[v1] * yv[v2] - xv[v2] * yv[v1]

            areas[i] = 0.5 * abs(area_sum)

        return areas


def get_area(ds):
    """Calculate the area of each cell in the model grid.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset.

    Returns
    -------
    area : xr.DataArray
        area of each cell
    """
    if ds.gridtype == "structured":
        area = xr.DataArray(
            np.outer(get_delc(ds), get_delr(ds)),
            dims=("y", "x"),
            coords={"y": ds["y"], "x": ds["x"]},
        )
    elif ds.gridtype == "vertex":
        if _NUMBA_AVAILABLE:
            xv = ds["xv"].values
            yv = ds["yv"].values
            icvert = ds["icvert"].values
            fill_val = ds["icvert"].attrs.get("nodata", -1)
            area_np = _compute_vertex_areas_numba(xv, yv, icvert, fill_value=fill_val)
        else:
            area_np = np.zeros(ds["icell2d"].size)
            for icell2d in ds["icell2d"]:
                area_np[icell2d] = _shoelace_formula(
                    ds["xv"][ds["icvert"].isel(icell2d=icell2d)],
                    ds["yv"][ds["icvert"].isel(icell2d=icell2d)],
                )

        return xr.DataArray(
            area_np, dims=("icell2d"), coords={"icell2d": ds["icell2d"]}
        )
    else:
        raise ValueError("function only support structured or vertex gridtypes")
    return area
