import datetime as dt
import logging
import numbers

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from .. import util
from ..epsg28992 import EPSG_28992
from . import resample
from .layers import fill_nan_top_botm_kh_kv, set_idomain

logger = logging.getLogger(__name__)


def set_ds_attrs(ds, model_name, model_ws, mfversion="mf6", exe_name=None):
    """set the attribute of a model dataset.

    Parameters
    ----------
    ds : xarray dataset
        An existing model dataset
    model_name : str
        name of the model.
    model_ws : str or None
        workspace of the model. This is where modeldata is saved to.
    mfversion : str, optional
        modflow version. The default is "mf6".
    exe_name: str, optional
        path to modflow executable, default is None, which assumes binaries
        are available in nlmod/bin directory. Binaries can be downloaded
        using `nlmod.util.download_mfbinaries()`.

    Returns
    -------
    ds : xarray dataset
        model dataset.
    """

    if model_name is not None and len(model_name) > 16 and mfversion == "mf6":
        raise ValueError("model_name can not have more than 16 characters")
    ds.attrs["model_name"] = model_name
    ds.attrs["mfversion"] = mfversion
    fmt = "%Y%m%d_%H:%M:%S"
    ds.attrs["created_on"] = dt.datetime.now().strftime(fmt)

    if exe_name is None:
        exe_name = util.get_exe_path(mfversion)

    ds.attrs["exe_name"] = exe_name

    # add some directories
    if model_ws is not None:
        figdir, cachedir = util.get_model_dirs(model_ws)
        ds.attrs["model_ws"] = model_ws
        ds.attrs["figdir"] = figdir
        ds.attrs["cachedir"] = cachedir

    return ds


def to_model_ds(
    ds,
    model_name=None,
    model_ws=None,
    extent=None,
    delr=100.0,
    delc=None,
    fill_nan=True,
    extrapolate=True,
    anisotropy=10,
    fill_value_kh=1.0,
    fill_value_kv=0.1,
    xorigin=0.0,
    yorigin=0.0,
    angrot=0.0,
    drop_attributes=True,
    transport=False,
):
    """Transform an input dataset to a groundwater model dataset.

    Optionally select a different grid size.

    Parameters
    ----------
    ds : xarray.dataset
        A layer model dataset.
    model_name : str, optional
        name of the model. The default is None
    model_ws : str, optional
        workspace of the model. This is where modeldata is saved to. The
        default is None
    extent : list or tuple of length 4, optional
        The extent of the new grid. Get from ds when None. The default is None.
    delr : int, float, list, tuple or array, optional
        The gridsize along columns (dx). The default is 100. meter.
    delc : None, int, float, list, tuple or array, optional
        The gridsize along rows (dy). Set to delr when None. If None delc=delr
        The default is None.
    fill_nan : bool, optional
        if True nan values in the top, botm, kh and kv are filled using the
        fill_nan_top_botm_kh_kv function. Layers with only nan values in the
        botm are removed.
    extrapolate : bool, optional
        When true, extrapolate data-variables, into the sea or other areas with
        only nans. The default is True
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data. The default is 0.1.
    xorigin : int or float, optional
        lower left x coordinate of the model grid only used if angrot != 0.
        Default is 0.0.
    yorigin : int or float, optional
        lower left y coordinate of the model grid only used if angrot != 0.
        Default is 0.0.
    angrot : int or float, optinal
        the rotation of the grid in counter clockwise degrees, default is 0.0
    drop_attributes : bool, optional
        if True drop the attributes from the layer model dataset. Otherwise
        keep the attributes. Default is True.
    transport : bool, optional
        flag indicating whether dataset includes data for a groundwater
        transport model (GWT). Default is False, no transport.

    Returns
    -------
    ds : xarray.dataset
        the model Dataset.
    """
    if extent is None:
        extent = ds.attrs["extent"]

    # drop attributes
    if drop_attributes:
        ds = ds.copy()
        for attr in list(ds.attrs):
            del ds.attrs[attr]

    # convert dataset to grid
    logger.info("resample layer model data to structured modelgrid")
    ds = resample.ds_to_structured_grid(
        ds, extent, delr, delc, xorigin=xorigin, yorigin=yorigin, angrot=angrot
    )

    # add cell area variable
    if delc is None:
        delc = delr
    if isinstance(delr, (numbers.Number)) and isinstance(delc, (numbers.Number)):
        ds["area"] = ("y", "x"), ds.delr * ds.delc * np.ones(
            (ds.dims["y"], ds.dims["x"])
        )
    elif isinstance(delr, np.ndarray) and isinstance(delc, np.ndarray):
        ds["area"] = ("y", "x"), np.outer(delc, delr)
        ds["delr"] = ("x"), delr
        ds["delc"] = ("y"), delc
    else:
        raise TypeError("unexpected type for delr and/or delc")

    if extrapolate:
        ds = extrapolate_ds(ds)

    # add attributes
    ds = set_ds_attrs(ds, model_name, model_ws)
    ds.attrs["transport"] = int(transport)

    # fill nan's and add idomain
    if fill_nan:
        ds = fill_nan_top_botm_kh_kv(
            ds,
            anisotropy=anisotropy,
            fill_value_kh=fill_value_kh,
            fill_value_kv=fill_value_kv,
        )
    else:
        ds = set_idomain(ds, remove_nan_layers=False)

    return ds


def extrapolate_ds(ds, mask=None):
    """Fill missing data in layermodel based on nearest interpolation.

    Used for ensuring layer model contains data everywhere. Useful for
    filling in data beneath the sea for coastal groundwater models, or models
    near the border of the Netherlands.

    Parameters
    ----------
    ds : xarray.DataSet
        Model layer DataSet
    mask: np.ndarray, optional
        Boolean mask for each cell, with a value of True if its value needs to
        be determined. When mask is None, it is determined from the botm-
        variable. The default is None.

    Returns
    -------
    ds : xarray.DataSet
        filled layermodel
    """
    if mask is None:
        mask = np.isnan(ds["botm"]).all("layer").data
    if not mask.any():
        # all of the model cells are is inside the known area
        return ds
    if mask.all():
        raise (Exception("The model only contains NaNs"))
    if "gridtype" in ds.attrs and ds.gridtype == "vertex":
        x = ds.x.data
        y = ds.y.data
        dims = ("icell2d",)
    else:
        x, y = np.meshgrid(ds.x, ds.y)
        dims = ("y", "x")
    points = np.stack((x[~mask], y[~mask]), axis=1)
    xi = np.stack((x[mask], y[mask]), axis=1)
    # geneterate the tree only once, to increase speed
    tree = cKDTree(points)
    _, i = tree.query(xi)
    for key in ds:
        if not np.any([dim in ds[key].dims for dim in dims]):
            continue
        data = ds[key].data
        if ds[key].dims == dims:
            if np.isnan(data[mask]).sum() > 0:  # do not update if no NaNs
                data[mask] = data[~mask, i]
        elif ds[key].dims == ("layer",) + dims:
            for lay in range(len(ds["layer"])):
                if np.isnan(data[lay, mask]).sum() > 0:  # do not update if no NaNs
                    data[lay, mask] = data[lay, ~mask][i]
        else:
            logger.warning(
                f"Data variable '{key}' not extrapolated because "
                f"dimensions are not {dims}."
            )
            # raise (Exception(f"Dimensions {ds[key].dims} not supported"))
        # make sure to set the data (which for some reason is sometimes needed)
        ds[key].data = data
    return ds


def _get_structured_grid_ds(
    xedges,
    yedges,
    nlay=1,
    top=np.nan,
    botm=np.nan,
    xorigin=0.0,
    yorigin=0.0,
    angrot=0,
    attrs=None,
    crs=None,
):
    """Create an xarray dataset with structured grid geometry.

    Parameters
    ----------
    xedges : array_like
        A 1D array of the x coordinates of the grid edges.
    yedges : array_like
        A 1D array of the y coordinates of the grid edges.
    nlay : int, optional
        The number of layers in the grid. Default is 1.
    top : array_like, optional
        A 2D array of the top elevation of the grid cells. Default is NaN.
    botm : array_like, optional
        A 3D array of the bottom elevation of the grid cells. Default is NaN.
    xorigin : float, optional
        The x-coordinate origin of the grid. Default is 0.0.
    yorigin : float, optional
        The y-coordinate origin of the grid. Default is 0.0.
    angrot : float, optional
        The counter-clockwise rotation angle of the grid, in degrees.
        Default is 0.
    attrs : dict, optional
        A dictionary of attributes to add to the xarray dataset. Default is an
        empty dictionary.
    crs : dict or str, optional
        A dictionary or string describing the coordinate reference system of
        the grid. Default is None.

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset with the following data variables and coordinates:

        - top : a 2D array of the top elevation of the grid cells
        - botm : a 3D array of the bottom elevation of the grid cells
        - x : a 1D array of the x coordinates of the grid cell centers
        - y : a 1D array of the y coordinates of the grid cell centers
        - layer : a 1D array of the layer indices
        - xc : a 2D array of the x coordinates of the grid cell centers, after
          rotation if `angrot` is not 0.0 (optional)
        - yc : a 2D array of the y coordinates of the grid cell centers, after
          rotation if `angrot` is not 0.0 (optional)

        The dataset also includes the attributes specified in the `attrs`
        dictionary, and a coordinate reference system specified by `crs`, if
        provided.
    """

    if attrs is None:
        attrs = {}
    attrs.update({"gridtype": "structured"})

    # get extent from local grid edge coordinates
    extent = [
        np.min(xedges),
        np.max(xedges),
        np.min(yedges),
        np.max(yedges),
    ]

    # calculate centers
    xcenters = xedges[:-1] + np.diff(xedges) / 2.0
    ycenters = yedges[:-1] + np.diff(yedges) / 2.0

    resample._set_angrot_attributes(extent, xorigin, yorigin, angrot, attrs)

    coords = {
        "x": xcenters,
        "y": ycenters,
        "layer": range(nlay),
    }

    if angrot != 0.0:
        affine = resample.get_affine_mod_to_world(attrs)
        xc, yc = affine * np.meshgrid(xcenters, ycenters)
        coords["xc"] = (("y", "x"), xc)
        coords["yc"] = (("y", "x"), yc)
    else:
        coords["x"] += xorigin
        coords["y"] += yorigin

    dims = ("layer", "y", "x")
    ds = xr.Dataset(
        data_vars={
            "top": (dims[1:], top),
            "botm": (dims, botm),
        },
        coords=coords,
        attrs=attrs,
    )
    # set delr and delc
    delr = np.diff(xedges)
    if len(np.unique(delr)) == 1:
        ds.attrs["delr"] = np.unique(delr)[0]
    else:
        ds["delr"] = ("x"), delr
    delc = -np.diff(yedges)
    if len(np.unique(delc)) == 1:
        ds.attrs["delc"] = np.unique(delc)[0]
    else:
        ds["delc"] = ("y"), delc

    if crs is not None:
        ds.rio.set_crs(crs)
    return ds


def _get_vertex_grid_ds(
    x,
    y,
    xv,
    yv,
    cell2d,
    extent,
    nlay=1,
    top=np.nan,
    botm=np.nan,
    xorigin=0.0,
    yorigin=0.0,
    angrot=0.0,
    attrs=None,
    crs=None,
):
    """Create an xarray dataset with vertex-based grid geometry.

    Parameters
    ----------
    x : array_like
        A 1D array of the x coordinates of the grid cell centers.
    y : array_like
        A 1D array of the y coordinates of the grid cell centers.
    xv : array_like
        A 1D array of the x coordinates of the grid vertices.
    yv : array_like
        A 1D array of the y coordinates of the grid vertices.
    cell2d : array-like
        array-like with vertex grid cell2d info
    extent : list
        A list of [xmin, xmax, ymin, ymax] defining the extent of the model grid.
    nlay : int or sequence of ints, optional
        The number of layers in the grid, or a sequence of layer indices.
        Default is 1.
    top : array_like, optional
        A 2D array of the top elevation of the grid cells. Default is NaN.
    botm : array_like, optional
        A 3D array of the bottom elevation of the grid cells. Default is NaN.
    xorigin : float, optional
        The x-coordinate origin of the grid. Default is 0.0.
    yorigin : float, optional
        The y-coordinate origin of the grid. Default is 0.0.
    angrot : float, optional
        The counter-clockwise rotation angle of the grid, in degrees.
        Default is 0.0.
    attrs : dict, optional
        A dictionary of attributes to add to the xarray dataset. Default is an
        empty dictionary.
    crs : dict or str, optional
        A dictionary or string describing the coordinate reference system of
        the grid. Default is None.

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset with the following data variables and coordinates:

        - top : a 2D array of the top elevation of the grid cells
        - botm : a 3D array of the bottom elevation of the grid cells
        - x : a 1D array of the x coordinates of the grid cell centers
        - y : a 1D array of the y coordinates of the grid cell centers
        - layer : a 1D array of the layer indices
        - xv : a 1D array of the x coordinates of the grid vertices
        - yv : a 1D array of the y coordinates of the grid vertices

        The dataset also includes the attributes specified in the `attrs`
        dictionary, and a coordinate reference system specified by `crs`, if
        provided.
    """
    if attrs is None:
        attrs = {}

    attrs.update(
        {
            "extent": extent,
            "angrot": angrot,
            "xorigin": xorigin,
            "yorigin": yorigin,
            "gridtype": "vertex",
        }
    )

    if isinstance(nlay, int):
        layers = range(nlay)
    else:
        layers = nlay

    coords = {"layer": layers, "y": y, "x": x}
    dims = ("layer", "icell2d")
    ds = xr.Dataset(
        data_vars=dict(
            top=(dims[1:], top),
            botm=(dims, botm),
        ),
        coords=coords,
        attrs=attrs,
    )

    # add extra modelgrid information to ds
    ds["xv"] = ("iv", xv)
    ds["yv"] = ("iv", yv)

    # set extra grid information
    nodata = -1
    ncpl = len(x)
    ncvert_max = np.max([x[3] for x in cell2d])
    icvert = np.full((ncpl, ncvert_max), nodata)
    for i in range(ncpl):
        icvert[i, : cell2d[i][3]] = cell2d[i][4:]
    ds["icvert"] = ("icell2d", "icv"), icvert
    ds["icvert"].attrs["nodata"] = nodata

    if crs is not None:
        ds.rio.set_crs(crs)
    return ds


def get_ds(
    extent,
    delr=100.0,
    delc=None,
    model_name=None,
    model_ws=None,
    layer=None,
    top=0.0,
    botm=None,
    kh=10.0,
    kv=1.0,
    crs=EPSG_28992,
    xorigin=0.0,
    yorigin=0.0,
    angrot=0.0,
    attrs=None,
    extrapolate=True,
    fill_nan=True,
    transport=False,
    **kwargs,
):
    """Create a model dataset from scratch.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int, float, list, tuple or array, optional
        The gridsize along columns (dx). The default is 100. meter.
    delc : None, int, float, list, tuple or array, optional
        The gridsize along rows (dy). Set to delr when None. If None delc=delr. The
        default is None.
    model_name : str, optional
        name of the model. The default is None
    model_ws : str, optional
        workspace of the model. This is where modeldata is saved to. The default is
        None.
    layer : int, list, tuple or ndarray, optional
        The names or index of the layers of the model. When layer is an integer it is
        the number of layers. When layer is None, the number of layers is caluclated
        from botm. When botm is None as well, the number of layers is set to 10. The
        default is None.
    top : float, list or ndarray, optional
        The top of the model. It has to be of shape (len(y), len(x)) or it is
        transformed into that shape if top is a float. The default is 0.0.
    botm : list or ndarray, optional
        The botm of the model layers. It has to be of shape
        (len(layer), len(y), len(x)) or it is transformed to that shape if botm
        is or a list/array of len(layer). When botm is None, a botm is
        generated with a constant layer thickness of 10 meter. The default is
        None.
    kh : float, list or ndarray, optional
        The horizontal conductivity of the model layers. It has to be of shape
        (len(layer), len(y), len(x)) or it is transformed to that shape if kh
        is a float or a list/array of len(layer). The default is 10.0.
    kv : float, list or ndarray, optional
        The vertical conductivity of the model layers. It has to be of shape
        (len(layer), len(y), len(x)) or it is transformed to that shape if kv
        is a float or a list/array of len(layer). The default is 1.0.
    crs : int, optional
        The coordinate reference system of the model. The default is 28992.
    xorigin : float, optional
        x-position of the lower-left corner of the model grid. Only used when angrot is
        not 0. The defauls is 0.0.
    yorigin : float, optional
        y-position of the lower-left corner of the model grid. Only used when angrot is
        not 0. The defauls is 0.0.
    angrot : float, optional
        counter-clockwise rotation angle (in degrees) of the lower-left corner of the
        model grid. The default is 0.0
    attrs : dict, optional
        Attributes of the model dataset. The default is None.
    extrapolate : bool, optional
        When true, extrapolate data-variables, into the sea or other areas with
        only nans. The default is True
    fill_nan : bool, optional
        if True nan values in the top, botm, kh and kv are filled using the
        fill_nan_top_botm_kh_kv function. Layers with only nan values in the
        botm are removed.
    transport : bool, optional
        flag indicating whether dataset includes data for a groundwater
        transport model (GWT). Default is False, no transport.


    **kwargs : dict
        Kwargs are passed into mbase.to_ds. These can be the model_name
        or ds.

    Returns
    -------
    xr.Dataset
        The model dataset.
    """
    if delc is None:
        delc = delr

    if isinstance(delr, (tuple, list)):
        delr = np.asarray(delr)

    if isinstance(delc, (tuple, list)):
        delc = np.asarray(delc)

    if attrs is None:
        attrs = {}

    if layer is None:
        if botm is None:
            layer = 10
        else:
            layer = len(botm)
    if isinstance(layer, int):
        layer = np.arange(0, layer)
    if botm is None:
        botm = top - 10 * np.arange(1.0, len(layer) + 1)

    # check for nan
    for par in [top, botm, kh, kv]:
        if isinstance(par, numbers.Number):
            if np.isnan(par) and (extrapolate or fill_nan):
                raise ValueError(
                    "'extrapolate' and 'remove_nan_layer' should be "
                    "False when setting model parameters to NaN"
                )

    resample._set_angrot_attributes(extent, xorigin, yorigin, angrot, attrs)
    x, y = resample.get_xy_mid_structured(attrs["extent"], delr, delc)
    coords = dict(x=x, y=y, layer=layer)
    if angrot != 0.0:
        affine = resample.get_affine_mod_to_world(attrs)
        xc, yc = affine * np.meshgrid(x, y)
        coords["xc"] = (("y", "x"), xc)
        coords["yc"] = (("y", "x"), yc)

    def check_variable(var, shape):
        if isinstance(var, int):
            # the variable is a single integer
            var = float(var)
        if isinstance(var, float):
            # the variable is a single float
            var = np.full(shape, var)
        else:
            # assume the variable is an array of some kind
            if not isinstance(var, np.ndarray):
                var = np.array(var)
            if var.dtype != float:
                var = var.astype(float)
            if len(var.shape) == 1 and len(shape) == 3:
                # the variable is defined per layer
                assert len(var) == shape[0]
                var = var[:, np.newaxis, np.newaxis]
                var = np.repeat(np.repeat(var, shape[1], 1), shape[2], 2)
            else:
                assert var.shape == shape
        return var

    shape = (len(y), len(x))
    top = check_variable(top, shape)
    shape = (len(layer),) + shape
    botm = check_variable(botm, shape)
    kh = check_variable(kh, shape)
    kv = check_variable(kv, shape)

    dims = ["layer", "y", "x"]
    ds = xr.Dataset(
        data_vars=dict(
            top=(dims[1:], top),
            botm=(dims, botm),
            kh=(dims, kh),
            kv=(dims, kv),
        ),
        coords=coords,
        attrs=attrs,
    )
    ds = to_model_ds(
        ds,
        model_name=model_name,
        model_ws=model_ws,
        extent=extent,
        delr=delr,
        delc=delc,
        drop_attributes=False,
        extrapolate=extrapolate,
        fill_nan=fill_nan,
        transport=transport,
        **kwargs,
    )
    ds.rio.set_crs(crs)
    return ds
