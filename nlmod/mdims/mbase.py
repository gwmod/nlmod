import datetime as dt
import numpy as np
import xarray as xr
import logging


from scipy.spatial import cKDTree

from . import resample, mlayers
from .. import util

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
    ds.attrs["model_dataset_created_on"] = dt.datetime.now().strftime(fmt)

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
    remove_nan_layers=True,
    extrapolate=True,
    anisotropy=10,
    fill_value_kh=1.0,
    fill_value_kv=0.1,
    xorigin=0.0,
    yorigin=0.0,
    angrot=0.0,
    drop_attributes=True,
):
    """
    Transform a regis datset to a model dataset with another resolution.

    Parameters
    ----------
    ds : xarray.dataset
        A layer model dataset.
    model_name : str, optional
        name of the model. THe default is None
    model_ws : str, optional
        workspace of the model. This is where modeldata is saved to. The
        default is None
    extent : list or tuple of length 4, optional
        The extent of the new grid. Get from ds when None. The default is None.
    delr : float, optional
        The gridsize along columns. The default is 100. meter.
    delc : float, optional
        The gridsize along rows. Set to delr when None. The default is None.
    remove_nan_layers : bool, optional
        if True regis and geotop layers with only nans are removed from the
        model. if False nan layers are kept which might be usefull if you want
        to keep some layers that exist in other models. The default is True.
    extrapolate : bool, optional
        When true, extrapolate data-variables, into the sea or other areas with
        only nans. THe default is True
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 1.0.

    Raises
    ------
    ValueError
        if the supplied extent does not fit delr and delc

    Returns
    -------
    ds : xarray.dataset
        THe model Dataset.

    """
    if extent is None:
        extent = ds.attrs["extent"]

    # drop attributes
    if drop_attributes:
        ds = ds.copy()
        for attr in list(ds.attrs):
            del ds.attrs[attr]

    # convert regis dataset to grid
    logger.info("resample layer model data to structured modelgrid")
    ds = resample.resample_dataset_to_structured_grid(
        ds, extent, delr, delc, xorigin=xorigin, yorigin=yorigin, angrot=angrot
    )

    if extrapolate:
        ds = extrapolate_ds(ds)

    # add attributes
    ds = set_ds_attrs(ds, model_name, model_ws)

    # fill nan's and add idomain
    ds = mlayers.fill_nan_top_botm_kh_kv(
        ds,
        anisotropy=anisotropy,
        fill_value_kh=fill_value_kh,
        fill_value_kv=fill_value_kv,
        remove_nan_layers=remove_nan_layers,
    )
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
    if ds.gridtype == "vertex":
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
            data[mask] = data[~mask][i]
        elif ds[key].dims == ("layer",) + dims:
            for lay in range(len(ds["layer"])):
                data[lay][mask] = data[lay][~mask][i]
        else:
            raise (Exception(f"Dimensions {ds[key].dims} not supported"))
        # make sure to set the data (which for some reason is sometimes needed)
        ds[key].data = data
    return ds


def get_default_ds(
    extent,
    delr=100.0,
    delc=None,
    model_name=None,
    model_ws=None,
    layer=10,
    top=0.0,
    botm=None,
    kh=10.0,
    kv=1.0,
    crs=28992,
    xorigin=0.0,
    yorigin=0.0,
    angrot=0.0,
    attrs=None,
    **kwargs,
):
    """
    Create a model dataset from scratch, so without a layer model.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : float, optional
        The gridsize along columns. The default is 100. meter.
    delc : float, optional
        The gridsize along rows. Set to delr when None. The default is None.
    layer : int, list, tuple or ndarray, optional
        The layers of the model. When layer is an integer it is the number of
        layers. The default is 10.
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
        THe coordinate reference system of the model. The default is 28992.
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
    if attrs is None:
        attrs = {}
    if isinstance(layer, int):
        layer = np.arange(1, layer + 1)
    if botm is None:
        botm = top - 10 * np.arange(1.0, len(layer) + 1)
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
    shape = (len(layer), len(y), len(x))
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
        **kwargs,
    )
    ds.rio.set_crs(crs)
    return ds
