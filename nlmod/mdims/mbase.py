import datetime as dt
import numpy as np
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
