# -*- coding: utf-8 -*-
"""function to project regis, or a combination of regis and geotop, data on a
modelgrid."""
import datetime as dt
import logging

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from .. import cache, mdims
from . import geotop

logger = logging.getLogger(__name__)

REGIS_URL = "http://www.dinodata.nl:80/opendap/REGIS/REGIS.nc"
# REGIS_URL = 'https://www.dinodata.nl/opendap/hyrax/REGIS/REGIS.nc'


@cache.cache_netcdf
def get_combined_layer_models(extent, regis_botm_layer="AKc",
                              use_regis=True, use_geotop=True):
    """combine layer models into a single layer model.

    Possibilities so far include:
        - use_regis -> full model based on regis
        - use_regis and use_geotop -> holoceen of REGIS is filled with geotop


    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    regis_botm_layer : binary str, optional
        regis layer that is used as the bottom of the model. This layer is
        included in the model. the Default is 'AKc' which is the bottom
        layer of regis. call nlmod.regis.get_layer_names() to get a list of
        regis names.
    use_regis : bool, optional
        True if part of the layer model should be REGIS. The default is True.
    use_geotop : bool, optional
        True if part of the layer model should be geotop. The default is True.

    Returns
    -------
    combined_ds : xarray dataset
        combination of layer models.

    Raises
    ------
    ValueError
        if an invalid combination of layers is used.
    """

    if use_regis:
        regis_ds = get_regis(extent, regis_botm_layer)
    else:
        raise ValueError("layer models without REGIS not supported")

    if use_geotop:
        geotop_ds = geotop.get_geotop(extent, regis_ds)

    if use_regis and use_geotop:
        regis_geotop_ds = add_geotop_to_regis_hlc(regis_ds, geotop_ds)

        combined_ds = regis_geotop_ds
    elif use_regis:
        combined_ds = regis_ds
    else:
        raise ValueError("combination of model layers not supported")

    return combined_ds


@cache.cache_netcdf
def get_regis(extent, botm_layer="AKc"):
    """get a regis dataset projected on the modelgrid.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int or float, optional
        cell size along rows, equal to dx. The default is 100 m.
    delc : int or float, optional
        cell size along columns, equal to dy. The default is 100 m.
    botm_layer : str, optional
        regis layer that is used as the bottom of the model. This layer is
        included in the model. the Default is "AKc" which is the bottom
        layer of regis. call nlmod.regis.get_layer_names() to get a list of
        regis names.

    Returns
    -------
    regis_ds : xarray dataset
        dataset with regis data projected on the modelgrid.
    """

    ds = xr.open_dataset(REGIS_URL, decode_times=False)

    # set x and y dimensions to cell center
    ds["x"] = ds.x_bounds.mean("bounds")
    ds["y"] = ds.y_bounds.mean("bounds")

    # slice extent
    ds = ds.sel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))

    # make sure layer names are regular strings
    ds["layer"] = ds["layer"].astype(str)

    # slice layers
    if botm_layer is not None:
        ds = ds.sel(layer=slice(botm_layer))

    # rename bottom to botm, as it is called in FloPy
    ds = ds.rename_vars({"bottom": "botm"})

    # slice data vars
    ds = ds[["top", "botm", "kh", "kv"]]

    ds.attrs["extent"] = extent
    for datavar in ds:
        ds[datavar].attrs["source"] = "REGIS"
        ds[datavar].attrs["url"] = REGIS_URL
        ds[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
        if datavar in ["top", "botm"]:
            ds[datavar].attrs["units"] = "mNAP"
        elif datavar in ["kh", "kv"]:
            ds[datavar].attrs["units"] = "m/day"
        # if "_FillValue" in ds[datavar].attrs:
            # remove _FillValue as this may cause problems with caching
        #     del ds[datavar].attrs["_FillValue"]
    return ds


def to_model_ds(ds, model_name=None, model_ws=None, extent=None, delr=100.,
                delc=None, remove_nan_layers=True, extrapolate=True,
                anisotropy=10, fill_value_kh=1., fill_value_kv=0.1):
    """
    Transform a regis datset to a model dataset with another resultion.

    Parameters
    ----------
    ds : xarray.dataset
        The regis dataset.
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
        THe gridsize along rows. Set to delr when None. The default is None.
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
    if delc is None:
        delc = delr
    # check extent
    extent2, _, _ = fit_extent_to_regis(extent, delr, delc)
    for coord1, coord2 in zip(extent, extent2):
        if coord1 != coord2:
            raise ValueError(
                ("extent not fitted to regis please fit to regis first, "
                 "use the nlmod.regis.fit_extent_to_regis function"))

    if remove_nan_layers:
        nlay, lay_sel = get_non_nan_layers(ds)
        ds = ds.sel(layer=lay_sel)
        logger.info(f"removing {nlay} nan layers from the model")

    # convert regis dataset to grid
    logger.info("resample regis data to structured modelgrid")
    ds = mdims.resample_dataset_to_structured_grid(ds, extent, delr, delc)

    # drop attributes
    for attr in list(ds.attrs):
        del ds.attrs[attr]

    # and add new attributes
    ds.attrs["gridtype"] = "structured"
    ds.attrs["extent"] = extent
    ds.attrs["delr"] = delr
    ds.attrs["delc"] = delc

    if extrapolate:
        ds = extrapolate_ds(ds)

    # add attributes
    ds = mdims.mbase.set_ds_attrs(ds, model_name, model_ws)
    # fill nan's and add idomain
    ds = mdims.mlayers.complete_ds(ds, anisotropy=anisotropy,
                                   fill_value_kh=fill_value_kh,
                                   fill_value_kv=fill_value_kv)
    return ds


def extrapolate_ds(ds, mask=None):
    """Extrapolate data-variables (into the sea or other areas with only nans)"""
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
            raise(Exception(f"Dimensions {ds[key].dims} not supported"))
        # make sure to set the data (which for some reason is sometimes needed)
        ds[key].data = data
    return ds


def add_geotop_to_regis_hlc(regis_ds, geotop_ds, float_correction=0.001):
    """Combine geotop and regis in such a way that the holoceen in Regis is
    replaced by the geo_eenheden of geotop.

    Parameters
    ----------
    regis_ds: xarray.DataSet
        regis dataset
    geotop_ds: xarray.DataSet
        geotop dataset
    float_correction: float
        due to floating point precision some floating point numbers that are
        the same are not recognised as the same. Therefore this correction is
        used.

    Returns
    -------
    regis_geotop_ds: xr.DataSet
        combined dataset
    """
    regis_geotop_ds = xr.Dataset()

    # find holoceen (remove all layers above Holoceen)
    layer_no = np.where((regis_ds.layer == "HLc").values)[0][0]
    new_layers = np.append(
        geotop_ds.layer.data, regis_ds.layer.data[layer_no + 1 :].astype("<U8")
    ).astype("O")

    top = xr.DataArray(
        dims=("layer", "y", "x"),
        coords={"y": geotop_ds.y, "x": geotop_ds.x, "layer": new_layers},
    )
    bot = xr.DataArray(
        dims=("layer", "y", "x"),
        coords={"y": geotop_ds.y, "x": geotop_ds.x, "layer": new_layers},
    )
    kh = xr.DataArray(
        dims=("layer", "y", "x"),
        coords={"y": geotop_ds.y, "x": geotop_ds.x, "layer": new_layers},
    )
    kv = xr.DataArray(
        dims=("layer", "y", "x"),
        coords={"y": geotop_ds.y, "x": geotop_ds.x, "layer": new_layers},
    )

    # haal overlap tussen geotop en regis weg
    logger.info("cut geotop layer based on regis holoceen")
    for lay in range(geotop_ds.dims["layer"]):
        # Alle geotop cellen die onder de onderkant van het holoceen liggen worden inactief
        mask1 = geotop_ds["top"][lay] <= (
            regis_ds["botm"][layer_no] - float_correction)
        geotop_ds["top"][lay] = xr.where(mask1, np.nan, geotop_ds["top"][lay])
        geotop_ds["botm"][lay] = xr.where(
            mask1, np.nan, geotop_ds["botm"][lay])
        geotop_ds["kh"][lay] = xr.where(mask1, np.nan, geotop_ds["kh"][lay])
        geotop_ds["kv"][lay] = xr.where(mask1, np.nan, geotop_ds["kv"][lay])

        # Alle geotop cellen waarvan de bodem onder de onderkant van het holoceen ligt, krijgen als bodem de onderkant van het holoceen
        mask2 = geotop_ds["botm"][lay] < regis_ds["botm"][layer_no]
        geotop_ds["botm"][lay] = xr.where(
            mask2 * (~mask1), regis_ds["botm"][layer_no], geotop_ds["botm"][lay])

        # Alle geotop cellen die boven de bovenkant van het holoceen liggen worden inactief
        mask3 = geotop_ds["botm"][lay] >= (
            regis_ds["top"][layer_no] - float_correction)
        geotop_ds["top"][lay] = xr.where(mask3, np.nan, geotop_ds["top"][lay])
        geotop_ds["botm"][lay] = xr.where(
            mask3, np.nan, geotop_ds["botm"][lay])
        geotop_ds["kh"][lay] = xr.where(mask3, np.nan, geotop_ds["kh"][lay])
        geotop_ds["kv"][lay] = xr.where(mask3, np.nan, geotop_ds["kv"][lay])

        # Alle geotop cellen waarvan de top boven de top van het holoceen ligt, krijgen als top het holoceen van regis
        mask4 = geotop_ds["top"][lay] >= regis_ds["top"][layer_no]
        geotop_ds["top"][lay] = xr.where(
            mask4 * (~mask3), regis_ds["top"][layer_no], geotop_ds["top"][lay]
        )

        # overal waar holoceen inactief is, wordt geotop ook inactief
        mask5 = regis_ds["botm"][layer_no].isnull()
        geotop_ds["top"][lay] = xr.where(mask5, np.nan, geotop_ds["top"][lay])
        geotop_ds["botm"][lay] = xr.where(
            mask5, np.nan, geotop_ds["botm"][lay])
        geotop_ds["kh"][lay] = xr.where(mask5, np.nan, geotop_ds["kh"][lay])
        geotop_ds["kv"][lay] = xr.where(mask5, np.nan, geotop_ds["kv"][lay])
        if (mask2 * (~mask1)).sum() > 0:
            logger.info(
                f"regis holoceen snijdt door laag {geotop_ds.layer[lay].values}"
            )

    top[: len(geotop_ds.layer), :, :] = geotop_ds["top"].data
    top[len(geotop_ds.layer) :, :, :] = regis_ds["top"].data[layer_no + 1 :]

    bot[:len(geotop_ds.layer), :, :] = geotop_ds["botm"].data
    bot[len(geotop_ds.layer):, :, :] = regis_ds["botm"].data[layer_no + 1:]

    kh[: len(geotop_ds.layer), :, :] = geotop_ds["kh"].data
    kh[len(geotop_ds.layer) :, :, :] = regis_ds["kh"].data[layer_no + 1 :]

    kv[: len(geotop_ds.layer), :, :] = geotop_ds["kv"].data
    kv[len(geotop_ds.layer) :, :, :] = regis_ds["kv"].data[layer_no + 1 :]

    regis_geotop_ds["top"] = top
    regis_geotop_ds["botm"] = bot
    regis_geotop_ds["kh"] = kh
    regis_geotop_ds["kv"] = kv

    _ = [
        regis_geotop_ds.attrs.update({key: item})
        for key, item in regis_ds.attrs.items()
    ]

    # maak top, bot, kh en kv nan waar de laagdikte 0 is
    mask = (regis_geotop_ds["top"] -
            regis_geotop_ds["botm"]) < float_correction
    for key in ["top", "botm", "kh", "kv"]:
        regis_geotop_ds[key] = xr.where(mask, np.nan, regis_geotop_ds[key])
        regis_geotop_ds[key].attrs["source"] = "REGIS/geotop"
        regis_geotop_ds[key].attrs["regis_url"] = regis_ds[key].url
        regis_geotop_ds[key].attrs["geotop_url"] = geotop_ds[key].url
        regis_geotop_ds[key].attrs["date"] = dt.datetime.now().strftime(
            "%Y%m%d")
        if key in ["top", "botm"]:
            regis_geotop_ds[key].attrs["units"] = "mNAP"
        elif key in ["kh", "kv"]:
            regis_geotop_ds[key].attrs["units"] = "m/day"

    return regis_geotop_ds


def fit_extent_to_regis(extent, delr, delc, cs_regis=100.0):
    """redifine extent and calculate the number of rows and columns.

    The extent will be redefined so that the borders of the grid (xmin, xmax,
    ymin, ymax) correspond with the borders of the regis grid.

    Parameters
    ----------
    extent : list, tuple or np.array
        original extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    cs_regis : int or float, optional
        cell size of regis grid. The default is 100..

    Returns
    -------
    extent : list, tuple or np.array
        adjusted extent
    nrow : int
        number of rows.
    ncol : int
        number of columns.
    """
    if isinstance(extent, list):
        extent = extent.copy()
    elif isinstance(extent, (tuple, np.ndarray)):
        extent = list(extent)
    else:
        raise TypeError(
            f"expected extent of type list, tuple or np.ndarray, got {type(extent)}"
        )

    logger.info(f"redefining current extent: {extent}, fit to regis raster")

    for d in [delr, delc]:
        available_cell_sizes = [
            10.0,
            20.0,
            25.0,
            50.0,
            100.0,
            200.0,
            400.0,
            500.0,
            800.0,
        ]
        if float(d) not in available_cell_sizes:
            raise NotImplementedError(
                "only this cell sizes can be used for " f"now -> {available_cell_sizes}"
            )

    # if xmin ends with 100 do nothing, otherwise fit xmin to regis cell border
    if extent[0] % cs_regis != 0:
        extent[0] -= extent[0] % cs_regis

    # get number of columns
    ncol = int(np.ceil((extent[1] - extent[0]) / delr))
    extent[1] = extent[0] + (ncol * delr)  # round xmax up to close grid

    # if ymin ends with 100 do nothing, otherwise fit ymin to regis cell border
    if extent[2] % cs_regis != 0:
        extent[2] -= extent[2] % cs_regis

    nrow = int(np.ceil((extent[3] - extent[2]) / delc))  # get number of rows
    extent[3] = extent[2] + (nrow * delc)  # round ymax up to close grid

    logger.info(f"new extent is {extent} model has {nrow} rows and {ncol} columns")

    return extent, nrow, ncol


def get_non_nan_layers(raw_layer_mod, data_var="botm"):
    """get number and name of layers based on the number of non-nan layers.

    Parameters
    ----------
    raw_layer_mod : xarray.Dataset
        dataset with raw layer model from regis or geotop.
    data_var : str
        data var that is used to check if layer mod contains nan values

    Returns
    -------
    nlay : int
        number of active layers within regis_ds_raw.
    lay_sel : list of str
        names of the active layers.
    """
    logger.info("find active layers in raw layer model")

    bot_raw_all = raw_layer_mod[data_var]
    lay_sel = []
    for lay in bot_raw_all.layer.data:
        if not bot_raw_all.sel(layer=lay).isnull().all():
            lay_sel.append(lay)
    nlay = len(lay_sel)

    logger.info(f"there are {nlay} active layers within the extent")

    return nlay, lay_sel


def get_layer_names():
    """get all the available regis layer names.

    Returns
    -------
    layer_names : np.array
        array with names of all the regis layers.
    """

    layer_names = xr.open_dataset(REGIS_URL).layer.values

    return layer_names


def extrapolate_regis(regis_ds):
    """Fill missing data in layermodel based on nearest interpolation.

    Used for ensuring layer model contains data everywhere. Useful for
    filling in data beneath the sea for coastal groundwater models.

    Parameters
    ----------
    regis_ds : xarray.DataSet
        REGIS DataSet

    Returns
    -------
    regis_ds : xarray.DataSet
        filled REGIS layermodel with nearest interpolation
    """
    # fill layermodel with nearest interpolation (usually for filling in data
    # under the North Sea)
    mask = np.isnan(regis_ds["top"]).all("layer")
    if not np.any(mask):
        # all of the model are is inside
        logger.info("No missing data to extrapolate")
        return regis_ds
    x, y = np.meshgrid(regis_ds.x, regis_ds.y)
    points = (x[~mask], y[~mask])
    xi = (x[mask], y[mask])
    for key in list(regis_ds.keys()):
        data = regis_ds[key].data
        for lay in range(len(regis_ds.layer)):
            values = data[lay][~mask]
            data[lay][mask] = griddata(points, values, xi, method="nearest")
    return regis_ds
