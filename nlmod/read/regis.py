# -*- coding: utf-8 -*-
"""function to project regis, or a combination of regis and geotop, data on a
modelgrid."""
import datetime as dt
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from .. import cache
from . import geotop

logger = logging.getLogger(__name__)

REGIS_URL = "http://www.dinodata.nl:80/opendap/REGIS/REGIS.nc"
# REGIS_URL = 'https://www.dinodata.nl/opendap/hyrax/REGIS/REGIS.nc'


@cache.cache_netcdf
def get_combined_layer_models(
    extent,
    regis_botm_layer="AKc",
    use_regis=True,
    use_geotop=True,
    remove_nan_layers=True,
):
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
    remove_nan_layers : bool, optional
        When True, layers which contain only NaNs for the botm array are removed.
        The default is True.

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
        regis_ds = get_regis(
            extent, regis_botm_layer, remove_nan_layers=remove_nan_layers
        )
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
def get_regis(
    extent,
    botm_layer="AKc",
    variables=("top", "botm", "kh", "kv"),
    remove_nan_layers=True,
):
    """get a regis dataset projected on the modelgrid.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    botm_layer : str, optional
        regis layer that is used as the bottom of the model. This layer is
        included in the model. the Default is "AKc" which is the bottom
        layer of regis. call nlmod.read.regis.get_layer_names() to get a list
        of regis names.
    variables : tuple, optional
        a tuple of the variables to keep from the regis Dataset. Possible
        entries in the list are 'top', 'botm', 'kD', 'c', 'kh', 'kv', 'sdh' and
        'sdv'. The default is ("top", "botm", "kh", "kv").
    remove_nan_layers : bool, optional
        When True, layers which contain only NaNs for the botm array are removed.
        The default is True.

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

    if remove_nan_layers:
        # only keep layers with at least one active cell
        ds = ds.sel(layer=~(np.isnan(ds["botm"])).all(ds["botm"].dims[1:]))

    # slice data vars
    ds = ds[list(variables)]

    ds.attrs["extent"] = extent
    for datavar in ds:
        ds[datavar].attrs["grid_mapping"] = "crs"
        ds[datavar].attrs["source"] = "REGIS"
        ds[datavar].attrs["url"] = REGIS_URL
        ds[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
        if datavar in ["top", "botm"]:
            ds[datavar].attrs["units"] = "mNAP"
        elif datavar in ["kh", "kv"]:
            ds[datavar].attrs["units"] = "m/day"
        # set _FillValue to NaN, otherise problems with caching will arise
        ds[datavar].encoding["_FillValue"] = np.NaN

    # set the crs to dutch rd-coordinates
    ds.rio.set_crs(28992)

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
            regis_ds["botm"][layer_no] - float_correction
        )
        geotop_ds["top"][lay] = xr.where(mask1, np.nan, geotop_ds["top"][lay])
        geotop_ds["botm"][lay] = xr.where(
            mask1, np.nan, geotop_ds["botm"][lay]
        )
        geotop_ds["kh"][lay] = xr.where(mask1, np.nan, geotop_ds["kh"][lay])
        geotop_ds["kv"][lay] = xr.where(mask1, np.nan, geotop_ds["kv"][lay])

        # Alle geotop cellen waarvan de bodem onder de onderkant van het holoceen ligt, krijgen als bodem de onderkant van het holoceen
        mask2 = geotop_ds["botm"][lay] < regis_ds["botm"][layer_no]
        geotop_ds["botm"][lay] = xr.where(
            mask2 * (~mask1),
            regis_ds["botm"][layer_no],
            geotop_ds["botm"][lay],
        )

        # Alle geotop cellen die boven de bovenkant van het holoceen liggen worden inactief
        mask3 = geotop_ds["botm"][lay] >= (
            regis_ds["top"][layer_no] - float_correction
        )
        geotop_ds["top"][lay] = xr.where(mask3, np.nan, geotop_ds["top"][lay])
        geotop_ds["botm"][lay] = xr.where(
            mask3, np.nan, geotop_ds["botm"][lay]
        )
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
            mask5, np.nan, geotop_ds["botm"][lay]
        )
        geotop_ds["kh"][lay] = xr.where(mask5, np.nan, geotop_ds["kh"][lay])
        geotop_ds["kv"][lay] = xr.where(mask5, np.nan, geotop_ds["kv"][lay])
        if (mask2 * (~mask1)).sum() > 0:
            logger.info(
                f"regis holoceen snijdt door laag {geotop_ds.layer[lay].values}"
            )

    top[: len(geotop_ds.layer), :, :] = geotop_ds["top"].data
    top[len(geotop_ds.layer) :, :, :] = regis_ds["top"].data[layer_no + 1 :]

    bot[: len(geotop_ds.layer), :, :] = geotop_ds["botm"].data
    bot[len(geotop_ds.layer) :, :, :] = regis_ds["botm"].data[layer_no + 1 :]

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
    mask = (
        regis_geotop_ds["top"] - regis_geotop_ds["botm"]
    ) < float_correction
    for key in ["top", "botm", "kh", "kv"]:
        regis_geotop_ds[key] = xr.where(mask, np.nan, regis_geotop_ds[key])
        regis_geotop_ds[key].attrs["source"] = "REGIS/geotop"
        regis_geotop_ds[key].attrs["regis_url"] = regis_ds[key].url
        regis_geotop_ds[key].attrs["geotop_url"] = geotop_ds[key].url
        regis_geotop_ds[key].attrs["date"] = dt.datetime.now().strftime(
            "%Y%m%d"
        )
        if key in ["top", "botm"]:
            regis_geotop_ds[key].attrs["units"] = "mNAP"
        elif key in ["kh", "kv"]:
            regis_geotop_ds[key].attrs["units"] = "m/day"

    return regis_geotop_ds


def get_layer_names():
    """get all the available regis layer names.

    Returns
    -------
    layer_names : np.array
        array with names of all the regis layers.
    """

    layer_names = xr.open_dataset(REGIS_URL).layer.astype(str).values

    return layer_names


def get_legend():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, "..", "data", "regis_2_2.gleg")
    leg = pd.read_csv(
        fname,
        sep="\t",
        header=None,
        names=["naam", "beschrijving", "r", "g", "b", "a", "x"],
    )
    leg["naam"] = leg["naam"].str.replace("-", "")
    leg.set_index("naam", inplace=True)
    clrs = np.array(leg.loc[:, ["r", "g", "b"]])
    clrs = [tuple(rgb / 255.0) for rgb in clrs]
    leg["color"] = clrs
    leg = leg.drop(["x", "r", "g", "b", "a"], axis=1)
    return leg
