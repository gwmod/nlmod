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
from ..dims.layers import calculate_thickness
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
    geotop_layers="HLc",
    geotop_k=None,
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
        layer of regis. call nlmod.read.regis.get_layer_names() to get a list of
        regis names.
    use_regis : bool, optional
        True if part of the layer model should be REGIS. The default is True.
    use_geotop : bool, optional
        True if part of the layer model should be geotop. The default is True.
    remove_nan_layers : bool, optional
        When True, layers which contain only NaNs for the botm array are removed.
        The default is True.
    geotop_layers : str or list of strings
        The regis layers to be replaced by geotop-layers
    geotop_k : pd.DataFrame, optional
        The DataFrame with information about kh and kv of the GeoTOP-data. This
        DataFrame must at least contain columns 'lithok' and 'kh'.

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
        geotop_ds = geotop.get_geotop(extent)

    if use_regis and use_geotop:
        combined_ds = add_geotop_to_regis_layers(
            regis_ds,
            geotop_ds,
            layers=geotop_layers,
            geotop_k=geotop_k,
            remove_nan_layers=remove_nan_layers,
        )

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

    if len(ds.x) == 0 or len(ds.y) == 0:
        msg = "No data found. Please supply valid extent in the Netherlands in RD-coordinates"
        raise (Exception(msg))

    # make sure layer names are regular strings
    ds["layer"] = ds["layer"].astype(str)

    # make sure y is descending
    if (ds["y"].diff("y") > 0).all():
        ds = ds.isel(y=slice(None, None, -1))

    # slice layers
    if botm_layer is not None:
        ds = ds.sel(layer=slice(botm_layer))

    # rename bottom to botm, as it is called in FloPy
    ds = ds.rename_vars({"bottom": "botm"})

    if remove_nan_layers:
        # only keep layers with at least one active cell
        ds = ds.sel(layer=~(np.isnan(ds["botm"])).all(ds["botm"].dims[1:]))
        if len(ds.layer) == 0:
            msg = "No data found. Please supply valid extent in the Netherlands in RD-coordinates"
            raise (Exception(msg))

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


def add_geotop_to_regis_layers(
    rg, gt, layers="HLc", geotop_k=None, remove_nan_layers=True
):
    """Combine geotop and regis in such a way that the one or more layers in
    Regis are replaced by the geo_eenheden of geotop.

    Parameters
    ----------
    rg : xarray.DataSet
        regis dataset
    gt : xarray.DataSet
        geotop dataset
    layers : str or list of strings
        The regis layers to be replaced by geotop-layers
    geotop_k : pd.DataFrame, optional
        The DataFrame with information about kh and kv of the GeoTOP-data. This
        DataFrame must at least contain columns 'lithok' and 'kh'.
    remove_nan_layers : bool, optional
        When True, layers with only 0 or NaN thickness are removed. The default is True.

    Returns
    -------
    gt: xr.DataSet
        combined dataset
    """
    if isinstance(layers, str):
        layers = [layers]
    if geotop_k is None:
        geotop_k = geotop.get_lithok_props()
    for layer in layers:
        # transform geotop data into layers
        gtl = geotop.to_model_layers(gt)

        # make sure top is 3d
        assert "layer" in rg["top"].dims, "Top of regis must be 3d"
        assert "layer" in gtl["top"].dims, "Top of geotop layers must be 3d"

        # only keep the part of layers inside the regis layer
        top = rg["top"].loc[layer]
        bot = rg["botm"].loc[layer]
        gtl["top"] = gtl["top"].where(top > gtl["top"], top)
        gtl["top"] = gtl["top"].where(bot < gtl["top"], bot)
        gtl["botm"] = gtl["botm"].where(top > gtl["botm"], top)
        gtl["botm"] = gtl["botm"].where(bot < gtl["botm"], bot)

        if remove_nan_layers:
            # drop layers with a remaining thickness of 0 (or NaN) everywhere
            th = calculate_thickness(gtl)
            gtl = gtl.sel(layer=(th > 0).any(th.dims[1:]))

        # add kh and kv to gt
        gt = geotop.add_kh_and_kv(gt, geotop_k)

        # add kh and kv from gt to gtl
        gtl = geotop.aggregate_to_ds(gt, gtl)

        # add gtl-layers to rg-layers
        if rg.layer.data[0] == layer:
            layer_order = np.concatenate([gtl.layer, rg.layer])
        elif rg.layer.data[-1] == layer:
            layer_order = np.concatenate([rg.layer, gtl.layer])
        else:
            lay = np.where(rg.layer == layer)[0][0]
            layer_order = np.concatenate(
                [rg.layer[:lay], gtl.layer, rg.layer[lay + 1 :]]
            )
        # call xr.concat with rg first, so we keep attributes of rg
        rg = xr.concat((rg.sel(layer=rg.layer[rg.layer != layer]), gtl), "layer")
        # we will then make sure the layer order is right
        rg = rg.reindex({"layer": layer_order})
    return rg


def get_layer_names():
    """get all the available regis layer names.

    Returns
    -------
    layer_names : np.array
        array with names of all the regis layers.
    """

    layer_names = xr.open_dataset(REGIS_URL).layer.astype(str).values

    return layer_names


def get_legend(kind="REGIS"):
    """Get a legend (DataFrame) with the colors of REGIS-layers.

    These colors can be used when plotting cross-sections.
    """
    allowed_kinds = ["REGIS", "GeoTOP", "combined"]
    if kind not in allowed_kinds:
        raise (Exception(f"Only allowed values for kind are {allowed_kinds}"))
    if kind in ["REGIS", "combined"]:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fname = os.path.join(dir_path, "..", "data", "regis_2_2.gleg")
        leg_regis = read_gleg(fname)
        if kind == "REGIS":
            return leg_regis
    if kind in ["GeoTOP", "combined"]:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fname = os.path.join(dir_path, "..", "data", "geotop", "geotop.gleg")
        leg_geotop = read_gleg(fname)
        if kind == "GeoTOP":
            return leg_geotop
    # return a combination of regis and geotop
    leg = pd.concat((leg_regis, leg_geotop))
    # drop duplicates, keeping first occurrences (from regis)
    leg = leg.loc[~leg.index.duplicated(keep="first")]
    return leg


def get_legend_lithoclass():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, "..", "data", "geotop", "Lithoklasse.voleg")
    leg = read_voleg(fname)
    return leg


def get_legend_lithostratigraphy():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, "..", "data", "geotop", "Lithostratigrafie.voleg")
    leg = read_voleg(fname)
    return leg


def read_gleg(fname):
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


def read_voleg(fname):
    leg = pd.read_csv(
        fname,
        sep="\t",
        header=None,
        names=["code", "naam", "r", "g", "b", "a", "beschrijving"],
    )
    leg.set_index("code", inplace=True)
    clrs = np.array(leg.loc[:, ["r", "g", "b"]])
    clrs = [tuple(rgb / 255.0) for rgb in clrs]
    leg["color"] = clrs
    leg = leg.drop(["r", "g", "b", "a"], axis=1)
    return leg
