import datetime as dt
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from .. import cache
from ..dims.layers import calculate_thickness, remove_layer_dim_from_top
from . import geotop

logger = logging.getLogger(__name__)

REGIS_URL = "https://dinodata.nl/opendap/REGIS/REGIS.nc"


@cache.cache_netcdf()
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


@cache.cache_netcdf()
def get_regis(
    extent,
    botm_layer="AKc",
    variables=("top", "botm", "kh", "kv"),
    remove_nan_layers=True,
    drop_layer_dim_from_top=True,
    probabilities=False,
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
        When True, layers that do not occur in the requested extent (layers that contain
        only NaN values for the botm array) are removed. The default is True.
    drop_layer_dim_from_top : bool, optional
        When True, fill NaN values in top and botm and drop the layer dimension from
        top. This will transform top and botm to the data model in MODFLOW. An advantage
        of this data model is that the layer model is consistent by definition, with no
        possibilities of gaps between layers. The default is True.
    probabilities : bool, optional
        if True, also download probability data. The default is False.

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
        raise (ValueError(msg))

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

    if drop_layer_dim_from_top:
        ds = remove_layer_dim_from_top(ds)

    # slice data vars
    if variables is not None:
        if probabilities:
            variables = variables + ("sdh", "sdv")
        ds = ds[list(variables)]

    ds.attrs["gridtype"] = "structured"
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

    # set the crs to dutch rd-coordinates
    ds.rio.set_crs(28992)

    return ds


def add_geotop_to_regis_layers(
    rg,
    gt,
    layers="HLc",
    geotop_k=None,
    remove_nan_layers=True,
    anisotropy=1.0,
    gt_layered=None,
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
    anisotropy : float, optional
        The anisotropy value (kh/kv) used when there are no kv values in df. The
        default is 1.0.
    gt_layered : xarray.Dataset
        A layered representation of the geotop-dataset. By supplying this parameter, the
        user can change the GeoTOP-layering, which is usueally defined by
        nlmod.read.geotop.to_model_layers(gt).

    Returns
    -------
    gt: xr.DataSet
        combined dataset
    """
    if isinstance(layers, str):
        layers = [layers]

    # make sure geotop dataset contains kh and kv
    if "kh" not in gt or "kv" not in gt:
        if "kv" in gt:
            logger.info(
                f"Calculating kh of geotop by multiplying kv with an anisotropy of {anisotropy}"
            )
            gt["kh"] = gt["kv"] * anisotropy
        elif "kh" in gt:
            logger.info(
                f"Calculating kv of geotop by dividing kh by an anisotropy of {anisotropy}"
            )
            gt["kv"] = gt["kh"] / anisotropy
        else:
            # add kh and kv to gt
            if geotop_k is None:
                geotop_k = geotop.get_lithok_props()
            gt = geotop.add_kh_and_kv(gt, geotop_k, anisotropy=anisotropy)

    # copy the regis-dataset, before altering it
    rg = rg.copy(deep=True)
    if "layer" in rg["top"].dims:
        msg = "Top in rg has a layer dimension. add_geotop_to_regis_layers will remove the layer dimension from top in rg."
        logger.warning(msg)
    else:
        # temporarily add layer dimension to top in rg
        rg["top"] = rg["botm"] + calculate_thickness(rg)

    for layer in layers:
        if gt_layered is not None:
            gtl = gt_layered.copy(deep=True)
        else:
            # transform geotop data into layers
            gtl = geotop.to_model_layers(gt)

        # temporarily add layer dimension to top in gtl
        gtl["top"] = gtl["botm"] + calculate_thickness(gtl)

        # only keep the part of layers inside the regis layer
        top = rg["top"].loc[layer]
        bot = rg["botm"].loc[layer]
        gtl["top"] = gtl["top"].where(gtl["top"].isnull() | (gtl["top"] < top), top)
        gtl["top"] = gtl["top"].where(gtl["top"].isnull() | (gtl["top"] > bot), bot)
        gtl["botm"] = gtl["botm"].where(gtl["botm"].isnull() | (gtl["botm"] < top), top)
        gtl["botm"] = gtl["botm"].where(gtl["botm"].isnull() | (gtl["botm"] > bot), bot)

        if remove_nan_layers:
            # drop layers with a remaining thickness of 0 (or NaN) everywhere
            th = calculate_thickness(gtl)
            gtl = gtl.sel(layer=(th > 0).any(th.dims[1:]))

        # add kh and kv from gt to gtl
        gtl = geotop.aggregate_to_ds(gt, gtl)

        # add gtl-layers to rg-layers
        lay = np.where(rg.layer == layer)[0][0]
        layer_order = np.concatenate([rg.layer[:lay], gtl.layer, rg.layer[lay + 1 :]])

        # call xr.concat with rg first, so we keep attributes of rg
        rg = xr.concat((rg, gtl), "layer")
        # we will then make sure the layer order is right
        rg = rg.reindex({"layer": layer_order})

    # remove the layer dimension from top again
    rg = remove_layer_dim_from_top(rg)
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
    """Get a legend (DataFrame) with the colors of REGIS and/or GeoTOP layers.

    These colors can be used when plotting cross-sections.
    """
    allowed_kinds = ["REGIS", "GeoTOP", "combined"]
    if kind not in allowed_kinds:
        raise (ValueError(f"Only allowed values for kind are {allowed_kinds}"))
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
