import logging
import warnings
from collections import OrderedDict

import numpy as np
import xarray as xr

from ..util import LayerError, MissingValueError
from . import resample

logger = logging.getLogger(__name__)


def calculate_thickness(ds, top="top", bot="botm"):
    """Calculate thickness from dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing information about top and bottom elevations
        of layers
    top : str, optional
        name of data variable containing tops, by default "top"
    bot : str, optional
        name of data variable containing bottoms, by default "botm"

    Returns
    -------
    thickness : xarray.DataArray
        DataArray containing thickness information
    """
    # calculate thickness
    if ds[top].ndim == ds[bot].ndim and ds[top].ndim in [2, 3]:
        if ds[top].shape[0] == ds[bot].shape[0]:
            # top is 3D, every layer has top and bot
            thickness = ds[top] - ds[bot]
        else:
            raise ValueError("3d top and bot should have same number of layers")
    elif ds[top].ndim == (ds[bot].ndim - 1) and ds[top].ndim in [1, 2]:
        if ds[top].shape[-1] == ds[bot].shape[-1]:
            # top is only top of first layer
            thickness = xr.zeros_like(ds[bot])
            for lay, _ in enumerate(thickness):
                if lay == 0:
                    thickness[lay] = ds[top] - ds[bot][lay]
                else:
                    thickness[lay] = ds[bot][lay - 1] - ds[bot][lay]
        else:
            raise ValueError("2d top should have same last dimension as bot")
    if isinstance(ds[bot], xr.DataArray):
        if hasattr(ds[bot], "long_name"):
            thickness.attrs["long_name"] = "thickness"
        if hasattr(ds[bot], "standard_name"):
            thickness.attrs["standard_name"] = "thickness_of_layer"
        if hasattr(ds[bot], "units"):
            if ds[bot].units == "mNAP":
                thickness.attrs["units"] = "m"
            else:
                thickness.attrs["units"] = ds[bot].units

    return thickness


def calculate_transmissivity(
    ds, kh="kh", thickness="thickness", top="top", botm="botm"
):
    """calculate the transmissivity (T) as the product of the horizontal
    conductance (kh) and the thickness (D).

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing information about top and bottom elevations
        of layers
    kh : str, optional
        name of data variable containing horizontal conductivity, by default 'kh'
    thickness : str, optional
        name of data variable containing thickness, if this data variable does not exists
        thickness is calculated using top and botm. By default 'thickness'
    top : str, optional
        name of data variable containing tops, only used to calculate thickness if not
        available in dataset. By default "top"
    botm : str, optional
        name of data variable containing bottoms, only used to calculate thickness if not
        available in dataset. By default "botm"

    Returns
    -------
    T : xarray.DataArray
        DataArray containing transmissivity (T). NaN where layer thickness is zero
    """

    if thickness in ds:
        thickness = ds[thickness]
    else:
        thickness = calculate_thickness(ds, top=top, bot=botm)

    # nan where layer does not exist (thickness is 0)
    thickness_nan = xr.where(thickness == 0, np.nan, thickness)

    # calculate transmissivity
    T = thickness_nan * ds[kh]

    if hasattr(T, "long_name"):
        T.attrs["long_name"] = "transmissivity"
    if hasattr(T, "standard_name"):
        T.attrs["standard_name"] = "T"
    if hasattr(thickness, "units"):
        if hasattr(ds[kh], "units"):
            if ds[kh].units == "m/day" and thickness.units in ["m", "mNAP"]:
                T.attrs["units"] = "m2/day"
            else:
                T.attrs["units"] = ""
        else:
            T.attrs["units"] = ""

    return T


def calculate_resistance(ds, kv="kv", thickness="thickness", top="top", botm="botm"):
    """calculate vertical resistance (c) between model layers from the vertical
    conductivity (kv) and the thickness. The resistance between two layers is assigned
    to the top layer. The bottom model layer gets a resistance of infinity.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing information about top and bottom elevations
        of layers
    kv : str, optional
        name of data variable containing vertical conductivity, by default 'kv'
    thickness : str, optional
        name of data variable containing thickness, if this data variable does not exists
        thickness is calculated using top and botm. By default 'thickness'
    top : str, optional
        name of data variable containing tops, only used to calculate thickness if not
        available in dataset. By default "top"
    botm : str, optional
        name of data variable containing bottoms, only used to calculate thickness if not
        available in dataset. By default "botm"

    Returns
    -------
    c : xarray.DataArray
        DataArray containing vertical resistance (c). NaN where layer thickness is zero
    """

    if thickness in ds:
        thickness = ds[thickness]
    else:
        thickness = calculate_thickness(ds, top=top, bot=botm)

    # nan where layer does not exist (thickness is 0)
    thickness_nan = xr.where(thickness == 0, np.nan, thickness)
    kv_nan = xr.where(thickness == 0, np.nan, ds[kv])

    # backfill thickness and kv to get the right value for the layer below
    thickness_bfill = thickness_nan.bfill(dim="layer")
    kv_bfill = kv_nan.bfill(dim="layer")

    # calculate resistance
    c = xr.zeros_like(thickness)
    for ilay in range(ds.dims["layer"] - 1):
        ctop = (thickness_nan.sel(layer=ds.layer[ilay]) * 0.5) / kv_nan.sel(
            layer=ds.layer[ilay]
        )
        cbot = (thickness_bfill.sel(layer=ds.layer[ilay + 1]) * 0.5) / kv_bfill.sel(
            layer=ds.layer[ilay + 1]
        )
        c[ilay] = ctop + cbot
    c[ilay + 1] = np.inf

    if hasattr(c, "long_name"):
        c.attrs["long_name"] = "resistance"
    if hasattr(c, "standard_name"):
        c.attrs["standard_name"] = "c"
    if hasattr(thickness, "units"):
        if hasattr(ds[kv], "units"):
            if ds[kv].units == "m/day" and thickness.units in ["m", "mNAP"]:
                c.attrs["units"] = "day"
            else:
                c.attrs["units"] = ""
        else:
            c.attrs["units"] = ""

    return c


def split_layers_ds(
    ds, split_dict, layer="layer", top="top", bot="botm", return_reindexer=False
):
    """Split layers based in Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset containing information about layers (layers, top and bot)
    split_dict : dict
        dictionary with name (string) or index (integer) of layers to split as keys.
        There are two options for the values of the dictionary, to indicate how to split
        up layer: an iterable of factors. E.g. {'BXk1': [1, 3]} will split layer 'BXk1'
        into 2 layers, with the first layer equal to 0.25 of the original thickness and
        the second layer equal to 0.75 of the original thickness.
        The second option would be to set the value to the number of layers to split the
        layer into, e.g. {'BXk1': 2}, which is equal to {'BXk1': [0.5, 0.5]}.
    layer : str, optional
        name of layer dimension, by default 'layer'
    top : str, optional
        name of data variable containing top of layers, by default 'top'
    bot : str, optional
        name of data variable containing bottom of layers, by default 'botm'
    return_reindexer : bool, optional
        Return a OrderedDict that can be used to reindex variables from the original
        layer-dimension to the new layer-dimension when True. The default is False.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with new tops and bottoms taking into account split layers, and filled
        data for other variables.
    """

    layers = list(ds.layer.data)

    # do some input-checking on split_dict
    for lay0 in list(split_dict):
        if isinstance(lay0, int) & (ds.layer.dtype != int):
            # if layer is an integer, and ds.layer is not of integer type
            # replace lay0 by the name of the layer
            split_dict[layers[lay0]] = split_dict.pop(lay0)
            lay0 = layers[lay0]
        if isinstance(split_dict[lay0], int):
            # If split_dict[lay0] is of integer type
            # split the layer in evenly thick layers
            split_dict[lay0] = [1 / split_dict[lay0]] * split_dict[lay0]
        else:
            # make sure the fractions add up to 1
            split_dict[lay0] = split_dict[lay0] / np.sum(split_dict[lay0])

    logger.info(f"Splitting layers {list(split_dict)}")

    layers_org = layers.copy()
    # add extra layers (keep the original ones for now, as we will copy data first)
    for lay0 in split_dict:
        for i, _ in enumerate(split_dict[lay0]):
            index = layers.index(lay0)
            layers.insert(index, lay0 + "_" + str(i + 1))
            layers_org.insert(index, lay0)
    ds = ds.reindex({"layer": layers})

    # calclate a new top and botm, and fill other variables with original data
    th = calculate_thickness(ds, top=top, bot=bot)
    for lay0 in split_dict:
        logger.info(f"Split '{lay0}' into {len(split_dict[lay0])} sub-layers")
        th0 = th.loc[lay0]
        for var in ds:
            if layer not in ds[var].dims:
                continue
            if lay0 == list(split_dict)[0] and var not in [top, bot]:
                logger.info(
                    f"Fill values for variable '{var}' in split"
                    " layers with the values from the original layer."
                )
            ds = _split_var(ds, var, lay0, th0, split_dict[lay0], top, bot)

    # drop the original layers
    ds = ds.drop_sel(layer=list(split_dict))

    if return_reindexer:
        # determine reindexer
        reindexer = OrderedDict(zip(layers, layers_org))
        for lay0 in split_dict:
            reindexer.pop(lay0)
        return ds, reindexer
    return ds


def _split_var(ds, var, layer, thickness, fctrs, top, bot):
    """Internal method to split a variable of one layer in multiple layers."""
    for i in range(len(fctrs)):
        name = layer + "_" + str(i + 1)
        if var == top:
            # take orignal top and subtract thickness of higher splitted layers
            ds[var].loc[name] = ds[var].loc[layer] - np.sum(fctrs[:i]) * thickness
        elif var == bot:
            # take original bottom and add thickness of lower splitted layers
            ds[var].loc[name] = ds[var].loc[layer] + np.sum(fctrs[i + 1 :]) * thickness
        else:
            # take data from the orignal layer
            ds[var].loc[name] = ds[var].loc[layer]
    return ds


def layer_combine_top_bot(ds, combine_layers, layer="layer", top="top", bot="botm"):
    """Calculate new tops and bottoms for combined layers.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset containing information about layers
        (layers, top and bot)
    combine_layers : list of tuple of ints
        list of tuples, with each tuple containing integers indicating
        layer indices to combine into one layer. E.g. [(0, 1), (2, 3)] will
        combine layers 0 and 1 into a single layer (with index 0) and layers
        2 and 3 into a second layer (with index 1).
    layer : str, optional
        name of layer dimension, by default 'layer'
    top : str, optional
        name of data variable containing top of layers, by default 'top'
    bot : str, optional
        name of data variable containing bottom of layers, by default 'botm'

    Returns
    -------
    new_top, new_bot : xarray.DataArrays
        DataArrays containing new tops and bottoms after splitting layers.
    reindexer : OrderedDict
        dictionary mapping new to old layer indices.
    """
    # calculate new number of layers
    new_nlay = (
        ds[layer].size - sum((len(c) for c in combine_layers)) + len(combine_layers)
    )

    # create new DataArrays for storing new top/bot
    new_bot = xr.DataArray(
        data=np.nan,
        dims=["layer", "y", "x"],
        coords={"layer": np.arange(new_nlay), "y": ds.y.data, "x": ds.x.data},
    )
    new_top = xr.DataArray(
        data=np.nan,
        dims=["layer", "y", "x"],
        coords={"layer": np.arange(new_nlay), "y": ds.y.data, "x": ds.x.data},
    )

    # dict to keep track of old and new layer indices
    reindexer = OrderedDict()

    j = 0  # new layer index
    icomb = 0  # combine layer index

    # loop over original layers
    for i in range(ds.layer.size):
        # check whether to combine layers
        if i in np.concatenate(combine_layers):
            # get indices of layers
            c = combine_layers[icomb]
            # store new and original layer indices
            reindexer[j] = c
            # only need to calculate new top/bot once for each merged layer
            if i == np.min(c):
                logger.debug(
                    f"{j:2d}: Merge layers {c} as layer {j}, calculate new top/bot."
                )
                tops = ds[top].data[c, :, :]
                bots = ds[bot].data[c, :, :]
                new_top.data[j] = np.nanmax(tops, axis=0)
                new_bot.data[j] = np.nanmin(bots, axis=0)

            elif i == np.max(c):
                # advance combine layer index after merging layers
                icomb += 1
                # advance new layer index
                j += 1
                continue
            else:
                # no need to merge more than once, so continue loop
                continue
        else:
            # do not merge, only map old layer index to new layer index
            logger.debug(
                f"{j:2d}: Do not merge, map old layer index to new layer index."
            )
            new_top.data[j] = ds[top].data[i]
            new_bot.data[j] = ds[bot].data[i]
            reindexer[j] = i
            j += 1

    return new_top, new_bot, reindexer


def sum_param_combined_layers(da, reindexer):
    """Calculate combined layer parameter with sum.

    Parameters
    ----------
    da : xarray.DataArray
        data array to calculate combined parameters for
    reindexer : OrderedDict
        dictionary mapping new layer indices to old layer indices

    Returns
    -------
    da_new : xarray.DataArray
        data array containing new parameters for combined layers and old
        parameters for unmodified layers.
    """
    da_new = xr.DataArray(
        data=np.nan,
        dims=["layer", "y", "x"],
        coords={
            "layer": np.arange(list(reindexer.keys())[-1] + 1),
            "y": da["y"],
            "x": da["x"],
        },
    )

    for k, v in reindexer.items():
        if isinstance(v, tuple):
            psum = np.sum(da.data[v, :, :], axis=0)
        else:
            psum = da.data[v]
        da_new.data[k] = psum
    return da_new


def kheq_combined_layers(kh, thickness, reindexer):
    """Calculate equivalent horizontal hydraulic conductivity.

    Parameters
    ----------
    kh : xarray.DataArray
        data array containing horizontal hydraulic conductivity
    thickness : xarray.DataArray
            data array containing thickness per layer
    reindexer : OrdererDict
        dictionary mapping new layer indices to old layer indices

    Returns
    -------
    da_kh : xarray.DataArray
        data array containing equivalent horizontal hydraulic conductivity
        for combined layers and original hydraulic conductivity in unmodified
        layers
    """
    da_kh = xr.DataArray(
        data=np.nan,
        dims=["layer", "y", "x"],
        coords={
            "layer": np.arange(list(reindexer.keys())[-1] + 1),
            "y": kh["y"],
            "x": kh["x"],
        },
    )

    for k, v in reindexer.items():
        if isinstance(v, tuple):
            kheq = np.nansum(
                thickness.data[v, :, :] * kh.data[v, :, :], axis=0
            ) / np.nansum(thickness.data[v, :, :], axis=0)
        else:
            kheq = kh.data[v]
        da_kh.data[k] = kheq
    return da_kh


def kveq_combined_layers(kv, thickness, reindexer):
    """Calculate equivalent vertical hydraulic conductivity.

    Parameters
    ----------
    kv : xarray.DataArray
        data array containing vertical hydraulic conductivity
    thickness : xarray.DataArray
        data array containing thickness per layer
    reindexer : OrdererDict
        dictionary mapping new layer indices to old layer indices

    Returns
    -------
    da_kv : xarray.DataArray
        data array containing equivalent vertical hydraulic conductivity
        for combined layers and original hydraulic conductivity in unmodified
        layers
    """
    da_kv = xr.DataArray(
        data=np.nan,
        dims=["layer", "y", "x"],
        coords={
            "layer": np.arange(list(reindexer.keys())[-1] + 1),
            "y": kv["y"],
            "x": kv["x"],
        },
    )

    for k, v in reindexer.items():
        if isinstance(v, tuple):
            kveq = np.nansum(thickness.data[v, :, :], axis=0) / np.nansum(
                thickness.data[v, :, :] / kv.data[v, :, :], axis=0
            )
        else:
            kveq = kv.data[v]
        da_kv.data[k] = kveq
    return da_kv


def combine_layers_ds(
    ds,
    combine_layers,
    layer="layer",
    top="top",
    bot="botm",
    kh="kh",
    kv="kv",
    kD="kD",
    c="c",
):
    """Combine layers in Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset containing information about layers
        (layers, top and bot)
    combine_layers : list of tuple of ints
        list of tuples, with each tuple containing integers indicating
        layer indices to combine into one layer. E.g. [(0, 1), (2, 3)] will
        combine layers 0 and 1 into a single layer (with index 0) and layers
        2 and 3 into a second layer (with index 1).
    layer : str, optional
        name of layer dimension, by default 'layer'
    top : str, optional
        name of data variable containing top of layers, by default 'top'
    bot : str, optional
        name of data variable containing bottom of layers, by default 'botm'
    kh : str, optional
        name of data variable containg horizontal hydraulic conductivity,
        by default 'kh'. Not parsed if set to None.
    kv : str, optional
        name of data variable containg vertical hydraulic conductivity,
        by default 'kv'. Not parsed if set to None.
    kD : str, optional
        name of data variable containg transmissivity or kD,
        by default 'kD'. Not parsed if set to None.
    c : str, optional
        name of data variable containg resistance or c,
        by default 'c'. Not parsed if set to None.

    Returns
    -------
    ds_combine : xarray.Dataset
        Dataset with new tops and bottoms taking into account combined layers,
        and recalculated values for parameters (kh, kv, kD, c).
    """

    data_vars = []
    for dv in [kh, kv, kD, c]:
        if dv is not None:
            data_vars.append(dv)
    parsed_dv = set([top, bot] + data_vars)

    dropped_dv = set(ds.data_vars.keys()) - parsed_dv
    if len(dropped_dv) > 0:
        logger.warning(f"Following data variables will be dropped: {dropped_dv}")

    # calculate new tops/bots
    logger.info("Calculating new layer tops and bottoms...")

    da_dict = {}

    new_top, new_bot, reindexer = layer_combine_top_bot(
        ds, combine_layers, layer=layer, top=top, bot=bot
    )
    da_dict[top] = new_top
    da_dict[bot] = new_bot

    # calculate original thickness
    thickness = calculate_thickness(ds, top=top, bot=bot)

    # calculate equivalent kh/kv
    if kh is not None:
        logger.info(f"Calculate equivalent '{kh}' for combined layers.")
        da_dict[kh] = kheq_combined_layers(ds[kh], thickness, reindexer)
    if kv is not None:
        logger.info(f"Calculate equivalent '{kv}' for combined layers.")
        da_dict[kv] = kveq_combined_layers(ds[kv], thickness, reindexer)
    if kD is not None:
        logger.info(f"Calculate value '{kD}' for combined layers with sum.")
        da_dict[kD] = sum_param_combined_layers(ds[kD], reindexer)
    if c is not None:
        logger.info(f"Calculate value '{c}' for combined layers with sum.")
        da_dict[c] = sum_param_combined_layers(ds[c], reindexer)

    # get new layer names, based on first sub-layer from each combined layer
    layer_names = []
    for _, i in reindexer.items():
        if isinstance(i, tuple):
            i = i[0]
        layercode = ds[layer].data[i]
        layer_names.append(layercode)

    # assign new layer names
    for k, da in da_dict.items():
        da_dict[k] = da.assign_coords(layer=layer_names)

    # add reindexer to attributes
    attrs = ds.attrs.copy()
    attrs["combine_reindexer"] = reindexer

    # create new dataset
    logger.info("Done! Created new dataset with combined layers!")
    ds_combine = xr.Dataset(da_dict, attrs=attrs)

    return ds_combine


def add_kh_kv_from_ml_layer_to_ds(
    ml_layer_ds, ds, anisotropy, fill_value_kh, fill_value_kv
):
    """add kh and kv from a model layer dataset to the model dataset.

    Supports structured and vertex grids.

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with kh and kv
    ds : xarray.Dataset
        dataset with model data where kh and kv are added to
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in the layer model. The
        default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in the layer model. The
        default is 1.0.

    Returns
    -------
    ds : xarray.Dataset
        dataset with model data with new kh and kv

    Notes
    -----
    some model dataset, such as regis, also have 'c' and 'kd' values. These
    are ignored at the moment
    """
    warnings.warn(
        "add_kh_kv_from_ml_layer_to_ds is deprecated. Please use nlmod.grid.update_ds_from_layer_ds instead.",
        DeprecationWarning,
    )

    ds.attrs["anisotropy"] = anisotropy
    ds.attrs["fill_value_kh"] = fill_value_kh
    ds.attrs["fill_value_kv"] = fill_value_kv

    logger.info("add kh and kv from model layer dataset to modflow model")

    kh, kv = get_kh_kv(
        ml_layer_ds["kh"],
        ml_layer_ds["kv"],
        anisotropy,
        fill_value_kh=fill_value_kh,
        fill_value_kv=fill_value_kv,
    )

    ds["kh"] = kh
    ds["kv"] = kv

    return ds


def set_model_top(ds, top, min_thickness=0.0):
    """Set the model top, changing layer bottoms when necessary as well.

    If the new top is higher than the previous top, the extra thickness is added to the
    highest layer with a thickness larger than 0.

    Parameters
    ----------
    ds : xarray.Dataset
        The model dataset, containing the current top.
    top : xarray.DataArray
        The new model top, with the same shape as the current top.

    Returns
    -------
    ds : xarray.Dataset
        The model dataset, containing the new top.
    """
    if "gridtype" not in ds.attrs:
        raise (KeyError("Make sure the Dataset is built by nlmod"))
    if isinstance(top, (float, int)):
        top = xr.full_like(ds["top"], top)
    if not top.shape == ds["top"].shape:
        raise (
            ValueError("Please make sure the new top has the same shape as the old top")
        )
    if np.any(np.isnan(top)):
        raise (ValueError("Please make sure the new top does not contain nans"))
    # where the botm is equal to the top, the layer is inactive
    # set the botm to the new top at these locations
    ds["botm"] = ds["botm"].where(ds["botm"] != ds["top"], top)
    # make sure the botm is never higher than the new top
    ds["botm"] = ds["botm"].where(top - ds["botm"] > min_thickness, top)
    # change the current top
    ds["top"] = top
    # recalculate idomain
    ds = set_idomain(ds)
    return ds


def set_layer_top(ds, layer, top):
    """Set the top of a layer."""
    assert layer in ds.layer
    lay = np.where(ds.layer == layer)[0][0]
    if lay == 0:
        # change the top of the model
        ds["top"] = top
        # make sure the botm of all layers is never higher than the new top
        ds["botm"] = ds["botm"].where(ds["botm"] < top, top)
    else:
        # change the botm of the layer above
        ds["botm"][lay - 1] = top
        # make sure the top of the layers above is never lower than the new top
        ds["top"] = ds["top"].where(ds["top"] > top, top)
        # make sure the botm of the layers above is never higher than the new top
        ds["botm"][: lay - 1] = ds["botm"][: lay - 1].where(
            ds["botm"][: lay - 1] > top, top
        )
        # make sure the botms of lower layers are lower than top
        ds["botm"][lay:] = ds["botm"][lay:].where(ds["botm"][lay:] < top, top)
    ds = set_idomain(ds)
    return ds


def set_layer_botm(ds, layer, botm):
    """Set the bottom of a layer."""
    assert layer in ds.layer
    lay = np.where(ds.layer == layer)[0][0]
    # if lay > 0 and np.any(botm > ds["botm"][lay - 1]):
    #    raise (Exception("set_layer_botm cannot change botm of higher layers yet"))
    ds["botm"][:lay] = ds["botm"][:lay].where(ds["botm"][:lay] > botm, botm)
    ds["botm"][lay] = botm
    # make sure the botm of the layers below is never higher than the new botm
    mask = ds["botm"][lay + 1 :] < botm
    ds["botm"][lay + 1 :] = ds["botm"][lay + 1 :].where(mask, botm)
    # make sure the botm of the layers above is lever lower than the new botm

    ds = set_idomain(ds)
    return ds


def set_layer_thickness(ds, layer, thickness, change="botm"):
    """Set the layer thickness by changing the bottom of the layer."""
    assert layer in ds.layer
    assert change == "botm", "Only change=botm allowed for now"
    lay = np.where(ds.layer == layer)[0][0]
    if lay == 0:
        top = ds["top"]
    else:
        top = ds["botm"][lay - 1]
    new_botm = top - thickness
    ds = set_layer_botm(ds, layer, new_botm)
    return ds


def set_minimum_layer_thickness(ds, layer, min_thickness, change="botm"):
    """Make sure layer has a minimum thickness by lowering the botm of the
    layer where neccesary."""
    assert layer in ds.layer
    assert change == "botm", "Only change=botm allowed for now"
    lay = np.where(ds.layer == layer)[0][0]
    if lay == 0:
        top = ds["top"]
    else:
        top = ds["botm"][lay - 1]
    botm = ds["botm"][lay]

    mask = (top - botm) > min_thickness
    new_botm = botm.where(mask, top - min_thickness)
    ds = set_layer_botm(ds, layer, new_botm)
    return ds


def get_kh_kv(kh, kv, anisotropy, fill_value_kh=1.0, fill_value_kv=0.1, idomain=None):
    """create kh en kv grid data for flopy from existing kh, kv and anistropy
    grids with nan values (typically from REGIS).

    fill nans in kh grid in these steps:
    1. take kv and multiply by anisotropy, if this is nan:
    2. take fill_value_kh

    fill nans in kv grid in these steps:
    1. take kh and divide by anisotropy, if this is nan:
    2. take fill_value_kv

    Supports structured and vertex grids.

    Parameters
    ----------
    kh : xarray.DataArray
        kh from regis with nan values shape(nlay, nrow, ncol) or
        shape(nlay, len(icell2d))
    kv : xarray.DataArray
        kv from regis with nan values shape(nlay, nrow, ncol) or
        shape(nlay, len(icell2d))
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in kh, kv and
        anisotropy. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in kv, kh and
        anisotropy. The default is 1.0.
    idomain : xarray.DataArray, optional
        The idomain DataArray, used in log-messages, to report the number of active
        cells that are filled. When idomain is None, the total number of cells that are
        filled is reported, and not just the active cells. The default is None.

    Returns
    -------
    kh : np.ndarray
        kh without nan values (nlay, nrow, ncol) or shape(nlay, len(icell2d))
    kv : np.ndarray
        kv without nan values (nlay, nrow, ncol) or shape(nlay, len(icell2d))
    """
    for layer in kh.layer.data:
        if ~np.all(np.isnan(kh.loc[layer])):
            logger.debug(f"layer {layer} has a kh")
        elif ~np.all(np.isnan(kv.loc[layer])):
            logger.debug(f"layer {layer} has a kv")
        else:
            logger.info(f"kv and kh both undefined in layer {layer}")

    # fill kh by kv * anisotropy
    msg_suffix = f" of kh by multipying kv by an anisotropy of {anisotropy}"
    kh = _fill_var(kh, kv * anisotropy, idomain, msg_suffix)

    # fill kv by kh / anisotropy
    msg_suffix = f" of kv by dividing kh by an anisotropy of {anisotropy}"
    kv = _fill_var(kv, kh / anisotropy, idomain, msg_suffix)

    # fill kh by fill_value_kh
    msg_suffix = f" of kh with a value of {fill_value_kh}"
    if "units" in kh.attrs:
        msg_suffix = f"{msg_suffix} {kh.units}"
    kh = _fill_var(kh, fill_value_kh, idomain, msg_suffix)

    # fill kv by fill_value_kv
    msg_suffix = f" of kv with a value of {fill_value_kv}"
    if "units" in kv.attrs:
        msg_suffix = f"{msg_suffix} {kv.units}"
    kv = _fill_var(kv, fill_value_kv, idomain, msg_suffix)

    return kh, kv


def _fill_var(var, by, idomain, msg_suffix=""):
    mask = np.isnan(var)
    if isinstance(by, xr.DataArray):
        mask = mask & (~np.isnan(by))
    if mask.any():
        var = var.where(~mask, by)
        if idomain is not None:
            mask = mask & (idomain > 0)
            if mask.any():
                logger.info(
                    f"Filling {int(mask.sum())} values in active cells{msg_suffix}"
                )
        else:
            logger.info(f"Filling {int(mask.sum())} values {msg_suffix}")
    return var


def fill_top_bot_kh_kv_at_mask(ds, fill_mask):
    """Fill values in top, bot, kh and kv.

    Fill where:
    1. the cell is True in fill_mask
    2. the cell thickness is greater than 0

    Fill values:
    - top: 0
    - bot: minimum of bottom_filled or top
    - kh: kh_filled if thickness is greater than 0
    - kv: kv_filled if thickness is greater than 0

    Parameters
    ----------
    ds : xr.DataSet
        model dataset
    fill_mask : xr.DataArray
        1 where a cell should be replaced by masked value.

    Returns
    -------
    ds : xr.DataSet
        model dataset with adjusted data variables: 'top', 'botm', 'kh', 'kv'
    """

    # zee cellen hebben altijd een top gelijk aan 0
    ds["top"].values = np.where(fill_mask, 0, ds["top"])

    for lay in range(ds.dims["layer"]):
        bottom_nan = xr.where(fill_mask, np.nan, ds["botm"][lay])
        bottom_filled = resample.fillnan_da(bottom_nan, ds=ds)

        kh_nan = xr.where(fill_mask, np.nan, ds["kh"][lay])
        kh_filled = resample.fillnan_da(kh_nan, ds=ds)

        kv_nan = xr.where(fill_mask, np.nan, ds["kv"][lay])
        kv_filled = resample.fillnan_da(kv_nan, ds=ds)

        if lay == 0:
            # top ligt onder bottom_filled -> laagdikte wordt 0
            # top ligt boven bottom_filled -> laagdikte o.b.v. bottom_filled
            mask_top = ds["top"] < bottom_filled
            ds["botm"][lay] = xr.where(fill_mask * mask_top, ds["top"], bottom_filled)
        else:
            # top ligt onder bottom_filled -> laagdikte wordt 0
            # top ligt boven bottom_filled -> laagdikte o.b.v. bottom_filled
            mask_top = ds["botm"][lay - 1] < bottom_filled
            ds["botm"][lay] = xr.where(
                fill_mask * mask_top, ds["botm"][lay - 1], bottom_filled
            )
        ds["kh"][lay] = xr.where(fill_mask * mask_top, ds["kh"][lay], kh_filled)
        ds["kv"][lay] = xr.where(fill_mask * mask_top, ds["kv"][lay], kv_filled)

    return ds


def fill_nan_top_botm_kh_kv(
    ds,
    anisotropy=10.0,
    fill_value_kh=1.0,
    fill_value_kv=0.1,
    remove_nan_layers=True,
):
    """Update a model dataset, by removing nans and adding necessary info.

    Steps:

    1. Compute top and botm values, by filling nans by data from other layers
    2. Compute idomain from the layer thickness
    3. Compute kh and kv, filling nans with anisotropy or fill_values
    """

    # 1
    ds = fill_top_and_bottom(ds)

    # 2
    ds = set_idomain(ds, remove_nan_layers=remove_nan_layers)

    # 3
    ds["kh"], ds["kv"] = get_kh_kv(
        ds["kh"],
        ds["kv"],
        anisotropy,
        fill_value_kh=fill_value_kh,
        fill_value_kv=fill_value_kv,
        idomain=ds["idomain"],
    )
    return ds


def fill_top_and_bottom(ds, drop_layer_dim_from_top=True):
    """
    Remove Nans in botm variable, and change top from 3d to 2d if necessary.

    Parameters
    ----------
    ds : xr.DataSet
        model DataSet
    drop_layer_dim_from_top : bool, optional
        If True and top contains a layer dimension, set top to the top of the upper
        layer (line the definition in MODFLOW). This removes redundant data, as the top
        of all layers exept the most upper one is also defined as the bottom of previous
        layers. The default is True.

    Returns
    -------
    ds : xarray.Dataset
        dataset with filled top and bottom data according to modflow definition,
        with 2d top and 3d bottom.
    """

    if "layer" in ds["top"].dims:
        top_max = ds["top"].max("layer")
        if drop_layer_dim_from_top:
            ds["top"] = top_max
    else:
        top_max = ds["top"]

    botm = ds["botm"].data
    # remove nans from botm
    for lay in range(botm.shape[0]):
        mask = np.isnan(botm[lay])
        if lay == 0:
            # by setting the botm to top_max
            botm[lay, mask] = top_max.data[mask]
        else:
            # by setting the botm to the botm of the layer above
            botm[lay, mask] = botm[lay - 1, mask]
    if "layer" in ds["top"].dims:
        # remove nans from top by setting it equal to botm
        # which sets the layer thickness to 0
        top = ds["top"].data
        mask = np.isnan(top)
        top[mask] = botm[mask]

    return ds


def set_idomain(ds, remove_nan_layers=True):
    """Set idmomain in a model Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The model Dataset.
    remove_nan_layers : bool, optional
        Removes layers which only contain inactive cells. The default is True.

    Returns
    -------
    ds : xr.Dataset
        Dataset with added idomain-variable.
    """
    ds["idomain"] = get_idomain(ds)
    if remove_nan_layers:
        # only keep layers with at least one active cell
        ds = ds.sel(layer=(ds["idomain"] > 0).any(ds["idomain"].dims[1:]))

    return ds


def get_idomain(ds):
    """Get idmomain from a model Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The model Dataset.

    Returns
    -------
    ds : xr.DataArray
        DataArray of idomain-variable.
    """
    # set idomain with a default of -1 (pass-through)
    idomain = xr.full_like(ds["botm"], -1, int)
    idomain.name = None
    # drop attributes inherited from botm
    idomain.attrs.clear()
    # set idomain of cells  with a positive thickness to 1
    thickness = calculate_thickness(ds)
    idomain.data[thickness.data > 0.0] = 1
    # set idomain to 0 in the inactive part of the model
    if "active" in ds:
        idomain = idomain.where(ds["active"], 0)
    # TODO: set idomain above/below the first/last active layer to 0
    # TODO: remove 'active' and replace by logic of keeping inactive cells in idomain
    return idomain


def get_first_active_layer(ds, **kwargs):
    """Get the first active layer in each cell from a model ds.

    Parameters
    ----------
    ds : xr.DataSet
        Model Dataset with a variable idomain.
    **kwargs : dict
        Kwargs are passed on to get_first_active_layer_from_idomain.

    Returns
    -------
    first_active_layer : xr.DataArray
        raster in which each cell has the zero based number of the first
        active layer. Shape can be (y, x) or (icell2d)
    """
    return get_first_active_layer_from_idomain(ds["idomain"], **kwargs)


def get_first_active_layer_from_idomain(idomain, nodata=-999):
    """get the first (top) active layer in each cell from the idomain.

    Parameters
    ----------
    idomain : xr.DataArray
        idomain. Shape can be (layer, y, x) or (layer, icell2d)
    nodata : int, optional
        nodata value. used for cells that are inactive in all layers.
        The default is -999.

    Returns
    -------
    first_active_layer : xr.DataArray
        raster in which each cell has the zero based number of the first
        active layer. Shape can be (y, x) or (icell2d)
    """
    logger.debug("get first active modellayer for each cell in idomain")

    first_active_layer = xr.where(idomain[0] == 1, 0, nodata)
    for i in range(1, idomain.shape[0]):
        first_active_layer = xr.where(
            (first_active_layer == nodata) & (idomain[i] == 1),
            i,
            first_active_layer,
        )
    first_active_layer.attrs["nodata"] = nodata
    return first_active_layer


def get_last_active_layer_from_idomain(idomain, nodata=-999):
    """get the last (bottom) active layer in each cell from the idomain.

    Parameters
    ----------
    idomain : xr.DataArray
        idomain. Shape can be (layer, y, x) or (layer, icell2d)
    nodata : int, optional
        nodata value. used for cells that are inactive in all layers.
        The default is -999.

    Returns
    -------
    last_active_layer : xr.DataArray
        raster in which each cell has the zero based number of the last
        active layer. Shape can be (y, x) or (icell2d)
    """
    logger.debug("get last active modellayer for each cell in idomain")

    last_active_layer = xr.where(idomain[-1] == 1, 0, nodata)
    for i in range(idomain.shape[0] - 2, -1, -1):
        last_active_layer = xr.where(
            (last_active_layer == nodata) & (idomain[i] == 1),
            i,
            last_active_layer,
        )
    last_active_layer.attrs["nodata"] = nodata
    return last_active_layer


def update_idomain_from_thickness(idomain, thickness, mask):
    """get new idomain from thickness in the cells where mask is 1 (or True).

    Idomain becomes:
    -    1: if cell thickness is bigger than 0
    -    0: if cell thickness is 0 and it is the top layer
    -   -1: if cell thickness is 0 and the layer is in between active cells

    Parameters
    ----------
    idomain : xr.DataArray
        raster with idomain of each cell. dimensions should be (layer, y, x) or
        (layer, icell2d).
    thickness : xr.DataArray
        raster with thickness of each cell. dimensions should be (layer, y, x)
        or (layer, icell2d).
    mask : xr.DataArray
        raster with ones in cell where the ibound is adjusted. dimensions
        should be (y, x) or (icell2d).

    Returns
    -------
    idomain : xr.DataArray
        raster with adjusted idomain of each cell. dimensions should be
        (layer, y, x) or (layer, icell2d).
    """
    warnings.warn(
        "update_idomain_from_thickness is deprecated. Please use set_idomain instead.",
        DeprecationWarning,
    )
    for ilay, thick in enumerate(thickness):
        if ilay == 0:
            mask1 = (thick == 0) * mask
            idomain[ilay] = xr.where(mask1, 0, idomain[ilay])
            mask2 = (thick > 0) * mask
            idomain[ilay] = xr.where(mask2, 1, idomain[ilay])
        else:
            mask1 = (thick == 0) * mask * (idomain[ilay - 1] == 0)
            idomain[ilay] = xr.where(mask1, 0, idomain[ilay])

            mask2 = (thick == 0) * mask * (idomain[ilay - 1] != 0)
            idomain[ilay] = xr.where(mask2, -1, idomain[ilay])

            mask3 = (thick != 0) * mask
            idomain[ilay] = xr.where(mask3, 1, idomain[ilay])

    return idomain


def aggregate_by_weighted_mean_to_ds(ds, source_ds, var_name):
    """Aggregate source data to a model dataset using the weighted mean.

    The weighted average per model layer is calculated for the variable in the
    source dataset. The datasets must have the same grid.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset containing layer information (x, y, top, botm)
    source_ds : xr.Dataset
        dataset containing x, y, top, botm and a data variable to aggregate.
    var_name : str
        name of the data array to aggregate

    Returns
    -------
    da : xarray.DataArray
        data array containing aggregated values from source dataset

    Raises
    ------
    ValueError
        if source_ds does not have a layer dimension

    See also
    --------
    nlmod.read.geotop.aggregate_to_ds
    """
    msg = "x and/or y coordinates do not match between 'ds' and 'source_ds'"
    assert (ds.x == source_ds.x).all() and (ds.y == source_ds.y).all(), msg

    if "layer" in ds["top"].dims:
        # make sure there is no layer dimension in top
        ds["top"] = ds["top"].max(dim="layer")

    if "layer" not in source_ds.dims:
        raise ValueError("Requires 'source_ds' to have a 'layer' dimension!")

    agg_ar = []

    for ilay in range(len(ds.layer)):
        if ilay == 0:
            top = ds["top"]
        else:
            top = ds["botm"][ilay - 1].drop_vars("layer")
        bot = ds["botm"][ilay].drop_vars("layer")

        s_top = source_ds.top
        s_bot = source_ds.bottom
        s_top = s_top.where(s_top < top, top)
        s_top = s_top.where(s_top > bot, bot)
        s_bot = s_bot.where(s_bot < top, top)
        s_bot = s_bot.where(s_bot > bot, bot)
        s_thk = s_top - s_bot

        agg_ar.append(
            (s_thk * source_ds[var_name]).sum("layer")
            / s_thk.where(~np.isnan(source_ds[var_name])).sum("layer")
        )

    return xr.concat(agg_ar, ds.layer)


def check_elevations_consistency(ds):
    if "layer" in ds["top"].dims:
        tops = ds["top"].data
        top_ref = np.full(tops.shape[1:], np.NaN)
        for lay, layer in zip(range(tops.shape[0]), ds.layer.data):
            top = tops[lay]
            mask = ~np.isnan(top)
            higher = top[mask] > top_ref[mask]
            if np.any(higher):
                n = int(higher.sum())
                logger.warning(
                    f"The top of layer {layer} is higher than the top of a previous layer in {n} cells"
                )
            top_ref[mask] = top[mask]

    bots = ds["botm"].data
    bot_ref = np.full(bots.shape[1:], np.NaN)
    for lay, layer in zip(range(bots.shape[0]), ds.layer.data):
        bot = bots[lay]
        mask = ~np.isnan(bot)
        higher = bot[mask] > bot_ref[mask]
        if np.any(higher):
            n = int(higher.sum())
            logger.warning(
                f"The bottom of layer {layer} is higher the bottom of a previous layer in {n} cells"
            )
        bot_ref[mask] = bot[mask]

    thickness = calculate_thickness(ds)
    mask = thickness < 0.0
    if mask.any():
        logger.warning(f"Thickness of layers is negative in {mask.sum()} cells.")


def insert_layer(ds, name, top, bot, kh=None, kv=None, copy=True):
    """
    Inserts a layer in a model Dataset, burning it in an existing layer model.

    This method loops over the existing layers, and checks if (part of) the new layer
    needs to be inserted above the existing layer, and if the top or bottom of the
    existing layer needs to be altered.

    For now, this method needs a layer model with a 3d-top, like you get using the
    method `nlmod.read.get_regis()`, and does not function for a model Dataset with a 2d
    (structured) or 1d (vertex) top.

    When comparing the height of the new layer with an existing layer, there are 7
    options:

    1 The new layer is entirely above the existing layer: layer is added completely
    above existing layer. When the bottom of the new layer is above the top of the
    existing layer (which can happen for the first layer), this creates a gap in the
    layer model.

    2 part of the new layer lies within an existing layer, bottom is never below: layer
    is added above the existing layer, and the top of existing layer is lowered.

    3 there are locations where the new layer is above the bottom of the existing layer,
    but below the top of the existing layer. The new layer splits the existing layer
    into two sub-layers. This is not supported (yet) and raises an Exception.

    4 part of the new layer lies above the bottom of the existing layer, while at other
    locations the new layer is below the existing layer. The new layer is split, part
    of the layer is added above the existing layer, and part of the new layer is added
    to the layer model in the next iteration(s) (above the next layer).

    5 Only the upper part of the new layer overlaps with the existing layer: the layer
    is not added above the extsing layer, but the bottom of the existing layer is
    raised because of the overlap.

    6 The new layer is below the existing layer everywhere. Nothing happens, move on to
    the next existing layer.

    When (part of) the new layer is not added to the layer model after comparison
    with the last existing layer, the (remaining part of) the new layer is added below
    the existing layers, at the bottom of the model.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset containing information about layers
    name : string
        The name of the new layer.
    top : xr.DataArray
        The top of the new layer.
    bot : xr.DataArray
        The bottom of the new layer..
    kh : xr.DataArray, optional
        The horizontal conductivity of the new layer. The default is None.
    kv : xr.DataArray, optional
        The vertical conductivity of the new layer. The default is None.
    copy : bool, optional
        If copy=True, data in the return value is always copied, so the original Dataset
        is not altered. The default is True.

    Returns
    -------
    ds : xarray.Dataset
        xarray Dataset containing the new layer(s)

    """
    shape = ds["botm"].shape[1:]
    assert top.shape == shape
    assert bot.shape == shape
    msg = "Inserting layers is only supported with a 3d top for now"
    assert "layer" in ds["top"].dims, msg
    if kh is not None:
        assert kh.shape == shape
    if kv is not None:
        assert kv.shape == shape
    todo = ~(np.isnan(top.data) | np.isnan(bot.data)) & ((top - bot).data > 0)
    if not todo.any():
        logger.warning(f"Thickness of new layer {name} is never larger than 0")
    if copy:
        # make a copy, so we are sure we do not alter the original DataSet
        ds = ds.copy(deep=True)
    isplit = None
    for layer in ds.layer.data:
        if not todo.any():
            continue
        # determine the top and bottom of layer, taking account they could be NaN
        # we assume a zero thickness when top or bottom is NaN
        top_layer = ds["top"].loc[layer:].max("layer").data
        bot_layer = ds["botm"].loc[layer].data
        mask = np.isnan(bot_layer)
        bot_layer[mask] = top_layer[mask]

        top_higher_than_bot = top.data > bot_layer
        if not top_higher_than_bot[todo].any():
            # 6 the new layer is entire below the existing layer, do nothing
            continue
        bot_lower_than_top = bot.data < top_layer
        bot_lower_than_bot = bot.data < bot_layer
        if not bot_lower_than_top[todo].any():
            # 1 the new layer can be added on top of the existing layer
            if isplit is not None:
                isplit += 1
            ds = _insert_layer_above(
                ds, layer, name, isplit, todo, top, bot, kh, kv, copy
            )
            todo[todo] = False
            continue
            # do not increase top of layer to bottom of new layer
        if bot_lower_than_top[todo].any():
            # the new layer can be added on top of the existing layer,
            # possibly only partly
            if not bot_lower_than_bot[todo].any():
                # 2 the top of the existing layer needs to be lowered
                mask = todo & bot_lower_than_top
                new_top_layer = ds["top"].loc[layer]
                new_top_layer.data[mask] = bot.data[mask]
                ds["top"].loc[layer] = new_top_layer
                # the new layer can be added on top of the existing layer
                if isplit is not None:
                    isplit += 1
                ds = _insert_layer_above(
                    ds, layer, name, isplit, todo, top, bot, kh, kv, copy
                )
                todo[todo] = False
                continue
            if not bot_lower_than_bot[todo].all():
                bot_higher_than_bot = bot.data > bot_layer
                if not bot_higher_than_bot[todo].any():
                    continue
                top_lower_than_top = top.data < top_layer
                if (todo & bot_higher_than_bot & top_lower_than_top).any():
                    # 3 the existing layer needs to be split,
                    # as part of it is below and part is above the new layer
                    msg = (
                        f"Existing layer {layer} exists in some cells both above and "
                        f"below the inserted layer {name}. Therefore existing layer "
                        f"{layer} needs to be split in two, which is not supported."
                    )
                    raise (LayerError(msg))
                # 4 the new layer needs to be split, as part of the new layer is
                # above the bottom of the existing layer, and part of it is below the
                # existing layer
                if isplit is None:
                    isplit = 1
                else:
                    isplit += 1
                # the top of the existing layer needs to be lowered
                mask = todo & bot_higher_than_bot & bot_lower_than_top
                new_top_layer = ds["top"].loc[layer]
                new_top_layer.data[mask] = bot.data[mask]
                ds["top"].loc[layer] = new_top_layer
                # and we insert the new layer
                mask = todo & bot_higher_than_bot
                ds = _insert_layer_above(
                    ds, layer, name, isplit, mask, top, bot, kh, kv, copy
                )
                todo[mask] = False

        mask = todo & top_higher_than_bot
        if mask.any():
            # 5 when the new layer is not added above the existing layer, as the bottom
            # of the new layer is always lower than the bottom of the existing
            # layer: the bottom of the existing layer needs to be raised to the top
            # of the new layer
            new_bot_layer = ds["botm"].loc[layer]
            new_bot_layer.data[mask] = top.data[mask]
            ds["botm"].loc[layer] = new_bot_layer

    if todo.any():
        # 7 the new layer needs to be added to the bottom of the model
        if isplit is not None:
            isplit += 1
        ds = _insert_layer_below(ds, None, name, isplit, mask, top, bot, kh, kv, copy)
    return ds


def _insert_layer_above(ds, above_layer, name, isplit, mask, top, bot, kh, kv, copy):
    new_layer_name = _get_new_layer_name(name, isplit)
    layers = list(ds.layer.data)
    if above_layer is None:
        above_layer = layers[0]
    layers.insert(layers.index(above_layer), new_layer_name)
    ds = ds.reindex({"layer": layers}, copy=copy)
    ds = _set_new_layer_values(ds, new_layer_name, mask, top, bot, kh, kv)
    return ds


def _insert_layer_below(ds, below_layer, name, isplit, mask, top, bot, kh, kv, copy):
    new_layer_name = _get_new_layer_name(name, isplit)
    layers = list(ds.layer.data)
    if below_layer is None:
        below_layer = layers[-1]
    layers.insert(layers.index(below_layer) + 1, new_layer_name)
    ds = ds.reindex({"layer": layers}, copy=copy)
    ds = _set_new_layer_values(ds, new_layer_name, mask, top, bot, kh, kv)
    return ds


def _set_new_layer_values(ds, new_layer_name, mask, top, bot, kh, kv):
    ds["top"].loc[new_layer_name].data[mask] = top.data[mask]
    ds["botm"].loc[new_layer_name].data[mask] = bot.data[mask]
    if kh is not None:
        ds["kh"].loc[new_layer_name].data[mask] = kh.data[mask]
    if kv is not None:
        ds["kv"].loc[new_layer_name].data[mask] = kv.data[mask]
    return ds


def _get_new_layer_name(name, isplit):
    new_layer_name = name
    if isplit is not None:
        new_layer_name = new_layer_name + "_" + str(isplit)
    return new_layer_name


def remove_layer(ds, layer):
    """Removes a layer from a Dataset, without changing elevations of other layers.

    This will create gaps in the layer model.
    """
    layers = list(ds.layer.data)
    if layer not in layers:
        raise (MissingValueError(f"layer '{layer}' not present in Dataset"))
    if "layer" not in ds["top"].dims:
        index = layers.index(layer)
        if index == 0:
            # lower the top to the second layer
            ds["top"] = ds["botm"].loc[layers[1]]
    layers.remove(layer)
    ds = ds.reindex({"layer": layers})
    return ds
