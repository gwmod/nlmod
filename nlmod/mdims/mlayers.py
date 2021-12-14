import logging
from collections import OrderedDict

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def calculate_thickness(ds, top="top", bot="bot"):
    """Calculate thickness from dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing information about top and bottom elevations
        of layers
    top : str, optional
        name of data variable containing tops, by default "top"
    bot : str, optional
        name of data variable containing bottoms, by default "bot"

    Returns
    -------
    thickness : xarray.DataArray
        DataArray containing thickness information
    top3d : bool
        boolean whether each layer has a top elevation (True),
        or top only indicates surface level and bottom of layer i
        is top of layer i+1 (False)
    """
    # calculate thickness
    if ds[top].ndim == ds[bot].ndim and ds[top].ndim in [2,3]:
        if ds[top].shape[0] == ds[bot].shape[0]:
            # top is 3D, every layer has top and bot
            thickness = ds[top] - ds[bot]
            top3d = True
        else:
            raise ValueError('3d top and bot should have same number of layers')
    elif ds[top].ndim == (ds[bot].ndim -1) and ds[top].ndim in [1,2]:
        if ds[top].shape[-1] == ds[bot].shape[-1]:
            # top is only top of first layer
            thickness = xr.zeros_like(ds[bot])
            for lay in range(len(bot)):
                if lay == 0:
                    thickness[lay] = ds[top] - ds[bot][lay]
                else:
                    thickness[lay] = ds[bot][lay - 1] - ds[bot][lay]
            top3d = False
        else:
            raise ValueError('2d top should have same last dimension as bot')
    if isinstance(ds[bot], xr.DataArray):
        if hasattr(ds[bot], 'units'):
            if ds[bot].units == 'mNAP':
                thickness.attrs['units'] = 'm'
            else:
                thickness.attrs['units'] = ds[bot].units

    return thickness, top3d


def layer_split_top_bot(ds, split_dict, layer='layer', top='top', bot='bot'):
    """Calculate new tops and bottoms for split layers.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset containing information about layers
        (layers, top and bot)
    split_dict : dict
        dictionary with index of layers to split as keys and iterable
        of fractions that add up to 1 to indicate how to split up layer.
        E.g. {0: [0.25, 0.75]} will split layer 0 into 2 layers, with first
        layer equal to 0.25 of original thickness and second layer 0.75 of
        original thickness.
    layer : str, optional
        name of layer dimension, by default 'layer'
    top : str, optional
        name of data variable containing top of layers, by default 'top'
    bot : str, optional
        name of data variable containing bottom of layers, by default 'bot'

    Returns
    -------
    new_top, new_bot : xarray.DataArrays
        DataArrays containing new tops and bottoms after splitting layers.
    reindexer : OrderedDict
        dictionary mapping new to old layer indices.
    """

    # calculate thickness
    thickness, top3d = calculate_thickness(ds, top=top, bot=bot)

    # calculate new number of layers
    new_nlay = ds[layer].size + \
        sum([len(sf) for sf in split_dict.values()]) - len(split_dict)

    # create new DataArrays for storing new top/bot
    new_bot = xr.DataArray(data=np.nan,
                           dims=["layer", "y", "x"],
                           coords={"layer": np.arange(new_nlay),
                                   "y": ds.y.data,
                                   "x": ds.x.data})
    new_top = xr.DataArray(data=np.nan,
                           dims=["layer", "y", "x"],
                           coords={"layer": np.arange(new_nlay),
                                   "y": ds.y.data,
                                   "x": ds.x.data})

    # dict to keep track of old and new layer indices
    reindexer = OrderedDict()

    j = 0  # new layer index
    isplit = 0  # split layer index

    # loop over original layers
    for i in range(ds[layer].size):

        # check if layer should be split
        if i in split_dict:

            # set new top based on old top
            if top3d:
                new_top.data[j] = ds[top].data[i]
            else:
                if i == 0:
                    new_top.data[j] = ds[top].data
                else:
                    new_top.data[j] = ds[bot].data[i - 1]

            # get split factors
            sf = split_dict[i]

            # check if factors add up to 1
            if np.sum(sf) != 1.0:
                raise ValueError(
                    "Sum of split factors for layer must equal 1.0!")
            logger.debug(f"{i}: Split layer {i} into {len(sf)} layers "
                         f"with fractions: {sf}")

            # loop over split factors
            for isf, factor in enumerate(sf):
                logger.debug(f"  - {isf}: Calculate new top/bot for "
                             f"new layer index {j}")

                # calculate new bot and new top
                new_bot.data[j] = new_top.data[j] - (factor * thickness[i])
                new_top.data[j + 1] = new_bot.data[j]

                # store new and old layer index
                reindexer[j] = i

                # increase new index
                j += 1

            # go to next layer to split
            isplit += 1

        # no split, remap old layer to new layer index
        else:
            logger.debug(f"{i:2d}: No split: map layer {i} to "
                         f"new layer index {j}")
            if top3d:
                new_top.data[j] = ds[top].data[i]
            else:
                if i == 0:
                    new_top.data[j] = ds[top].data.squeeze()
                else:
                    new_top.data[j] = ds[bot].data[i - 1]

            new_bot.data[j] = ds[bot].data[i]
            reindexer[j] = i
            j += 1

    return new_top, new_bot, reindexer


def fill_data_split_layers(da, reindexer):
    """Fill data for split layers with values from original layer.

    Parameters
    ----------
    da : xarray.DataArray or numpy.ndarray
        original array with data
    reindexer : dict
        dictionary containing mapping between new layer index and
        original layer index.

    Returns
    -------
    da_new : xarray.DataArray or numpy.ndarray
        array with filled data for split layers
    """
    if isinstance(da, xr.DataArray):
        da_new = xr.DataArray(data=np.nan,
                              dims=["layer", "y", "x"],
                              coords={"layer": np.arange(
                                  list(reindexer.keys())[-1] + 1),
                                  "y": da["y"],
                                  "x": da["x"]})
        for k, v in reindexer.items():
            da_new.data[k] = da.data[v]
    elif isinstance(da, np.ndarray):
        da_new = np.zeros((list(reindexer.keys())[-1] + 1), *da.shape[1:])
        for k, v in reindexer.items():
            da_new[k] = da[v]
    else:
        raise TypeError(f"Cannot fill type: '{type(da)}'!")
    return da_new


def split_layers_ds(ds, split_dict, layer='layer',
                    top='top', bot='bot', kh='kh', kv='kv'):
    """Split layers based in Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset containing information about layers
        (layers, top and bot)
    split_dict : dict
        dictionary with index of layers to split as keys and iterable
        of fractions that add up to 1 to indicate how to split up layer.
        E.g. {0: [0.25, 0.75]} will split layer 0 into 2 layers, with first
        layer equal to 0.25 of original thickness and second layer 0.75 of
        original thickness.
    layer : str, optional
        name of layer dimension, by default 'layer'
    top : str, optional
        name of data variable containing top of layers, by default 'top'
    bot : str, optional
        name of data variable containing bottom of layers, by default 'bot'
    kh : str, opti
        name of data variable containg horizontal hydraulic conductivity,
        by default 'kh'
    kv : str, optional
        name of data variable containg vertical hydraulic conductivity,
        by default 'kv'

    Returns
    -------
    ds_split : xarray.Dataset
        Dataset with new tops and bottoms taking into account split layers,
        and filled data for hydraulic conductivities.
    """

    parsed_dv = set([top, bot, kh, kv])

    dropped_dv = set(ds.data_vars.keys()) - parsed_dv
    if len(dropped_dv) > 0:
        logger.warning("Warning! Following data variables "
                       f"will be dropped: {dropped_dv}")

    # calculate new tops/bots
    logger.info("Calculating new layer tops and bottoms...")

    new_top, new_bot, reindexer = layer_split_top_bot(
        ds, split_dict, layer=layer, top=top, bot=bot)

    # fill kh/kv
    logger.info(f"Fill value '{kh}' for split layers with "
                "value original layer.")
    da_kh = fill_data_split_layers(ds["kh"], reindexer)
    logger.info(f"Fill value '{kv}' for split layers with "
                "value original layer.")
    da_kv = fill_data_split_layers(ds["kv"], reindexer)

    # get new layer names
    layer_names = []
    for j, i in reindexer.items():
        layercode = ds[layer].data[i]

        if layercode in layer_names:
            if isinstance(layercode, str):
                ilay = np.sum([1 for ilay in layer_names
                               if ilay.startswith(layercode)]) + 1
                layercode += f"_{ilay}"
            else:
                layercode = j

        layer_names.append(layercode)

    # assign new layer names
    new_top = new_top.assign_coords(layer=layer_names)
    new_bot = new_bot.assign_coords(layer=layer_names)
    da_kh = da_kh.assign_coords(layer=layer_names)
    da_kv = da_kv.assign_coords(layer=layer_names)

    # add reindexer to attributes
    attrs = ds.attrs.copy()
    attrs["split_reindexer"] = reindexer

    # create new dataset
    logger.info("Done! Created new dataset with split layers!")
    ds_split = xr.Dataset({"top": new_top,
                           "bot": new_bot,
                           "kh": da_kh,
                           "kv": da_kv},
                          attrs=attrs)

    return ds_split


def layer_combine_top_bot(ds, combine_layers, layer='layer',
                          top='top', bot='bot'):
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
        name of data variable containing bottom of layers, by default 'bot'

    Returns
    -------
    new_top, new_bot : xarray.DataArrays
        DataArrays containing new tops and bottoms after splitting layers.
    reindexer : OrderedDict
        dictionary mapping new to old layer indices.
    """
    # calculate new number of layers
    new_nlay = (ds[layer].size
                - sum([len(c) for c in combine_layers])
                + len(combine_layers))

    # create new DataArrays for storing new top/bot
    new_bot = xr.DataArray(data=np.nan,
                           dims=["layer", "y", "x"],
                           coords={"layer": np.arange(new_nlay),
                                   "y": ds.y.data,
                                   "x": ds.x.data})
    new_top = xr.DataArray(data=np.nan,
                           dims=["layer", "y", "x"],
                           coords={"layer": np.arange(new_nlay),
                                   "y": ds.y.data,
                                   "x": ds.x.data})

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
                logger.debug(f"{j:2d}: Merge layers {c} as layer {j}, "
                             "calculate new top/bot.")
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
            logger.debug(f"{j:2d}: Do not merge, map old layer index "
                         "to new layer index.")
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
    da_new = xr.DataArray(data=np.nan,
                          dims=["layer", "y", "x"],
                          coords={"layer": np.arange(
                              list(reindexer.keys())[-1] + 1),
                              "y": da["y"],
                              "x": da["x"]})

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
    da_kh = xr.DataArray(data=np.nan,
                         dims=["layer", "y", "x"],
                         coords={"layer": np.arange(
                             list(reindexer.keys())[-1] + 1),
                             "y": kh["y"],
                             "x": kh["x"]})

    for k, v in reindexer.items():
        if isinstance(v, tuple):
            kheq = np.nansum(thickness[v, :, :] * kh.data[v, :, :], axis=0) / \
                np.nansum(thickness[v, :, :], axis=0)
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
    da_kv = xr.DataArray(data=np.nan,
                         dims=["layer", "y", "x"],
                         coords={"layer": np.arange(
                             list(reindexer.keys())[-1] + 1),
                             "y": kv["y"],
                             "x": kv["x"]})

    for k, v in reindexer.items():
        if isinstance(v, tuple):
            kveq = np.nansum(thickness[v, :, :], axis=0) / \
                np.nansum(thickness[v, :, :] / kv.data[v, :, :], axis=0)
        else:
            kveq = kv.data[v]
        da_kv.data[k] = kveq
    return da_kv


def combine_layers_ds(ds, combine_layers, layer='layer',
                      top='top', bot='bot',
                      kh="kh", kv="kv", kD="kD", c="c"):
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
        name of data variable containing bottom of layers, by default 'bot'
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
        logger.warning("Warning! Following data variables "
                       f"will be dropped: {dropped_dv}")

    # calculate new tops/bots
    logger.info("Calculating new layer tops and bottoms...")

    da_dict = {}

    new_top, new_bot, reindexer = layer_combine_top_bot(
        ds, combine_layers, layer=layer, top=top, bot=bot)
    da_dict[top] = new_top
    da_dict[bot] = new_bot

    # calculate original thickness
    thickness, _ = calculate_thickness(ds, top=top, bot=bot)

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
    ds_combine = xr.Dataset(da_dict,
                            attrs=attrs)

    return ds_combine
