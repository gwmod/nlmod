import logging
from collections import OrderedDict

import numpy as np
import xarray as xr

from . import resample, mgrid
from ..read import jarkus, rws
from .. import util

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
    if ds[top].ndim == ds[bot].ndim and ds[top].ndim in [2, 3]:
        if ds[top].shape[0] == ds[bot].shape[0]:
            # top is 3D, every layer has top and bot
            thickness = ds[top] - ds[bot]
            top3d = True
        else:
            raise ValueError(
                '3d top and bot should have same number of layers')
    elif ds[top].ndim == (ds[bot].ndim - 1) and ds[top].ndim in [1, 2]:
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
    new_nlay = (ds[layer].size -
                sum([len(c) for c in combine_layers]) +
                len(combine_layers))

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
            kheq = np.nansum(thickness.data[v, :, :] * kh.data[v, :, :], axis=0) / \
                np.nansum(thickness.data[v, :, :], axis=0)
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
            kveq = np.nansum(thickness.data[v, :, :], axis=0) / \
                np.nansum(thickness.data[v, :, :] / kv.data[v, :, :], axis=0)
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


def add_kh_kv_from_ml_layer_to_dataset(ml_layer_ds, model_ds, anisotropy,
                                       fill_value_kh, fill_value_kv):
    """add kh and kv from a model layer dataset to the model dataset.

    Supports structured and vertex grids.

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with kh and kv
    model_ds : xarray.Dataset
        dataset with model data where kh and kv are added to
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 1.0.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data with new kh and kv

    Notes
    -----
    some model dataset, such as regis, also have 'c' and 'kd' values. These
    are ignored at the moment
    """
    model_ds.attrs['anisotropy'] = anisotropy
    model_ds.attrs['fill_value_kh'] = fill_value_kh
    model_ds.attrs['fill_value_kv'] = fill_value_kv
    kh_arr = ml_layer_ds['kh'].data
    kv_arr = ml_layer_ds['kv'].data

    logger.info('add kh and kv from model layer dataset to modflow model')

    kh, kv = get_kh_kv(kh_arr, kv_arr, anisotropy,
                       fill_value_kh=fill_value_kh,
                       fill_value_kv=fill_value_kv)

    if model_ds.gridtype == 'structured':
        da_ones = util.get_da_from_da_ds(model_ds, dims=('layer', 'y', 'x'),
                                         data=1)
    elif model_ds.gridtype == 'vertex':
        da_ones = util.get_da_from_da_ds(model_ds, dims=('layer', 'icell2d'),
                                         data=1)
    else:
        raise ValueError(
            'function only support structured or vertex gridtypes')

    model_ds['kh'] = da_ones * kh

    model_ds['kv'] = da_ones * kv

    # keep attributes for bot en top
    for datavar in ['kh', 'kv']:
        for key, att in ml_layer_ds[datavar].attrs.items():
            model_ds[datavar].attrs[key] = att

    return model_ds


def get_kh_kv(kh_in, kv_in, anisotropy,
              fill_value_kh=1.0, fill_value_kv=1.0):
    """maak kh en kv rasters voor flopy vanuit een regis raster met nan
    waardes.

    vul kh raster door:
    1. pak kh uit regis, tenzij nan dan:
    2. pak kv uit regis vermenigvuldig met anisotropy, tenzij nan dan:
    3. pak fill_value_kh

    vul kv raster door:
    1. pak kv uit regis, tenzij nan dan:
    2. pak kh uit regis deel door anisotropy, tenzij nan dan:
    3. pak fill_value_kv

    Supports structured and vertex grids.

    Parameters
    ----------
    kh_in : np.ndarray
        kh from regis with nan values shape(nlay, nrow, ncol) or
        shape(nlay, len(icell2d))
    kv_in : np.ndarray
        kv from regis with nan values shape(nlay, nrow, ncol) or
        shape(nlay, len(icell2d))
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 1.0.

    Returns
    -------
    kh_out : np.ndarray
        kh without nan values (nlay, nrow, ncol) or shape(nlay, len(icell2d))
    kv_out : np.ndarray
        kv without nan values (nlay, nrow, ncol) or shape(nlay, len(icell2d))
    """
    kh_out = np.zeros_like(kh_in)
    for i, kh_lay in enumerate(kh_in):
        kh_new = kh_lay.copy()
        kv_new = kv_in[i].copy()
        if ~np.all(np.isnan(kh_new)):
            logger.debug(f'layer {i} has a kh')
            kh_out[i] = np.where(np.isnan(kh_new), kv_new * anisotropy, kh_new)
            kh_out[i] = np.where(np.isnan(kh_out[i]), fill_value_kh, kh_out[i])
        elif ~np.all(np.isnan(kv_new)):
            logger.debug(f'layer {i} has a kv')
            kh_out[i] = np.where(
                np.isnan(kv_new), fill_value_kh, kv_new * anisotropy)
        else:
            logger.info(f'kv and kh both undefined in layer {i}')
            kh_out[i] = fill_value_kh

    kv_out = np.zeros_like(kv_in)
    for i, kv_lay in enumerate(kv_in):
        kv_new = kv_lay.copy()
        kh_new = kh_in[i].copy()
        if ~np.all(np.isnan(kv_new)):
            logger.debug(f'layer {i} has a kv')
            kv_out[i] = np.where(np.isnan(kv_new), kh_new / anisotropy, kv_new)
            kv_out[i] = np.where(np.isnan(kv_out[i]), fill_value_kv, kv_out[i])
        elif ~np.all(np.isnan(kh_new)):
            logger.debug(f'layer {i} has a kh')
            kv_out[i] = np.where(
                np.isnan(kh_new), fill_value_kv, kh_new / anisotropy)
        else:
            logger.info(f'kv and kh both undefined in layer {i}')
            kv_out[i] = fill_value_kv

    return kh_out, kv_out


def fill_top_bot_kh_kv_at_mask(model_ds, fill_mask,
                               gridtype='structured'):
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
    model_ds : xr.DataSet
        model dataset, should contain 'first_active_layer'
    fill_mask : xr.DataArray
        1 where a cell should be replaced by masked value.
    gridtype : str, optional
        type of grid.

    Returns
    -------
    model_ds : xr.DataSet
        model dataset with adjusted data variables: 'top', 'bot', 'kh', 'kv'
    """

    # zee cellen hebben altijd een top gelijk aan 0
    model_ds['top'].values = np.where(fill_mask, 0, model_ds['top'])

    if gridtype == 'structured':
        fill_function = resample.fillnan_dataarray_structured_grid
        fill_function_kwargs = {}
    elif gridtype == 'vertex':
        fill_function = resample.fillnan_dataarray_vertex_grid
        fill_function_kwargs = {'model_ds': model_ds}

    for lay in range(model_ds.dims['layer']):
        bottom_nan = xr.where(fill_mask, np.nan, model_ds['bot'][lay])
        bottom_filled = fill_function(bottom_nan, **fill_function_kwargs)

        kh_nan = xr.where(fill_mask, np.nan, model_ds['kh'][lay])
        kh_filled = fill_function(kh_nan, **fill_function_kwargs)

        kv_nan = xr.where(fill_mask, np.nan, model_ds['kv'][lay])
        kv_filled = fill_function(kv_nan, **fill_function_kwargs)

        if lay == 0:
            # top ligt onder bottom_filled -> laagdikte wordt 0
            # top ligt boven bottom_filled -> laagdikte o.b.v. bottom_filled
            mask_top = model_ds['top'] < bottom_filled
            model_ds['bot'][lay] = xr.where(fill_mask * mask_top,
                                            model_ds['top'],
                                            bottom_filled)
            model_ds['kh'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kh'][lay],
                                           kh_filled)
            model_ds['kv'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kv'][lay],
                                           kv_filled)

        else:
            # top ligt onder bottom_filled -> laagdikte wordt 0
            # top ligt boven bottom_filled -> laagdikte o.b.v. bottom_filled
            mask_top = model_ds['bot'][lay - 1] < bottom_filled
            model_ds['bot'][lay] = xr.where(fill_mask * mask_top,
                                            model_ds['bot'][lay - 1],
                                            bottom_filled)
            model_ds['kh'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kh'][lay],
                                           kh_filled)
            model_ds['kv'][lay] = xr.where(fill_mask * mask_top,
                                           model_ds['kv'][lay],
                                           kv_filled)

    return model_ds


def update_model_ds_from_ml_layer_ds(model_ds, ml_layer_ds,
                                     gridtype='structured',
                                     keep_vars=None,
                                     add_northsea=True,
                                     anisotropy=10,
                                     fill_value_kh=1.,
                                     fill_value_kv=0.1,
                                     cachedir=None):
    """Update a model dataset with a model layer dataset.

    Steps:

    1. Add the data variables in 'keep_vars' from the model layer dataset
    to the model dataset
    2. add the attributes of the model layer dataset to the model dataset if
    they don't exist yet.
    3. compute idomain from the bot values in the model layer dataset, add
    to model dataset
    4. compute top and bots from model layer dataset, add to model dataset
    5. compute kh, kv from model layer dataset, add to model dataset
    6. if add_northsea is True:
        a) get cells from modelgrid that are within the northsea, add data
        variable 'northsea' to model_ds
        b) fill top, bot, kh and kv add northsea cell by extrapolation
        c) get bathymetry (northsea depth) from jarkus. Add datavariable
        bathymetry to model dataset


    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data, preferably without a grid definition.
    ml_layer_ds : xarray.Dataset
        dataset with model layer data corresponding to the modelgrid
    gridtype : str, optional
        type of grid, default is 'structured'
    keep_vars : list of str
        variables in ml_layer_ds that will be used in model_ds
    add_northsea : bool, optional
        if True the nan values at the northsea are filled using the
        bathymetry from jarkus
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 1.0.
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data
    """
    model_ds.attrs['gridtype'] = gridtype

    if keep_vars is None:
        keep_vars = []
    else:
        # update variables
        model_ds.update(ml_layer_ds[keep_vars])
        # update attributes
        for key, item in ml_layer_ds.attrs.items():
            if key not in model_ds.attrs.keys():
                model_ds.attrs.update({key: item})

    model_ds = add_idomain_from_bottom_to_dataset(ml_layer_ds['bot'],
                                                  model_ds)

    model_ds = add_top_bot_to_model_ds(ml_layer_ds, model_ds,
                                       gridtype=gridtype)

    model_ds = add_kh_kv_from_ml_layer_to_dataset(ml_layer_ds,
                                                  model_ds,
                                                  anisotropy,
                                                  fill_value_kh,
                                                  fill_value_kv)

    if add_northsea:
        logger.info(
            'nan values at the northsea are filled using the bathymetry from jarkus')

        # find grid cells with northsea
        model_ds.update(rws.get_northsea(model_ds,
                                         cachedir=cachedir,
                                         cachename='sea_model_ds.nc'))

        # fill top, bot, kh, kv at sea cells
        fill_mask = (model_ds['first_active_layer'] ==
                     model_ds.nodata) * model_ds['northsea']
        model_ds = fill_top_bot_kh_kv_at_mask(model_ds, fill_mask,
                                              gridtype=gridtype)

        # add bathymetry noordzee
        model_ds.update(jarkus.get_bathymetry(model_ds,
                                              model_ds['northsea'],
                                              cachedir=cachedir,
                                              cachename='bathymetry_model_ds.nc'))

        model_ds = jarkus.add_bathymetry_to_top_bot_kh_kv(model_ds,
                                                          model_ds['bathymetry'],
                                                          fill_mask)

        # update idomain on adjusted tops and bots
        model_ds['thickness'], _ = calculate_thickness(model_ds)
        model_ds['idomain'] = update_idomain_from_thickness(model_ds['idomain'],
                                                            model_ds['thickness'],
                                                            model_ds['northsea'])
        model_ds['first_active_layer'] = mgrid.get_first_active_layer_from_idomain(
            model_ds['idomain'])

    else:
        model_ds['thickness'], _ = calculate_thickness(model_ds)
        model_ds['first_active_layer'] = mgrid.get_first_active_layer_from_idomain(
            model_ds['idomain'])

    return model_ds


def add_idomain_from_bottom_to_dataset(bottom, model_ds, nodata=-999):
    """add idomain and first_active_layer to model_ds The active layers are
    defined as the layers where the bottom is not nan.

    Parameters
    ----------
    bottom : xarray.DataArray
        DataArray with bottom values of each layer. Nan values indicate
        inactive cells.
    model_ds : xarray.Dataset
        dataset with model data where idomain and first_active_layer
        are added to.
    nodata : int, optional
        nodata value used in integer arrays. For float arrays np.nan is use as
        nodata value. The default is -999.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including idomain and first_active_layer
    """
    logger.info('get active cells (idomain) from bottom DataArray')

    idomain = xr.where(bottom.isnull(), -1, 1)

    # if the top cell is inactive set idomain = 0, for other inactive cells
    # set idomain = -1
    idomain[0] = xr.where(idomain[0] == -1, 0, idomain[0])
    for i in range(1, bottom.shape[0]):
        idomain[i] = xr.where((idomain[i - 1] == 0)
                              & (idomain[i] == -1), 0, idomain[i])

    model_ds['idomain'] = idomain
    model_ds['first_active_layer'] = mgrid.get_first_active_layer_from_idomain(idomain,
                                                                               nodata=nodata)

    model_ds.attrs['nodata'] = nodata

    return model_ds


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


def add_top_bot_to_model_ds(ml_layer_ds, model_ds,
                            nodata=None,
                            gridtype='structured'):
    """add top and bot from a model layer dataset to THE model dataset.

    Supports structured and vertex grids.

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with a top and bottom
    model_ds : xarray.Dataset
        dataset with model data where top and bot are added to
    nodata : int, optional
        if the first_active_layer data array in model_ds has this value,
        it means this cell is inactive in all layers. If nodata is None the
        nodata value in model_ds is used.
        the default is None
    gridtype : str, optional
        type of grid, options are 'structured' and 'vertex'.
        The default is 'structured'.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including top and bottom
    """
    if nodata is None:
        nodata = model_ds.attrs['nodata']

    logger.info(
        'using top and bottom from model layers dataset for modflow model')
    logger.info('replace nan values for inactive layers with dummy value')

    if gridtype == 'structured':
        model_ds = add_top_bot_structured(ml_layer_ds, model_ds,
                                          nodata=nodata)

    elif gridtype == 'vertex':
        model_ds = add_top_bot_vertex(ml_layer_ds, model_ds,
                                      nodata=nodata)

    return model_ds


def add_top_bot_vertex(ml_layer_ds, model_ds, nodata=-999):
    """Voeg top en bottom vanuit layer dataset toe aan de model dataset.

    Deze functie is bedoeld voor vertex arrays in modflow 6. Supports
    only vertex grids.

    Stappen:

    1. Zorg dat de onderste laag altijd een bodemhoogte heeft, als de bodem
       van alle bovenliggende lagen nan is, pak dan 0.
    2. Zorg dat de top van de bovenste laag altijd een waarde heeft, als de
       top van alle onderligende lagen nan is, pak dan 0.
    3. Vul de nan waarden in alle andere lagen door:
        a) pak bodem uit regis, tenzij nan dan:
        b) gebruik bodem van de laag erboven (of de top voor de bovenste laag)

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with a top and bottom
    model_ds : xarray.Dataset
        dataset with model data where top and bottom are added to
    nodata : int, optional
        if the first_active_layer data array in model_ds has this value,
        it means this cell is inactive in all layers

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including top and bottom
    """
    # step 1:
    # set nan-value in bottom array
    # set to zero if value is nan in all layers
    # set to minimum value of all layers if there is any value in any layer
    active_domain = model_ds['first_active_layer'].data != nodata

    lowest_bottom = ml_layer_ds['bot'].data[-1].copy()
    if np.any(~active_domain):
        percentage = 100 * (~active_domain).sum() / \
            (active_domain.shape[0])
        if percentage > 80:
            logger.warning(f'{percentage:0.1f}% of all cells have nan '
                           'values in every layer there is probably a '
                           'problem with your extent.')

        # set bottom to zero if bottom in a cell is nan in all layers
        lowest_bottom = np.where(active_domain, lowest_bottom, 0)

    if np.any(np.isnan(lowest_bottom)):
        # set bottom in a cell to lowest bottom of all layers
        i_nan = np.where(np.isnan(lowest_bottom))
        for i in i_nan:
            val = np.nanmin(ml_layer_ds['bot'].data[:, i])
            lowest_bottom[i] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Artesia')

    # step 2: get highest top values of all layers without nan values
    highest_top = ml_layer_ds['top'].data[0].copy()
    if np.any(np.isnan(highest_top)):
        highest_top = np.where(active_domain, highest_top, 0)

    if np.any(np.isnan(highest_top)):
        i_nan = np.where(np.isnan(highest_top))
        for i in i_nan:
            val = np.nanmax(ml_layer_ds['top'].data[:, i])
            highest_top[i] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Artesia')

    # step 3: fill nans in all layers
    nlay = model_ds.dims['layer']
    top_bot_raw = np.ones((nlay + 1, model_ds.dims['icell2d']))
    top_bot_raw[0] = highest_top
    top_bot_raw[1:-1] = ml_layer_ds['bot'].data[:-1].copy()
    top_bot_raw[-1] = lowest_bottom
    top_bot = np.ones_like(top_bot_raw)
    for i_from_bot, blay in enumerate(top_bot_raw[::-1]):
        i_from_top = nlay - i_from_bot
        new_lay = blay.copy()
        if np.any(np.isnan(new_lay)):
            lay_from_bot = i_from_bot
            lay_from_top = nlay - lay_from_bot
            while np.any(np.isnan(new_lay)):
                new_lay = np.where(np.isnan(new_lay),
                                   top_bot_raw[lay_from_top],
                                   new_lay)
                lay_from_bot += 1
                lay_from_top = nlay - lay_from_bot

        top_bot[i_from_top] = new_lay

    model_ds['bot'] = xr.DataArray(top_bot[1:], dims=('layer', 'icell2d'),
                                   coords={'icell2d': model_ds.icell2d.data,
                                           'layer': model_ds.layer.data})
    model_ds['top'] = xr.DataArray(top_bot[0], dims=('icell2d',),
                                   coords={'icell2d': model_ds.icell2d.data})

    # keep attributes for bot en top
    for datavar in ['top', 'bot']:
        for key, att in ml_layer_ds[datavar].attrs.items():
            model_ds[datavar].attrs[key] = att

    return model_ds


def add_top_bot_structured(ml_layer_ds, model_ds, nodata=-999):
    """Voeg top en bottom vanuit een layer dataset toe aan de model dataset.

    Deze functie is bedoeld voor structured arrays in modflow 6. Supports
    only structured grids.

    Stappen:

    1. Zorg dat de onderste laag altijd een bodemhoogte heeft, als de bodem
       van alle bovenliggende lagen nan is, pak dan 0.
    2. Zorg dat de top van de bovenste laag altijd een waarde heeft, als de
       top van alle onderligende lagen nan is, pak dan 0.
    3. Vul de nan waarden in alle andere lagen door:
        a) pak bodem uit de model layer dataset, tenzij nan dan:
        b) gebruik bodem van de laag erboven (of de top voor de bovenste laag)

    Parameters
    ----------
    ml_layer_ds : xarray.Dataset
        dataset with model layer data with a top and bottom
    model_ds : xarray.Dataset
        dataset with model data where top and bottom are added to
    nodata : int, optional
        if the first_active_layer data array in model_ds has this value,
        it means this cell is inactive in all layers

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with model data including top and bottom
    """

    active_domain = model_ds['first_active_layer'].data != nodata

    # step 1:
    # set nan-value in bottom array
    # set to zero if value is nan in all layers
    # set to minimum value of all layers if there is any value in any layer
    lowest_bottom = ml_layer_ds['bot'].data[-1].copy()
    if np.any(~active_domain):
        percentage = 100 * (~active_domain).sum() / \
            (active_domain.shape[0] * active_domain.shape[1])
        if percentage > 80:
            logger.warning(f'{percentage:0.1f}% of all cells have nan '
                           'values in every layer there is probably a '
                           'problem with your extent.')
        # set bottom to zero if bottom in a cell is nan in all layers
        lowest_bottom = np.where(active_domain, lowest_bottom, 0)

    if np.any(np.isnan(lowest_bottom)):
        # set bottom in a cell to lowest bottom of all layers
        rc_nan = np.where(np.isnan(lowest_bottom))
        for row, col in zip(rc_nan[0], rc_nan[1]):
            val = np.nanmin(ml_layer_ds['bot'].data[:, row, col])
            lowest_bottom[row, col] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Onno')

    # step 2: get highest top values of all layers without nan values
    highest_top = ml_layer_ds['top'].data[0].copy()
    if np.any(np.isnan(highest_top)):
        # set top to zero if top in a cell is nan in all layers
        highest_top = np.where(active_domain, highest_top, 0)

    if np.any(np.isnan(highest_top)):
        # set top in a cell to highest top of all layers
        rc_nan = np.where(np.isnan(highest_top))
        for row, col in zip(rc_nan[0], rc_nan[1]):
            val = np.nanmax(ml_layer_ds['top'].data[:, row, col])
            highest_top[row, col] = val
            if np.isnan(val):
                raise ValueError(
                    'this should never happen please contact Onno')

    # step 3: fill nans in all layers
    nlay = model_ds.dims['layer']
    nrow = model_ds.dims['y']
    ncol = model_ds.dims['x']
    top_bot_raw = np.ones((nlay + 1, nrow, ncol))
    top_bot_raw[0] = highest_top
    top_bot_raw[1:-1] = ml_layer_ds['bot'].data[:-1].copy()
    top_bot_raw[-1] = lowest_bottom
    top_bot = np.ones_like(top_bot_raw)
    for i_from_bot, blay in enumerate(top_bot_raw[::-1]):
        i_from_top = nlay - i_from_bot
        new_lay = blay.copy()
        if np.any(np.isnan(new_lay)):
            lay_from_bot = i_from_bot
            lay_from_top = nlay - lay_from_bot
            while np.any(np.isnan(new_lay)):
                new_lay = np.where(np.isnan(new_lay),
                                   top_bot_raw[lay_from_top],
                                   new_lay)
                lay_from_bot += 1
                lay_from_top = nlay - lay_from_bot

        top_bot[i_from_top] = new_lay

    model_ds['bot'] = xr.DataArray(top_bot[1:], dims=('layer', 'y', 'x'),
                                   coords={'x': model_ds.x.data,
                                           'y': model_ds.y.data,
                                           'layer': model_ds.layer.data})

    model_ds['top'] = xr.DataArray(top_bot[0], dims=('y', 'x'),
                                   coords={'x': model_ds.x.data,
                                           'y': model_ds.y.data})

    # keep attributes for bot en top
    for datavar in ['top', 'bot']:
        for key, att in ml_layer_ds[datavar].attrs.items():
            model_ds[datavar].attrs[key] = att

    return model_ds
