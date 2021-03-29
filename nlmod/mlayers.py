from collections import OrderedDict

import numpy as np
import xarray as xr


def layer_split_top_bot(ds, split_dict, layer='layer', top='top', bot='bot',
                        verbose=True):
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
    verbose : bool, optional
        if True print information to console, by default True

    Returns
    -------
    new_top, new_bot : xarray.DataArrays
        DataArrays containing new tops and bottoms after splitting layers.
    """

    # calculate thickness
    if ds.top.ndim == 3 and ds[top].shape[0] == ds[bot].shape[0]:
        # top is 3D, every layer has top and bot
        thickness = ds[top].data - ds[bot].data
        top3d = True
    else:
        # top is only top of first layer
        thickness = -1 * \
            np.diff(np.concatenate(
                [ds[top].data, ds[bot].data], axis=0), axis=0)
        top3d = False

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
            if verbose:
                print(f"{i}: Split layer {i} into {len(sf)} layers "
                      f"with fractions: {sf}")

            # loop over split factors
            for isf, factor in enumerate(sf):
                if verbose:
                    print(f"  - {isf}: Calculate new top/bot for "
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
            if verbose:
                print(f"{i:2d}: No split: map layer {i} to "
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
                    top='top', bot='bot', kh='kh', kv='kv',
                    verbose=True):
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
    verbose : bool, optional
        if True print information to console, by default True

    Returns
    -------
    ds_split : xarray.Dataset
        Dataset with new tops and bottoms taking into account split layers,
        and filled data for hydraulic conductivities.
    """

    parsed_dv = set([top, bot, kh, kv])

    if verbose:
        dropped_dv = set(ds.data_vars.keys()) - parsed_dv
        if len(dropped_dv) > 0:
            print("Warning! Following data variables "
                  f"will be dropped: {dropped_dv}")

    # calculate new tops/bots
    if verbose:
        print("Calculating new layer tops and bottoms...")

    new_top, new_bot, reindexer = layer_split_top_bot(
        ds, split_dict, layer=layer, top=top, bot=bot, verbose=verbose)

    # fill kh/kv
    if verbose:
        print(f"Fill value '{kh}' for split layers with "
              "value original layer.")
    da_kh = fill_data_split_layers(ds["kh"], reindexer)
    if verbose:
        print(f"Fill value '{kv}' for split layers with "
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
    if verbose:
        print("Done! Created new dataset with split layers!")
    ds_split = xr.Dataset({"top": new_top,
                           "bot": new_bot,
                           "kh": da_kh,
                           "kv": da_kv},
                          attrs=attrs)

    return ds_split
