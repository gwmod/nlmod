import datetime as dt
import logging
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .. import NLMOD_DATADIR, cache
from ..dims.layers import insert_layer, remove_layer, remove_layer_dim_from_top
from ..util import MissingValueError

logger = logging.getLogger(__name__)

GEOTOP_URL = "https://dinodata.nl/opendap/GeoTOP/geotop.nc"


def get_lithok_props(rgb_colors=True):
    fname = os.path.join(NLMOD_DATADIR, "geotop", "litho_eenheden.csv")
    df = pd.read_csv(fname, index_col=0)
    if rgb_colors:
        df["color"] = pd.Series(get_lithok_colors())
    return df


def get_lithok_colors():
    colors = {
        0: (200, 200, 200),
        1: (157, 78, 64),
        2: (0, 146, 0),
        3: (194, 207, 92),
        5: (255, 255, 0),
        6: (243, 225, 6),
        7: (231, 195, 22),
        8: (216, 163, 32),
        9: (95, 95, 255),
    }
    colors = {key: tuple(x / 255 for x in color) for key, color in colors.items()}
    return colors


def get_strat_props():
    fname = os.path.join(NLMOD_DATADIR, "geotop", "REF_GTP_STR_UNIT.csv")
    df = pd.read_csv(fname, keep_default_na=False, na_values="")
    # rename the columns to previously used values
    # so existing nlmod-code will keep working
    df = df.rename(
        columns={"STR_UNIT_CD": "code", "VOXEL_NR": "strat", "DESCRIPTION": "name"}
    )
    # calculate color from red, green and blue columns
    color = {}
    for index in df.index:
        color[index] = (
            df.at[index, "RED_DEC"] / 255,
            df.at[index, "GREEN_DEC"] / 255,
            df.at[index, "BLUE_DEC"] / 255,
        )
    df["color"] = color
    df = df.drop(columns=["RED_DEC", "GREEN_DEC", "BLUE_DEC"]).set_index("strat")

    return df


def get_kh_kv_table(kind="Brabant"):
    if kind == "Brabant":
        fname = os.path.join(
            NLMOD_DATADIR,
            "geotop",
            "hydraulische_parameterisering_geotop_noord-brabant_en_noord-_en_midden-limburg.csv",
        )
        df = pd.read_csv(fname)
    else:
        raise (ValueError(f"Unknown kind in get_kh_kv_table: '{kind}'"))
    return df


@cache.cache_netcdf()
def to_model_layers(
    geotop_ds,
    strat_props=None,
    method_geulen="add_to_layer_below",
    drop_layer_dim_from_top=True,
    **kwargs,
):
    """Convert geotop voxel dataset to layered dataset.

    Converts geotop data to dataset with layer elevations and hydraulic conductivities.
    Optionally uses hydraulic conductivities provided present in geotop_ds.

    Parameters
    ----------
    geotop_ds : xr.DataSet
        geotop voxel dataset (download using `get_geotop(extent)`)
    strat_props : pd.DataFrame, optional
        The properties (code and name) of the stratigraphic units. Load with
        get_strat_props() when None. The default is None.
    method_geulen : str, optional
        strat-units >=6000 are so-called 'geulen' (paleochannels, gullies). These are
        difficult to add to the layer model, because they can occur above and/or below
        any other unit. Multiple methods are available to handle these 'geulen'.
        The method "add_to_layer_below" adds the thickness of the 'geul' to the layer
        with a positive thickness below the 'geul'. The method "add_to_layer_above"
        adds the thickness of the 'geul' to the layer with a positive thickness above
        the 'geul'. The method "add_as_layer" tries to add the 'geulen' as one or more
        layers, which can fail if a 'geul' is locally both below the top and above the
        bottom of another layer (splitting the layer in two, which is not supported).
        The default is "add_to_layer_below".
    drop_layer_dim_from_top : bool, optional
        When True, fill NaN values in top and botm and drop the layer dimension from
        top. This will transform top and botm to the data model in MODFLOW. An advantage
        of this data model is that the layer model is consistent by definition, with no
        possibilities of gaps between layers. The default is True.
    kwargs : dict
        Kwargs are passed to `aggregate_to_ds()`

    Returns
    -------
    ds: xr.DataSet
        dataset with top and botm (and optionally kh and kv) per geotop layer
    """
    if strat_props is None:
        strat_props = get_strat_props()

    # get all strat-units in Dataset
    strat = geotop_ds["strat"].values
    units = np.unique(strat)
    units = units[~np.isnan(units)].astype(int)
    shape = (len(units), len(geotop_ds.y), len(geotop_ds.x))

    if "SEQ_NR" in strat_props.columns:
        # sort units based on SEQ_NR in strat_props
        units = strat_props.loc[units, "SEQ_NR"].sort_values().index.values
    else:
        # stratigraphy unit (geo eenheid) 2000 is above 1130
        if (2000 in units) and (1130 in units):
            units[(units == 2000) + (units == 1130)] = [2000, 1130]

    # fill top and bot
    top = np.full(shape, np.nan)
    bot = np.full(shape, np.nan)

    z = (
        geotop_ds["z"]
        .data[:, np.newaxis, np.newaxis]
        .repeat(len(geotop_ds.y), 1)
        .repeat(len(geotop_ds.x), 2)
    )
    layers = []
    geulen = []
    for layer, unit in enumerate(units):
        mask = strat == unit
        top[layer] = np.nanmax(np.where(mask, z, np.nan), 0) + 0.25
        bot[layer] = np.nanmin(np.where(mask, z, np.nan), 0) - 0.25
        if int(unit) in strat_props.index:
            layers.append(strat_props.at[unit, "code"])
        else:
            logger.warning(f"Unknown strat-value: {unit}")
            layers.append(str(unit))
        if unit >= 6000:
            geulen.append(layers[-1])

    dims = ("layer", "y", "x")
    coords = {"layer": layers, "y": geotop_ds.y, "x": geotop_ds.x}
    ds = xr.Dataset({"top": (dims, top), "botm": (dims, bot)}, coords=coords)

    if method_geulen is None:
        pass
    elif method_geulen == "add_as_layer":
        top = ds["top"].copy(deep=True)
        bot = ds["botm"].copy(deep=True)
        for geul in geulen:
            ds = remove_layer(ds, geul)
        for geul in geulen:
            ds = insert_layer(ds, geul, top.loc[geul], bot.loc[geul])
    elif method_geulen == "add_to_layer_below":
        top = ds["top"].copy(deep=True)
        bot = ds["botm"].copy(deep=True)
        for geul in geulen:
            ds = remove_layer(ds, geul)
        for geul in geulen:
            todo = (top.loc[geul] - bot.loc[geul]) > 0.0
            for layer in ds.layer:
                if not todo.any():
                    continue
                # adds the thickness of the geul to the layer below the geul
                mask = (top.loc[geul] > bot.loc[layer]) & todo
                if mask.any():
                    ds["top"].loc[layer].data[mask] = np.maximum(
                        top.loc[geul].data[mask], top.loc[layer].data[mask]
                    )
                    todo.data[mask] = False
            if todo.any():
                # unless the geul is the bottom layer
                # then its thickness is added to the last active layer
                # idomain = get_idomain(ds)
                # fal = get_last_active_layer_from_idomain(idomain)
                logger.warning(
                    f"Geul {geul} is at the bottom of the GeoTOP-dataset in {int(todo.sum())} cells, where it is ignored"
                )

    elif method_geulen == "add_to_layer_above":
        top = ds["top"].copy(deep=True)
        bot = ds["botm"].copy(deep=True)
        for geul in geulen:
            ds = remove_layer(ds, geul)
        for geul in geulen:
            todo = (top.loc[geul] - bot.loc[geul]) > 0.0
            for layer in reversed(ds.layer):
                if not todo.any():
                    continue
                # adds the thickness of the geul to the layer above the geul
                mask = (bot.loc[geul] < top.loc[layer]) & todo
                if mask.any():
                    ds["botm"].loc[layer].data[mask] = np.minimum(
                        bot.loc[geul].data[mask], bot.loc[layer].data[mask]
                    )
                    todo.data[mask] = False
            if todo.any():
                # unless the geul is the top layer
                # then its thickness is added to the last active layer
                # idomain = get_idomain(ds)
                # fal = get_first_active_layer_from_idomain(idomain)
                logger.warning(
                    f"Geul {geul} is at the top of the GeoTOP-dataset in {int(todo.sum())} cells, where it is ignored"
                )
    else:
        raise (ValueError(f"Unknown method to deal with geulen: '{method_geulen}'"))

    ds.attrs["geulen"] = geulen

    if drop_layer_dim_from_top:
        ds = remove_layer_dim_from_top(ds)

    if "kh" in geotop_ds and "kv" in geotop_ds:
        aggregate_to_ds(geotop_ds, ds, **kwargs)

    # add atributes
    for datavar in ds:
        ds[datavar].attrs["source"] = "Geotop"
        ds[datavar].attrs["url"] = GEOTOP_URL
        ds[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
        if datavar in ["top", "bot"]:
            ds[datavar].attrs["units"] = "mNAP"
        elif datavar in ["kh", "kv"]:
            ds[datavar].attrs["units"] = "m/day"

    return ds


def get_geotop(*args, **kwargs):
    """Get a slice of the geotop netcdf url within the extent, set the x and y
    coordinates to match the cell centers and keep only the strat and lithok data
    variables.

    .. deprecated:: 0.10.0
        `get_geotop` will be removed in nlmod 1.0.0, it is replaced by
        `download_geotop` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    url : str, optional
        url of geotop netcdf file. The default is
        http://www.dinodata.nl/opendap/GeoTOP/geotop.nc
    probabilities : bool, optional
        if True, also download probability data. The default is False.

    Returns
    -------
    gt : xarray Dataset
        slices geotop netcdf.
    """
    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.geotop.download_geotop() in the future.",
        DeprecationWarning,
    )
    return download_geotop(*args, **kwargs)


@cache.cache_netcdf()
def download_geotop(extent, url=GEOTOP_URL, probabilities=False, chunks="auto"):
    """Get a slice of the geotop netcdf url within the extent, set the x and y
    coordinates to match the cell centers and keep only the strat and lithok data
    variables.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    url : str, optional
        url of geotop netcdf file. The default is
        http://www.dinodata.nl/opendap/GeoTOP/geotop.nc
    probabilities : bool, optional
        if True, also download probability data. The default is False.
    chunks : int, dict, 'auto' or None. The default is 'auto'.
        If provided, used to load the data into dask arrays.
        - ``chunks="auto"`` will use dask ``auto`` chunking.
        - ``chunks=None`` skips using dask. This uses xarray's internally private lazy
          indexing classes, but data is eagerly loaded into memory as numpy arrays when
          accessed. This can be more efficient for smaller arrays or when large arrays
          are sliced before computation.

        See dask chunking for more details.


    Returns
    -------
    gt : xarray Dataset
        slices geotop netcdf.
    """
    gt = xr.open_dataset(url, chunks=chunks)

    # only download requisite data
    data_vars = ["strat", "lithok"]
    if probabilities:
        data_vars += [
            "kans_1",
            "kans_2",
            "kans_3",
            "kans_4",
            "kans_5",
            "kans_6",
            "kans_7",
            "kans_8",
            "kans_9",
            "onz_lk",
            "onz_ls",
        ]

    # set x and y dimensions to cell center
    for dim in ["x", "y"]:
        old_dim = gt[dim].values
        gt[dim] = old_dim + (old_dim[1] - old_dim[0]) / 2

    # get data vars and slice extent
    gt = gt[data_vars].sel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))

    # change order of dimensions from x, y, z to z, y, x
    gt = gt.transpose("z", "y", "x")

    # flip z, and y coordinates
    gt = gt.isel(z=slice(None, None, -1), y=slice(None, None, -1))

    # add missing value
    # gt.strat.attrs["missing_value"] = -127

    return gt


def add_top_and_botm(ds):
    """Add the top and bottom of the voxels to the GeoTOP Dataset.

    This makes sure the structure of the geotop dataset is more like regis, and we can
    use the cross-section class (DatasetCrossSection from nlmod.

    Parameters
    ----------
    ds : xr.Dataset
        The geotop-dataset.

    Returns
    -------
    ds : xr.Dataset
        The geotop-dataset, with added variables "top" and "botm".
    """
    bottom = np.expand_dims(ds.z.values - 0.25, axis=(1, 2))
    bottom = np.repeat(np.repeat(bottom, len(ds.y), 1), len(ds.x), 2)
    bottom[np.isnan(ds.strat.values)] = np.nan
    ds["botm"] = ("z", "y", "x"), bottom

    top = np.expand_dims(ds.z.values + 0.25, axis=(1, 2))
    top = np.repeat(np.repeat(top, len(ds.y), 1), len(ds.x), 2)
    top[np.isnan(ds.strat.values)] = np.nan
    ds["top"] = ("z", "y", "x"), top
    return ds


def add_kh_and_kv(
    gt,
    df,
    stochastic=None,
    kh_method="arithmetic_mean",
    kv_method="harmonic_mean",
    anisotropy=1.0,
    kh="kh",
    kv="kv",
    kh_df="kh",
    kv_df="kv",
):
    """Add kh and kv variables to the voxels of the GeoTOP Dataset.

    Parameters
    ----------
    gt : xr.Dataset
        The geotop dataset, at least with variable lithok.
    df : pd.DataFrame
        A DataFrame with information about the kh and optionally kv, for different
        lithoclasses or stratigraphic units. The DataFrame must contain the columns
        'lithok' and 'kh', and optionally 'strat' and 'kv'. As an example see
        nlmod.read.geotop.get_kh_kv_table().
    stochastic : bool, str or None, optional
        When stochastic is True or a string, use the stochastic data of GeoTOP. The only
        supported method right now is "linear", which means kh and kv are determined
        from a linear weighted mean of the voxels. For kh the method from kh_method is
        used to calculate the mean. For kv the method from kv_method is used. When
        stochastic is False or None, the stochastic data of GeoTOP is not used. The
        default is None.
    kh_method : str, optional
        The method to calculate the weighted mean of kh values when stochastic is True
        or "linear". Allowed values are "arithmetic_mean" and "harmonic_mean". The
        default is "arithmetic_mean".
    kv_method : str, optional
        The method to calculate the weighted mean of kv values when stochastic is True
        or "linear". Allowed values are "arithmetic_mean" and "harmonic_mean". The
        default is "arithmetic_mean".
    anisotropy : float, optional
        The anisotropy value used when there are no kv values in df. The default is 1.0.
    kh : str, optional
        THe name of the new variable with kh values in gt. The default is "kh".
    kv : str, optional
        The name of the new variable with kv values in gt. The default is "kv".
    kh_df : str, optional
        The name of the column with kh values in df. The default is "kh".
    kv_df : str, optional
        THe name of the column with kv values in df. The default is "kv".

    Raises
    ------
        DESCRIPTION.

    Returns
    -------
    gt : xr.Dataset
        Datset with voxel-data, with the added variables 'kh' and 'kv'.
    """
    if isinstance(stochastic, bool):
        if stochastic:
            stochastic = "linear"
        else:
            stochastic = None
    if kh_method not in ["arithmetic_mean", "harmonic_mean"]:
        raise (ValueError("Unknown kh_method: {kh_method}"))
    if kv_method not in ["arithmetic_mean", "harmonic_mean"]:
        raise (ValueError("Unknown kv_method: {kv_method}"))
    strat = gt["strat"].values
    msg = "Determining kh and kv of geotop-data based on lithoclass"
    if df.index.name in ["lithok", "strat"]:
        df = df.reset_index()
    if "strat" in df:
        msg = f"{msg} and stratigraphy"
    logger.info(msg)
    if kh_df not in df:
        raise (MissingValueError(f"No {kh_df} defined in df"))
    if kv_df not in df:
        logger.info(f"Setting kv equal to kh / {anisotropy}")
    if stochastic is None:
        # calculate kh and kv from most likely lithoclass
        lithok = gt["lithok"].values
        kh_ar = np.full(lithok.shape, np.nan)
        kv_ar = np.full(lithok.shape, np.nan)
        if "strat" in df:
            combs = np.column_stack((strat.ravel(), lithok.ravel()))
            # drop nans
            combs = combs[~np.isnan(combs).any(1)].astype(int)
            # get unique combinations of strat and lithok
            combs_un = np.unique(combs, axis=0)
            for istrat, ilithok in combs_un:
                mask = (strat == istrat) & (lithok == ilithok)
                kh_ar[mask], kv_ar[mask] = _get_kh_kv_from_df(
                    df, ilithok, istrat, anisotropy=anisotropy, mask=mask
                )
        else:
            for ilithok in np.unique(lithok[~np.isnan(lithok)]):
                mask = lithok == ilithok
                kh_ar[mask], kv_ar[mask] = _get_kh_kv_from_df(
                    df,
                    ilithok,
                    anisotropy=anisotropy,
                    mask=mask,
                )
    elif stochastic == "linear":
        strat_un = np.unique(strat[~np.isnan(strat)])
        kh_ar = np.full(strat.shape, 0.0)
        kv_ar = np.full(strat.shape, 0.0)
        probability_total = np.full(strat.shape, 0.0)
        for ilithok in df["lithok"].unique():
            if ilithok == 0:
                # there are no probabilities defined for lithoclass 'antropogeen'
                continue
            probability = gt[f"kans_{ilithok}"].values
            if "strat" in df:
                khi, kvi = _handle_nans_in_stochastic_approach(
                    np.nan, np.nan, kh_method, kv_method
                )
                khi = np.full(strat.shape, khi)
                kvi = np.full(strat.shape, kvi)
                for istrat in strat_un:
                    mask = (strat == istrat) & (probability > 0)
                    if not mask.any():
                        continue
                    kh_sel, kv_sel = _get_kh_kv_from_df(
                        df, ilithok, istrat, anisotropy=anisotropy, mask=mask
                    )
                    if np.isnan(kh_sel):
                        probability[mask] = 0.0
                    kh_sel, kv_sel = _handle_nans_in_stochastic_approach(
                        kh_sel, kv_sel, kh_method, kv_method
                    )
                    khi[mask], kvi[mask] = kh_sel, kv_sel
            else:
                khi, kvi = _get_kh_kv_from_df(df, ilithok, anisotropy=anisotropy)
                if np.isnan(khi):
                    probability[:] = 0.0
                khi, kvi = _handle_nans_in_stochastic_approach(
                    khi, kvi, kh_method, kv_method
                )
            if kh_method == "arithmetic_mean":
                kh_ar = kh_ar + probability * khi
            else:
                kh_ar = kh_ar + (probability / khi)
            if kv_method == "arithmetic_mean":
                kv_ar = kv_ar + probability * kvi
            else:
                kv_ar = kv_ar + (probability / kvi)
            probability_total += probability
        if kh_method == "arithmetic_mean":
            kh_ar = kh_ar / probability_total
        else:
            kh_ar = probability_total / kh_ar
        if kv_method == "arithmetic_mean":
            kv_ar = kv_ar / probability_total
        else:
            kv_ar = probability_total / kv_ar
    else:
        raise (ValueError(f"Unsupported value for stochastic: '{stochastic}'"))

    dims = gt["strat"].dims
    gt[kh] = dims, kh_ar
    gt[kv] = dims, kv_ar
    return gt


def _get_kh_kv_from_df(df, ilithok, istrat=None, anisotropy=1.0, mask=None):
    mask_df = df["lithok"] == ilithok
    if istrat is not None:
        mask_df = mask_df & (df["strat"] == istrat)
    if not np.any(mask_df):
        msg = f"No conductivities found for stratigraphic unit {istrat}"
        if istrat is not None:
            msg = f"{msg} and lithoclass {ilithok}"
        if mask is None:
            msg = f"{msg}. Setting values of voxels to NaN."
        else:
            msg = f"{msg}. Setting values of {mask.sum()} voxels to NaN."
        logger.warning(msg)
        return np.nan, np.nan

    kh = df.loc[mask_df, "kh"].mean()
    if "kv" in df:
        kv = df.loc[mask_df, "kv"].mean()
        if np.isnan(kv):
            kv = kh / anisotropy
        if np.isnan(kh):
            kh = kv * anisotropy
    else:
        kv = kh / anisotropy

    return kh, kv


def _handle_nans_in_stochastic_approach(kh, kv, kh_method, kv_method):
    if np.isnan(kh):
        if kh_method == "arithmetic_mean":
            kh = 0.0
        else:
            kh = np.inf
    if np.isnan(kv):
        if kv_method == "arithmetic_mean":
            kv = 0.0
        else:
            kv = np.inf
    return kh, kv


def aggregate_to_ds(
    gt, ds, kh="kh", kv="kv", kd="kD", c="c", kh_gt="kh", kv_gt="kv", add_kd_and_c=False
):
    """Aggregate voxels from GeoTOP to layers in a model DataSet with top and botm, to
    calculate kh and kv.

    Parameters
    ----------
    gt : xr.Dataset
        A Dataset containing the Geotop voxel data.
    ds : xr.Dataset
        A Dataset containing the top and botm of the layers.
    kh : str, optional
        The name of the new variable for the horizontal conductivity in ds. The default
        is "kh".
    kv : str, optional
        The name of the new variable for the vertical conductivity in ds. The default is
        "kv".
    kd : str, optional
        The name of the variable for the horizontal transmissivity. Only used when
        add_kd_and_c is True. The default is "kD".
    c : str, optional
        The name of the variable for the vertical reistance. Only used when add_kd_and_c
        is True The default is "c".
    kh_gt : str, optional
        The name of the variable for the horizontal conductivity in gt. The default is
        "kh".
    kv_gt : str, optional
        The name of the variable for the vertical conductivity in gt. The default is
        "kv".
    add_kd_and_c : bool, optional
        Add the variables kd and c to ds. The default is False.

    Returns
    -------
    ds : xr.Dataset
        The Dataset ds, with added variables kh and kv (and optionally kd and c).
    """
    assert (ds.x == gt.x).all() and (ds.y == gt.y).all()
    msg = "Please add '{}' to geotop-Dataset first, using add_kh_and_kv()"
    if kh_gt not in gt:
        raise (MissingValueError(msg.format(kh_gt)))
    if kv_gt not in gt:
        raise (MissingValueError(msg.format(kv_gt)))
    kD_ar = []
    c_ar = []
    kh_ar = []
    kv_ar = []
    for ilay in range(len(ds.layer)):
        if ilay == 0:
            top = ds["top"]
            if "layer" in top.dims:
                top = top[0].drop_vars("layer")
        else:
            if "layer" in ds["top"].dims:
                top = ds["top"][ilay].drop_vars("layer")
            else:
                top = ds["botm"][ilay - 1].drop_vars("layer")
        bot = ds["botm"][ilay].drop_vars("layer")

        gt_top = (gt["z"] + 0.25).broadcast_like(gt[kh_gt])
        gt_bot = (gt["z"] - 0.25).broadcast_like(gt[kh_gt])
        gt_top = gt_top.where(gt_top < top, top)
        gt_top = gt_top.where(gt_top > bot, bot)
        gt_bot = gt_bot.where(gt_bot < top, top)
        gt_bot = gt_bot.where(gt_bot > bot, bot)
        gt_thk = gt_top - gt_bot
        # kD is the sum of thickness multiplied by conductivity
        kD_ar.append((gt_thk * gt[kh_gt]).sum("z"))
        # c is the sum of thickness devided by conductivity
        c_ar.append((gt_thk / gt[kv_gt]).sum("z"))
        # caluclate kh and hv
        d_gt = gt_top - gt_bot
        # use only the thickness with valid kh-values
        D = d_gt.where(~np.isnan(gt[kh_gt])).sum("z")
        kh_ar.append(kD_ar[-1] / D)
        # use only the thickness with valid kv-values
        D = d_gt.where(~np.isnan(gt[kv_gt])).sum("z")
        kv_ar.append(D / c_ar[-1])
    if add_kd_and_c:
        ds[kd] = xr.concat(kD_ar, ds.layer)
        ds[c] = xr.concat(c_ar, ds.layer)
    ds[kh] = xr.concat(kh_ar, ds.layer)
    ds[kv] = xr.concat(kv_ar, ds.layer)
    return ds


def _save_excel_files_as_csv():
    """
    This method takes the files REF_GTP_STR_UNIT.xlsx and REF_GTP_LITHO_CLASS.xlsx that
    are taken from the GeoTOP 1.6 zipfile downloaded from DINOloket, and saves them as
    csv-files. In this way version-control can better process the changes in future
    versions of GeoTOP.

    Returns
    -------
    None.

    """
    for name in ["REF_GTP_STR_UNIT.xlsx", "REF_GTP_LITHO_CLASS.xlsx"]:
        fname = os.path.join(NLMOD_DATADIR, "geotop", name)
        df = pd.read_excel(fname, keep_default_na=False)
        df.to_csv(fname.replace(".xlsx", ".csv"), index=False)
