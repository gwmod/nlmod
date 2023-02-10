import datetime as dt
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

from .. import NLMOD_DATADIR, cache
from ..dcs import DatasetCrossSection

logger = logging.getLogger(__name__)

GEOTOP_URL = r"http://www.dinodata.nl/opendap/GeoTOP/geotop.nc"


def get_lithok_props(rgb_colors=True):
    df = pd.read_csv(
        os.path.join(NLMOD_DATADIR, "geotop", "litho_eenheden.csv"),
        index_col=0,
    )
    if rgb_colors:
        df["color"] = get_lithok_colors()
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
    for key in colors:
        colors[key] = tuple([x / 255 for x in colors[key]])
    return colors


def get_lithok_translateion():
    lithok_translation = {
        0: "antropogeen",
        1: "organisch materiaal (veen)",
        2: "klei",
        3: "klei zandig, zandige klei en leem",
        5: "zand fijn",
        6: "zand midden",
        7: "zand grof",
        8: "grind",
        9: "schelpen",
    }
    return lithok_translation


def get_strat_props():
    geo_eenheid_translate_df = pd.read_csv(
        os.path.join(NLMOD_DATADIR, "geotop", "geo_eenheden.csv"),
        index_col=0,
        keep_default_na=False,
    )
    return geo_eenheid_translate_df


@cache.cache_netcdf
def get_geotop(extent, regis_ds, regis_layer="HLc"):
    """get a model layer dataset for modflow from geotop within a certain
    extent and grid.

    if regis_ds and regis_layer are defined the geotop model is only created
    to replace this regis_layer in a regis layer model.


    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    regis_ds: xarray.DataSet
        regis dataset used to cut geotop to the same x and y coordinates
    regis_layer: str, optional
        layer of regis dataset that will be filled with geotop. The default is
        'HLc'.

    Returns
    -------
    geotop_ds: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid
    """
    geotop_ds_raw1 = get_geotop_raw_within_extent(extent, GEOTOP_URL)

    litho_translate_df = get_lithok_props()
    geo_eenheid_translate_df = get_strat_props()

    ds = convert_geotop_to_ml_layers(
        geotop_ds_raw1,
        regis_ds=regis_ds,
        regis_layer=regis_layer,
        litho_translate_df=litho_translate_df,
        geo_eenheid_translate_df=geo_eenheid_translate_df,
    )

    ds.attrs["extent"] = extent

    for datavar in ds:
        ds[datavar].attrs["source"] = "Geotop"
        ds[datavar].attrs["url"] = GEOTOP_URL
        ds[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")
        if datavar in ["top", "bot"]:
            ds[datavar].attrs["units"] = "mNAP"
        elif datavar in ["kh", "kv"]:
            ds[datavar].attrs["units"] = "m/day"

    return ds


def get_geotop_raw_within_extent(extent, url=GEOTOP_URL, drop_probabilities=True):
    """Get a slice of the geotop netcdf url within the extent, set the x and y
    coordinates to match the cell centers and keep only the strat and lithok
    data variables.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    url : str, optional
        url of geotop netcdf file. The default is
        http://www.dinodata.nl/opendap/GeoTOP/geotop.nc

    Returns
    -------
    gt : xarray Dataset
        slices geotop netcdf.
    """
    gt = xr.open_dataset(url)

    # set x and y dimensions to cell center
    for dim in ["x", "y"]:
        old_dim = gt[dim].values
        gt[dim] = old_dim + (old_dim[1] - old_dim[0]) / 2

    # slice extent
    gt = gt.sel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))

    if drop_probabilities:
        gt = gt[["strat", "lithok"]]

    return gt


def convert_geotop_to_ml_layers(
    geotop_ds_raw1,
    regis_ds=None,
    regis_layer=None,
    litho_translate_df=None,
    geo_eenheid_translate_df=None,
):
    """does the following steps to obtain model layers based on geotop:

        1. slice by regis layer (if not None)
        2. compute kh from lithoklasse
        3. create a layer model based on geo-eenheden

    Parameters
    ----------
    geotop_ds_raw1: xr.Dataset
        dataset with geotop netcdf
    regis_ds: xarray.DataSet
        regis dataset used to cut geotop to the same x and y coördinates
    regis_layer: str, optional
        layer of regis dataset that will be filled with geotop
    litho_translate_df: pandas.DataFrame
        horizontal conductance (kh)
    geo_eenheid_translate_df: pandas.DataFrame
        dictionary to translate geo_eenheid to a geo name

    Returns
    -------
    geotop_ds_raw: xarray.DataSet
        geotop dataset with added horizontal conductance
    """

    # stap 1
    if (regis_ds is not None) and (regis_layer is not None):
        logger.info(f"slice geotop with regis layer {regis_layer}")
        top_rl = regis_ds["top"].sel(layer=regis_layer)
        bot_rl = regis_ds["botm"].sel(layer=regis_layer)

        geotop_ds_raw = geotop_ds_raw1.sel(
            z=slice(np.floor(bot_rl.min().data), np.ceil(top_rl.max().data))
        )

    # stap 2 maak kh matrix a.d.v. lithoklasse
    logger.info("create kh matrix from lithoklasse and csv file")
    kh_from_litho = xr.zeros_like(geotop_ds_raw.lithok)
    for i, row in litho_translate_df.iterrows():
        kh_from_litho = xr.where(
            geotop_ds_raw.lithok == i,
            row["hor_conductance_default"],
            kh_from_litho,
        )
    geotop_ds_raw["kh_from_litho"] = kh_from_litho

    # stap 3 maak een laag per geo-eenheid
    geotop_ds_mod = get_top_bot_from_geo_eenheid(
        geotop_ds_raw, geo_eenheid_translate_df
    )

    return geotop_ds_mod


def get_top_bot_from_geo_eenheid(geotop_ds_raw, geo_eenheid_translate_df):
    """get top, botm and kh of each geo-eenheid in geotop dataset.

    Parameters
    ----------
    geotop_ds_raw: xr.DataSet
        geotop dataset with added horizontal conductance
    geo_eenheid_translate_df: pandas.DataFrame
        dictionary to translate geo_eenheid to a geo name

    Returns
    -------
    geotop_ds_mod: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid

    Note
    ----
    the 'geo_eenheid' >6000 are 'stroombanen' these are difficult to add because
    they can occur above and below any other 'geo_eenheid' therefore they are
    added to the geo_eenheid below the stroombaan.
    """

    # vindt alle geo-eenheden in model_extent
    geo_eenheden = np.unique(geotop_ds_raw.strat.data)
    geo_eenheden = geo_eenheden[np.isfinite(geo_eenheden)]
    stroombaan_eenheden = geo_eenheden[geo_eenheden < 5999]
    geo_eenheden = geo_eenheden[geo_eenheden < 5999]

    # geo eenheid 2000 zit boven 1130
    if (2000.0 in geo_eenheden) and (1130.0 in geo_eenheden):
        geo_eenheden[(geo_eenheden == 2000.0) + (geo_eenheden == 1130.0)] = [
            2000.0,
            1130.0,
        ]

    geo_names = [
        geo_eenheid_translate_df.loc[float(geo_eenh), "Code (lagenmodel en boringen)"]
        for geo_eenh in geo_eenheden
    ]

    # fill top and bot
    shape = (len(geo_names), len(geotop_ds_raw.y), len(geotop_ds_raw.x))
    top = np.full(shape, np.nan)
    bot = np.full(shape, np.nan)
    lay = 0
    logger.info("creating top and bot per geo eenheid")
    for geo_eenheid in geo_eenheden:
        logger.debug(geo_eenheid)

        mask = geotop_ds_raw.strat == geo_eenheid
        geo_z = xr.where(mask, geotop_ds_raw.z, np.nan)

        top[lay] = geo_z.max(dim="z").T + 0.5
        bot[lay] = geo_z.min(dim="z").T

        lay += 1

    geotop_ds_mod = add_stroombanen_and_get_kh(geotop_ds_raw, top, bot, geo_names)

    geotop_ds_mod.attrs["stroombanen"] = stroombaan_eenheden

    return geotop_ds_mod


def add_stroombanen_and_get_kh(geotop_ds_raw, top, bot, geo_names, f_anisotropy=0.25):
    """add stroombanen to tops and bots of geo_eenheden, also computes kh per
    geo_eenheid. Kh is computed by taking the average of all kh's of a
    geo_eenheid within a cell (e.g. if one geo_eenheid has a thickness of 1,5m
    in a certain cell the kh of the cell is calculated as the mean of the 3
    cells in geotop)

    Parameters
    ----------
    geotop_ds_raw: xr.DataSet
        geotop dataset with added horizontal conductance
    top: np.array
        raster with top of each geo_eenheid, shape(nlay,nrow,ncol)
    bot: np.array
        raster with bottom of each geo_eenheid, shape(nlay,nrow,ncol)
    geo_names: list of str
        names of each geo_eenheid
    f_anisotropy: float, optional
        anisotropy factor kv/kh, ratio between vertical and horizontal
        hydraulic conductivities, by default 0.25.


    Returns
    -------
    geotop_ds_mod: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid
    """
    shape = (len(geo_names), len(geotop_ds_raw.y), len(geotop_ds_raw.x))
    kh = np.full(shape, np.nan)
    thickness = np.full(shape, np.nan)
    z = xr.ones_like(geotop_ds_raw.lithok) * geotop_ds_raw.z
    logger.info("adding stroombanen to top and bot of each layer")
    logger.info("get kh for each layer")

    for lay in range(top.shape[0]):
        logger.info(geo_names[lay])
        if lay == 0:
            top[0] = np.nanmax(top, axis=0)
        else:
            top[lay] = bot[lay - 1]
        bot[lay] = np.where(np.isnan(bot[lay]), top[lay], bot[lay])
        thickness[lay] = top[lay] - bot[lay]

        # check which geotop voxels are within the range of the layer
        bool_z = xr.zeros_like(z)
        for i in range(z.z.shape[0]):
            mask = (z[:, :, i] >= bot[lay].T) * (z[:, :, i] < top[lay].T)
            bool_z[:, :, i] = np.where(mask, True, False)

        kh_geo = xr.where(bool_z, geotop_ds_raw["kh_from_litho"], np.nan)
        kh[lay] = kh_geo.mean(dim="z").T

    dims = ("layer", "y", "x")
    coords = {"layer": geo_names, "y": geotop_ds_raw.y, "x": geotop_ds_raw.x}
    da_top = xr.DataArray(data=top, dims=dims, coords=coords)
    da_bot = xr.DataArray(data=bot, dims=dims, coords=coords)
    da_kh = xr.DataArray(data=kh, dims=dims, coords=coords)
    da_thick = xr.DataArray(data=thickness, dims=dims, coords=coords)

    geotop_ds_mod = xr.Dataset()

    geotop_ds_mod["top"] = da_top
    geotop_ds_mod["botm"] = da_bot
    geotop_ds_mod["kh"] = da_kh
    geotop_ds_mod["kv"] = geotop_ds_mod["kh"] * f_anisotropy
    geotop_ds_mod["thickness"] = da_thick

    return geotop_ds_mod


def add_top_and_botm(ds):
    """
    Adds the top and bottom of the voxels to the geotop Dataset

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
    # make ready for DataSetCrossSection
    ds = ds.transpose("z", "y", "x")
    ds = ds.sortby("z", ascending=False)

    bottom = np.expand_dims(ds.z.data - 0.25, axis=(1, 2))
    bottom = np.repeat(np.repeat(bottom, len(ds.y), 1), len(ds.x), 2)
    bottom[np.isnan(ds.strat.data)] = np.NaN
    ds["botm"] = ("z", "y", "x"), bottom

    top = np.expand_dims(ds.z.data + 0.25, axis=(1, 2))
    top = np.repeat(np.repeat(top, len(ds.y), 1), len(ds.x), 2)
    top[np.isnan(ds.strat.data)] = np.NaN
    ds["top"] = ("z", "y", "x"), top
    return ds


def add_kh_and_kv(
    gt, df, stochastic=None, kh_method="arithmetic_mean", kv_method="harmonic_mean"
):
    if isinstance(stochastic, bool):
        if stochastic:
            stochastic = "linear"
        else:
            stochastic = None
    if kh_method not in ["arithmetic_mean", "harmonic_mean"]:
        raise (Exception("Unknown kh_method: {kh_method}"))
    if kv_method not in ["arithmetic_mean", "harmonic_mean"]:
        raise (Exception("Unknown kv_method: {kv_method}"))
    strat = gt["strat"].data
    msg = "Determining kh and kv of geotop-data based on lithoclass"
    if "strat" in df:
        msg = f"{msg} and stratigraphy"
    logging.info(msg)
    if "kh" not in df:
        raise (Exception("No kh defined in df"))
    if "kv" not in df:
        logging.info("Setting kv equal to kh")
    if stochastic is None:
        # calculate kh and kv from most likely lithoclass
        lithok = gt["lithok"].data
        kh = np.full(lithok.shape, np.NaN)
        kv = np.full(lithok.shape, np.NaN)
        if "strat" in df:
            combs = np.column_stack((strat.ravel(), lithok.ravel()))
            # drop nans
            combs = combs[~np.isnan(combs).any(1)].astype(int)
            # get unique combinations of strat and lithok
            combs_un = np.unique(combs, axis=0)
            for istrat, ilithok in combs_un:
                mask = (strat == istrat) & (lithok == ilithok)
                kh[mask], kv[mask] = _get_kh_kv_from_df(df, ilithok, istrat)
        else:
            for ilithok in np.unique(lithok[~np.isnan(lithok)]):
                mask = lithok == ilithok
                kh[mask], kv[mask] = _get_kh_kv_from_df(df, ilithok)
    elif stochastic == "linear":
        strat_un = np.unique(strat[~np.isnan(strat)])
        kh = np.full(strat.shape, 0.0)
        kv = np.full(strat.shape, 0.0)
        probality_total = np.full(strat.shape, 0.0)
        for ilithok in df["lithok"].unique():
            if ilithok == 0:
                # there are no probabilities defined for lithoclass 'antropogeen'
                continue
            probality = gt[f"kans_{ilithok}"].data
            if "strat" in df:
                khi = np.full(strat.shape, np.NaN)
                kvi = np.full(strat.shape, np.NaN)
                for istrat in strat_un:
                    mask = strat == istrat
                    kh_sel, kv_sel = _get_kh_kv_from_df(df, ilithok, istrat)
                    if np.isnan(kh_sel):
                        probality[mask] = 0.0
                    kh_sel, kv_sel = _handle_nans_in_stochastic_approach(
                        kh_sel, kv_sel, kh_method, kv_method
                    )
                    khi[mask], kvi[mask] = kh_sel, kv_sel
            else:
                khi, kvi = _get_kh_kv_from_df(df, ilithok)
                if np.isnan(khi):
                    probality[:] = 0.0
                khi, kvi = _handle_nans_in_stochastic_approach(
                    khi, kvi, kh_method, kv_method
                )
            if kh_method == "arithmetic_mean":
                kh = kh + probality * khi
            else:
                kh = kh + (probality / khi)
            if kv_method == "arithmetic_mean":
                kv = kv + probality * kvi
            else:
                kv = kv + (probality / kvi)
            probality_total += probality
        if kh_method == "arithmetic_mean":
            kh = kh / probality_total
        else:
            kh = probality_total / kh
        if kv_method == "arithmetic_mean":
            kv = kv / probality_total
        else:
            kv = probality_total / kv
    else:
        raise (Exception(f"Unsupported value for stochastic: {stochastic}"))

    dims = gt["strat"].dims
    gt["kh"] = dims, kh
    gt["kv"] = dims, kv
    return gt


def _get_kh_kv_from_df(df, ilithok, istrat=None):
    mask = df["lithok"] == ilithok
    if istrat is not None:
        mask = mask & (df["strat"] == istrat)
    if not np.any(mask):
        logging.warning(
            f"No conductivities found for stratigraphy-unit {istrat} and lithoclass "
            f"{ilithok}"
        )
        return np.NaN, np.NaN

    kh = df.loc[mask, "kh"].mean()
    if "kv" in df:
        kv = df.loc[mask, "kv"].mean()
        if np.isnan(kv):
            kv = kh
        if np.isnan(kh):
            kh = kv
    else:
        kv = kh

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


def _add_flow_properties(
    gt,
    k_dict,
    kv_dict=None,
    stochastic=None,
    kh="kh",
    kv="kv",
):
    lithok_translation = get_lithok_props()["lithologie"]
    if isinstance(list(k_dict)[0], str):
        lith2float = {v: k for k, v in lithok_translation.items()}
        k_dict = {lith2float[key]: float(k_dict[key]) for key in k_dict}
    if kv_dict is None:
        kv_dict = k_dict
    if isinstance(list(kv_dict)[0], str):
        lith2float = {v: k for k, v in lithok_translation.items()}
        kv_dict = {lith2float[key]: float(kv_dict[key]) for key in kv_dict}
    if stochastic is None:
        gt_k = xr.full_like(gt["lithok"], np.NaN)
        k_data = gt_k.data
        lithok_data = gt["lithok"].data
        for key in np.unique(lithok_data[~np.isnan(lithok_data)]):
            mask = lithok_data == key
            k_data[mask] = k_dict[key]
        gt_kv = gt_k
    elif stochastic == "linear":
        gt_k = xr.full_like(gt["lithok"], 0.0)
        gt_c = xr.full_like(gt["lithok"], 0.0)
        k_data = gt_k.data
        c_data = gt_c.data
        kans_totaal = xr.full_like(gt["lithok"], 0.0).data
        for key in k_dict.keys():
            var = f"kans_{key}"
            if var not in gt:
                logging.debug(f"GeoTOP does not contain {var}")
                continue
            kans = gt[var].data
            k_data += kans * k_dict[key]
            c_data += kans * 1 / kv_dict[key]
            kans_totaal += kans

        mask = kans_totaal == 0.0
        if mask.any():
            # antropogeen does not contain any stochastic information
            assert np.all(gt["lithok"].data[mask] == 0)
            k_data[mask] = k_dict[0]
            c_data[mask] = 1 / kv_dict[0]
            kans_totaal[mask] = 1.0
        k_data /= kans_totaal
        c_data /= kans_totaal
        gt_kv = 1 / gt_c
    else:
        raise (Exception(f"Unsupported value for stochastic: {stochastic}"))


def aggregate_to_ds(gt, ds, kh="kh", kv="kv", kd="kD", c="c", add_kD_and_c=False):
    assert (ds.x == gt.x).all() and (ds.y == gt.y).all()
    msg = "Please add {} to gt-Dataset first, using add_kh_and_kv()"
    if kh not in gt:
        raise (Exception(msg.format(kh)))
    if kv not in gt:
        raise (Exception(msg.format(kv)))
    kD_ar = []
    c_ar = []
    kh_ar = []
    kv_ar = []
    for ilay in range(len(ds.layer)):
        if ilay == 0:
            top = ds["top"]
        else:
            top = ds["botm"][ilay - 1].drop_vars("layer")
        bot = ds["botm"][ilay].drop_vars("layer")

        gt_top = (gt["z"] + 0.25).broadcast_like(gt[kh])
        gt_bot = (gt["z"] - 0.25).broadcast_like(gt[kh])
        gt_top = gt_top.where(gt_top < top, top)
        gt_top = gt_top.where(gt_top > bot, bot)
        gt_bot = gt_bot.where(gt_bot < top, top)
        gt_bot = gt_bot.where(gt_bot > bot, bot)
        gt_thk = gt_top - gt_bot
        # kD is the sum of thickness multiplied by conductivity
        kD_ar.append((gt_thk * gt[kh]).sum("z"))
        # c is the sum of thickness devided by conductivity
        c_ar.append((gt_thk / gt[kv]).sum("z"))
        # caluclate kh and hv
        d_gt = gt_top - gt_bot
        # use only the thickness with valid kh-values
        D = d_gt.where(~np.isnan(gt["kh"])).sum("z")
        kh_ar.append(kD_ar[-1] / D)
        # use only the thickness with valid kv-values
        D = d_gt.where(~np.isnan(gt["kv"])).sum("z")
        kv_ar.append(D / c_ar[-1])
    if add_kD_and_c:
        ds[kd] = xr.concat(kD_ar, ds.layer)
        ds[c] = xr.concat(c_ar, ds.layer)
    ds[kh] = xr.concat(kh_ar, ds.layer)
    ds[kv] = xr.concat(kv_ar, ds.layer)
    return ds


def plot_cross_section(line, gt=None, ax=None, legend=True, legend_loc=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if gt is None:
        # download geotop
        x = [coord[0] for coord in line.coords]
        y = [coord[1] for coord in line.coords]
        extent = [min(x), max(x), min(y), max(y)]
        gt = get_geotop_raw_within_extent(extent)

    if "top" not in gt or "botm" not in gt:
        gt = add_top_and_botm(gt)

    cs = DatasetCrossSection(gt, line, layer="z", ax=ax, **kwargs)
    lithoks = gt["lithok"].data
    lithok_un = np.unique(lithoks[~np.isnan(lithoks)])
    array = np.full(lithoks.shape, np.NaN)
    lithok_colors = get_lithok_props()["color"]
    colors = []
    for i, lithok in enumerate(lithok_un):
        array[lithoks == lithok] = i
        colors.append(lithok_colors[lithok])
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.Normalize(-0.5, np.nanmax(array) + 0.5)
    cs.plot_array(array, norm=norm, cmap=cmap)
    if legend:
        # make a legend with dummy handles
        handles = []
        lithok_translation = get_lithok_props()["lithologie"]
        for i, lithok in enumerate(lithok_un):
            label = lithok_translation[lithok]
            handles.append(matplotlib.patches.Patch(facecolor=colors[i], label=label))
        ax.legend(handles=handles, loc=legend_loc)

    return ax
