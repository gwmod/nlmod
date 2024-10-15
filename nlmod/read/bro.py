import logging

import flopy
import hydropandas as hpd
import numpy as np
from shapely.geometry import Point

from .. import cache, dims, gwf

logger = logging.getLogger(__name__)


def add_modelled_head(oc, ml=None, ds=None, method="linear"):
    """Add modelled heads as seperate observations to the ObsCollection.

    Parameters
    ----------
    oc : ObsCollection
        Set of observed groundwater heads
    ml : flopy.modflow.mf.model, optional
        modflow model, by default None
    ds : xr.DataSet, optional
        dataset with relevant model information, by default None
    method : str, optional
        type of interpolation used to get heads. For now only 'linear' and
        'nearest' are supported. The default is 'linear'.

    Returns
    -------
    ObsCollection
        combination of observed and modelled groundwater heads.
    """
    oc["modellayer"] = oc.gwobs.get_modellayers(gwf=ml)
    if ds is not None and "heads" in ds:
        heads = ds["heads"]
    else:
        heads = gwf.get_heads_da(ds=ds, gwf=ml)

    # this function requires a flopy model object, see
    # https://github.com/ArtesiaWater/hydropandas/issues/146
    if ds.time.dtype.kind != "M":
        raise TypeError("add modelled head requires a datetime64[ns] time index")

    oc_modflow = hpd.read_modflow(oc, ml, heads.values, ds.time.values, method=method)

    if ds.gridtype == "vertex":
        gi = flopy.utils.GridIntersect(dims.modelgrid_from_ds(ds), method="vertex")

    obs_list = []
    for name in oc.index:
        o = oc.loc[name, "obs"]
        modellayer = oc.loc[name, "modellayer"]
        if "qualifier" in o.columns:
            o = o[o["qualifier"] == "goedgekeurd"]
        o_resampled = o.resample("D").last().sort_index()
        modelled = oc_modflow.loc[name, "obs"]

        if ds.gridtype == "structured":
            bot = ds["botm"].interp(x=o.x, y=o.y, method="nearest").values[modellayer]
            if modellayer == 0:
                top = ds["top"].interp(x=o.x, y=o.y, method="nearest")
            else:
                top = (
                    ds["botm"]
                    .interp(x=o.x, y=o.y, method="nearest")
                    .values[modellayer - 1]
                )
        elif ds.gridtype == "vertex":
            icelld2 = gi.intersect(Point(o.x, o.y))["cellids"][0]
            bot = ds["botm"].values[modellayer, icelld2]
            if modellayer == 0:
                top = ds["top"].values[icelld2]
            else:
                top = ds["botm"].values[modellayer - 1, icelld2]
        else:
            raise ValueError("unexpected gridtype")

        modelled = hpd.GroundwaterObs(
            modelled.rename(columns={0: "values"}),
            name=f"{o.name}_model",
            x=o.x,
            y=o.y,
            tube_nr=o.tube_nr,
            screen_top=top,
            screen_bottom=bot,
            monitoring_well=o.monitoring_well,
            source="MODFLOW",
            unit="m NAP",
            metadata_available=o.metadata_available,
        )
        obs_list.append(o_resampled)
        obs_list.append(modelled)
    oc_compare = hpd.ObsCollection(obs_list, name="meting+model")

    return oc_compare


def get_bro(
    extent,
    regis_layers=None,
    max_screen_top=None,
    min_screen_bot=None,
    max_dx=10000,
    max_dy=10000,
    only_metadata=False,
    cachedir=None,
):
    """Get bro groundwater measurements within an extent.

    Parameters
    ----------
    extent : list or tuple
        get bro groundwater measurements within this extent,
        (xmin, xmax, ymin, ymax).
    regis_layers : str, list or tuple, optional
        get only measurements within these regis layers, by default None
    max_screen_top : int or float, optional
        get only measurements with a screen top lower than this, by default None
    min_screen_bot : int or float, optional
        get only measurements with a screen bottom higher than this, by default
        None.
    max_dx : int, optional
        maximum distance in y direction that can be downloaded at once, by
        default 20000 meters (20 km)
    max_dy : int, optional
        maximum distance in x direction that can be downloaded at once, by
        default 20000 meters (20 km)
    only_metadata : bool, optional
        if True an ObsCollection with only metadata is returned. The default is False.
    cachedir : str, optional
        path to cache the bro observation collections

    Returns
    -------
    ObsCollection
        obsevations
    """
    oc_meta = _get_bro_metadata(extent, max_dx=max_dx, max_dy=max_dy, cachedir=cachedir)
    oc_meta = oc_meta.loc[~oc_meta["gld_ids"].isna()]
    if oc_meta.empty:
        logger.warning("none of the observation wells have measurements")
        return oc_meta

    if regis_layers is not None:
        if isinstance(regis_layers, str):
            regis_layers = [regis_layers]

        oc_meta["regis_layer"] = oc_meta.gwobs.get_regis_layers()
        oc_meta = oc_meta[oc_meta["regis_layer"].isin(regis_layers)]
        if oc_meta.empty:
            logger.warning(
                f"none of the regis layers {regis_layers} found in the observation wells"
            )
            return oc_meta

    if max_screen_top is not None:
        oc_meta = oc_meta[oc_meta["screen_top"] < max_screen_top]
        if oc_meta.empty:
            logger.warning(
                f"none of the observation wells have a screen lower than {max_screen_top}"
            )
            return oc_meta

    if min_screen_bot is not None:
        oc_meta = oc_meta[oc_meta["screen_bottom"] > min_screen_bot]
        if oc_meta.empty:
            logger.warning(
                f"none of the observation wells have a screen higher than {min_screen_bot}"
            )
            return oc_meta

    if only_metadata:
        return oc_meta

    # download measurements
    new_obs_list = []
    for gld_ids in oc_meta["gld_ids"]:
        for gld_id in gld_ids:
            new_o = hpd.GroundwaterObs.from_bro(gld_id)
            if not new_o.empty:
                new_obs_list.append(new_o)
    oc = hpd.ObsCollection.from_list(new_obs_list)

    oc = _get_bro_measurement(oc_meta, cachedir=cachedir, cachename="bro.pklz")

    return oc


def _get_bro_metadata(extent, max_dx=10000, max_dy=10000, cachedir=None):
    """Wrapper around hpd.read_bro that deals with large extents and only returns
    metadata (location, tube top/bot, ground level, ..) of the wells and no actual
    measurements. This is useful when the extent is too big to obtain all measurements
    at once.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    max_dx : int, optional
        maximum distance in y direction that can be downloaded at once, by
        default 20000 meters (20 km)
    max_dy : int, optional
        maximum distance in x direction that can be downloaded at once, by
        default 20000 meters (20 km)

    Returns
    -------
    ObsCollection
    """
    # check if extent is within limits
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]

    # check if size exceeds maxsize
    if dx > max_dx:
        x_segments = int(np.ceil(dx / max_dx))
    else:
        x_segments = 1

    if dy > max_dy:
        y_segments = int(np.ceil(dy / max_dy))
    else:
        y_segments = 1

    if (x_segments * y_segments) > 1:
        st = f"""requested bro dataset width or height bigger than {max_dx} or {max_dy}
            -> splitting extent into {x_segments} * {y_segments} tiles"""
        logger.info(st)
        d = {}

        for tx in range(x_segments):
            for ty in range(y_segments):
                xmin = extent[0] + tx * max_dx
                xmax = min(extent[1], extent[0] + (tx + 1) * max_dx)
                ymin = extent[2] + ty * max_dy
                ymax = min(extent[3], extent[2] + (ty + 1) * max_dx)

                logger.debug(
                    f"reading bro within extent {xmin}, {xmax}, {ymin}, {ymax}"
                )
                oc = _get_bro_metadata_oc(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    cachedir=cachedir,
                    cachename=f"bro_{xmin}_{xmax}_{ymin}_{ymax}.pklz",
                )
                d[f"{tx}_{ty}"] = oc

        # merge datasets
        i = 0
        for item in d.values():
            if i == 0:
                oc = item
            else:
                if not item.empty:
                    oc = oc.add_obs_collection(item)
            i += 1
    else:
        oc = hpd.read_bro(extent, only_metadata=True, name="BRO", ignore_max_obs=True)

    oc = oc.add_meta_to_df("gld_ids")

    if oc.empty:
        logger.warning("no observation wells within extent")

    return oc


@cache.cache_pickle
def _get_bro_metadata_oc(xmin, xmax, ymin, ymax):
    """get observation collection with only metadata in extent.

    Parameters
    ----------
    xmin : int, float
        lower x  bound, RD coordinate
    xmax : int, float
        upper x  bound, RD coordinate
    ymin : int, float
        lower y  bound, RD coordinate
    ymax : int, float
        upper y  bound, RD coordinate

    Returns
    -------
    ObsCollection
        BRO metadata within extent
    """
    oc = hpd.read_bro(
        (xmin, xmax, ymin, ymax), only_metadata=True, name="BRO", ignore_max_obs=True
    )

    return oc


@cache.cache_pickle
def _get_bro_measurement(oc_meta):
    """add measurements to BRO ObsCollection with only metadata.

    Parameters
    ----------
    oc_meta : ObsCollection
        collection with only metadata and a column 'gld_ids'

    Returns
    -------
    ObsCollection
        collection with measurements
    """

    obs_list = []
    for _, row in oc_meta.iterrows():
        o = hpd.GroundwaterObs.from_bro(row["monitoring_well"], tube_nr=row["tube_nr"])
        if not o.empty:
            obs_list.append(o)
    oc = hpd.ObsCollection.from_list(obs_list)

    return oc
