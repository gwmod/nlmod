import logging
import warnings

import numpy as np
import pandas as pd

from pyproj import Transformer

from .. import cache

logger = logging.getLogger(__name__)


def get_bro(*args, **kwargs):
    """Wrapper around hpd.read_bro that deals with large extents.

    .. deprecated:: 0.10.0
        `get_bro` will be removed in nlmod 1.0.0, it is replaced by
        `download_bro_groundwater` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    max_dx : int, optional
        If the extent is bigger in x (longitude) direction than this the extent is split
        in multiple tiles. By default 0.1 degrees (~10 km)
    max_dy : int, optional
        If the extent is bigger in y (latitude) direction than this the extent is split
        in multiple tiles. By default 0.1 degrees (~10 km)
    epsg : int, optional
        crs
    cachedir : str or None, optional
        If not None every (sub)extent is cached.

    Returns
    -------
    ObsCollection
    """

    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.bro.download_bro_groundwater() in the future.",
        DeprecationWarning,
    )

    return download_bro_groundwater(*args, **kwargs)


def download_bro_groundwater(
    extent,
    max_dx=0.1,
    max_dy=0.1,
    epsg=28992,
    cachedir=None,
    ignore_errors=True,
    **kwargs,
):
    """Wrapper around hpd.read_bro that deals with large extents.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    max_dx : int, optional
        If the extent is bigger in x (longitude) direction than this the extent is split
        in multiple tiles. By default 0.1 degrees (~10 km)
    max_dy : int, optional
        If the extent is bigger in y (latitude) direction than this the extent is split
        in multiple tiles. By default 0.1 degrees (~10 km)
    epsg : int, optional
        crs
    cachedir : str or None, optional
        If not None every (sub)extent is cached.

    Returns
    -------
    ObsCollection
    """
    # convert extent to epsg 4326
    if epsg != 4326:
        transformer = Transformer.from_crs(epsg, 4326)
        lat1, lon1 = transformer.transform(extent[0], extent[2])
        lat2, lon2 = transformer.transform(extent[1], extent[3])
        extent = (lon1, lon2, lat1, lat2)

    # check if extent exceeds maxsize
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]
    if dx > max_dx:
        x_segments = int(np.ceil(dx / max_dx))
    else:
        x_segments = 1

    if dy > max_dy:
        y_segments = int(np.ceil(dy / max_dy))
    else:
        y_segments = 1

    # split in tiles and download per tile
    if (x_segments * y_segments) > 1:
        st = (
            "requested bro dataset width or height bigger than maxsize "
            f"splitting extent into {x_segments} * {y_segments} tiles"
        )
        logger.info(st)
        oc_list = []
        for tx in range(x_segments):
            for ty in range(y_segments):
                xmin = extent[0] + tx * max_dx
                xmax = min(extent[1], extent[0] + (tx + 1) * max_dx)
                ymin = extent[2] + ty * max_dy
                ymax = min(extent[3], extent[2] + (ty + 1) * max_dx)

                logger.debug(
                    f"reading bro within extent {xmin}, {xmax}, {ymin}, {ymax}"
                )
                name = f"BRO_{xmin}_{xmax}_{ymin}_{ymax}"
                if ignore_errors:
                    try:
                        oc = _get_bro_within_extent(
                            (xmin, xmax, ymin, ymax),
                            name=name,
                            ignore_max_obs=True,
                            epsg=4326,
                            cachedir=cachedir,
                            cachename=name,
                            **kwargs,
                        )
                        oc_list.append(oc)
                    except Exception as e:
                        logger.error(
                            f"could not download BRO data in extent {xmin}, {xmax}, {ymin}, {ymax}"
                        )
                        logger.error(e)
                else:
                    oc = _get_bro_within_extent(
                        (xmin, xmax, ymin, ymax),
                        name=name,
                        ignore_max_obs=True,
                        epsg=4326,
                        cachedir=cachedir,
                        cachename=name,
                        **kwargs,
                    )
                    oc_list.append(oc)
        oc = pd.concat(oc_list)
    else:
        name = "BRO_" + "_".join(map(str, extent))
        oc = _get_bro_within_extent(
            extent,
            name=name,
            ignore_max_obs=True,
            epsg=4326,
            cachedir=cachedir,
            cachename=name,
            **kwargs,
        )
    if oc.empty:
        logger.warning("no observation wells within extent")

    return oc


@cache.cache_pickle
def _get_bro_within_extent(extent, name, ignore_max_obs, epsg, **kwargs):
    """Get bro groundwater measurements within extent

    Parameters
    ----------
    extent : tuple
        extent
    name : str
        name of the ObsCollection
    ignore_max_obs : bool, optional
        by default you get a prompt if you want to download over a 1000
        observations at once. if ignore_max_obs is True you won't get the
        prompt. The default is False
    epsg : int
        crs

    Returns
    -------
    hpd.ObsCollection
        _description_
    """
    try:
        import hydropandas as hpd
    except ImportError:
        raise ImportError(
            "hydropandas is required for nlmod.read.bro.download_bro_groundwater(), "
            "please install it using 'pip install hydropandas'"
        )
    return hpd.read_bro(
        extent, name=name, ignore_max_obs=ignore_max_obs, epsg=epsg, **kwargs
    )
