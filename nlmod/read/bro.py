import logging
import flopy
import hydropandas as hpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from .. import cache, dims, gwf

logger = logging.getLogger(__name__)


def get_bro(extent, max_dx=0.1, max_dy=0.1, epsg=28992, cachedir=None, ignore_errors=True,
            **kwargs):
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
    if not epsg == 4326:
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
        st = ("requested bro dataset width or height bigger than maxsize "
              f"splitting extent into {x_segments} * {y_segments} tiles")
        logger.info(st)
        l = []
        for tx in range(x_segments):
            for ty in range(y_segments):
                xmin = extent[0] + tx * max_dx
                xmax = min(extent[1], extent[0] + (tx + 1) * max_dx)
                ymin = extent[2] + ty * max_dy
                ymax = min(extent[3], extent[2] + (ty + 1) * max_dx)

                logger.debug(
                    f"reading bro within extent {xmin}, {xmax}, {ymin}, {ymax}"
                )
                name = f'BRO_{xmin}_{xmax}_{ymin}_{ymax}'
                if ignore_errors:
                    try:
                        oc = _get_bro_within_extent((xmin, xmax, ymin, ymax),
                                                    name=name,
                                                    ignore_max_obs=True,
                                                    epsg=4326,
                                                    cachedir=cachedir,
                                                    cachename=name,
                                                    **kwargs)
                        l.append(oc)
                    except Exception as e:
                        logger.error(
                        f"could not download BRO data in extent {xmin}, {xmax}, {ymin}, {ymax}"
                        )
                        logger.error(e)
                else:
                    oc = _get_bro_within_extent((xmin, xmax, ymin, ymax),
                                                    name=name,
                                                    ignore_max_obs=True,
                                                    epsg=4326,
                                                    cachedir=cachedir,
                                                    cachename=name,
                                                    **kwargs)
                    l.append(oc)
        oc = pd.concat(l)
    else:
        name = 'BRO_' + '_'.join(map(str,extent))
        oc = _get_bro_within_extent(extent,
                                    name=name,
                                    ignore_max_obs=True,
                                    epsg=4326,
                                    cachedir=cachedir,
                                    cachename=name,
                                    **kwargs)
    if oc.empty:
        logger.warning("no observation wells within extent")

    return oc


@cache.cache_pickle
def _get_bro_within_extent(extent, name, ignore_max_obs, epsg, **kwargs):
    """get bro groundwater measurements within extent

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

    return hpd.read_bro(extent,
                        name=name,
                        ignore_max_obs=ignore_max_obs,
                        epsg=epsg, **kwargs)