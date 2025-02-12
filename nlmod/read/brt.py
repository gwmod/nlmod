import json
import logging
import requests

import geopandas as gpd
import numpy as np
import pandas as pd

from io import BytesIO
from shapely.geometry import mapping, shape, Polygon
from zipfile import ZipFile

from ..util import extent_to_polygon

logger = logging.getLogger(__name__)

def get_brt(
    extent,
    layer="waterdeel_lijn",
    crs=28992,
    pagesize=1000
):
    """
    Get geometries within an extent or polygon from the Basis Registratie
    Topografie (BRT). Some useful links:
    https://geoforum.nl/t/pdok-lanceert-de-brt-top10nl-in-ogc-api-s-als-demo/9821/5
    https://api.pdok.nl/brt/top10nl/ogc/v1/api

    Parameters
    ----------
    extent : list or tuple of int or float
        The extent (xmin, xmax, ymin, ymax) for which shapes are requested.
    layer : string, optional
        The layer for which shapes are requested. The default is 'waterdeel_lijn'.
    crs : int, optional
        The coordinate reference system. The default is 28992.
    pagesize : int, optional
        The number of features that is reqeusted per page. The default is 1000.

    Returns
    -------
    gdf : GeoPandas GeoDataFrame or requests.models.Response
        A GeoDataFrame containing all geometries and properties.

    """

    if isinstance(layer, (list, tuple, np.ndarray)):
        # recursively call this function for all layers
        gdfs = [get_brt(extent=extent, layer=l, crs=crs, pagesize=pagesize) for l in layer]
        return pd.concat(gdfs)

    api_url = "https://api.pdok.nl"
    url = f"{api_url}/brt/top10nl/ogc/v1/collections/{layer}/items?"

    params = {'bbox':f'{extent[0]},{extent[2]},{extent[1]},{extent[3]}',
              'f':'json',
              'limit':pagesize}

    if crs==28992:
        params['crs']      = "http://www.opengis.net/def/crs/EPSG/0/28992"
        params['bbox-crs'] = "http://www.opengis.net/def/crs/EPSG/0/28992"
    elif crs!=4326:
        raise ValueError('invalid crs, please choose between 28992 (RD) or 4326 (WGS84)')

    response = requests.get(url, params=params)
    response.raise_for_status()

    gdf = gpd.GeoDataFrame.from_features(response.json()['features'])

    return gdf