import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def get_brt(extent, layer="waterdeel_lijn", crs=28992, limit=1000, apif="json", timeout=1200):
    """
    Get geometries within an extent from the Basis Registratie Topografie (BRT).
    Some useful links:
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
    limit : int, optional
        The maximum number of features that is requested. The default is 1000.
    apif : str, optional
        The output format of the api. The default is 'json'.
    timeout: int optional
        The amount of time in seconds to wait for the server to send data before giving
        up. The default is 1200 (20 minutes).

    Returns
    -------
    gdf : GeoPandas GeoDataFrame
        A GeoDataFrame containing all geometries and properties.

    """
    if isinstance(layer, (list, tuple, np.ndarray)):
        # recursively call this function for all layers
        gdfs = [
            get_brt(extent=extent, layer=lay, crs=crs, limit=limit, apif=apif, timeout=timeout)
            for lay in layer
        ]
        return pd.concat(gdfs)

    api_url = "https://api.pdok.nl"
    url = f"{api_url}/brt/top10nl/ogc/v1/collections/{layer}/items?"

    params = {
        "bbox": f"{extent[0]},{extent[2]},{extent[1]},{extent[3]}",
        "f": apif,
        "limit": limit,
    }

    if crs == 28992:
        params["crs"] = "http://www.opengis.net/def/crs/EPSG/0/28992"
        params["bbox-crs"] = "http://www.opengis.net/def/crs/EPSG/0/28992"
    elif crs != 4326:
        raise ValueError(
            "invalid crs, please choose between 28992 (RD) or 4326 (WGS84)"
        )

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    gdf = gpd.GeoDataFrame.from_features(response.json()["features"])

    if gdf.shape[0] == limit:
        msg = f'the number of features in your extent is probably higher than {limit}, consider increasing the "limit" argument in "get_brt"'
        logger.warning(msg)

    return gdf
