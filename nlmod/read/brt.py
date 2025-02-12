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



def get_brt_v2(
    extent,
    layer="waterdeel",
    cut_by_extent=True,
    make_valid=False,
    fname=None,
    geometry=None,
    remove_expired=True,
    add_bronhouder_names=True,
    timeout=1200,
):
    """Get geometries within an extent or polygon from the Basis Registratie
    Topografie (BRT). Useful links:
    https://api.pdok.nl/brt/top10nl/download/v1_0/ui

    Parameters
    ----------
    extent : list or tuple of length 4 or shapely Polygon
        The extent (xmin, xmax, ymin, ymax) or polygon for which shapes are
        requested.
    layer : string or list of strings, optional
        The layer(s) for which shapes are requested. When layer is "all", all layers are
        requested. The default is "waterdeel".
    cut_by_extent : bool, optional
        Only return the intersection with the extent if True. The default is
        True
    make_valid : bool, optional
        Make geometries valid by appying a buffer of 0 m when True. The default is
        False.
    fname : string, optional
        Save the zipfile that is received by the request to file. The default
        is None, which does not save anything to file.
    geometry: string, optional
        When geometry is specified, this attribute is used as the geometry of the
        resulting GeoDataFrame. Some layers have multiple geometry-attributes. An
        example is the layer 'pand', where each buidling (polygon) also contains a
        Point-geometry for the label. When geometry is None, the last attribute starting
        with the word "geometrie" is used as the geometry. The default is None.
    remove_expired: bool, optional
        Remove expired items (that contain a value for 'eindRegistratie') when True. The
        default is True.
    add_bronhouder_names: bool, optional
        Add bronhouder in a column called 'bronhouder_name. names when True. The default
        is True.
    timeout: int optional
        The amount of time in seconds to wait for the server to send data before giving
        up. The default is 1200 (20 minutes).

    Returns
    -------
    gdf : GeoPandas GeoDataFrame or dict of GeoPandas GeoDataFrame
        A GeoDataFrame (when only one layer is requested) or a dict of GeoDataFrames
        containing all geometries and properties.
    """

    if layer == "all":
        layer = get_brt_layers()
    if isinstance(layer, str):
        layer = [layer]

    api_url = "https://api.pdok.nl"
    url = f"{api_url}/brt/top10nl/download/v1_0/full/custom"
    body = {"format": "gml", "featuretypes": layer}

    if isinstance(extent, Polygon):
        polygon = extent
    else:
        polygon = extent_to_polygon(extent)

    body["geofilter"] = polygon.wkt

    headers = {"content-type": "application/json"}
    
    response = requests.post(
        url, headers=headers, data=json.dumps(body), timeout=timeout
    )  # 20 minutes

    # check api-status, if completed, download
    if response.status_code in range(200, 300):
        running = True
        href = response.json()["_links"]["status"]["href"]
        url = f"{api_url}{href}"

        while running:
            response = requests.get(url, timeout=timeout)
            if response.status_code in range(200, 300):
                status = response.json()["status"]
                if status == "COMPLETED":
                    running = False
                else:
                    time.sleep(2)
            else:
                running = False
    else:
        msg = f"Download of brt-data failed: {response.text}"
        raise (Exception(msg))

    href = response.json()["_links"]["download"]["href"]
    response = requests.get(f"{api_url}{href}", timeout=timeout)

    if fname is not None:
        with open(fname, "wb") as file:
            file.write(response.content)

    raise NotImplementedError('this part is not yet implemented please use "read_brt"')

    zipfile = BytesIO(response.content)
    gdf = read_brt_zipfile(
        zipfile,
        geometry=geometry,
        cut_by_extent=cut_by_extent,
        make_valid=make_valid,
        extent=polygon,
        remove_expired=remove_expired,
        add_bronhouder_names=add_bronhouder_names,
    )

    if len(layer) == 1:
        gdf = gdf[layer[0]]

    return 


def get_brt_layers(timeout=1200):
    """
    Get the layers in the Basis Registratie Topografie (BRT)

    Parameters
    ----------
    timeout: int optional
        The amount of time in seconds to wait for the server to send data before giving
        up. The default is 1200 (20 minutes).

    Returns
    -------
    list
        A list with the layer names.

    """
    url = "https://api.pdok.nl/brt/top10nl/ogc/v1/collections/waterdeel_lijn"
    resp = requests.get(url, timeout=timeout)
    data = resp.json()
    return [x["featuretype"] for x in data["timeliness"]]


def read_brt_zipfile(
    fname,
    geometry=None,
    files=None,
    cut_by_extent=True,
    make_valid=False,
    extent=None,
    remove_expired=True,
    add_bronhouder_names=True,
):
    """Read data from a zipfile that was downloaded using get_bgt().

    Parameters
    ----------
    fname : string
        The filename of the zip-file containing the BGT-data.
    geometry : str, optional
        DESCRIPTION. The default is None.
    files : string of list of strings, optional
        The files to read from the zipfile. Read all files when files is None. The
        default is None.
    cut_by_extent : bool, optional
        Cut the geoemetries by the supplied extent. When no extent is supplied,
        cut_by_extent is set to False. The default is True.
    make_valid : bool, optional
        Make geometries valid by appying a buffer of 0 m when True. THe defaults is
        False.
    extent : list or tuple of length 4 or shapely Polygon
        The extent (xmin, xmax, ymin, ymax) or polygon by which the geometries are
        clipped. Only used when cut_by_extent is True. The defult is None.
    remove_expired: bool, optional
        Remove expired items (that contain a value for 'eindRegistratie') when True. The
        default is True.
    add_bronhouder_names: bool, optional
        Add bronhouder in a column called 'bronhouder_name. names when True. The default
        is True.

    Returns
    -------
    gdf : dict of GeoPandas GeoDataFrame
        A dict of GeoDataFrames containing all geometries and properties.
    """
    zf = ZipFile(fname)
    gdf = {}
    if files is None:
        files = zf.namelist()
    elif isinstance(files, str):
        files = [files]
    if extent is None:
        cut_by_extent = False
    else:
        if isinstance(extent, Polygon):
            polygon = extent
        else:
            polygon = extent_to_polygon(extent)
    for file in files:
        key = file[4:-4]
        gdf[key] = read_brt_gml(zf.open(file), geometry=geometry)

        if remove_expired and gdf[key] is not None and "eindRegistratie" in gdf[key]:
            # remove double features
            # by removing features with an eindRegistratie
            gdf[key] = gdf[key][gdf[key]["eindRegistratie"].isna()]

        if make_valid and isinstance(gdf[key], gpd.GeoDataFrame):
            gdf[key].geometry = gdf[key].geometry.buffer(0.0)

        if cut_by_extent and isinstance(gdf[key], gpd.GeoDataFrame):
            gdf[key].geometry = gdf[key].intersection(polygon)
            gdf[key] = gdf[key][~gdf[key].is_empty]

    if add_bronhouder_names:
        bgt_bronhouder_names = get_bronhouder_names()
        for gdf_layer in gdf.values():
            if gdf_layer is None or "bronhouder" not in gdf_layer.columns:
                continue
            gdf_layer["bronhouder_name"] = gdf_layer["bronhouder"].map(
                bgt_bronhouder_names
            )

    return gdf


endpoint = ["https://api.pdok.nl/brt/top10nl/download/v1_0/full/custom",
            "https://api.pdok.nl/brt/top10nl/ogc/v1"]
