import json
import logging
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from io import BytesIO
from shapely.geometry import mapping, shape, Polygon, LineString
from xml.etree import ElementTree
from zipfile import ZipFile

from ..util import extent_to_polygon

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



def get_brt_v2(
    extent,
    layer="waterdeel",
    cut_by_extent=True,
    make_valid=False,
    fname=None,
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

    # raise NotImplementedError('this part is not yet implemented please use "read_brt"')

    zipfile = BytesIO(response.content)
    gdf_dic = read_brt_zipfile(
        zipfile,
        cut_by_extent=cut_by_extent,
        make_valid=make_valid,
        extent=polygon,
        remove_expired=remove_expired,
        add_bronhouder_names=add_bronhouder_names,
    )

    if len(layer) == 1:
        gdf_dic = gdf_dic[layer[0]]

    return gdf_dic


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

    Returns
    -------
    gdf : dict of GeoPandas GeoDataFrame
        A dict of GeoDataFrames containing all geometries and properties.
    """
    zf = ZipFile(fname)
    gdf_dic = {}
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
        key = file[8:-4]
        gdf_dic[key] = read_brt_gml(zf.open(file))

        if remove_expired and gdf_dic[key] is not None and "eindRegistratie" in gdf_dic[key]:
            # remove double features
            # by removing features with an eindRegistratie
            gdf_dic[key] = gdf_dic[key][gdf_dic[key]["eindRegistratie"].isna()]

        if make_valid and isinstance(gdf_dic[key], gpd.GeoDataFrame):
            gdf_dic[key].geometry = gdf_dic[key].geometry.buffer(0.0)

        if cut_by_extent and isinstance(gdf_dic[key], gpd.GeoDataFrame):
            gdf_dic[key].geometry = gdf_dic[key].intersection(polygon)
            gdf_dic[key] = gdf_dic[key][~gdf_dic[key].is_empty]

    return gdf_dic



def read_brt_gml(fname, crs="epsg:28992"):
    def get_xy(text):
        xy = [float(val) for val in text.split()]
        xy = np.array(xy).reshape(int(len(xy) / 2), 2)
        return xy

    def get_ring_xy(exterior):
        assert len(exterior) == 1
        if exterior[0].tag.rpartition('}')[-1] == "LinearRing":
            lr = exterior.find("gml:LinearRing",ns)
            xy = get_xy(lr.find("gml:posList",ns).text)
        else:
            raise Exception(f"Unknown exterior type: {exterior[0].tag}")
        return xy

    def read_polygon(polygon):
        exterior = polygon.find("gml:exterior",ns)
        shell = get_ring_xy(exterior)
        holes = []
        for interior in polygon.findall("gml:interior",ns):
            holes.append(get_ring_xy(interior))
        return shell, holes

    def read_linestring(linestring):
        return get_xy(linestring.find("gml:posList",ns).text)

    ns = {'top10nl': "http://register.geostandaarden.nl/gmlapplicatieschema/top10nl/1.2.0",
          'brt': "http://register.geostandaarden.nl/gmlapplicatieschema/brt-algemeen/1.2.0",
          'gml': "http://www.opengis.net/gml/3.2"}
    tree = ElementTree.parse(fname)

    data = []
    for com in tree.findall("top10nl:FeatureMember", ns):
        assert len(com) == 1
        bp = com[0]
        d = {}
        for key, name in bp.attrib.items():
            d[key.rpartition('}')[-1]] = name

        for child in bp:
            tag = child.tag.rpartition('}')[-1]
            if tag == 'geometrie':
                geom = child.find('brt:BRTVlakLijnOfPunt',ns)
                assert len(geom) == 1
                for child in geom:
                    tag = child.tag.rpartition('}')[-1]
                if tag =='lijnGeometrie':
                    d['geometry'] = LineString(read_linestring(child[0]))
                elif tag =='vlakGeometrie':
                    d['geometry'] = Polygon(*read_polygon(child[0]))
                else:
                    raise RuntimeError('unexpected geometry type')
            elif tag == 'identificatie':
                loc_id = child.find(f"brt:NEN3610ID",ns).find(f"brt:lokaalID",ns)
                d['lokaalID'] = loc_id.text
            else:
                d[tag] = child.text
        data.append(d)

    if len(data) > 0:
        return gpd.GeoDataFrame(data, geometry='geometry', crs=crs)
    else:
        return None

# endpoint = ["https://api.pdok.nl/brt/top10nl/download/v1_0/full/custom",
#             "https://api.pdok.nl/brt/top10nl/ogc/v1"]