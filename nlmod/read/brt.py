import json
import logging
import time
import warnings
from io import BytesIO
from xml.etree import ElementTree
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from tqdm import tqdm

from ..util import extent_to_polygon

logger = logging.getLogger(__name__)

NS = {
    "top10nl": "http://register.geostandaarden.nl/gmlapplicatieschema/top10nl/1.2.0",
    "brt": "http://register.geostandaarden.nl/gmlapplicatieschema/brt-algemeen/1.2.0",
    "gml": "http://www.opengis.net/gml/3.2",
}
def get_brt(*args, **kwargs):
    """Get geometries within an extent/polygon from the Basis Registratie Topografie.

    .. deprecated:: 0.10.0
          `get_brt` will be removed in nlmod 1.0.0, it is replaced by
          `download_brt_gdf` because of new naming convention 
          https://github.com/gwmod/nlmod/issues/47

    Useful links:
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
        example is the layer 'wegdeel', where each feature (point) has a hartGeometry
        and a hoofdGeometry. When geometry is None, the last tag in the xml ending
        with the word "geometrie" of "Geometrie" is used as the geometry. The default
        is None.
    timeout: int optional
        The amount of time in seconds to wait for the server to send data before giving
        up. The default is 1200 (20 minutes).

    Returns
    -------
    gdf : GeoPandas GeoDataFrame or dict of GeoPandas GeoDataFrame
        A GeoDataFrame (when only one layer is requested) or a dict of GeoDataFrames
        containing all geometries and properties.
    """
    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.administrative.download_waterboards_gdf() in the future.",
        DeprecationWarning,
    )

    return download_brt_gdf(*args, **kwargs)



def download_brt_gdf(
    extent,
    layer="waterdeel",
    cut_by_extent=True,
    make_valid=False,
    fname=None,
    geometry=None,
    timeout=1200,
):
    """Get geometries within an extent/polygon from the Basis Registratie Topografie.

    Useful links:
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
        example is the layer 'wegdeel', where each feature (point) has a hartGeometry
        and a hoofdGeometry. When geometry is None, the last tag in the xml ending
        with the word "geometrie" of "Geometrie" is used as the geometry. The default
        is None.
    timeout: int optional
        The amount of time in seconds to wait for the server to send data before giving
        up. The default is 1200 (20 minutes).

    Returns
    -------
    gdf : GeoPandas GeoDataFrame or dict of GeoPandas GeoDataFrame
        A GeoDataFrame (when only one layer is requested) or a dict of GeoDataFrames
        containing all geometries and properties.
    """
    if layer.lower() == "all":
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

    msg = f"Downloading BRT data layers {layer} within {body['geofilter']}"
    logger.info(msg)

    response = requests.post(
        url, headers=headers, data=json.dumps(body), timeout=timeout
    )

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
        response.raise_for_status()

    href = response.json()["_links"]["download"]["href"]
    response = requests.get(f"{api_url}{href}", timeout=timeout)

    if fname is not None:
        with open(fname, "wb") as file:
            file.write(response.content)

    zipfile = BytesIO(response.content)
    gdf_dic = read_brt_zipfile(
        zipfile,
        geometry=geometry,
        cut_by_extent=cut_by_extent,
        make_valid=make_valid,
        extent=polygon,
    )

    if len(layer) == 1:
        gdf_dic = gdf_dic[layer[0]]

    return gdf_dic


def get_brt_layers(timeout=1200):
    """
    Get the layers in the Basis Registratie Topografie (BRT).

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
    url = "https://api.pdok.nl/brt/top10nl/download/v1_0/dataset"
    resp = requests.get(url, timeout=timeout)
    data = resp.json()
    return [x["featuretype"] for x in data["timeliness"]]


def read_brt_zipfile(
    fname, geometry=None, files=None, cut_by_extent=True, make_valid=False, extent=None
):
    """Read data from a zipfile that was downloaded using get_brt().

    Parameters
    ----------
    fname : string
        The filename of the zip-file containing the BRT-data.
    geometry: string, optional
        When geometry is specified, this attribute is used as the geometry of the
        resulting GeoDataFrame. Some layers have multiple geometry-attributes. An
        example is the layer 'wegdeel', where each feature (point) has a hartGeometry
        and a hoofdGeometry. When geometry is None, the last tag in the xml ending
        with the word "geometrie" of "Geometrie" is used as the geometry. The default
        is None.
    files : string of list of strings, optional
        The files to read from the zipfile. Read all files when files is None. The
        default is None.
    cut_by_extent : bool, optional
        Cut the geoemetries by the supplied extent. When no extent is supplied,
        cut_by_extent is set to False. The default is True.
    make_valid : bool, optional
        Make geometries valid by appying a buffer of 0 m when True. The default is
        False.
    extent : list or tuple of length 4 or shapely Polygon
        The extent (xmin, xmax, ymin, ymax) or polygon by which the geometries are
        clipped. Only used when cut_by_extent is True. The defult is None.

    Returns
    -------
    gdf_dic : dict of GeoPandas GeoDataFrame
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
        lay = file[8:-4]
        if lay == "relief":
            logger.warning("Cannot read relief data, not implemented yet")
            continue

        logger.debug(f"reading brt layer {file}")
        gdf_dic[lay] = read_brt_gml(zf.open(file), lay=lay, geometry=geometry)

        if make_valid and isinstance(gdf_dic[lay], gpd.GeoDataFrame):
            gdf_dic[lay].geometry = gdf_dic[lay].geometry.buffer(0.0)

        if cut_by_extent and isinstance(gdf_dic[lay], gpd.GeoDataFrame):
            no_ft = gdf_dic[lay].shape[0]
            gdf_dic[lay].geometry = gdf_dic[lay].intersection(polygon)
            gdf_dic[lay] = gdf_dic[lay][~gdf_dic[lay].is_empty]
            rm_ft = no_ft - gdf_dic[lay].shape[0]
            logger.info(f"removed {rm_ft} features that are outside the extent")

    return gdf_dic


def read_brt_gml(fname, lay="waterdeel", geometry=None, crs="epsg:28992"):
    """Read an xml file with features from the BRT.

    Parameters
    ----------
    fname : str
        filename
    lay : str, optional
        The layer to read. The default is "waterdeel".
    geometry: string, optional
        When geometry is specified, this attribute is used as the geometry of the
        resulting GeoDataFrame. Some layers have multiple geometry-attributes. An
        example is the layer 'wegdeel', where each feature (point) has a hartGeometry
        and a hoofdGeometry. When geometry is None, the last tag in the xml ending
        with the word "geometrie" of "Geometrie" is used as the geometry. The default
        is None.
    crs : str, optional
        coordinate reference system, by default "epsg:28992"

    Returns
    -------
    GeoDataFrame or None
        with BRT feature data
    """
    tree = ElementTree.parse(fname)
    ft_members = tree.findall("top10nl:FeatureMember", NS)
    data = [
        _read_single_brt_feature(com, lay=lay, geometry=geometry)
        for com in tqdm(ft_members, desc=f"Downloading features of layer {lay}")
    ]

    if len(data) > 0:
        return gpd.GeoDataFrame(data, geometry="geometry", crs=crs)
    else:
        return None


def _read_single_brt_feature(com, lay="waterdeel", geometry=None):
    """Read a single feature from an xml with multiple features.

    Parameters
    ----------
    com : Element
        xml reference to a feature of the BRT
    lay : str, optional
        The layer to read. The default is "waterdeel".
    geometry: string, optional
        When geometry is specified, this attribute is used as the geometry of the
        resulting GeoDataFrame. Some layers have multiple geometry-attributes. An
        example is the layer 'wegdeel', where each feature (point) has a hartGeometry
        and a hoofdGeometry. When geometry is None, the last tag in the xml ending
        with the word "geometrie" of "Geometrie" is used as the geometry. The default
        is None.


    Returns
    -------
    dict
        dictionary with data from a single feature

    Raises
    ------
    RuntimeError
        if the feature is not a 'lijn' or 'vlak' geometry
    """
    if geometry is None:
        geometry = "geometrie"
    else:
        geometry = geometry.lower()  # make sure geometry is lower case

    assert len(com) == 1
    bp = com[0]
    d = {}
    for key, name in bp.attrib.items():
        d[key.rpartition("}")[-1]] = name

    for child in bp:
        tag = child.tag.rpartition("}")[-1]
        if tag.lower().endswith("geometrie"):
            geom = child[0]
            d["geometry"] = _read_geometry(geom)
        elif tag == "geometrieVlak" and lay == "terrein":
            d["geometry"] = _read_geometry(child)
        elif tag == "identificatie":
            loc_id = child.find("brt:NEN3610ID", NS).find("brt:lokaalID", NS)
            d["lokaalID"] = loc_id.text
        else:
            d[tag] = child.text

    return d


def _read_geometry(geom):
    """Read geometry from a geometry xml tag.

    Parameters
    ----------
    geom : Element
        xml reference to a geometry tag in the BRT

    Returns
    -------
    shapely.Geometry or None
    """
    assert len(geom) == 1
    feature_geom = geom[0]

    geom_tag = feature_geom.tag.rpartition("}")[-1]
    if geom_tag == "lijnGeometrie":
        return LineString(_read_linestring(feature_geom[0]))
    elif geom_tag == "puntGeometrie":
        return Point(_read_point(feature_geom[0]))
    elif geom_tag == "vlakGeometrie":
        return Polygon(*_read_polygon(feature_geom[0]))
    elif geom_tag == "hartGeometrie":
        return Polygon(*_read_polygon(feature_geom[0]))
    elif geom_tag == "Polygon":
        return Polygon(*_read_polygon(feature_geom))
    elif geom_tag == "multivlakGeometrie":
        ms = feature_geom.find("gml:MultiSurface", NS)
        polygons = []
        for sm in ms:
            assert len(sm) == 1
            polygon = sm.find("gml:Polygon", NS)
            polygons.append(_read_polygon(polygon))
        return MultiPolygon(polygons)
    else:
        logger.warning(f"cannot read geometry type {geom_tag}, skipping these features")
        return None


def _get_xy(text):
    """Get x and y coordinates from a tag text.

    Parameters
    ----------
    text : str
        x and y values in str format

    Returns
    -------
    np.ndarray
        x and y coordinates
    """
    xy = [float(val) for val in text.split()]
    xy = np.array(xy).reshape(int(len(xy) / 2), 2)
    return xy


def _get_ring_xy(exterior):
    """Get x and y coordinates for a LinearRing.

    Parameters
    ----------
    exterior : Element
        xml reference to an exterior tag of the BRT

    Returns
    -------
    np.ndarray
        x and y coordinates

    Raises
    ------
    NotImplementedError
        for unknown exterior types
    """
    assert len(exterior) == 1
    if exterior[0].tag.rpartition("}")[-1] == "LinearRing":
        lr = exterior.find("gml:LinearRing", NS)
        xy = _get_xy(lr.find("gml:posList", NS).text)
    else:
        raise NotImplementedError(f"Unknown exterior type: {exterior[0].tag}")
    return xy


def _read_polygon(polygon):
    """Read polygon geometry from an xml tag.

    Parameters
    ----------
    polygon : Element
        xml reference to a polygon tag.

    Returns
    -------
    shell, holes
        polygon definition
    """
    exterior = polygon.find("gml:exterior", NS)
    shell = _get_ring_xy(exterior)
    holes = []
    for interior in polygon.findall("gml:interior", NS):
        holes.append(_get_ring_xy(interior))
    return shell, holes


def _read_linestring(linestring):
    """Read linestring geometry from an xml tag.

    Parameters
    ----------
    linestring : Element
        xml reference to a linestring tag.

    Returns
    -------
    np.ndarray
        x and y coordinates
    """
    return _get_xy(linestring.find("gml:posList", NS).text)


def _read_point(point):
    """Read point geometry from an xml tag.

    Parameters
    ----------
    point : Element
        xml reference to a point tag.

    Returns
    -------
    list
        x and y coordinates
    """
    xy = [float(x) for x in point.find("gml:pos", NS).text.split()]
    return xy


def get_brt_ogc(
    extent, layer="waterdeel_lijn", crs=28992, limit=1000, apif="json", timeout=1200
):
    """Get geometries within an extent from the Basis Registratie Topografie (BRT).

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
        The maximum number of features that is requested. The default is 1000. This api
        does not seem to work properly when there are more than 1000 features in your
        extent.
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
            get_brt_ogc(
                extent=extent,
                layer=lay,
                crs=crs,
                limit=limit,
                apif=apif,
                timeout=timeout,
            )
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
            "invalid crs, please choose between 28992 (RD) and 4326 (WGS84)"
        )

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    gdf = gpd.GeoDataFrame.from_features(response.json()["features"])

    if gdf.shape[0] == limit:
        msg = (
            f"the number of features in your extent is probably higher than {limit},"
            ' consider increasing the "limit" argument in "get_brt"'
        )
        logger.error(msg)

    return gdf
