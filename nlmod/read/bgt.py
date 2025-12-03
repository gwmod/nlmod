import json
import os
import time
import warnings
from io import BytesIO
from typing import Dict
from xml.etree import ElementTree
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from nlmod import NLMOD_DATADIR

from .. import cache
from ..util import extent_to_polygon


def get_bgt(*args, **kwargs):
    """Get geometries within an extent or polygon from the Basis Registratie
    Grootschalige Topografie (BGT)

    .. deprecated:: 0.10.0
        `get_bgt` will be removed in nlmod 1.0.0, it is replaced by
        `download_bgt` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47


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

    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.bgt.download_bgt() in the future.",
        DeprecationWarning,
    )

    return download_bgt(*args, **kwargs)


@cache.cache_pickle
def download_bgt(
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
    Grootschalige Topografie (BGT)

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
        layer = get_bgt_layers()
    if isinstance(layer, str):
        layer = [layer]

    api_url = "https://api.pdok.nl"
    url = f"{api_url}/lv/bgt/download/v1_0/full/custom"
    body = {"format": "citygml", "featuretypes": layer}

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
        msg = f"Download of bgt-data failed: {response.text}"
        raise (Exception(msg))

    href = response.json()["_links"]["download"]["href"]
    response = requests.get(f"{api_url}{href}", timeout=timeout)

    if fname is not None:
        with open(fname, "wb") as file:
            file.write(response.content)

    zipfile = BytesIO(response.content)
    gdf = read_bgt_zipfile(
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

    return gdf


def read_bgt_zipfile(
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
        gdf[key] = read_bgt_gml(zf.open(file), geometry=geometry)

        if remove_expired and gdf[key] is not None and "eindRegistratie" in gdf[key]:
            # remove double features
            # by removing features with an eindRegistratie
            gdf[key] = gdf[key][gdf[key]["eindRegistratie"].isna()]

        if make_valid and isinstance(gdf[key], gpd.GeoDataFrame):
            gdf[key].geometry = gdf[key].make_valid()

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


def read_bgt_gml(fname, geometry="geometrie2dGrondvlak", crs="epsg:28992"):
    def get_xy(text):
        xy = [float(val) for val in text.split()]
        xy = np.array(xy).reshape(int(len(xy) / 2), 2)
        return xy

    def get_ring_xy(exterior):
        ns = "{http://www.opengis.net/gml}"
        assert len(exterior) == 1
        if exterior[0].tag == f"{ns}LinearRing":
            lr = exterior.find(f"{ns}LinearRing")
            xy = get_xy(lr.find(f"{ns}posList").text)
        elif exterior[0].tag == f"{ns}Ring":
            cm = exterior.find(f"{ns}Ring").find(f"{ns}curveMember")
            xy = read_curve(cm.find(f"{ns}Curve"))
        else:
            raise Exception(f"Unknown exterior type: {exterior[0].tag}")
        return xy

    def read_polygon(polygon):
        ns = "{http://www.opengis.net/gml}"
        exterior = polygon.find(f"{ns}exterior")
        shell = get_ring_xy(exterior)
        holes = []
        for interior in polygon.findall(f"{ns}interior"):
            holes.append(get_ring_xy(interior))
        return shell, holes

    def read_point(point):
        ns = "{http://www.opengis.net/gml}"
        xy = [float(x) for x in point.find(f"{ns}pos").text.split()]
        return xy

    def read_curve(curve):
        xy = np.empty((0, 2))
        for segment in curve.find(f"{ns}segments"):
            xy = np.vstack((xy, get_xy(segment.find(f"{ns}posList").text)))
        return xy

    def read_linestring(linestring):
        return get_xy(linestring.find(f"{ns}posList").text)

    def read_label(child, d):
        ns = "{http://www.geostandaarden.nl/imgeo/2.1}"
        label = child.find(f"{ns}Label")
        d["label"] = label.find(f"{ns}tekst").text
        positie = label.find(f"{ns}positie").find(f"{ns}Labelpositie")
        xy = read_point(
            positie.find(f"{ns}plaatsingspunt").find(
                "{http://www.opengis.net/gml}Point"
            )
        )
        d["label_plaatsingspunt"] = Point(xy)
        d["label_hoek"] = float(positie.find(f"{ns}hoek").text)

    tree = ElementTree.parse(fname)
    ns = "{http://www.opengis.net/citygml/2.0}"
    data = []
    for com in tree.findall(f".//{ns}cityObjectMember"):
        assert len(com) == 1
        bp = com[0]
        d = {}
        for key in bp.attrib:
            d[key.split("}", 1)[1]] = bp.attrib[key]
        for child in bp:
            key = child.tag.split("}", 1)[1]
            if len(child) == 0:
                d[key] = child.text
            else:
                if key == "identificatie":
                    ns = "{http://www.geostandaarden.nl/imgeo/2.1}"
                    loc_id = child.find(f"{ns}NEN3610ID").find(f"{ns}lokaalID")
                    d[key] = loc_id.text
                elif key.startswith("geometrie"):
                    if geometry is None:
                        geometry = key
                    assert len(child) == 1
                    ns = "{http://www.opengis.net/gml}"
                    if child[0].tag == f"{ns}MultiSurface":
                        ms = child.find(f"{ns}MultiSurface")
                        polygons = []
                        for sm in ms:
                            assert len(sm) == 1
                            polygon = sm.find(f"{ns}Polygon")
                            polygons.append(read_polygon(polygon))
                        d[key] = MultiPolygon(polygons)
                    elif child[0].tag == f"{ns}Polygon":
                        d[key] = Polygon(*read_polygon(child[0]))
                    elif child[0].tag == f"{ns}Curve":
                        d[key] = LineString(read_curve(child[0]))
                    elif child[0].tag == f"{ns}LineString":
                        d[key] = LineString(read_linestring(child[0]))
                    elif child[0].tag == f"{ns}Point":
                        d[key] = Point(read_point(child[0]))
                    else:
                        raise (ValueError((f"Unsupported tag: {child[0].tag}")))
                elif key == "nummeraanduidingreeks":
                    ns = "{http://www.geostandaarden.nl/imgeo/2.1}"
                    nar = child.find(f"{ns}Nummeraanduidingreeks").find(
                        f"{ns}nummeraanduidingreeks"
                    )
                    read_label(nar, d)
                elif key.startswith("kruinlijn"):
                    ns = "{http://www.opengis.net/gml}"
                    if child[0].tag == f"{ns}LineString":
                        ls = child.find(f"{ns}LineString")
                        d[key] = LineString(read_linestring(ls))
                    elif child[0].tag == f"{ns}Curve":
                        d[key] = LineString(read_curve(child[0]))
                    else:
                        raise (ValueError((f"Unsupported tag: {child[0].tag}")))
                elif key == "openbareRuimteNaam":
                    read_label(child, d)
                else:
                    raise (KeyError((f"Unknown key: {key}")))
        data.append(d)
    if len(data) > 0:
        if geometry is None:
            return pd.DataFrame(data)
        else:
            return gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    else:
        return None


def get_bgt_layers(timeout=1200):
    """
    Get the layers in the Basis Registratie Grootschalige Topografie (BGT)

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
    url = "https://api.pdok.nl/lv/bgt/download/v1_0/dataset"
    resp = requests.get(url, timeout=timeout)
    data = resp.json()
    return [x["featuretype"] for x in data["timeliness"]]


def get_bronhouder_names() -> Dict[str, str]:
    """Get the names of the bronhouders of the BGT data.

    Returns
    -------
    dict
        A dictionary with the bronhouder codes as keys and the names as values.

    Notes
    -----
        The bronhouder names are retrieved from
        https://www.kadaster.nl/-/bgt-bronhoudercodes. The `Toelichting` sheet
        in the .ods file gives the changes compared to the old file. If changes
        are made, please add them manually to the bgt_bronhouder_names
        dictionary. A test is added for to check if the dictionary is up to
        date with the latest file from the Kadaster up to date with the .ods
        file from 2024-01-01.
    """
    fname = os.path.join(NLMOD_DATADIR, "bgt", "bronhouder_names.json")
    with open(fname, "r", encoding="utf-8") as fo:
        bgt_bronhouder_names = json.load(fo)

    return bgt_bronhouder_names
