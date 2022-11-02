# -*- coding: utf-8 -*-
"""Created on Wed Apr 20 17:01:07 2022.

@author: Ruben
"""

import json
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from ..dims.resample import extent_to_polygon


def get_bgt(extent, layer="waterdeel", cut_by_extent=True, fname=None, geometry=None):
    """Get geometries within an extent or polygon from the Basis Registratie
    Grootschalige Topografie (BGT)

    Parameters
    ----------
    extent : list or tuple of length 4 or shapely Polygon
        The extent (xmin, xmax, ymin, ymax) or polygon for which shapes are
        requested.
    layer : string, optional
        The layer for which shapes are requested. The default is "waterdeel".
    cut_by_extent : bool, optional
        Only return the intersection with the extent if True. The default is
        True
    fname : string, optional
        Save the zipfile that is received by the request to file. The default
        is None, which does not save anything to file.
    geometry: string, optional
        When geometry is specified, the gml inside the received zipfile is read
        using an xml-reader (instead of fiona). For the layer 'waterdeel' the
        geometry-field is 'geometrie2dWaterdeel', and for the layer 'pand' the
        geometry-field is 'geometrie2dGrondvlak'. To determine the geometry
        field of other layers, use fname to save a response and inspect the
        gml-file. The default is None, which results in fiona reading the data.
        This can cause problems when there are multiple geometrie-fields inside
        each object. This happens in the layer 'pand', where each buidling
        (polygon) also contains a Point-geometry for the label.

    Returns
    -------
    gdf : GeoPandas GeoDataFrame
        A GeoDataFrame containing all geometries and properties.
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
        url, headers=headers, data=json.dumps(body), timeout=1200
    )  # 20 minutes

    # check api-status, if completed, download
    if response.status_code in range(200, 300):
        running = True
        href = response.json()["_links"]["status"]["href"]
        url = f"{api_url}{href}"

        while running:
            response = requests.get(url, timeout=1200)  # 20 minutes
            if response.status_code in range(200, 300):
                status = response.json()["status"]
                if status == "COMPLETED":
                    running = False
                else:
                    time.sleep(2)
            else:
                running = False
    else:
        msg = "Download of bgt-data failed: {response.text}"
        raise (Exception(msg))

    href = response.json()["_links"]["download"]["href"]
    response = requests.get(f"{api_url}{href}", timeout=1200)  # 20 minutes

    if fname is not None:
        with open(fname, "wb") as file:
            file.write(response.content)

    gdf = {}

    zipfile = BytesIO(response.content)
    gdf = read_bgt_zipfile(zipfile, geometry=geometry)

    for key in gdf:
        if gdf[key] is not None and "eindRegistratie" in gdf[key]:
            # remove double features
            # by removing features with an eindRegistratie
            gdf[key] = gdf[key][gdf[key]["eindRegistratie"].isna()]

        if cut_by_extent and isinstance(gdf[key], gpd.GeoDataFrame):
            try:
                gdf[key].geometry = gdf[key].intersection(polygon)
                gdf[key] = gdf[key][~gdf[key].is_empty]
            except shapely.geos.TopologicalError:
                print(f"Cutting by extent failed for {key}")
    if len(layer) == 1:
        gdf = gdf[layer[0]]
    return gdf


def read_bgt_zipfile(fname, geometry=None, files=None):
    zf = ZipFile(fname)
    gdf = {}
    if files is None:
        files = zf.namelist()
    elif isinstance(files, str):
        files = [files]
    for file in files:
        key = file[4:-4]
        gdf[key] = read_bgt_gml(zf.open(file), geometry=geometry)
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

    def _read_label(child, d):
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

    tree = ET.parse(fname)
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
                        raise (Exception((f"Unsupported tag: {child[0].tag}")))
                elif key == "nummeraanduidingreeks":
                    ns = "{http://www.geostandaarden.nl/imgeo/2.1}"
                    nar = child.find(f"{ns}Nummeraanduidingreeks").find(
                        f"{ns}nummeraanduidingreeks"
                    )
                    _read_label(nar, d)
                elif key in [
                    "kruinlijnBegroeidTerreindeel",
                    "kruinlijnOnbegroeidTerreindeel",
                    "kruinlijnOndersteunendWegdeel",
                ]:
                    ns = "{http://www.opengis.net/gml}"
                    if child[0].tag == f"{ns}LineString":
                        ls = child.find(f"{ns}LineString")
                        d[key] = LineString(read_linestring(ls))
                    elif child[0].tag == f"{ns}Curve":
                        d[key] = LineString(read_curve(child[0]))
                    else:
                        raise (Exception((f"Unsupported tag: {child[0].tag}")))
                elif key == "openbareRuimteNaam":
                    _read_label(child, d)
                else:
                    raise (Exception((f"Unknown key: {key}")))
        data.append(d)
    if len(data) > 0:
        if geometry is None:
            return pd.DataFrame(data)
        else:
            return gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    else:
        return None


def get_bgt_layers():
    url = "https://api.pdok.nl/lv/bgt/download/v1_0/dataset"
    resp = requests.get(url, timeout=1200)  # 20 minutes
    data = resp.json()
    return [x["featuretype"] for x in data["timeliness"]]
