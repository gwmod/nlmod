# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:54:02 2022

@author: Ruben
"""

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon

# from owslib.wfs import WebFeatureService


def arcrest(url, layer, extent=None, sr=28992, f="geojson", max_record_count=None):
    """Download data from an arcgis rest FeatureServer"""
    params = {
        "f": f,
        "outFields": "*",
        "outSR": sr,
        "where": "1=1",
    }
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        params["spatialRel"] = "esriSpatialRelIntersects"
        params["geometry"] = f"{xmin},{ymin},{xmax},{ymax}"
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = sr
    r = requests.get(url, params={"f": "json"})
    if not r.ok:
        raise (Exception("Request not successful"))
    props = r.json()
    if max_record_count is None:
        max_record_count = props["maxRecordCount"]
    else:
        max_record_count = min(max_record_count, props["maxRecordCount"])
    params["returnIdsOnly"] = True
    r = requests.get(f"{url}/{layer}/query", params=params)
    if not r.ok:
        raise (Exception("Request not successful"))
    props = r.json()
    params.pop("returnIdsOnly")
    if "objectIds" in props:
        object_ids = props["objectIds"]
        object_id_field_name = props["objectIdFieldName"]
    else:
        object_ids = props["properties"]["objectIds"]
        object_id_field_name = props["properties"]["objectIdFieldName"]
    if len(object_ids) > max_record_count:
        object_ids.sort()
        n_d = int(np.ceil((len(object_ids) / max_record_count)))
        features = []
        for i_d in tqdm(range(n_d)):
            i_min = i_d * max_record_count
            i_max = min(i_min + max_record_count - 1, len(object_ids) - 1)
            where = "{}>={} and {}<={}".format(
                object_id_field_name,
                object_ids[i_min],
                object_id_field_name,
                object_ids[i_max],
            )
            params["where"] = where
            r = requests.get(f"{url}/{layer}/query", params=params)
            if not r.ok:
                raise (Exception("Request not successful"))
            features.extend(r.json()["features"])
    else:
        r = requests.get(f"{url}/{layer}/query", params=params)
        if not r.ok:
            raise (Exception("Request not successful"))
        features = r.json()["features"]
    if f == "json":
        # convert to geometry
        data = []
        for feature in features:
            if len(feature["geometry"]) > 1 or "rings" not in feature["geometry"]:
                raise (Exception("Not supported yet"))
            if len(feature["geometry"]["rings"]) == 1:
                geometry = Polygon(feature["geometry"]["rings"][0])
            else:
                pols = [Polygon(xy) for xy in feature["geometry"]["rings"]]
                keep = [0]
                for i in range(1, len(pols)):
                    if pols[i].within(pols[keep[-1]]):
                        pols[keep[-1]] = pols[keep[-1]].difference(pols[i])
                    else:
                        keep.append(i)
                if len(keep) == 1:
                    geometry = pols[keep[0]]
                else:
                    geometry = MultiPolygon([pols[i] for i in keep])
            feature["attributes"]["geometry"] = geometry
            data.append(feature["attributes"])
        gdf = gpd.GeoDataFrame(data)
    else:
        gdf = gpd.GeoDataFrame.from_features(features)
    return gdf


def wfs(url, layer, extent=None, version="2.0.0", paged=True, max_record_count=None):
    """Download data from a wfs server"""
    params = dict(version=version, request="GetFeature", typeName=layer)
    if extent is not None:
        params["bbox"] = f"{extent[0]},{extent[2]},{extent[1]},{extent[3]}"
    if paged:
        # wfs = WebFeatureService(url)
        # get the maximum number of features
        r = requests.get(f"{url}&request=getcapabilities")
        if not r.ok:
            raise (Exception("Request not successful"))
        root = ET.fromstring(r.text)
        ns = {"ows": "http://www.opengis.net/ows/1.1"}

        constraints = {}

        def add_constrains(elem, constraints):
            for child in elem.findall("ows:Constraint", ns):
                key = child.attrib["name"]
                dv = child.find("ows:DefaultValue", ns)
                if not hasattr(dv, "text"):
                    continue
                value = dv.text
                if value[0].isdigit():
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                elif value.lower() in ["true", "false"]:
                    value = bool(value)
                constraints[key] = value

        om = root.find("ows:OperationsMetadata", ns)
        add_constrains(om, constraints)
        ops = om.findall("ows:Operation", ns)
        for op in ops:
            if op.attrib["name"] == "GetFeature":
                add_constrains(op, constraints)

        if max_record_count is None:
            max_record_count = constraints["CountDefault"]
        else:
            max_record_count = min(max_record_count, constraints["CountDefault"])

        # get the number of features
        params["resultType"] = "hits"
        r = requests.get(url, params=params)
        params.pop("resultType")
        root = ET.fromstring(r.text)
        if version == "1.1.0":
            n = int(root.attrib["numberOfFeatures"])
        else:
            n = int(root.attrib["numberMatched"])
        if n <= max_record_count:
            paged = False

    if paged:
        # download the features per page
        gdfs = []
        params["count"] = max_record_count
        for ip in range(int(np.ceil(n / max_record_count))):
            params["startindex"] = ip * max_record_count
            q = requests.Request("GET", url, params=params).prepare().url
            gdfs.append(gpd.read_file(q))
        gdf = pd.concat(gdfs)
    else:
        # download all features in one go
        q = requests.Request("GET", url, params=params).prepare().url
        gdf = gpd.read_file(q)

    return gdf
