import logging
import xml.etree.ElementTree as ET
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray
from owslib.wcs import WebCoverageService
from rasterio import merge
from rasterio.io import MemoryFile
from requests.exceptions import HTTPError
from shapely.geometry import MultiPolygon, Point, Polygon
from tqdm import tqdm

# from owslib.wfs import WebFeatureService

logger = logging.getLogger(__name__)


def arcrest(
    url,
    layer,
    extent=None,
    sr=28992,
    f="geojson",
    max_record_count=None,
    timeout=120,
    **kwargs,
):
    """Download data from an arcgis rest FeatureServer."""
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
    props = _get_data(url, {"f": "json"}, timeout=timeout, **kwargs)
    if max_record_count is None:
        max_record_count = props["maxRecordCount"]
    else:
        max_record_count = min(max_record_count, props["maxRecordCount"])

    params["returnIdsOnly"] = True
    url_query = f"{url}/{layer}/query"
    props = _get_data(url_query, params, timeout=timeout, **kwargs)
    params.pop("returnIdsOnly")
    if "objectIds" in props:
        object_ids = props["objectIds"]
        object_id_field_name = props["objectIdFieldName"]
    else:
        object_ids = props["properties"]["objectIds"]
        object_id_field_name = props["properties"]["objectIdFieldName"]
    if object_ids is not None and len(object_ids) > max_record_count:
        # download in batches
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
            data = _get_data(url_query, params, timeout=timeout, **kwargs)
            features.extend(data["features"])
    else:
        # download all data in one go
        data = _get_data(url_query, params, timeout=timeout, **kwargs)
        features = data["features"]
    if f == "json" or f == "pjson":
        # Interpret the geometry field
        data = []
        for feature in features:
            if "rings" in feature["geometry"]:
                if len(feature["geometry"]) > 1:
                    raise (NotImplementedError("Multiple rings not supported yet"))
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
            elif (
                len(feature["geometry"]) == 2
                and "x" in feature["geometry"]
                and "y" in feature["geometry"]
            ):
                geometry = Point(feature["geometry"]["x"], feature["geometry"]["y"])
            else:
                raise (Exception("Not supported yet"))
            feature["attributes"]["geometry"] = geometry
            data.append(feature["attributes"])
        if len(data) == 0:
            # Assigning CRS to a GeoDataFrame without a geometry column is not supported
            gdf = gpd.GeoDataFrame()
        else:
            gdf = gpd.GeoDataFrame(data, crs=sr)
    else:
        # for geojson-data we can transform to GeoDataFrame right away
        if len(features) == 0:
            # Assigning CRS to a GeoDataFrame without a geometry column is not supported
            gdf = gpd.GeoDataFrame()
        else:
            gdf = gpd.GeoDataFrame.from_features(features, crs=sr)
    return gdf


def _get_data(url, params, timeout=120, **kwargs):
    r = requests.get(url, params=params, timeout=timeout, **kwargs)
    if not r.ok:
        raise (HTTPError(f"Request not successful: {r.url}"))
    data = r.json()
    if "error" in data:
        code = data["error"]["code"]
        message = data["error"]["message"]
        raise (Exception(f"Error code {code}: {message}"))
    return data


def wfs(
    url,
    layer,
    extent=None,
    version="2.0.0",
    paged=True,
    max_record_count=None,
    driver="GML",
    timeout=120,
):
    """Download data from a wfs server."""
    params = {"version": version, "request": "GetFeature"}
    if version == "2.0.0":
        params["typeNames"] = layer
    else:
        params["typeName"] = layer
    if extent is not None:
        params["bbox"] = f"{extent[0]},{extent[2]},{extent[1]},{extent[3]}"
    if paged:
        # wfs = WebFeatureService(url)
        # get the maximum number of features
        r = requests.get(f"{url}&request=getcapabilities", timeout=120)
        if not r.ok:
            raise (HTTPError(f"Request not successful: {r.url}"))
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

        if "CountDefault" not in constraints:
            logger.info("Cannot find CountDefault. Setting CountDefault to inf")
            constraints["CountDefault"] = np.inf
        if max_record_count is None:
            max_record_count = constraints["CountDefault"]
        else:
            max_record_count = min(max_record_count, constraints["CountDefault"])

        # get the number of features
        params["resultType"] = "hits"
        r = requests.get(url, params=params, timeout=timeout)
        if not r.ok:
            raise (HTTPError(f"Request not successful: {r.url}"))
        params.pop("resultType")
        root = ET.fromstring(r.text)
        if "ExceptionReport" in root.tag:
            raise Exception(root[0].attrib)
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
            r = requests.get(url, params=params, timeout=timeout)
            if not r.ok:
                raise (HTTPError(f"Request not successful: {r.url}"))
            gdfs.append(gpd.read_file(BytesIO(r.content), driver=driver))
        gdf = pd.concat(gdfs).reset_index(drop=True)
    else:
        # download all features in one go
        r = requests.get(url, params=params, timeout=timeout)
        if not r.ok:
            raise (HTTPError(f"Request not successful: {r.url}"))
        gdf = gpd.read_file(BytesIO(r.content), driver=driver)

    return gdf


def wcs(
    url,
    extent,
    res,
    identifier=None,
    version="1.0.0",
    fmt="GEOTIFF_FLOAT32",
    crs="EPSG:28992",
    maxsize=2000,
):
    """Download data from a web coverage service (WCS), return a MemoryFile.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    res : float, optional
        resolution of wcs raster
    url : str
        webservice url.
    identifier : str
        identifier.
    version : str
        version of wcs service, options are '1.0.0' and '2.0.1'.
    fmt : str, optional
        geotif format
    crs : str, optional
        coördinate reference system

    Raises
    ------
    Exception
        wrong version

    Returns
    -------
    memfile : rasterio.io.MemoryFile
        MemoryFile.
    """
    # check if wcs is within limits
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]

    # check if size exceeds maxsize
    if (dx / res) > maxsize:
        x_segments = int(np.ceil((dx / res) / maxsize))
    else:
        x_segments = 1

    if (dy / res) > maxsize:
        y_segments = int(np.ceil((dy / res) / maxsize))
    else:
        y_segments = 1

    if (x_segments * y_segments) > 1:
        st = f"""requested wcs raster width or height bigger than {maxsize*res}
            -> splitting extent into {x_segments} * {y_segments} tiles"""
        logger.info(st)
        memfile = _split_wcs_extent(
            extent,
            x_segments,
            y_segments,
            maxsize,
            res,
            url,
            identifier,
            version,
            fmt,
            crs,
        )
        da = rioxarray.open_rasterio(memfile.open(), mask_and_scale=True)[0]
    else:
        memfile = _download_wcs(extent, res, url, identifier, version, fmt, crs)
        da = rioxarray.open_rasterio(memfile.open(), mask_and_scale=True)[0]
        # load the data from memfile otherwise lazy loading of xarray causes problems
        da.load()

    return da


def _split_wcs_extent(
    extent,
    x_segments,
    y_segments,
    maxsize,
    res,
    url,
    identifier,
    version,
    fmt,
    crs,
):
    """There is a max height and width limit for the wcs server. This function
    splits your extent in chunks smaller than the limit. It returns a list of
    Memory files.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    res : float
        The resolution of the requested output-data
    x_segments : int
        number of tiles on the x axis
    y_segments : int
        number of tiles on the y axis
    maxsize : int or float
        maximum widht or height of wcs tile

    Returns
    -------
    MemoryFile
        Rasterio MemoryFile of the merged data
    Notes
    -----
    1. The resolution is used to obtain the data from the wcs server. Not sure
    what kind of interpolation is used to resample the original grid.
    """

    # write tiles
    datasets = []
    start_x = extent[0]
    pbar = tqdm(total=x_segments * y_segments)
    for tx in range(x_segments):
        if (tx + 1) == x_segments:
            end_x = extent[1]
        else:
            end_x = start_x + maxsize * res
        start_y = extent[2]
        for ty in range(y_segments):
            if (ty + 1) == y_segments:
                end_y = extent[3]
            else:
                end_y = start_y + maxsize * res
            subextent = [start_x, end_x, start_y, end_y]
            logger.debug(
                f"segment x {tx+1} of {x_segments}, segment y {ty+1} of {y_segments}"
            )

            memfile = _download_wcs(subextent, res, url, identifier, version, fmt, crs)

            datasets.append(memfile)
            start_y = end_y
            pbar.update(1)

        start_x = end_x

    pbar.close()
    memfile = MemoryFile()
    merge.merge([b.open() for b in datasets], dst_path=memfile)

    return memfile


def _download_wcs(extent, res, url, identifier, version, fmt, crs):
    """Download the wcs-data, return a MemoryFile.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent
    res : float, optional
        resolution of wcs raster
    url : str
        webservice url.
    identifier : str
        identifier.
    version : str
        version of wcs service, options are '1.0.0' and '2.0.1'.
    fmt : str, optional
        geotif format
    crs : str, optional
        coördinate reference system

    Raises
    ------
    Exception
        wrong version

    Returns
    -------
    memfile : rasterio.io.MemoryFile
        MemoryFile.
    """
    # download file
    logger.debug(
        f"- download wcs between: x ({str(extent[0])}, {str(extent[1])}); "
        f"y ({str(extent[2])}, {str(extent[3])})"
    )
    wcs = WebCoverageService(url, version=version)
    if identifier is None:
        identifiers = list(wcs.contents)
        if len(identifiers) > 1:
            raise (ValueError("wcs contains more than 1 identifier. Please specify."))
        identifier = identifiers[0]
    if version == "1.0.0":
        bbox = (extent[0], extent[2], extent[1], extent[3])
        output = wcs.getCoverage(
            identifier=identifier,
            bbox=bbox,
            format=fmt,
            crs=crs,
            resx=res,
            resy=res,
        )
    elif version == "2.0.1":
        # bbox, resx and resy do nothing in version 2.0.1
        subsets = [("x", extent[0], extent[1]), ("y", extent[2], extent[3])]
        output = wcs.getCoverage(
            identifier=[identifier], subsets=subsets, format=fmt, crs=crs
        )
    else:
        raise NotImplementedError(f"Version {version} not yet supported")
    if "xml" in output.info()["Content-Type"]:
        root = ET.fromstring(output.read())
        raise (Exception(f"Download failed: {root[0].text}"))
    memfile = MemoryFile(output.read())
    return memfile
