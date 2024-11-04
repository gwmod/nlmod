import json
import time
from io import BytesIO
from typing import Dict
from xml.etree import ElementTree
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from ..util import extent_to_polygon


def get_bgt(
    extent,
    layer="waterdeel",
    cut_by_extent=True,
    make_valid=False,
    fname=None,
    geometry=None,
    remove_expired=True,
):
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
    make_valid : bool, optional
        Make geometries valid by appying a buffer of 0 m when True. THe defaults is
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

    zipfile = BytesIO(response.content)
    gdf = read_bgt_zipfile(
        zipfile,
        geometry=geometry,
        cut_by_extent=cut_by_extent,
        make_valid=make_valid,
        extent=polygon,
        remove_expired=remove_expired,
    )

    if len(layer) == 1:
        gdf = gdf[layer[0]]
    bgt_bronhouder_names = get_bronhouder_names()
    gdf["bronhouder_name"] = gdf["bronhouder"].map(bgt_bronhouder_names)

    return gdf


def read_bgt_zipfile(
    fname,
    geometry=None,
    files=None,
    cut_by_extent=True,
    make_valid=False,
    extent=None,
    remove_expired=True,
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

        if make_valid:
            gdf[key].geometry = gdf[key].geometry.buffer(0.0)

        if cut_by_extent and isinstance(gdf[key], gpd.GeoDataFrame):
            gdf[key].geometry = gdf[key].intersection(polygon)
            gdf[key] = gdf[key][~gdf[key].is_empty]

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


def get_bgt_layers():
    url = "https://api.pdok.nl/lv/bgt/download/v1_0/dataset"
    resp = requests.get(url, timeout=1200)  # 20 minutes
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
        file from 2024.
    """
    bgt_bronhouder_names = {
        "G0003": "Appingedam",
        "G0005": "Bedum",
        "G0007": "Bellingwedde",
        "G0009": "Ten Boer",
        "G0010": "Delfzijl",
        "G0014": "Groningen",
        "G0015": "Grootegast",
        "G0017": "Haren",
        "G0018": "Hoogezand-Sappemeer",
        "G0022": "Leek",
        "G0024": "Loppersum",
        "G0025": "Marum",
        "G0034": "Almere",
        "G0037": "Stadskanaal",
        "G0040": "Slochteren",
        "G0047": "Veendam",
        "G0048": "Vlagtwedde",
        "G0050": "Zeewolde",
        "G0051": "Skarsterlân",
        "G0053": "Winsum",
        "G0055": "Boarnsterhim",
        "G0056": "Zuidhorn",
        "G0058": "Dongeradeel",
        "G0059": "Achtkarspelen",
        "G0060": "Ameland",
        "G0063": "het Bildt",
        "G0070": "Franekeradeel",
        "G0072": "Harlingen",
        "G0074": "Heerenveen",
        "G0079": "Kollumerland en Nieuwkruisland",
        "G0080": "Leeuwarden",
        "G0081": "Leeuwarderadeel",
        "G0082": "Lemsterland",
        "G0085": "Ooststellingwerf",
        "G0086": "Opsterland",
        "G0088": "Schiermonnikoog",
        "G0090": "Smallingerland",
        "G0093": "Terschelling",
        "G0096": "Vlieland",
        "G0098": "Weststellingwerf",
        "G0106": "Assen",
        "G0109": "Coevorden",
        "G0114": "Emmen",
        "G0118": "Hoogeveen",
        "G0119": "Meppel",
        "G0140": "Littenseradiel",
        "G0141": "Almelo",
        "G0147": "Borne",
        "G0148": "Dalfsen",
        "G0150": "Deventer",
        "G0153": "Enschede",
        "G0158": "Haaksbergen",
        "G0160": "Hardenberg",
        "G0163": "Hellendoorn",
        "G0164": "Hengelo (O)",
        "G0166": "Kampen",
        "G0168": "Losser",
        "G0171": "Noordoostpolder",
        "G0173": "Oldenzaal",
        "G0175": "Ommen",
        "G0177": "Raalte",
        "G0180": "Staphorst",
        "G0183": "Tubbergen",
        "G0184": "Urk",
        "G0189": "Wierden",
        "G0193": "Zwolle",
        "G0196": "Rijnwaarden",
        "G0197": "Aalten",
        "G1924": "Goeree-Overflakkee",
        "G1927": "Molenwaard",
        "G0200": "Apeldoorn",
        "G0202": "Arnhem",
        "G0203": "Barneveld",
        "G0209": "Beuningen",
        "G0213": "Brummen",
        "G0214": "Buren",
        "G0216": "Culemborg",
        "G0221": "Doesburg",
        "G0222": "Doetinchem",
        "G0225": "Druten",
        "G0226": "Duiven",
        "G0228": "Ede",
        "G0230": "Elburg",
        "G0232": "Epe",
        "G0233": "Ermelo",
        "G0236": "Geldermalsen",
        "G0241": "Groesbeek",
        "G0243": "Harderwijk",
        "G0244": "Hattem",
        "G0246": "Heerde",
        "G0252": "Heumen",
        "G0262": "Lochem",
        "G0263": "Maasdriel",
        "G0265": "Millingen aan de Rijn",
        "G0267": "Nijkerk",
        "G0268": "Nijmegen",
        "G0269": "Oldebroek",
        "G0273": "Putten",
        "G0274": "Renkum",
        "G0275": "Rheden",
        "G0277": "Rozendaal",
        "G0279": "Scherpenzeel",
        "G0281": "Tiel",
        "G0282": "Ubbergen",
        "G0285": "Voorst",
        "G0289": "Wageningen",
        "G0293": "Westervoort",
        "G0294": "Winterswijk",
        "G0296": "Wijchen",
        "G0297": "Zaltbommel",
        "G0299": "Zevenaar",
        "G0301": "Zutphen",
        "G0302": "Nunspeet",
        "G0303": "Dronten",
        "G0304": "Neerijnen",
        "G0307": "Amersfoort",
        "G0308": "Baarn",
        "G0310": "De Bilt",
        "G0312": "Bunnik",
        "G0313": "Bunschoten",
        "G0317": "Eemnes",
        "G0321": "Houten",
        "G0327": "Leusden",
        "G0331": "Lopik",
        "G0335": "Montfoort",
        "G0339": "Renswoude",
        "G0340": "Rhenen",
        "G0342": "Soest",
        "G0344": "Utrecht",
        "G0345": "Veenendaal",
        "G0351": "Woudenberg",
        "G0352": "Wijk bij Duurstede",
        "G0353": "IJsselstein",
        "G0355": "Zeist",
        "G0356": "Nieuwegein",
        "G0358": "Aalsmeer",
        "G0361": "Alkmaar",
        "G0362": "Amstelveen",
        "G0363": "Amsterdam",
        "G0365": "Graft-De Rijp",
        "G0370": "Beemster",
        "G0373": "Bergen (NH)",
        "G0375": "Beverwijk",
        "G0376": "Blaricum",
        "G0377": "Bloemendaal",
        "G0381": "Bussum",
        "G0383": "Castricum",
        "G0384": "Diemen",
        "G0385": "Edam-Volendam",
        "G0388": "Enkhuizen",
        "G0392": "Haarlem",
        "G0393": "Haarlemmerliede en Spaarnwoude",
        "G0394": "Haarlemmermeer",
        "G0395": "Harenkarspel",
        "G0396": "Heemskerk",
        "G0397": "Heemstede",
        "G0398": "Heerhugowaard",
        "G0399": "Heiloo",
        "G0400": "Den Helder",
        "G0402": "Hilversum",
        "G0405": "Hoorn",
        "G0406": "Huizen",
        "G0415": "Landsmeer",
        "G0416": "Langedijk",
        "G0417": "Laren",
        "G0420": "Medemblik",
        "G0424": "Muiden",
        "G0425": "Naarden",
        "G0431": "Oostzaan",
        "G0432": "Opmeer",
        "G0437": "Ouder-Amstel",
        "G0439": "Purmerend",
        "G0441": "Schagen",
        "G0448": "Texel",
        "G0450": "Uitgeest",
        "G0451": "Uithoorn",
        "G0453": "Velsen",
        "G0457": "Weesp",
        "G0458": "Schermer",
        "G0473": "Zandvoort",
        "G0476": "Zijpe",
        "G0478": "Zeevang",
        "G0479": "Zaanstad",
        "G0482": "Alblasserdam",
        "G0484": "Alphen aan den Rijn",
        "G0489": "Barendrecht",
        "G0491": "Bergambacht",
        "G0498": "Drechterland",
        "G0499": "Boskoop",
        "G0501": "Brielle",
        "G0502": "Capelle aan den IJssel",
        "G0503": "Delft",
        "G0504": "Dirksland",
        "G0505": "Dordrecht",
        "G0511": "Goedereede",
        "G0512": "Gorinchem",
        "G0513": "Gouda",
        "G0518": "'s-Gravenhage",
        "G0523": "Hardinxveld-Giessendam",
        "G0530": "Hellevoetsluis",
        "G0531": "Hendrik-Ido-Ambacht",
        "G0532": "Stede Broec",
        "G0534": "Hillegom",
        "G0537": "Katwijk",
        "G0542": "Krimpen aan den IJssel",
        "G0545": "Leerdam",
        "G0546": "Leiden",
        "G0547": "Leiderdorp",
        "G0553": "Lisse",
        "G0556": "Maassluis",
        "G0559": "Middelharnis",
        "G0568": "Bernisse",
        "G0569": "Nieuwkoop",
        "G0571": "Nieuw-Lekkerland",
        "G0575": "Noordwijk",
        "G0576": "Noordwijkerhout",
        "G0579": "Oegstgeest",
        "G0580": "Oostflakkee",
        "G0584": "Oud-Beijerland",
        "G0585": "Binnenmaas",
        "G0588": "Korendijk",
        "G0589": "Oudewater",
        "G0590": "Papendrecht",
        "G0597": "Ridderkerk",
        "G0599": "Rotterdam",
        "G0603": "Rijswijk",
        "G0606": "Schiedam",
        "G0608": "Schoonhoven",
        "G0610": "Sliedrecht",
        "G0611": "Cromstrijen",
        "G0612": "Spijkenisse",
        "G0613": "Albrandswaard",
        "G0614": "Westvoorne",
        "G0617": "Strijen",
        "G0620": "Vianen",
        "G0622": "Vlaardingen",
        "G0623": "Vlist",
        "G0626": "Voorschoten",
        "G0627": "Waddinxveen",
        "G0629": "Wassenaar",
        "G0632": "Woerden",
        "G0637": "Zoetermeer",
        "G0638": "Zoeterwoude",
        "G0642": "Zwijndrecht",
        "G0643": "Nederlek",
        "G0644": "Ouderkerk",
        "G0653": "Gaasterlân-Sleat",
        "G0654": "Borsele",
        "G0664": "Goes",
        "G0668": "West Maas en Waal",
        "G0677": "Hulst",
        "G0678": "Kapelle",
        "G0687": "Middelburg",
        "G0689": "Giessenlanden",
        "G0693": "Graafstroom",
        "G0694": "Liesveld",
        "G0703": "Reimerswaal",
        "G0707": "Zederik",
        "G0715": "Terneuzen",
        "G0716": "Tholen",
        "G0717": "Veere",
        "G0718": "Vlissingen",
        "G0733": "Lingewaal",
        "G0736": "De Ronde Venen",
        "G0737": "Tytsjerksteradiel",
        "G0738": "Aalburg",
        "G0743": "Asten",
        "G0744": "Baarle-Nassau",
        "G0748": "Bergen op Zoom",
        "G0753": "Best",
        "G0755": "Boekel",
        "G0756": "Boxmeer",
        "G0757": "Boxtel",
        "G0758": "Breda",
        "G0762": "Deurne",
        "G0765": "Pekela",
        "G0766": "Dongen",
        "G0770": "Eersel",
        "G0772": "Eindhoven",
        "G0777": "Etten-Leur",
        "G0779": "Geertruidenberg",
        "G0784": "Gilze en Rijen",
        "G0785": "Goirle",
        "G0786": "Grave",
        "G0788": "Haaren",
        "G0794": "Helmond",
        "G0796": "'s-Hertogenbosch",
        "G0797": "Heusden",
        "G0798": "Hilvarenbeek",
        "G0809": "Loon op Zand",
        "G0815": "Mill en Sint Hubert",
        "G0820": "Nuenen, Gerwen en Nederwetten",
        "G0823": "Oirschot",
        "G0824": "Oisterwijk",
        "G0826": "Oosterhout",
        "G0828": "Oss",
        "G0840": "Rucphen",
        "G0844": "Schijndel",
        "G0845": "Sint-Michielsgestel",
        "G0846": "Sint-Oedenrode",
        "G0847": "Someren",
        "G0848": "Son en Breugel",
        "G0851": "Steenbergen",
        "G0852": "Waterland",
        "G0855": "Tilburg",
        "G0856": "Uden",
        "G0858": "Valkenswaard",
        "G0860": "Veghel",
        "G0861": "Veldhoven",
        "G0865": "Vught",
        "G0866": "Waalre",
        "G0867": "Waalwijk",
        "G0870": "Werkendam",
        "G0873": "Woensdrecht",
        "G0874": "Woudrichem",
        "G0879": "Zundert",
        "G0880": "Wormerland",
        "G0881": "Onderbanken",
        "G0882": "Landgraaf",
        "G0888": "Beek",
        "G0889": "Beesel",
        "G0893": "Bergen (L)",
        "G0899": "Brunssum",
        "G0907": "Gennep",
        "G0917": "Heerlen",
        "G0928": "Kerkrade",
        "G0935": "Maastricht",
        "G0938": "Meerssen",
        "G0944": "Mook en Middelaar",
        "G0946": "Nederweert",
        "G0951": "Nuth",
        "G0957": "Roermond",
        "G0962": "Schinnen",
        "G0965": "Simpelveld",
        "G0971": "Stein",
        "G0981": "Vaals",
        "G0983": "Venlo",
        "G0984": "Venray",
        "G0986": "Voerendaal",
        "G0988": "Weert",
        "G0994": "Valkenburg aan de Geul",
        "G0995": "Lelystad",
        "G1507": "Horst aan de Maas",
        "G1509": "Oude IJsselstreek",
        "G1525": "Teylingen",
        "G1581": "Utrechtse Heuvelrug",
        "G1586": "Oost Gelre",
        "G1598": "Koggenland",
        "G1621": "Lansingerland",
        "G1640": "Leudal",
        "G1641": "Maasgouw",
        "G1651": "Eemsmond",
        "G1652": "Gemert-Bakel",
        "G1655": "Halderberge",
        "G1658": "Heeze-Leende",
        "G1659": "Laarbeek",
        "G1663": "De Marne",
        "G1667": "Reusel-De Mierden",
        "G1669": "Roerdalen",
        "G1671": "Maasdonk",
        "G1672": "Rijnwoude",
        "G1674": "Roosendaal",
        "G1676": "Schouwen-Duiveland",
        "G1680": "Aa en Hunze",
        "G1681": "Borger-Odoorn",
        "G1684": "Cuijk",
        "G1685": "Landerd",
        "G1690": "De Wolden",
        "G1695": "Noord-Beveland",
        "G1696": "Wijdemeren",
        "G1699": "Noordenveld",
        "G1700": "Twenterand",
        "G1701": "Westerveld",
        "G1702": "Sint Anthonis",
        "G1705": "Lingewaard",
        "G1706": "Cranendonck",
        "G1708": "Steenwijkerland",
        "G1709": "Moerdijk",
        "G1711": "Echt-Susteren",
        "G1714": "Sluis",
        "G1719": "Drimmelen",
        "G1721": "Bernheze",
        "G1722": "Ferwerderadiel",
        "G1723": "Alphen-Chaam",
        "G1724": "Bergeijk",
        "G1728": "Bladel",
        "G1729": "Gulpen-Wittem",
        "G1730": "Tynaarlo",
        "G1731": "Midden-Drenthe",
        "G1734": "Overbetuwe",
        "G1735": "Hof van Twente",
        "G1740": "Neder-Betuwe",
        "G1742": "Rijssen-Holten",
        "G1771": "Geldrop-Mierlo",
        "G1773": "Olst-Wijhe",
        "G1774": "Dinkelland",
        "G1783": "Westland",
        "G1842": "Midden-Delfland",
        "G1859": "Berkelland",
        "G1876": "Bronckhorst",
        "G1883": "Sittard-Geleen",
        "G1884": "Kaag en Braassem",
        "G1891": "Dantumadiel",
        "G1892": "Zuidplas",
        "G1894": "Peel en Maas",
        "G1895": "Oldambt",
        "G1896": "Zwartewaterland",
        "G1900": "Súdwest-Fryslân",
        "G1901": "Bodegraven-Reeuwijk",
        "G1903": "Eijsden-Margraten",
        "G1904": "Stichtse Vecht",
        "G1908": "Menameradiel",
        "G1911": "Hollands Kroon",
        "G1916": "Leidschendam-Voorburg",
        "G1921": "De Friese Meren",
        "G1926": "Pijnacker-Nootdorp",
        "G1955": "Montferland",
        "G1987": "Menterwolde",
        "G1930": "Nissewaard",
        "G1931": "Krimpenerwaard",
        "G1940": "De Fryske Marren",
        "G1942": "Gooise Meren",
        "G1945": "Berg en Dal",
        "G1948": "Meierijstad",
        "G1949": "Waadhoeke",
        "G1950": "Westerwolde",
        "G1952": "Midden-Groningen",
        "G1954": "Beekdaelen",
        "G1959": "Altena",
        "G1960": "West Betuwe",
        "G1961": "Vijfheerenlanden",
        "G1963": "Hoeksche Waard",
        "G1966": "Het Hogeland",
        "G1969": "Westerkwartier",
        "G1970": "Noardeast-Fryslân",
        "G1978": "Molenlanden",
        "G1979": "Eemsdelta",
        "G1980": "Dijk en Waard",
        "G1982": "Land van Cuijk",
        "G1991": "Maashorst",
        "G1992": "Voorne aan Zee",
        "P0020": "Groningen",
        "P0021": "Fryslân",
        "P0022": "Drenthe",
        "P0023": "Overijssel",
        "P0024": "Flevoland",
        "P0025": "Gelderland",
        "P0026": "Utrecht",
        "P0027": "Noord-Holland",
        "P0028": "Zuid-Holland",
        "P0029": "Zeeland",
        "P0030": "Noord-Brabant",
        "P0031": "Limburg",
        "W0151": "Waterschap Groot Salland",
        "W0152": "Waterschap Rijn en IJssel",
        "W0153": "Waterschap Veluwe",
        "W0154": "Waterschap Vallei & Eem",
        "W0155": "Waterschap Amstel, Gooi en Vecht",
        "W0201": "Waterschap Regge en Dinkel",
        "W0372": "Hoogheemraadschap van Delfland",
        "W0524": "Waterschap Zeeuws-Vlaanderen",
        "W0539": "Waterschap De Dommel",
        "W0585": "Waterschap Roer en Overmaas",
        "W0621": "Waterschap Rivierenland",
        "W0616": "Hoogheemraadschap van Rijnland",
        "W0636": "Hoogheemraadschap De Stichtse Rijnlanden",
        "W0638": "Waterschap Peel en Maasvallei",
        "W0646": "Waterschap Hunze en Aa's",
        "W0647": "Waterschap Noorderzijlvest",
        "W0648": "Waterschap Reest en Wieden",
        "W0649": "Waterschap Velt en Vecht",
        "W0650": "Waterschap Zuiderzeeland",
        "W0651": "Hoogheemraadschap Hollands Noorderkwartier",
        "W0652": "Waterschap Brabantse Delta",
        "W0653": "Wetterskip Fryslân",
        "W0654": "Waterschap Aa en Maas",
        "W0655": "Waterschap Hollandse Delta",
        "W0656": "Hoogheemraadschap van Schieland en de Krimpenerwaard",
        "W0659": "Waterschapsbedrijf Limburg",
        "W0661": "Waterschap Scheldestromen",
        "W0662": "Waterschap Vallei en Veluwe",
        "W0663": "Waterschap Vechtstromen",
        "W0664": "Waterschap Drents Overijsselse Delta",
        "W0665": "Waterschap Limburg",
        "L0001": "Ministerie van Landbouw, Natuur en Voedselkwaliteit (LNV)",
        "L0002": "Ministerie van Infrastructuur en Waterstaat (IenW)",
        "L0003": "Ministerie van Defensie",
        "L0004": "Prorail",
        "S0001": "Samenwerkingsverband voor Bronhouders",
        "K0001": "Kadaster",
    }

    return bgt_bronhouder_names
