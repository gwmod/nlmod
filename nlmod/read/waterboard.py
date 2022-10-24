import numpy as np
import logging
from . import webservices

logger = logging.getLogger(__name__)


def get_polygons(**kwargs):
    """Get the location of the Waterboards as a Polygon GeoDataFrame"""
    url = "https://services.arcgis.com/nSZVuSZjHpEZZbRo/arcgis/rest/services/Waterschapsgrenzen/FeatureServer"
    layer = 0
    ws = webservices.arcrest(url, layer, **kwargs)
    # remove different prefixes
    ws["waterschap"] = ws["waterschap"].str.replace("HH van ", "")
    ws["waterschap"] = ws["waterschap"].str.replace("HHS van ", "")
    ws["waterschap"] = ws["waterschap"].str.replace("HH ", "")
    ws["waterschap"] = ws["waterschap"].str.replace("Waterschap ", "")
    ws["waterschap"] = ws["waterschap"].str.replace("Wetterskip ", "")
    ws = ws.set_index("waterschap")

    return ws


def get_configuration():
    """Get the configuration of of the data sources of the Waterboards"""
    config = {}

    config["Aa en Maas"] = {
        "bgt_code": "W0654",
        "watercourses": {
            "url": "https://gisservices.aaenmaas.nl/arcgis/rest/services/EXTERN/Oppervlaktewater_L/MapServer",
            "layer": 8,
            "bottom_width": "BODEMBREEDTE",
            "bottom_height": [["BODEMHOOGTE_BOS", "BODEMHOOGTE_BES"]],
        },
        "level_areas": {
            # "server_kind": "wfs",
            # "url": "https://maps.aaenmaas.nl/services/DAMO_S/wfs?",
            # "layer": "WS_PEILGEBIED",
            "url": "https://gisservices.aaenmaas.nl/arcgis/rest/services/EXTERN/Oppervlaktewater_B/MapServer",
            "layer": 5,
            "summer_stage": "ZOMERPEIL",
            "winter_stage": "WINTERPEIL",
        },
    }

    config["Amstel, Gooi en Vecht"] = {
        "bgt_code": "W0155",
        "watercourses": {
            "url": "https://maps.waternet.nl/arcgis/rest/services/AGV_Legger/AGV_Onderh_Secundaire_Watergangen/MapServer",
            "layer": 40,
            "bottom_width": "BODEMBREEDTE",
            "bottom_height": "BODEMHOOGTE",
            "water_depth": "WATERDIEPTE",
        },
        "level_areas": {
            "url": "https://maps.waternet.nl/arcgis/rest/services/AGV_Legger/Vastgestelde_Waterpeilen/MapServer",
            "layer": 209,
            "summer_stage": [
                "ZOMERPEIL",
                "FLEXIBEL_ZOMERPEIL_BOVENGR",
                "VAST_PEIL",
            ],
            "winter_stage": [
                "WINTERPEIL",
                "FLEXIBEL_WINTERPEIL_BOVENGR",
                "VAST_PEIL",
            ],
        },
    }

    config["Brabantse Delta"] = {
        "bgt_code": "W0652",
        "watercourses": {
            # legger
            "url": "https://geoservices.brabantsedelta.nl/arcgis/rest/services/EXTERN/WEB_Vastgestelde_Legger_Oppervlaktewaterlichamen/FeatureServer",
            "layer": 11,  # categorie A
            # "layer": 12,  # categorie B
            # beheer
            # "url": "https://geoservices.brabantsedelta.nl/arcgis/rest/services/EXTERN/WEB_Beheerregister_Waterlopen_en_Kunstwerken/FeatureServer",
            # "layer": 13,  # categorie A
            # "layer": 14,  # categorie B
            # "layer": 15,  # categorie C
        },
        "level_areas": {
            # "url": "https://geoservices.brabantsedelta.nl/arcgis/rest/services/EXTERN/WEB_Beheerregister_Waterlopen_en_Kunstwerken/FeatureServer",
            # "layer": 19,
            "url": "https://maps.brabantsedelta.nl/arcgis/rest/services/Extern/Legger/MapServer",
            "layer": 6,
            "summer_stage": [
                "WS_ZOMERPEIL",
                "WS_VAST_PEIL",
                "WS_STREEFPEIL",
                "WS_MAXIMUM_PEIL",
                "WS_MINIMUM_PEIL",
            ],
            "winter_stage": [
                "WS_WINTERPEIL",
                "WS_VAST_PEIL",
                "WS_STREEFPEIL",
                "WS_MINIMUM_PEIL",
                "WS_MAXIMUM_PEIL",
            ],
        },
    }

    config["De Dommel"] = {
        "bgt_code": "W0539",
        "watercourses": {
            "url": "https://services8.arcgis.com/dmR647kStmcYa6EN/arcgis/rest/services/LW_2021_20211110/FeatureServer",
            "layer": 9,  # LOW_2021_A_Water
            # "layer": 10,  # LOW_2021_A_Water_Afw_Afv
            # "layer": 11,  # LOW_2021_B_Water
            # "layer": 2,  # LOW_2021_Profielpunt
            # "layer": 13,  # LOW_2021_Profiellijn
            # "index": "WS_PROFIELID",
        },
    }

    config["De Stichtse Rijnlanden"] = {
        "bgt_code": "W0636",
        "watercourses": {
            "url": "https://services1.arcgis.com/1lWKHMyUIR3eKHKD/ArcGIS/rest/services/Keur_2020/FeatureServer",
            "layer": 39,  # Leggervak
            # "layer": 43, # Leggervak droge sloot
        },
        "level_areas": {
            "url": "https://geoservices.hdsr.nl/arcgis/rest/services/Extern/PeilbesluitenExtern/FeatureServer",
            "layer": 1,
            "index": "WS_PGID",
            "summer_stage": ["WS_ZP", "WS_BP", "WS_OP", "WS_VP"],
            "winter_stage": ["WS_WP", "WS_OP", "WS_BP", "WS_VP"],
        },
    }

    config["Delfland"] = {
        "bgt_code": "W0372",
        "watercourses": {
            "url": "https://services.arcgis.com/f6rHQPZpXXOzhDXU/arcgis/rest/services/Leggerkaart_Delfland_definitief/FeatureServer",
            "layer": 39,  # primair
            # "layer": 40,  # secundair
        },
        "level_areas": {
            "url": "https://services.arcgis.com/f6rHQPZpXXOzhDXU/arcgis/rest/services/Peilbesluiten2/FeatureServer",
            "summer_stage": "WS_HOOGPEIL",
            "winter_stage": "WS_LAAGPEIL",
        },
        "level_deviations": {
            "url": "https://services.arcgis.com/f6rHQPZpXXOzhDXU/arcgis/rest/services/Peilbesluiten2/FeatureServer",
            "layer": 2,
        },
    }

    config["Drents Overijsselse Delta"] = {
        "bgt_code": "W0664",
        "watercourses": {
            "url": "https://services6.arcgis.com/BZiPrSbS4NknjGsQ/arcgis/rest/services/Primaire_watergang_20_3_2018/FeatureServer",
            "index": "OVKIDENT",
        },
        "level_areas": {
            "url": "https://services6.arcgis.com/BZiPrSbS4NknjGsQ/arcgis/rest/services/Peilgebieden_opendata/FeatureServer",
            "index": "GPGIDENT",
            "summer_stage": "GPGZMRPL",
            "winter_stage": "GPGWNTPL",
        },
    }

    config["Fryslân"] = {
        "bgt_code": "W0653",
        "watercourses": {
            "url": "https://gis.wetterskipfryslan.nl/arcgis/rest/services/BeheerregisterWaterlopen/MapServer",
            "layer": 0,  # # Wateren (primair, secundair)
            "index": "OVKIDENT",
            "bottom_height": "AVVBODH",
            "water_depth": "AVVDIEPT",
            # "url": "https://gis.wetterskipfryslan.nl/arcgis/rest/services/Legger_vastgesteld__2019/MapServer",
            # "layer": 604,  # Wateren legger
            # "index": "BLAEU.LEG_VL_GW_OVK.OVKIDENT",
            # "bottom_height": "BLAEU.LEG_VL_GW_OVK.AVVBODH",
        },
        "level_areas": {
            # "url": "https://gis.wetterskipfryslan.nl/arcgis/rest/services/Peilbelsuit_Friese_boezem/MapServer",
            # "index": "BLAEU_WFG_GPG_BEHEER_PBHIDENT",
            "url": "https://gis.wetterskipfryslan.nl/arcgis/rest/services/Peilen/MapServer",
            "layer": 1,  # PeilenPeilenbeheerkaart - Peilen
            "index": "PBHIDENT",
            # "layer": 4,  # Peilbesluitenkaart
            # "index": "GPGIDENT",
            "summer_stage": "HOOGPEIL",
            "winter_stage": "LAAGPEIL",
        },
    }

    config["Hollands Noorderkwartier"] = {
        "bgt_code": "W0651",
        "watercourses": {
            "url": "https://kaarten.hhnk.nl/arcgis/rest/services/od_legger/od_legger_wateren_2022_oppervlaktewateren_ti/MapServer",
            "bottom_height": "WS_BODEMHOOGTE",
        },
        "level_areas": {
            "url": "https://kaarten.hhnk.nl/arcgis/rest/services/NHFLO/Peilgebied_beheerregister/MapServer",
            "summer_stage": [
                "ZOMER",
                "STREEFPEIL_ZOMER",
                "BOVENGRENS_JAARROND",
                "ONDERGRENS_JAARROND",
                "VAST",
                "STREEFPEIL_JAARROND",
            ],
            "winter_stage": [
                "WINTER",
                "STREEFPEIL_WINTER",
                "ONDERGRENS_JAARROND",
                "BOVENGRENS_JAARROND",
                "VAST",
                "STREEFPEIL_JAARROND",
            ],
        },
        "level_deviations": {
            "url": "https://kaarten.hhnk.nl/arcgis/rest/services/NHFLO/Peilafwijking_gebied/MapServer"
        },
    }

    config["Hollandse Delta"] = {
        "bgt_code": "W0655",
        "watercourses": {
            "url": "https://geoportaal.wshd.nl/arcgis/rest/services/Geoportaal/Legger2014waterbeheersing_F_transparant/FeatureServer",
        },
        "level_areas": {
            "url": "https://geoportaal.wshd.nl/arcgis/rest/services/Watersysteem/Peilgebieden/MapServer",
            "layer": 31,
            "f": "json",
            "summer_stage": [
                "REKENPEIL_ZOMER",
                "BOVENGRENS_BEHEERMARGE_ZOMER",
                "ONDERGRENS_BEHEERMARGE_ZOMER",
            ],
            "winter_stage": [
                "REKENPEIL_WINTER",
                "BOVENGRENS_BEHEERMARGE_WINTER",
                "ONDERGRENS_BEHEERMARGE_WINTER",
            ],
        },
    }

    config["Hunze en Aa's"] = {
        "bgt_code": "W0646",
    }

    config["Limburg"] = {
        "bgt_code": "W0665",
        "watercourses": {
            # "url": "https://maps.waterschaplimburg.nl/arcgis/rest/services/Legger/Leggerwfs/MapServer",
            # "layer": 1,  # primair
            # "layer": 2, # secundair
            "url": "https://maps.waterschaplimburg.nl/arcgis/rest/services/Legger/Legger/MapServer",
            "layer": 22,  # primair
            # "layer": 23,  # secunair
            # "layer": 24,  # Waterplas
        },
    }

    config["Noorderzijlvest"] = {
        "bgt_code": "W0647",
        "watercourses": {
            "url": "https://arcgis.noorderzijlvest.nl/server/rest/services/Legger/Legger_Watergangen_2012/MapServer",
            "index": "OVKIDENT",
        },
        "level_areas": {
            "url": "https://arcgis.noorderzijlvest.nl/server/rest/services/Peilbeheer/Peilgebieden/MapServer",
            "layer": 3,
            "index": "GPGIDENT",
            "summer_stage": "OPVAFWZP",
            "winter_stage": "OPVAFWWP",
        },
    }

    config["Rijn en IJssel"] = {
        "bgt_code": "W0152",
        "watercourses": {
            "url": "https://opengeo.wrij.nl/arcgis/rest/services/VigerendeLegger/MapServer",
            # "layer": 12,
            # "index": "OWAIDENT",
            # "layer": 11,
            # "index": "OBJECTID",
            "layer": 10,
            "index": "OVKIDENT",
            # "f": "json",
            "bottom_height": ["IWS_AVVHOBOS_L", "IWS_AVVHOBES_L"],
            "bottom_width": "AVVBODDR",
        },
    }

    config["Rijnland"] = {
        "bgt_code": "W0616",
        "watercourses": {
            "url": "https://rijnland.enl-mcs.nl/arcgis/rest/services/Leggers/Legger_Oppervlaktewater_Vigerend/MapServer",
            "layer": 1,
            "water_depth": "WATERDIEPTE",
        },
        "level_areas": {
            "url": "https://rijnland.enl-mcs.nl/arcgis/rest/services/Peilgebied_vigerend_besluit/MapServer",
            "summer_stage": [
                "ZOMERPEIL",
                "VASTPEIL",
                "FLEXZOMERPEILBOVENGRENS",
            ],
            "winter_stage": [
                "WINTERPEIL",
                "VASTPEIL",
                "FLEXWINTERPEILBOVENGRENS",
            ],
        },
        "level_deviations": {
            "url": "https://rijnland.enl-mcs.nl/arcgis/rest/services/Peilafwijking_praktijk/MapServer"
        },
    }

    config["Rivierenland"] = {
        "bgt_code": "W0621",
        "watercourses": {
            "url": "https://kaarten.wsrl.nl/arcgis/rest/services/Kaarten/WatersysteemLeggerVastgesteld/MapServer",
            # "layer": 13,  # profiellijn
            "layer": 14,  # waterloop
            "index": "code",
        },
        "level_areas": {
            # "url": "https://kaarten.wsrl.nl/arcgis/rest/services/Kaarten/Peilgebieden_praktijk/FeatureServer",
            "url": "https://kaarten.wsrl.nl/arcgis/rest/services/Kaarten/Peilgebieden_vigerend/FeatureServer",
            "summer_stage": [
                "ZOMERPEIL",
                "MIN_PEIL",
                "STREEFPEIL",
                "VASTPEIL",
            ],
            "winter_stage": [
                "WINTERPEIL",
                "MAX_PEIL",
                "STREEFPEIL",
                "VASTPEIL",
            ],
        },
    }

    config["Scheldestromen"] = {
        "bgt_code": "W0661",
        "watercourses": {
            "url": "https://geo.scheldestromen.nl/arcgis/rest/services/Extern/EXT_WB_Legger_Oppervlaktewaterlichamen_Vastgesteld/MapServer",
            "layer": 6,
            "index": "OAFIDENT",
        },
        "level_areas": {
            "url": "https://geo.scheldestromen.nl/arcgis/rest/services/Extern/EXT_WB_Waterbeheer/FeatureServer",
            "layer": 14,  # Peilgebieden (praktijk)
            "index": "GPGIDENT",
            # "layer": 15,  # Peilgebieden (juridisch)
            # "index": "GJPIDENT",
            "f": "json",  # geojson does not return GPGZP and GPGWP
            "summer_stage": "GPGZP",
            "winter_stage": "GPGWP",
            "nan_values": [-99, 99],
        },
    }

    config["Schieland en de Krimpenerwaard"] = {
        "bgt_code": "W0656",
        "watercourses": {
            "url": "https://services.arcgis.com/OnnVX2wGkBfflKqu/arcgis/rest/services/HHSK_Legger_Watersysteem/FeatureServer",
            "layer": 11,  # Hoofdwatergang
            # "layer": 12,  # Overig Water
            "water_depth": "DIEPTE",
        },
        "level_areas": {
            # "url": "https://services.arcgis.com/OnnVX2wGkBfflKqu/ArcGIS/rest/services/Peilbesluiten/FeatureServer",
            "url": "https://services.arcgis.com/OnnVX2wGkBfflKqu/ArcGIS/rest/services/VigerendePeilgebiedenEnPeilafwijkingen_HHSK/FeatureServer",
            "summer_stage": ["BOVENPEIL", "VASTPEIL"],
            "winter_stage": ["ONDERPEIL", "VASTPEIL"],
            "nan_values": 9999,
        },
        "level_deviations": {
            "url": "https://services.arcgis.com/OnnVX2wGkBfflKqu/ArcGIS/rest/services/VigerendePeilgebiedenEnPeilafwijkingen_HHSK/FeatureServer",
            "layer": 1,
        },
    }

    config["Vallei & Veluwe"] = {
        "bgt_code": "W0662",
        "watercourses": {
            "url": "https://services1.arcgis.com/ug8NBKcLHVNmdmdt/ArcGIS/rest/services/Legger_Watersysteem/FeatureServer",
            "layer": 16,  # A-water
            # "layer": 17,  # B-water
            # "layer": 18, # A-water
        },
        "level_areas": {
            "url": "https://services1.arcgis.com/ug8NBKcLHVNmdmdt/arcgis/rest/services/Peilvakken/FeatureServer",
            "summer_stage": "WS_MAX_PEIL",
            "winter_stage": "WS_MIN_PEIL",
            "nan_values": 999,
        },
    }

    config["Vechtstromen"] = {
        "bgt_code": "W0663",
        "watercourses": {
            "url": "https://services1.arcgis.com/3RkP6F5u2r7jKHC9/arcgis/rest/services/Legger_publiek_Vastgesteld_Openbaar/FeatureServer",
            "layer": 11,
            "index": "GLOBALID",
        },
        "level_areas": {
            "url": "https://services1.arcgis.com/3RkP6F5u2r7jKHC9/arcgis/rest/services/WBP_Peilen/FeatureServer",
            "layer": 0,  # Peilgebieden voormalig Velt en Vecht
            "index": "GPG_ID",
            "summer_stage": "GPGZMRPL",
            "winter_stage": "GPGWNTPL",
            "nan_values": 0,
            # "layer": 1,  # Peilregister voormalig Regge en Dinkel
            # "index": None,
        },
    }

    config["Zuiderzeeland"] = {
        "bgt_code": "W0650",
        "watercourses": {
            # "url": "https://services.arcgis.com/84oM5NriBghHdQ3Z/ArcGIS/rest/services/leggerkavelsloten/FeatureServer",
            "url": "https://services.arcgis.com/84oM5NriBghHdQ3Z/arcgis/rest/services/legger_concept/FeatureServer",
            "layer": 12,  # Profiel (lijnen)
            # "layer": 13,  # Oppervlaktewater (vlakken)
            "index": "IDENT",
        },
        "level_areas": {
            "url": "https://services.arcgis.com/84oM5NriBghHdQ3Z/arcgis/rest/services/zzl_Peilgebieden/FeatureServer",
            "index": "GPGIDENT",
            "summer_stage": "GPGZMRPL",
            "winter_stage": "GPGWNTPL",
            "nan_values": -999,
        },
    }

    return config


def get_data(wb, data_kind, extent=None, max_record_count=None, config=None, **kwargs):
    """
    Get the data for a Waterboard and a specific data_kind

    Parameters
    ----------
    ws : str
        The name of the waterboard.
    data_kind : str
        The kind of data you like to download. Possible values are
        'watercourses', 'level_areas' and 'level_deviations'
    extent : tuple or list of length 4, optional
        THe extent of the data you like to donload: (xmin, xmax, ymin, ymax).
        Download everything when extent is None. The default is None.
    max_record_count : int, optional
        THe maximum number of records that are downloaded in each call to the
        webservice. When max_record_count is None, the maximum is set equal to
        the maximum of the server. The default is None.
    config : dict, optional
        A dictionary with properties of the data sources of the Waterboards.
        When None, the configuration is retreived from the method
        get_configuration(). The default is None.
    **kwargs : dict
        Optional arguments which are passed onto arcrest() or wfs().

    Raises
    ------

        DESCRIPTION.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing data from the waterboard (polygons for
        level_areas/level_deviations and lines for watercourses).

    """
    if config is None:
        config = get_configuration()
    # some default values
    layer = 0
    index = "CODE"
    server_kind = "arcrest"
    f = "geojson"

    if wb not in config:
        raise (Exception(f"No configuration available for {wb}"))
    if data_kind not in config[wb]:
        raise (Exception(f"{data_kind} not available for {wb}"))
    conf = config[wb][data_kind]
    url = conf["url"]
    if "layer" in conf:
        layer = conf["layer"]
    if "index" in conf:
        index = conf["index"]
    if "server_kind" in conf:
        server_kind = conf["server_kind"]
    if "f" in conf:
        f = conf["f"]

    # % download and plot data
    if server_kind == "arcrest":
        gdf = webservices.arcrest(
            url,
            layer,
            extent,
            f=f,
            max_record_count=max_record_count,
            **kwargs,
        )
    elif server_kind == "wfs":
        gdf = webservices.wfs(
            url, layer, extent, max_record_count=max_record_count, **kwargs
        )
    else:
        raise (Exception("Unknown server-kind: {server_kind}"))
    if index is not None:
        if index not in gdf:
            logger.warning(f"Cannot find {index} in {data_kind} of {wb}")
        else:
            gdf = gdf.set_index(index)
    if data_kind == "level_areas":
        summer_stage = []
        if "summer_stage" in conf:
            summer_stage = conf["summer_stage"]
        gdf = _set_column_from_columns(gdf, "summer_stage", summer_stage)
        winter_stage = []
        if "winter_stage" in conf:
            winter_stage = conf["winter_stage"]
        gdf = _set_column_from_columns(gdf, "winter_stage", winter_stage)
    elif data_kind == "watercourses":
        bottom_height = []
        if "bottom_height" in conf:
            bottom_height = conf["bottom_height"]
        gdf = _set_column_from_columns(gdf, "bottom_height", bottom_height)
        water_depth = []
        if "water_depth" in conf:
            water_depth = conf["water_depth"]
        gdf = _set_column_from_columns(gdf, "water_depth", water_depth)
    return gdf


def _set_column_from_columns(gdf, set_column, from_columns, nan_values=None):
    """Retrieve values from one or more Geo)DataFrame-columns and set these
    values as another column"""
    if set_column in gdf.columns:
        raise (Exception(f"Column {set_column} allready exists"))
    gdf[set_column] = np.NaN
    if from_columns is None:
        return gdf
    if isinstance(from_columns, str):
        from_columns = [from_columns]
    for from_column in from_columns:
        if from_column not in gdf:
            logger.warning(
                f"Cannot find column {from_column} as source for {set_column}"
            )
            continue
        mask = gdf[set_column].isna()
        if not mask.any():
            break
        mask = mask & ~gdf[from_column].isna()
        if not mask.any():
            continue
        if isinstance(from_column, list):
            gdf.loc[mask, set_column] = gdf.loc[mask, from_column].mean(1)
        else:
            gdf.loc[mask, set_column] = gdf.loc[mask, from_column]
        if nan_values is not None:
            if isinstance(nan_values, (float, int)):
                nan_values = [nan_values]
            gdf.loc[gdf[set_column].isin(nan_values), set_column] = np.NaN
    return gdf
