from . import waterboard, webservices
from .. import cache


@cache.cache_pickle
def get_municipalities(source="cbs", drop_water=True, **kwargs):
    """Get the location of the Dutch municipalities as a Polygon GeoDataFrame.

    Parameters
    ----------
    source : str, optional
        'cbs' or 'kadaster'
    drop_water : bool, optional
        drop water
    **kwargs
        passed to webservices.wfs
    
    Returns
    -------
    gpd.GeoDataFrame
        polygons of municipalities
    """
    if source == "kadaster":
        url = (
            "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?service=WFS"
        )
        layer = "Gemeentegebied"
        gdf = webservices.wfs(url, layer, **kwargs)
        gdf = gdf.set_index("naam")
    elif source == "cbs":
        # more course:
        # url = "https://service.pdok.nl/cbs/gebiedsindelingen/2023/wfs/v1_0?service=WFS"
        # layer = "gemeente_gegeneraliseerd"
        # more detail:
        url = "https://service.pdok.nl/cbs/wijkenbuurten/2022/wfs/v1_0?&service=WFS"
        layer = "gemeenten"

        gdf = webservices.wfs(url, layer, **kwargs)
        if drop_water:
            gdf = gdf[gdf["water"] == "NEE"]
        gdf = gdf.set_index("gemeentenaam")
    else:
        raise ValueError(f"Unknown source: {source}")
    return gdf


@cache.cache_pickle
def get_provinces(source="cbs", **kwargs):
    """Get the location of the Dutch provinces as a Polygon GeoDataFrame.
    
    Parameters
    ----------
    source : str, optional
        'cbs' or 'kadaster'
    **kwargs
        passed to webservices.wfs

    Returns
    -------
    gpd.GeoDataFrame
        polygons of provinces
    """
    if source == "kadaster":
        url = (
            "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?service=WFS"
        )
        layer = "Provinciegebied"
        gdf = webservices.wfs(url, layer, **kwargs)
        gdf = gdf.set_index("naam")
    elif source == "cbs":
        url = "https://service.pdok.nl/cbs/gebiedsindelingen/2023/wfs/v1_0?service=WFS"
        layer = "provincie_gegeneraliseerd"
        gdf = webservices.wfs(url, layer, **kwargs)
        gdf = gdf.set_index("statnaam")
    else:
        raise (ValueError(f"Unknown source: {source}"))
    return gdf


@cache.cache_pickle
def get_netherlands(source="cbs", **kwargs):
    """Get the location of the Dutch border as a Polygon GeoDataFrame.
    
    Parameters
    ----------
    source : str, optional
        'cbs' or 'kadaster'
    **kwargs
        passed to webservices.wfs

    Returns
    -------
    gpd.GeoDataFrame
        polygons of the Netherlands
    """
    if source == "kadaster":
        url = (
            "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?service=WFS"
        )
        layer = "Landgebied"
        gdf = webservices.wfs(url, layer, **kwargs)
        gdf = gdf.set_index("naam")
    else:
        url = "https://service.pdok.nl/cbs/gebiedsindelingen/2023/wfs/v1_0?service=WFS"
        layer = "landsdeel_gegeneraliseerd"
        gdf = webservices.wfs(url, layer, **kwargs)
        gdf = gdf.set_index("statnaam")
    return gdf


@cache.cache_pickle
def get_waterboards(**kwargs):
    """Get the location of the Dutch Waterboards as a Polygon GeoDataFrame.
     
    Parameters
    ----------
    **kwargs
        passed to waterboard.get_polygons

    Returns
    -------
    gpd.GeoDataFrame
        polygons of the Netherlands
    """
    return waterboard.get_polygons(**kwargs)
