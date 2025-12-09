import warnings

from . import waterboard, webservices
from .. import cache


def get_municipalities(*args, **kwargs):
    """Get the location of the Dutch municipalities as a Polygon GeoDataFrame.

    .. deprecated:: 0.10.0
          `get_municipalities` will be removed in nlmod 1.0.0, it is replaced by
          `download_municipalities_gdf` because of new naming convention
          https://github.com/gwmod/nlmod/issues/47

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
    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.administrative.download_municipalities_gdf() in the future.",
        DeprecationWarning,
    )

    return download_municipalities_gdf(*args, **kwargs)


@cache.cache_pickle
def download_municipalities_gdf(source="cbs", drop_water=True, **kwargs):
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


def get_provinces(*args, **kwargs):
    """Get the location of the Dutch provinces as a Polygon GeoDataFrame.

    .. deprecated:: 0.10.0
        `get_provinces` will be removed in nlmod 1.0.0, it is replaced by
        `download_provinces_gdf` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

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

    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.administrative.download_provinces_gdf() in the future.",
        DeprecationWarning,
    )

    return download_provinces_gdf(*args, **kwargs)


@cache.cache_pickle
def download_provinces_gdf(source="cbs", **kwargs):
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


def get_netherlands(*args, **kwargs):
    """Get the location of the Dutch border as a Polygon GeoDataFrame.

    .. deprecated:: 0.10.0
        `get_netherlands` will be removed in nlmod 1.0.0, it is replaced by
        `download_netherlands_gdf` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

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

    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.administrative.download_netherlands_gdf() in the future.",
        DeprecationWarning,
    )

    return download_netherlands_gdf(*args, **kwargs)


@cache.cache_pickle
def download_netherlands_gdf(source="cbs", **kwargs):
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


def get_waterboards(**kwargs):
    """Get the location of the Dutch Waterboards as a Polygon GeoDataFrame.

    .. deprecated:: 0.10.0
          `get_waterboards` will be removed in nlmod 1.0.0, it is replaced by
          `download_waterboards_gdf` because of new naming convention
          https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    **kwargs
        passed to waterboard.get_polygons

    Returns
    -------
    gpd.GeoDataFrame
        polygons of the Netherlands
    """
    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.administrative.download_waterboards_gdf() in the future.",
        DeprecationWarning,
    )

    return waterboard.get_polygons(**kwargs)


@cache.cache_pickle
def download_waterboards_gdf(**kwargs):
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
