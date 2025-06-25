import warnings

from . import webservices


def get_percelen(*args, **kwargs):
    """Get a gdf from the Basisregistratie Percelen.

    .. deprecated:: 0.10.0
        `get_percelen` will be removed in nlmod 1.0.0, it is replaced by
        `download_percelen_gdf` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    """

    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.brp.download_percelen_gdf() in the future.",
        DeprecationWarning,
    )

    return download_percelen_gdf(*args, **kwargs)


def download_percelen_gdf(extent, year=None):
    """Get a gdf from the Basisregistratie Percelen."""
    if year is None:
        url = "https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0?service=WFS"
        layer = "BrpGewas"
        gdf = webservices.wfs(url, layer, extent)
        gdf = gdf.set_index("fuuid")
    else:
        if year < 2009 or year > 2021:
            raise (ValueError("Only data available from 2009 up to and including 2021"))
        url = f"https://services.arcgis.com/nSZVuSZjHpEZZbRo/ArcGIS/rest/services/BRP_{year}/FeatureServer"
        gdf = webservices.arcrest(url, 0, extent=extent)
    return gdf
