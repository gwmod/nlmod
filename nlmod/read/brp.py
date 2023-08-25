from . import webservices


def get_percelen(extent, year=None):
    """Get the Basisregistrayie Percelen."""
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
