from . import webservices


def get_percelen(extent):
    """Get the Basisregistrayie Percelen"""
    url = "https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0?service=WFS"
    layer = "BrpGewas"
    return webservices.wfs(url, layer, extent)
