import geopandas as gpd

import nlmod


def test_brt():
    extent = [119900, 120000, 440000, 440100]
    brt = nlmod.read.brt.get_brt(extent, layer="waterdeel")

    assert isinstance(brt, gpd.GeoDataFrame)
