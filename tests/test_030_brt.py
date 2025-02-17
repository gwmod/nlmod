import nlmod
import geopandas as gpd

def test_brt():

    nlmod.util.get_color_logger("DEBUG")
    extent = [119900, 120000, 440000, 440100]
    brt = nlmod.read.brt.get_brt(extent, layer='waterdeel')

    assert isinstance(brt, gpd.GeoDataFrame)