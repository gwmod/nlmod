import nlmod


def test_get_municipalities_cbs():
    extent = [100000, 110000, 400000, 410000]
    gdf = nlmod.read.administrative.download_municipalities_gdf(extent=extent)
    assert len(gdf) > 0


def test_get_municipalities_kadaster():
    extent = [100000, 110000, 400000, 410000]
    gdf = nlmod.read.administrative.download_municipalities_gdf(source="kadaster", extent=extent)
    assert len(gdf) > 0


def test_get_provinces_cbs():
    gdf = nlmod.read.administrative.download_provinces_gdf()
    assert len(gdf) > 0


def test_get_provinces_kadaster():
    gdf = nlmod.read.administrative.download_provinces_gdf(source="kadaster")
    assert len(gdf) > 0


def test_get_netherlands_cbs():
    gdf = nlmod.read.administrative.download_nlborder_gdf()
    assert len(gdf) > 0


def test_get_netherlands_kadaster():
    gdf = nlmod.read.administrative.download_nlborder_gdf(source="kadaster")
    assert len(gdf) > 0


def test_get_waterboards():
    gdf = nlmod.read.administrative.download_waterboards_gdf()
    assert len(gdf) > 0
