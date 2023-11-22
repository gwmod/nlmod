import nlmod


def test_get_municipalities_cbs():
    extent = [100000, 110000, 400000, 410000]
    gdf = nlmod.read.administrative.get_municipalities(extent=extent)
    assert len(gdf) > 0


def test_get_municipalities_kadaster():
    extent = [100000, 110000, 400000, 410000]
    gdf = nlmod.read.administrative.get_municipalities(source="kadaster", extent=extent)
    assert len(gdf) > 0


def test_get_provinces_cbs():
    gdf = nlmod.read.administrative.get_provinces()
    assert len(gdf) > 0


def test_get_provinces_kadaster():
    gdf = nlmod.read.administrative.get_provinces(source="kadaster")
    assert len(gdf) > 0


def test_get_netherlands_cbs():
    gdf = nlmod.read.administrative.get_netherlands()
    assert len(gdf) > 0


def test_get_netherlands_kadaster():
    gdf = nlmod.read.administrative.get_netherlands(source="kadaster")
    assert len(gdf) > 0


def test_get_waterboards():
    gdf = nlmod.read.administrative.get_waterboards()
    assert len(gdf) > 0
