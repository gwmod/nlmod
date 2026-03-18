import requests
import pytest

import nlmod


def test_get_municipalities_cbs():
    extent = [100000, 110000, 400000, 410000]
    gdf = nlmod.read.administrative.download_municipalities_gdf(extent=extent)
    assert len(gdf) > 0


def test_get_municipalities_kadaster():
    extent = [100000, 110000, 400000, 410000]
    gdf = nlmod.read.administrative.download_municipalities_gdf(
        source="kadaster", extent=extent
    )
    assert len(gdf) > 0


def test_get_provinces_cbs():
    gdf = nlmod.read.administrative.download_provinces_gdf()
    assert len(gdf) > 0


def test_get_provinces_kadaster():
    gdf = nlmod.read.administrative.download_provinces_gdf(source="kadaster")
    assert len(gdf) > 0


def test_get_netherlands_cbs():
    gdf = nlmod.read.administrative.download_netherlands_gdf()
    assert len(gdf) > 0


def test_get_netherlands_kadaster():
    gdf = nlmod.read.administrative.download_netherlands_gdf(source="kadaster")
    assert len(gdf) > 0


def test_download_kadaster_percelen():
    extent = [118_200, 118_300, 439_800, 439_900]
    gdf = nlmod.read.administrative.download_kadaster_percelen(extent=extent)
    assert len(gdf) > 0


def test_get_waterboards():
    try:
        gdf = nlmod.read.administrative.download_waterboards_gdf()
        assert len(gdf) > 0
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Network unavailable: {exc}")
