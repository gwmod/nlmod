import pytest

import nlmod


@pytest.mark.parametrize("resolution", [1, 20])
def test_bathymetry(resolution):
    xmin = 25_000.0
    ymin = 410_000.0
    xmax = xmin + 2 * resolution
    ymax = ymin + 2 * resolution
    extent = [xmin, xmax, ymin, ymax]
    nlmod.read.rws.get_bathymetry(extent, resolution=f"{resolution}m")
