import pytest
import xarray as xr

import nlmod


@pytest.mark.parametrize("resolution", [1, 20])
def test_bathymetry(resolution):
    xmin = 25_000.0
    ymin = 410_000.0
    if resolution == 1:
        xmin = 102_200.0
        ymin = 497_700.0
    xmax = xmin + 2 * resolution
    ymax = ymin + 2 * resolution
    extent = [xmin, xmax, ymin, ymax]
    da = nlmod.read.rws.download_bathymetry(extent, resolution=f"{resolution}m")
    assert isinstance(da, xr.DataArray)
    assert not da.isnull().all()
