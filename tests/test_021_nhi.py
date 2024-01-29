import os
import numpy as np
import geopandas as gpd
import tempfile
import nlmod
import pytest
import matplotlib.pyplot as plt

tmpdir = tempfile.gettempdir()


@pytest.mark.slow
def test_buidrainage():
    model_ws = os.path.join(tmpdir, "buidrain")
    ds = nlmod.get_ds([110_000, 130_000, 435_000, 445_000], model_ws=model_ws)
    ds = nlmod.read.nhi.add_buisdrainage(ds)

    # assert that all locations with a specified depth also have a positive conductance
    mask = ~ds["buisdrain_depth"].isnull()
    assert np.all(ds["buisdrain_cond"].data[mask] > 0)

    # assert that all locations with a positive conductance also have a specified depth
    mask = ds["buisdrain_cond"] > 0
    assert np.all(~np.isnan(ds["buisdrain_depth"].data[mask]))


def test_gwo():
    username = os.environ["NHI_GWO_USERNAME"]
    password = os.environ["NHI_GWO_PASSWORD"]

    # bijvoorbeeld: download ontrekkingen van Brabant Water
    wells = nlmod.read.nhi.get_gwo_wells(
        username=username, password=password, organisation="Brabant Water"
    )
    assert isinstance(wells, gpd.GeoDataFrame)

    # download extractions from well "13-PP016" of pomping station Veghel
    measurements, gdf = nlmod.read.nhi.get_gwo_measurements(
        username, password, well_site="veghel", filter__well__name="13-PP016"
    )
    assert measurements.reset_index()["Name"].isin(gdf.index).all()


@pytest.mark.skip("too slow")
def test_gwo_entire_pumping_station():
    username = os.environ["NHI_GWO_USERNAME"]
    password = os.environ["NHI_GWO_PASSWORD"]
    measurements, gdf = nlmod.read.nhi.get_gwo_measurements(
        username,
        password,
        well_site="veghel",
    )
    assert measurements.reset_index()["Name"].isin(gdf.index).all()

    ncols = 3
    nrows = int(np.ceil(len(gdf.index) / ncols))
    f, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(10, 10), sharex=True, sharey=True
    )
    axes = axes.ravel()
    for name, ax in zip(gdf.index, axes):
        measurements.loc[name, "Volume"].plot(ax=ax)
        ax.set_xlabel("")
        ax.set_title(name)
    f.tight_layout(pad=0.0)
