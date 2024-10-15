import os
import nlmod
import tempfile
import rioxarray
import geopandas as gpd

model_ws = os.path.join(tempfile.gettempdir(), "test_util")
extent = [98000.0, 99000.0, 489000.0, 490000.0]


def get_bgt():
    fname = os.path.join(model_ws, "bgt.gpkg")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        bgt = nlmod.read.bgt.get_bgt(extent)
        bgt.to_file(fname)
    return gpd.read_file(fname).set_index("identificatie")


def get_ahn():
    fname = os.path.join(model_ws, "ahn.tif")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        ahn = nlmod.read.ahn.get_ahn4(extent)
        ahn.rio.to_raster(fname)
    ahn = rioxarray.open_rasterio(fname, mask_and_scale=True)[0]
    return ahn


def test_add_min_ahn_to_gdf():
    bgt = get_bgt()
    ahn = get_ahn()
    bgt = nlmod.gwf.surface_water.add_min_ahn_to_gdf(bgt, ahn, buffer=100.0)


def test_zonal_statistics():
    bgt = get_bgt()
    ahn = get_ahn()
    statistics = ["min", "mean", "max"]
    all_touched = True
    bgt = nlmod.util.zonal_statistics(
        bgt,
        ahn,
        engine="geocube",
        columns="geocube",
        statistics=statistics,
        all_touched=all_touched,
    )
    bgt = nlmod.util.zonal_statistics(
        bgt,
        ahn,
        engine="rasterstats",
        columns="rasterstats",
        statistics=statistics,
        all_touched=all_touched,
    )

    for stat in statistics:
        diff = bgt[f"geocube_{stat}"] - bgt[f"rasterstats_{stat}"]
        assert (diff.abs() < 0.001).all()
