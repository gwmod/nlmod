import nlmod
import util
import os


def test_struc_da_to_gdf():
    ds = util.get_ds_structured()
    gdf = nlmod.gis.struc_da_to_gdf(ds, "top")
    return gdf


def test_vertex_da_to_gdf():
    ds = util.get_ds_vertex()
    gdf = nlmod.gis.vertex_da_to_gdf(ds, "top")
    return gdf


def test_ds_to_ugrid_nc_file():
    ds = util.get_ds_vertex()
    nlmod.gis.ds_to_ugrid_nc_file(ds, os.path.join("data", "ugrid_test.nc"))