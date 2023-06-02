import os

import util

import nlmod


def test_struc_da_to_gdf():
    ds = util.get_ds_structured()
    nlmod.gis.struc_da_to_gdf(ds, "top")


def test_vertex_da_to_gdf():
    ds = util.get_ds_vertex()
    nlmod.gis.vertex_da_to_gdf(ds, "top")


def test_ds_to_ugrid_nc_file():
    ds = util.get_ds_vertex()
    nlmod.gis.ds_to_ugrid_nc_file(ds, os.path.join("data", "ugrid_test.nc"))
