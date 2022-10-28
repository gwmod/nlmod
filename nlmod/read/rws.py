# -*- coding: utf-8 -*-
"""functions to add surface water to a mf model using the ghb package."""

import datetime as dt
import logging
import os

import geopandas as gpd
import nlmod
import xarray as xr

from .. import cache, mdims, util

from . import jarkus

logger = logging.getLogger(__name__)


def get_gdf_surface_water(ds):
    """read a shapefile with surface water as a geodataframe, cut by the extent
    of the model.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model information

    Returns
    -------
    gdf_opp_water : GeoDataframe
        surface water geodataframe.
    """
    # laad bestanden in
    fname = os.path.join(nlmod.NLMOD_DATADIR, "opp_water.shp")
    gdf_swater = gpd.read_file(fname)
    extent = mdims.get_extent(ds)
    gdf_swater = util.gdf_within_extent(gdf_swater, extent)

    return gdf_swater


@cache.cache_netcdf
def get_surface_water(ds, da_name):
    """create 3 data-arrays from the shapefile with surface water:

    - area: with the area of the shape in the cell
    - cond: with the conductance based on the area and bweerstand column in shapefile
    - peil: with the surface water lvl based on the peil column in the shapefile

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    da_name : str
        name of the polygon shapes, name is used to store data arrays in
        ds

    Returns
    -------
    ds : xarray.Dataset
        dataset with modelgrid data.
    """

    modelgrid = mdims.modelgrid_from_ds(ds)
    gdf = get_gdf_surface_water(ds)

    area = xr.zeros_like(ds["top"])
    cond = xr.zeros_like(ds["top"])
    peil = xr.zeros_like(ds["top"])
    for _, row in gdf.iterrows():
        area_pol = mdims.polygon_to_area(
            modelgrid,
            row["geometry"],
            xr.ones_like(ds["top"]),
            ds.gridtype,
        )
        cond = xr.where(area_pol > area, area_pol / row["bweerstand"], cond)
        peil = xr.where(area_pol > area, row["peil"], peil)
        area = xr.where(area_pol > area, area_pol, area)

    ds_out = util.get_ds_empty(ds)
    ds_out[f"{da_name}_area"] = area
    ds_out[f"{da_name}_area"].attrs["units"] = "m2"
    ds_out[f"{da_name}_cond"] = cond
    ds_out[f"{da_name}_cond"].attrs["units"] = "m2/day"
    ds_out[f"{da_name}_peil"] = peil
    ds_out[f"{da_name}_peil"].attrs["units"] = "mNAP"

    for datavar in ds_out:
        ds_out[datavar].attrs["source"] = "RWS"
        ds_out[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")

    return ds_out


@cache.cache_netcdf
def get_northsea(ds, da_name="northsea"):
    """Get Dataset which is 1 at the northsea and 0 everywhere else. Sea is
    defined by rws surface water shapefile.

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    da_name : str, optional
        name of the datavar that identifies sea cells

    Returns
    -------
    ds_out : xr.DataSet
        Dataset with a single DataArray, this DataArray is 1 at sea and 0
        everywhere else. Grid dimensions according to ds.
    """

    gdf_surf_water = get_gdf_surface_water(ds)

    # find grid cells with sea
    swater_zee = gdf_surf_water[
        gdf_surf_water["OWMNAAM"].isin(
            [
                "Rijn territoriaal water",
                "Waddenzee",
                "Waddenzee vastelandskust",
                "Hollandse kust (kustwater)",
                "Waddenkust (kustwater)",
            ]
        )
    ]

    modelgrid = mdims.modelgrid_from_ds(ds)
    ds_out = mdims.gdf_to_bool_dataset(ds, swater_zee, modelgrid, da_name)

    return ds_out


def add_northsea(ds, cachedir=None):
    """a) get cells from modelgrid that are within the northsea, add data
    variable 'northsea' to ds
    b) fill top, bot, kh and kv add northsea cell by extrapolation
    c) get bathymetry (northsea depth) from jarkus. Add datavariable
    bathymetry to model dataset"""

    logger.info(
        "nan values at the northsea are filled using the bathymetry from jarkus"
    )

    # find grid cells with northsea
    ds.update(get_northsea(ds, cachedir=cachedir, cachename="sea_ds.nc"))

    # fill top, bot, kh, kv at sea cells
    fal = mdims.get_first_active_layer(ds)
    fill_mask = (fal == fal.attrs["_FillValue"]) * ds["northsea"]
    ds = mdims.fill_top_bot_kh_kv_at_mask(ds, fill_mask, gridtype=ds.attrs["gridtype"])

    # add bathymetry noordzee
    ds.update(
        jarkus.get_bathymetry(
            ds,
            ds["northsea"],
            cachedir=cachedir,
            cachename="bathymetry_ds.nc",
        )
    )

    ds = jarkus.add_bathymetry_to_top_bot_kh_kv(ds, ds["bathymetry"], fill_mask)

    # update idomain on adjusted tops and bots
    ds = mdims.set_idomain(ds)
    return ds
