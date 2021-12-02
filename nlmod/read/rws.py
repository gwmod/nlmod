# -*- coding: utf-8 -*-
"""functions to add surface water to a mf model using the ghb package."""

import logging
import os
import geopandas as gpd
import nlmod
import datetime as dt
import xarray as xr

from .. import mdims, cache, util

logger = logging.getLogger(__name__)


def get_gdf_surface_water(model_ds):
    """read a shapefile with surface water as a geodataframe, cut by the extent
    of the model.

    Parameters
    ----------
    model_ds : xr.DataSet
        dataset containing relevant model information

    Returns
    -------
    gdf_opp_water : GeoDataframe
        surface water geodataframe.
    """
    # laad bestanden in
    fname = os.path.join(nlmod.NLMOD_DATADIR, 'opp_water.shp')
    gdf_swater = gpd.read_file(fname)
    gdf_swater = util.gdf_within_extent(gdf_swater, model_ds.extent)

    return gdf_swater


@cache.cache_netcdf
def surface_water_to_model_dataset(model_ds, da_name, gridprops=None):
    """create 3 data-arrays from the shapefile with surface water:

    - area: with the area of the shape in the cell
    - cond: with the conductance based on the area and bweerstand column in shapefile
    - peil: with the surface water lvl based on the peil column in the shapefile

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    da_name : str
        name of the polygon shapes, name is used to store data arrays in
        model_ds
    gridprops : dict, optional
        extra model properties when using unstructured grids. 
        The default is None. 

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with modelgrid data.
    """

    modelgrid = mdims.modelgrid_from_model_ds(model_ds, gridprops=gridprops)
    gdf = get_gdf_surface_water(model_ds)

    area = xr.zeros_like(model_ds['top'])
    cond = xr.zeros_like(model_ds['top'])
    peil = xr.zeros_like(model_ds['top'])
    for _, row in gdf.iterrows():
        area_pol = mdims.polygon_to_area(modelgrid, row['geometry'],
                                         xr.ones_like(model_ds['top']),
                                         model_ds.gridtype)
        cond = xr.where(area_pol > area, area_pol / row['bweerstand'], cond)
        peil = xr.where(area_pol > area, row['peil'], peil)
        area = xr.where(area_pol > area, area_pol, area)

    model_ds_out = util.get_model_ds_empty(model_ds)
    model_ds_out[f'{da_name}_area'] = area
    model_ds_out[f'{da_name}_area'].attrs['units'] = 'm2'
    model_ds_out[f'{da_name}_cond'] = cond
    model_ds_out[f'{da_name}_cond'].attrs['units'] = 'm2/day'
    model_ds_out[f'{da_name}_peil'] = peil
    model_ds_out[f'{da_name}_peil'].attrs['units'] = 'mNAP'

    for datavar in model_ds_out:
        model_ds_out[datavar].attrs['source'] = 'RWS'
        model_ds_out[datavar].attrs['date'] = dt.datetime.now().strftime('%Y%m%d')
        

    return model_ds_out
