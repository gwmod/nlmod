# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:50:28 2021

@author: oebbe
"""
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import requests
from flopy.discretization.structuredgrid import StructuredGrid

import nlmod
from . import util, mgrid, surface_water


def get_modelgrid_sea(model_ds,
                      gridtype='structured',
                      cachedir=None, use_cache=False,
                      verbose=False):
    """ Get DataArray which is 1 at sea and 0 overywhere else.
    Sea is defined by the geometries in gdf_sea
    grid is defined by mfgrid and model_ds

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        polygon shapes with surface water.
    modelgrid : flopy grid
        model grid.
    model_ds : xr.DataSet
        xarray with model data
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached sea data is used. The default is False.
    verbose : bool, optional
        print additional information to the screen. The default is False.

    Returns
    -------
    model_ds : xr.DataSet
        dataset with 'sea' DataVariable.

    """
    model_ds = util.get_cache_netcdf(use_cache, cachedir, 'sea_model_ds.nc',
                                     find_sea_cells, model_ds,
                                     da_name='northsea',
                                     check_time=False,
                                     verbose=verbose)

    return model_ds



def find_sea_cells(model_ds, da_name='northsea'):
    """ Get Dataset which is 1 at sea and 0 everywhere else.
    Sea is defined by opp_water shapefile
    grid is defined in model_ds

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    da_name : str
        name of the datavar

    Returns
    -------
    model_ds_out : xr.DataSet
        Dataset with a single DataArray, this DataArray is 1 at sea and 0 
        everywhere else. Grid dimensions according to model_ds.

    """
    
    gdf_opp_water = surface_water.get_gdf_opp_water(model_ds)
    
    # find grid cells with sea
    opp_water_zee = gdf_opp_water[gdf_opp_water['OWMNAAM'].isin(['Rijn territoriaal water',
                                                                 'Waddenzee',
                                                                 'Waddenzee vastelandskust',
                                                                 'Hollandse kust (kustwater)',
                                                                 'Waddenkust (kustwater)'])]
    mfgrid = StructuredGrid(delc=np.array([model_ds.delc] * model_ds.dims['y']),
                            delr=np.array([model_ds.delc] * model_ds.dims['x']),
                            xoff=model_ds.extent[0], yoff=model_ds.extent[2])
    
    model_ds_out = mgrid.gdf_to_bool_dataset(model_ds, opp_water_zee, mfgrid, da_name)
    
    return model_ds_out


def get_modelgrid_bathymetry(model_ds,
                             gridtype='structured',
                             gridprops=None,
                             cachedir=None, use_cache=False,
                             verbose=False):
    """ get bathymetry of the Northsea from the jarkus dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data where bathymetry is added to
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached jarkus data is used. The default is False.
    verbose : bool, optional
        print additional information to the screen. The default is False.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with bathymetry
    """

    model_ds = util.get_cache_netcdf(use_cache, cachedir, 'bathymetry_model_ds.nc',
                                     bathymetry_to_model_dataset, model_ds,
                                     verbose=verbose,
                                     gridprops=gridprops,
                                     check_time=False,
                                     )
    return model_ds


def bathymetry_to_model_dataset(model_ds,
                                gridprops=None):
    """ get bathymetry of the Northsea from the jarkus dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data where bathymetry is added to
    use_cache : bool, optional
        if True the cached jarkus data is used. The default is False.
    verbose : bool, optional
        print additional information to the screen. The default is False.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with bathymetry

    Notes
    -----
    Now the nan values in the original bathymetry are filled and than the 
    data is resampled to the modelgrid. Maybe we can speed up things by
    changing the order in which operations are executed.
    """
    try:
        jarkus = get_dataset_jarkus(model_ds.extent)
    except OSError:
        print('cannot access Jarkus netCDF link, copy file from google drive instead')
        fname_jarkus = os.path.join(model_ds.model_ws, 
                                    'bathymetry', 
                                    'jarkus_nhflopy.nc')
        util.download_file_from_google_drive('1uNy4THL3FmNFrTDTfizDAl0lxOH-yCEo', 
                                             fname_jarkus)
        jarkus = xr.open_dataset(fname_jarkus)
        
    da_bathymetry_raw = jarkus['z']

    # fill nan values in bathymetry
    da_bathymetry_filled = mgrid.fillnan_dataarray_structured_grid(
        da_bathymetry_raw)

    # bathymetrie mag nooit groter zijn dan NAP 0.0
    da_bathymetry_filled = xr.where(
        da_bathymetry_filled > 0, 0, da_bathymetry_filled)

    # bathymetry projected on model grid
    if model_ds.gridtype == 'structured':
        da_bathymetry = mgrid.resample_dataarray_to_structured_grid(da_bathymetry_filled,
                                                                    extent=model_ds.extent,
                                                                    delr=model_ds.delr,
                                                                    delc=model_ds.delc,
                                                                    xmid=model_ds.x.data,
                                                                    ymid=model_ds.y.data[::-1])[0]
    elif model_ds.gridtype == 'unstructured':
        da_bathymetry = mgrid.resample_dataarray_to_unstructured_grid(da_bathymetry_filled,
                                                                      gridprops=gridprops)[0]

    model_ds_out = util.get_model_ds_empty(model_ds)

    model_ds_out['bathymetry'] = xr.where(
        model_ds['northsea'], da_bathymetry, np.nan)

    return model_ds_out

def get_dataset_jarkus(extent, verbose=True):
    """ Get bathymetry from Jarkus within a certain extent. If all_tiles is 
    False, the following steps are used:
    1. find Jarkus tiles within the extent
    2. download netcdf files of Jarkus tiles
    3. read Jarkus tiles and combine the 'z' parameter of the last time step
    of each tile, to a dataarray.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid. Should be RD-new
        coördinates (EPSG:28992)
    verbose : bool, optional
        print additional information to the screen. The default is False.

    Returns
    -------
    z : xr.DataSet
        dataset containing bathymetry data

    """

    extent = [int(x) for x in extent]
    netcdf_tile_names = get_jarkus_tilenames(extent)
    tiles = [xr.open_dataset(name) for name in netcdf_tile_names]
    # only use the last timestep
    tiles = [tile.isel(time=-1) for tile in tiles]
    z_dataset = xr.combine_by_coords(tiles, combine_attrs='drop')

    return z_dataset

def get_jarkus_tilenames(extent):
    """ Find all Jarkus tilenames within a certain extent

    Parameters
    ----------
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid. Should be RD-new
        coördinates (EPSG:28992)

    Returns
    -------
    netcdf_urls : list of str
        list of the urls of all netcdf files of the tiles with Jarkus data.

    """
    ds_jarkus_catalog = xr.open_dataset(
        'http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/grids/catalog.nc')
    ew_x = ds_jarkus_catalog['projectionCoverage_x']
    sn_y = ds_jarkus_catalog['projectionCoverage_y']

    mask_ew = (ew_x[:, 1] > extent[0]) & (ew_x[:, 0] < extent[1])
    mask_sn = (sn_y[:, 1] > extent[2]) & (sn_y[:, 0] < extent[3])

    indices_tiles = np.where(mask_ew & mask_sn)[0]
    all_netcdf_tilenames = get_netcdf_tiles()

    netcdf_tile_names = [all_netcdf_tilenames[i] for i in indices_tiles]

    return netcdf_tile_names


def get_netcdf_tiles():
    """ Find all Jarkus netcdf tile names. 


    Returns
    -------
    netcdf_urls : list of str
        list of the urls of all netcdf files of the tiles with Jarkus data.

    Notes
    -----
    This function would be redundant if the jarkus catalog 
    (http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/grids/catalog.nc)
    had a proper way of displaying the url's of each tile. It seems like an
    attempt was made to do this because here is a data variable
    named 'urlPath' in the catalog. However the dataarray of 'urlPath' has the
    same string for each tile.
    """
    url = 'http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/grids/catalog.nc.ascii'
    req = requests.get(url)
    s = req.content.decode('ascii')
    start = s.find('urlPath', s.find('urlPath') + 1)
    end = s.find('projectionCoverage_x', s.find('projectionCoverage_x') + 1)
    netcdf_urls = list(eval(s[start + 12:end - 2]))
    return netcdf_urls