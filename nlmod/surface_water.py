# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:18:22 2020

@author: oebbe
"""
import json
import os
import time
import warnings

import fiona
import gdal
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from owslib.wfs import WebFeatureService
from shapely.geometry import mapping, shape, LineString, Polygon, Point
from shapely.strtree import STRtree
from tqdm import tqdm
import rasterio
import rasterio.features

import flopy
from flopy.discretization.structuredgrid import StructuredGrid

#from .arcrest import ArcREST
import nlmod
from . import mgrid, util#, rws, panden, ahn


def get_gdf_opp_water(model_ds):
    
    # laad bestanden in
    fname = os.path.join(nlmod.nlmod_datadir, r'opp_water.shp')
    gdf_opp_water = gpd.read_file(fname)
    gdf_opp_water = util.gdf_within_extent(gdf_opp_water, model_ds.extent)
    
    return gdf_opp_water

def get_general_head_boundary(model_ds, gdf,
                              modelgrid, name,
                              gridtype='structured',
                              cachedir=None,
                              use_cache=False,
                              verbose=False):
    """ Get general head boundary from surface water geodataframe

    Parameters
    ----------
    model_ds : xr.DataSet
        dataset containing relevant model grid information
    gdf : geopandas.GeoDataFrame
        polygon shapes with surface water.
    modelgrid : flopy grid
        model grid.
    name : str
        name of the polygon shapes, name is used to store data arrays in 
        model_ds
    gridtype : str, optional
        type of grid, options are 'structured' and 'unstructured'. The default is 'structured'.
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached ghb data is used. The default is False.
    verbose : bool, optional
        print additional information to the screen. The default is False.

    Returns
    -------
    model_ds : xr.DataSet
        dataset with spatial model data including the ghb rasters

    """
    model_ds = util.get_cache_netcdf(use_cache, cachedir, 'ghb_model_ds.nc',
                                     gdf_to_model_dataset,
                                     model_ds, verbose=verbose, gdf=gdf,
                                     modelgrid=modelgrid, name=name,
                                     gridtype=gridtype)

    return model_ds


def gdf_to_model_dataset(model_ds, gdf, modelgrid, name, gridtype='structured'):
    """ create 3 data-arrays from a geodataframe with oppervlaktewater:
    - area: with the area of the geodataframe in the cell
    - cond: with the conductance based on the area and bweerstand column in gdf
    - peil: with the surface water lvl based on the peil column in the gdf


    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    gdf : geopandas.GeoDataFrame
        polygon shapes with surface water.
    modelgrid : flopy grid
        model grid.
    name : str
        name of the polygon shapes, name is used to store data arrays in 
        model_ds

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with modelgrid data. Has 

    """
    area = xr.zeros_like(model_ds['top'])
    cond = xr.zeros_like(model_ds['top'])
    peil = xr.zeros_like(model_ds['top'])
    for i, row in gdf.iterrows():
        area_pol = mgrid.polygon_to_area(modelgrid, row['geometry'],
                                         xr.ones_like(model_ds['top']),
                                         gridtype)
        cond = xr.where(area_pol > area, area_pol / row['bweerstand'], cond)
        peil = xr.where(area_pol > area, row['peil'], peil)
        area = xr.where(area_pol > area, area_pol, area)

    model_ds_out = util.get_model_ds_empty(model_ds)
    model_ds_out[f'{name}_area'] = area
    model_ds_out[f'{name}_cond'] = cond
    model_ds_out[f'{name}_peil'] = peil

    return model_ds_out