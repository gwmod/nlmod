# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:11:03 2020

@author: oebbe
"""

import os
import tempfile
import requests
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

from . import mgrid, mtime

def get_model_dirs(model_ws):
    """ creates model directories if they do not exist yet
    

    Parameters
    ----------
    model_ws : str
        model workspace.

    Returns
    -------
    figdir : str
        figure directory inside model workspace.
    cachedir : str
        cache directory inside model workspace.

    """
    figdir = os.path.join(model_ws, 'figure')
    cachedir = os.path.join(model_ws, 'cache')
    if not os.path.exists(model_ws):
        os.makedirs(model_ws)

    if not os.path.exists(figdir):
        os.mkdir(figdir)

    if not os.path.exists(cachedir):
        os.mkdir(cachedir)
        
    return figdir, cachedir

def get_model_ds_empty(model_ds):
    """ get a copy of a model dataset with only grid and time information.

    Parameters
    ----------
    model_ds : xr.Dataset
        dataset with at least the variables layer, x, y and time

    Returns
    -------
    model_ds_out : xr.Dataset
        dataset with only model grid and time information

    """
    if model_ds.gridtype == 'structured':
        model_ds_out = model_ds[['layer', 'x', 'y', 'time']].copy()
        return model_ds_out
    elif model_ds.gridtype == 'unstructured':
        model_ds_out = model_ds[['cid', 'layer', 'x', 'y', 'time']].copy()
        return model_ds_out
    else:
        raise ValueError('no gridtype defined cannot compare model datasets')


def check_delr_delc_extent(dic, model_ds, verbose=False):
    """ checks if the delr, delc and extent in a dictionary equals the
    delr, delc and extent in a model dataset.


    Parameters
    ----------
    dic : dictionary
        dictionary with one or more of the keys 'delr', 'delc' or 'extent'.
    model_ds : xr.Dataset
        model dataset with the attributes 'delr', 'delc' or 'extent'.

    Returns
    -------
    check : bool
        True of the delr, delc and extent are the same.

    """
    check = True
    for key in ['delr', 'delc']:
        if key in dic.keys():
            key_check = dic[key] == model_ds.attrs[key]
            if verbose:
                if key_check:
                    print(f'{key} of current grid is the same as cached grid')
                else:
                    print(f'{key} of current grid differs from cached grid')
            check = check and key_check

    if 'extent' in dic.keys():
        key_check = (np.array(dic['extent']) == np.array(
            model_ds.attrs['extent'])).all()
        if verbose:
            if key_check:
                print('extent of current grid is the same as cached grid')
            else:
                print('extent of current grid differs from cached grid')
            check = check and key_check

    return check


def check_model_ds(model_ds, model_ds2, check_grid=True, check_time=True,
                   verbose=False):
    """ check if two model datasets have the same grid and time discretization.
    e.g. the same dimensions and coordinates.


    Parameters
    ----------
    model_ds : xr.Dataset
        dataset with model grid and time discretisation
    model_ds2 : xr.Dataset
        dataset with model grid and time discretisation. This is typically
        a cached dataset.
    check_grid : bool, optional
        if True the grids of both models are compared to check if they are
        the same
    check_time : bool, optional
        if True the time discretisation of both models are compared to check 
        if they are the same
    verbose : bool, optional
        print additional information. default is False

    Raises
    ------
    ValueError
        if the gridtype of model_ds is not structured or unstructured.

    Returns
    -------
    bool
        True if the two datasets have the same grid and time discretization.

    """

    if model_ds.gridtype == 'structured':
        try:
            if check_grid:
                #check x coordinates
                len_x = model_ds['x'].shape == model_ds2['x'].shape
                comp_x = model_ds['x'] == model_ds2['x']
                if (len(comp_x)>0) and comp_x.all() and len_x:
                    check_x = True
                else:
                    check_x = False
                
                #check y coordinates
                len_y = model_ds['y'].shape == model_ds2['y'].shape
                comp_y = model_ds['y'] == model_ds2['y']
                if (len(comp_y)>0) and comp_y.all() and len_y:
                    check_y = True
                else:
                    check_y = False

                # check layers
                comp_layer = model_ds['layer'] == model_ds2['layer']
                if (len(comp_layer) > 0) and comp_layer.all():
                    check_layer = True
                else:
                    check_layer = False

                # also check layer for dtype
                check_layer = check_layer and (
                    model_ds['layer'].dtype == model_ds2['layer'].dtype)
                if (check_x and check_y and check_layer):
                    if verbose:
                        print('cached data has same grid as current model')
                else:
                    if verbose:
                        print('cached data grid differs from current model')
                    return False
            if check_time:
                check_time = (model_ds['time'].data ==
                              model_ds2['time'].data).all()
                if check_time:
                    if verbose:
                        print(
                            'cached data has same time discretisation as current model')
                else:
                    if verbose:
                        print(
                            'cached data time discretisation differs from current model')
                    return False
            return True
        except KeyError:
            return False

    elif model_ds.gridtype == 'unstructured':
        try:
            if check_grid:
                check_cid = (model_ds['cid'] == model_ds2['cid']).all()
                check_x = (model_ds['x'] == model_ds2['x']).all()
                check_y = (model_ds['y'] == model_ds2['y']).all()
                check_layer = (model_ds['layer'] == model_ds2['layer']).all()
                # also check layer for dtype
                check_layer = check_layer and (
                    model_ds['layer'].dtype == model_ds2['layer'].dtype)
                if (check_cid and check_x and check_y and check_layer):
                    if verbose:
                        print('cached data has same grid as current model')
                else:
                    if verbose:
                        print('cached data grid differs from current model')
                    return False
            if check_time:
                check_time = (model_ds['time'].data ==
                              model_ds2['time'].data).all()
                if check_time:
                    if verbose:
                        print(
                            'cached data has same time discretisation as current model')
                else:
                    if verbose:
                        print(
                            'cached data time discretisation differs from current model')
                    return False
            return True
        except KeyError:
            return False
    else:
        raise ValueError('no gridtype defined cannot compare model datasets')


def get_cache_netcdf(use_cache, cachedir, cache_name, get_dataset_func,
                     model_ds=None, check_grid=True,
                     check_time=True, verbose=False, **get_kwargs):
    """ reate, read or modify cached netcdf files of a model dataset.

    following steps are done:
        1. Read cached dataset and merge this with the current model_ds if all 
        of the following conditions are satisfied:
            a. use_cache = True
            b. dataset exists in cachedir
            c. a model_ds is defined or delr, delc and the extent are defined.
            d. the grid and time discretisation of the cached dataset equals
            the grid and time discretisation of the model dataset
        2. if any of the conditions in step 1 is false the get_dataset_func is
        called (with the **get_kwargs arguments).
        3. the dataset from step 2 is written to the cachedir.
        4. the dataset from step 2 is merged with the current model_ds.


    Parameters
    ----------
    use_cache : bool
        if True an attempt is made to use the cached dataset.
    cachedir : str
        directory to store cached values, if None a temporary directory is
        used. default is None
    cache_name : str
        named of the cached netcdf file with the dataset.
    get_dataset_func : function
        this function is called to obtain a new dataset (and not use the 
        cached dataset).
    model_ds : xr.Dataset
        dataset where the cached or new dataset is added to.
    check_grid : bool, optional
        if True the grids of both models are compared to check if they are
        the same
    check_time : bool, optional
        if True the time discretisation of both models are compared to check 
        if they are the same
    verbose : bool, optional
        print additional information. default is False
    **get_kwargs : 
        keyword arguments are used when calling the get_dataset_func.

    Returns
    -------
    model_ds
        dataset with the cached or new dataset.

    """

    if cachedir is None:
        cachedir = tempfile.gettempdir()

    fname_model_ds = os.path.join(cachedir, cache_name)

    if use_cache:
        if os.path.exists(fname_model_ds):
            if verbose:
                print(f'found cached {cache_name}, loading cached dataset')

            cache_model_ds = xr.open_dataset(fname_model_ds)

            # trying to compare cache model grid to current model grid
            if (model_ds is None):
                # check if delr, delc and extent are in the get_kwargs dic
                pos_kwargs_check = len(set(get_kwargs.keys()).intersection(
                    ['delr', 'delc', 'extent'])) == 3
                if pos_kwargs_check:
                    if check_delr_delc_extent(get_kwargs, cache_model_ds,
                                              verbose=verbose):
                        return cache_model_ds
                    else:
                        cache_model_ds.close()
                else:
                    print('could not check if cached grid corresponds to current grid')
                    cache_model_ds.close()

            # check coordinates of model dataset
            elif check_model_ds(model_ds, cache_model_ds,
                                check_grid, check_time,
                                verbose=verbose):
                model_ds.update(cache_model_ds)
                cache_model_ds.close()
                return model_ds
            else:
                cache_model_ds.close()
    if verbose:
        print(f'creating and caching dataset {cache_name}\n')
    if model_ds is None:
        ds = get_dataset_func(**get_kwargs)
        ds.to_netcdf(fname_model_ds)
        return ds
    else:
        ds = get_dataset_func(model_ds, **get_kwargs)
        ds.to_netcdf(fname_model_ds)
        model_ds.update(ds)
        return model_ds


def get_cache_gdf(use_cache, cachedir, cache_name, get_gdf_func, model_ds=None,
                  check_grid=True, check_time=True, verbose=False,
                  get_args=(), **get_kwargs):
    """
    Create or read a cached GeoDataFrame.

    following steps are done:
        1. Read cached geodataframe if all of the following conditions are met:
            a. use_cache = True
            b. geodataframe exists in cachedir
            c. the grid and time discretisation of the cached dataset equals
            the grid and time discretisation of the model dataset
        2. if the conditions in step 1 are not met the get_gdf_func is
        called (with the **get_kwargs arguments).
        3. the geodatframe from step 2 is written to the cachedir, along with
        an empty model dataset so that the cache can be checked in the future.

    Parameters
    ----------
    use_cache : bool
        if True an attempt is made to use the cached dataset.
    cachedir : str
        directory to store cached values, if None a temporary directory is
        used. default is None
    cache_name : str
        named of the cached netcdf file with the dataset.
    get_gdf_func : function
        this function is called to obtain a new dataset (and not use the
        cached geodataframe)..
    model_ds : xr.Dataset
        dataset where the cached or new dataset is added to.
    check_grid : bool, optional
        if True the grids of both models are compared to check if they are
        the same
    check_time : bool, optional
        if True the time discretisation of both models are compared to check
        if they are the same
    verbose : bool, optional
        print additional information. default is False
    get_args : tuple
        arguments used when calling the get_gdf_func.
    **get_kwargs :
        keyword arguments are used when calling the get_gdf_func.

    Returns
    -------
    gdf : TYPE
        DESCRIPTION.

    """
    if cachedir is None:
        cachedir = tempfile.gettempdir()

    fname_gdf = os.path.join(cachedir, cache_name)
    fname_model_ds = os.path.join(cachedir, cache_name + '.nc')

    if use_cache and os.path.exists(fname_gdf):
        if model_ds is None or (not check_grid and not check_time):
            # just read from cache
            gdf = gpd.read_file(fname_gdf).set_index('index')
            # replace None values by nan
            gdf.fillna(value=np.nan, inplace=True)
            return gdf
        elif os.path.exists(fname_model_ds):
            cache_model_ds = xr.open_dataset(fname_model_ds)
            if check_model_ds(model_ds, cache_model_ds, check_grid, check_time,
                              verbose=verbose):
                cache_model_ds.close()
                gdf = pd.read_pickle(fname_gdf)
                # replace None values by nan
                gdf.fillna(value=np.nan, inplace=True)
                return gdf
            else:
                cache_model_ds.close()
    gdf = get_gdf_func(*get_args, **get_kwargs)
    # save result to cache
    gdf.to_pickle(fname_gdf)
    if model_ds is not None:
        # save an empty model_ds to check future runs against the cache
        get_model_ds_empty(model_ds).to_netcdf(fname_model_ds)

    return gdf





def find_most_recent_file(folder, name, extension='.pklz'):
    """ find the most recent file in a folder. File must startwith name and
    end width extension. If you want to look for the most recent folder use
    extension = ''.

    Parameters
    ----------
    folder : str
        path of folder to look for files
    name : str
        find only files that start with this name
    extension : str
        find only files with this extension

    Returns
    -------
    newest_file : str
        name of the most recent file
    """

    i = 0
    for file in os.listdir(folder):
        if file.startswith(name) and file.endswith(extension):
            if i == 0:
                newest_file = os.path.join(folder, file)
                time_prev_file = os.stat(newest_file).st_mtime
            else:
                check_file = os.path.join(folder, file)
                if os.stat(check_file).st_mtime > time_prev_file:
                    newest_file = check_file
                    time_prev_file = os.stat(check_file).st_mtime
            i += 1

    if i == 0:
        return None

    return newest_file


def compare_model_extents(extent1, extent2, verbose=False):
    """ check overlap between two model extents


    Parameters
    ----------
    extent1 : list, tuple or array
        first extent [xmin, xmax, ymin, ymax]
    extent2 : xr.DataSet
        second extent
    verbose : bool, optional
        if True additional information is printed. The default is False.

    Returns
    -------
    int
        several outcomes:
            1: extent1 is completely within extent2
            2: extent2 is completely within extent1

    """

    # option1 extent1 is completely within extent2
    check_xmin = extent1[0] >= extent2[0]
    check_xmax = extent1[1] <= extent2[1]
    check_ymin = extent1[2] >= extent2[2]
    check_ymax = extent1[3] <= extent2[3]

    if check_xmin and check_xmax and check_ymin and check_ymax:
        if verbose:
            print('extent1 is completely within extent2 ')
        return 1

    # option2 extent2 is completely within extent1
    if (not check_xmin) and (not check_xmax) and (not check_ymin) and (not check_ymax):
        if verbose:
            print('extent2 is completely within extent1')
        return 2

    # option 3 left bound
    if (not check_xmin) and check_xmax and check_ymin and check_ymax:
        if verbose:
            print('extent1 is completely within extent2 except for the left bound (xmin)')
        return 3

    # option 4 right bound
    if check_xmin and (not check_xmax) and check_ymin and check_ymax:
        if verbose:
            print(
                'extent1 is completely within extent2 except for the right bound (xmax)')
        return 4

    # option 10
    if check_xmin and (not check_xmax) and (not check_ymin) and (not check_ymax):
        if verbose:
            print('only the left bound of extent 1 is within extent 2')
        return 10

    raise NotImplementedError('other options are not yet implemented')


def gdf_within_extent(gdf, extent):
    """ select only parts of the geodataframe within the extent.
    Only works for polygon features.

    Parameters
    ----------
    gdf : geopandas GeoDataFrame
        dataframe with polygon features.
    extent : list or tuple
        extent to slice gdf, (xmin, xmax, ymin, ymax).

    Returns
    -------
    gdf : geopandas GeoDataFrame
        dataframe with only polygon features within the extent.

    """

    bbox = (extent[0], extent[2], extent[1], extent[3])
    geom_extent = box(*tuple(bbox))
    gdf_extent = gpd.GeoDataFrame(['extent'], geometry=[geom_extent],
                                  crs=gdf.crs)
    gdf = gpd.overlay(gdf, gdf_extent)

    return gdf


def get_google_drive_filename(id):
    if isinstance(id, requests.Response):
        response = id
    else:
        url = 'https://drive.google.com/uc?export=download&id=' + id
        response = requests.get(url)
    header = response.headers['Content-Disposition']
    file_name = re.search(r'filename="(.*)"', header).group(1)
    return file_name


def download_file_from_google_drive(id, destination=None):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    if destination is None:
        destination = get_google_drive_filename(id)
    else:
        if os.path.isdir(destination):
            filename = get_google_drive_filename(id)
            destination = os.path.join(destination, filename)

    save_response_content(response, destination)
