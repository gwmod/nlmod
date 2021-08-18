# -*- coding: utf-8 -*-
"""utility functions for nlmod.

Mostly functions to cache data and manage filenames and directories.
"""

import logging
import os
import re
import sys
import tempfile

import flopy
import geopandas as gpd
import numpy as np
import datetime as dt
import requests
import xarray as xr
from shapely.geometry import box
from shutil import copyfile

import __main__


logger = logging.getLogger(__name__)



def write_and_run_model(gwf, model_ds, write_model_ds=True,
                        nb_path=None):
    """ write modflow files and run the model. 2 extra options:
        1. write the model dataset to cache
        2. copy the modelscript (typically a Jupyter Notebook) to the model
        workspace with a timestamp.
    

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        groundwater flow model.
    model_ds : xarray.Dataset
        dataset with model data.
    write_model_ds : bool, optional
        if True the model dataset is cached. The default is True.
    nb_path : str or None, optional
        full path of the Jupyter Notebook (.ipynb) with the modelscript. The 
        default is None. Preferably this path does not have to be given
        manually but there is currently no good option to obtain the filename
        of a Jupyter Notebook from within the notebook itself.

    """
    
    if not nb_path is None:
        new_nb_fname = f'{dt.datetime.now().strftime("%Y%m%d")}' + os.path.split(nb_path)[-1]
        dst = os.path.join(model_ds.model_ws,  new_nb_fname)
        logger.info(f'write script {new_nb_fname} to model workspace')
        copyfile(nb_path, dst)
            
    
    if write_model_ds:
        logger.info('write model dataset to cache')
        model_ds.to_netcdf(os.path.join(model_ds.attrs['cachedir'], 
                                        'full_model_ds.nc'))
        model_ds.attrs['model_dataset_written_to_disk_on'] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    
    logger.info('write modflow files to model workspace')
    gwf.simulation.write_simulation()
    model_ds.attrs['model_data_written_to_disk_on'] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    
    logger.info('run model')
    assert gwf.simulation.run_simulation()[0]
    model_ds.attrs['model_ran_on'] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    



def get_model_dirs(model_ws):
    """ Creates a new model workspace directory, if it does not 
    exists yet. Within the model workspace directory a
    few subdirectories are created (if they don't exist yet):
    - figure
    - cache

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
    """get a copy of a model dataset with only grid and time information.

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


def check_delr_delc_extent(dic, model_ds):
    """checks if the delr, delc and extent in a dictionary equals the delr,
    delc and extent in a model dataset.

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
            if key_check:
                logger.info(f'{key} of current grid is the same as cached grid')
            else:
                logger.info(f'{key} of current grid differs from cached grid')
            check = check and key_check

    if 'extent' in dic.keys():
        key_check = (np.array(dic['extent']) == np.array(
            model_ds.attrs['extent'])).all()
        
        if key_check:
            logger.info('extent of current grid is the same as cached grid')
        else:
            logger.info('extent of current grid differs from cached grid')
        check = check and key_check

    return check


def check_model_ds(model_ds, model_ds2, check_grid=True, check_time=True):
    """check if two model datasets have the same grid and time discretization.
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
                # check x coordinates
                len_x = model_ds['x'].shape == model_ds2['x'].shape
                comp_x = model_ds['x'] == model_ds2['x']
                if (len(comp_x) > 0) and comp_x.all() and len_x:
                    check_x = True
                else:
                    check_x = False

                # check y coordinates
                len_y = model_ds['y'].shape == model_ds2['y'].shape
                comp_y = model_ds['y'] == model_ds2['y']
                if (len(comp_y) > 0) and comp_y.all() and len_y:
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
                    logger.info('cached data has same grid as current model')
                else:
                    logger.info('cached data grid differs from current model')
                    return False
            if check_time:
                if len(model_ds['time']) != len(model_ds2['time']):
                    check_time = False
                else:
                    check_time = (model_ds['time'].data ==
                                  model_ds2['time'].data).all()
                if check_time:
                    logger.info('cached data has same time discretisation as current model')
                else:
                    logger.info('cached data time discretisation differs from current model')
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
                    logger.info('cached data has same grid as current model')
                else:
                    logger.info('cached data grid differs from current model')
                    return False
            if check_time:
                # check length time series
                if model_ds.dims['time'] == model_ds2.dims['time']:
                    # check times time series
                    check_time = (model_ds['time'].data ==
                                  model_ds2['time'].data).all()
                else:
                    check_time = False
                if check_time:
                    logger.info(
                            'cached data has same time discretisation as current model')
                else:
                    logger.info(
                            'cached data time discretisation differs from current model')
                    return False
            return True
        except KeyError:
            return False
    else:
        raise ValueError('no gridtype defined cannot compare model datasets')


def get_cache_netcdf(use_cache, cachedir, cache_name, get_dataset_func,
                     model_ds=None, check_grid=True,
                     check_time=True, **get_kwargs):
    """Create, read or modify cached netcdf files of a model dataset.

    Steps:

    1. Read cached dataset and merge this with the current model_ds if all
    of the following conditions are satisfied:

       a) use_cache = True
       b) dataset exists in cachedir
       c) a model_ds is defined or delr, delc and the extent are defined.
       d) the grid and time discretisation of the cached dataset equals
          the grid and time discretisation of the model dataset

    2. if any of the conditions in step 1 is false the get_dataset_func is
    called (with the `**get_kwargs` arguments).

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
    **get_kwargs : dict, optional
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
            logger.info(f'found cached {cache_name}, loading cached dataset')

            cache_model_ds = xr.open_dataset(fname_model_ds)

            # trying to compare cache model grid to current model grid
            if (model_ds is None):
                # check if delr, delc and extent are in the get_kwargs dic
                pos_kwargs_check = len(set(get_kwargs.keys()).intersection(
                    ['delr', 'delc', 'extent'])) == 3
                if pos_kwargs_check:
                    if check_delr_delc_extent(get_kwargs, cache_model_ds):
                        return cache_model_ds
                    else:
                        cache_model_ds.close()
                else:
                    print('could not check if cached grid corresponds to current grid')
                    cache_model_ds.close()

            # check coordinates of model dataset
            elif check_model_ds(model_ds, cache_model_ds,
                                check_grid, check_time):
                model_ds.update(cache_model_ds)
                cache_model_ds.close()
                return model_ds
            else:
                cache_model_ds.close()
    logger.info(f'creating and caching dataset {cache_name}\n')
    if model_ds is None:
        ds = get_dataset_func(**get_kwargs)
        ds.to_netcdf(fname_model_ds)
        return ds
    else:
        ds = get_dataset_func(model_ds, **get_kwargs)
        ds.to_netcdf(fname_model_ds)
        model_ds.update(ds)
        return model_ds


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


def compare_model_extents(extent1, extent2):
    """check overlap between two model extents.

    Parameters
    ----------
    extent1 : list, tuple or array
        first extent [xmin, xmax, ymin, ymax]
    extent2 : xr.DataSet
        second extent

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
        logger.info('extent1 is completely within extent2 ')
        return 1

    # option2 extent2 is completely within extent1
    if (not check_xmin) and (not check_xmax) and (not check_ymin) and (not check_ymax):
        logger.info('extent2 is completely within extent1')
        return 2

    # option 3 left bound
    if (not check_xmin) and check_xmax and check_ymin and check_ymax:
        logger.info('extent1 is completely within extent2 except for the left bound (xmin)')
        return 3

    # option 4 right bound
    if check_xmin and (not check_xmax) and check_ymin and check_ymax:
        logger.info(
                'extent1 is completely within extent2 except for the right bound (xmax)')
        return 4

    # option 10
    if check_xmin and (not check_xmax) and (not check_ymin) and (not check_ymax):
        logger.info('only the left bound of extent 1 is within extent 2')
        return 10

    raise NotImplementedError('other options are not yet implemented')


def gdf_from_extent(extent, crs="EPSG:28992"):
    """create a geodataframe with a single polygon with the extent given.

    Parameters
    ----------
    extent : tuple, list or array
        extent.
    crs : str, optional
        coördinate reference system of the extent, default is EPSG:28992
        (RD new)

    Returns
    -------
    gdf_extent : GeoDataFrame
        geodataframe with extent.
    """

    bbox = (extent[0], extent[2], extent[1], extent[3])
    geom_extent = box(*tuple(bbox))
    gdf_extent = gpd.GeoDataFrame(geometry=[geom_extent],
                                  crs=crs)

    return gdf_extent


def gdf_within_extent(gdf, extent):
    """select only parts of the geodataframe within the extent. Only accepts
    Polygon and Linestring geometry types.

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
    # create geodataframe from the extent
    gdf_extent = gdf_from_extent(extent, crs=gdf.crs)

    # check type
    geom_types = gdf.geom_type.unique()
    if len(geom_types) > 1:
        # exception if geomtypes is a combination of Polygon and Multipolygon
        multipoly_check = ('Polygon' in geom_types) and (
            'MultiPolygon' in geom_types)
        if (len(geom_types) == 2) and multipoly_check:
            gdf = gpd.overlay(gdf, gdf_extent)
        else:
            raise TypeError(
                f'Only accepts single geometry type not {geom_types}')
    elif geom_types[0] == 'Polygon':
        gdf = gpd.overlay(gdf, gdf_extent)
    elif geom_types[0] == 'LineString':
        gdf = gpd.sjoin(gdf, gdf_extent)
    elif geom_types[0] == 'Point':
        gdf = gdf.loc[gdf.within(gdf_extent.geometry.values[0])]
    else:
        raise TypeError('Function is not tested for geometry type: '
                        f'{geom_types[0]}')

    return gdf


def get_google_drive_filename(id):
    """get the filename of a google drive file.

    Parameters
    ----------
    id : str
        google drive id name of a file.

    Returns
    -------
    file_name : str
        filename.
    """
    raise DeprecationWarning(
        'this function is no longer supported use the gdown package instead')

    if isinstance(id, requests.Response):
        response = id
    else:
        url = 'https://drive.google.com/uc?export=download&id=' + id
        response = requests.get(url)
    header = response.headers['Content-Disposition']
    file_name = re.search(r'filename="(.*)"', header).group(1)
    return file_name


def download_file_from_google_drive(id, destination=None):
    """download a file from google drive using it's id.

    Parameters
    ----------
    id : str
        google drive id name of a file.
    destination : str, optional
        location to save the file to. If destination is None the file is
        written to the current working directory. The default is None.
    """
    raise DeprecationWarning(
        'this function is no longer supported use the gdown package instead')

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


# %% helper functions (from USGS)


def get_platform(pltfrm):
    """Determine the platform in order to construct the zip file name.

    Source: USGS

    Parameters
    ----------
    pltfrm : str, optional
        check if platform string is correct for downloading binaries,
        default is None and will determine platform string based on system

    Returns
    -------
    pltfrm : str
        return platform string
    """
    if pltfrm is None:
        if sys.platform.lower() == 'darwin':
            pltfrm = 'mac'
        elif sys.platform.lower().startswith('linux'):
            pltfrm = 'linux'
        elif 'win' in sys.platform.lower():
            is_64bits = sys.maxsize > 2 ** 32
            if is_64bits:
                pltfrm = 'win64'
            else:
                pltfrm = 'win32'
        else:
            errmsg = ('Could not determine platform'
                      '.  sys.platform is {}'.format(sys.platform))
            raise Exception(errmsg)
    else:
        assert pltfrm in ['mac', 'linux', 'win32', 'win64']
    return pltfrm


def getmfexes(pth='.', version='', pltfrm=None):
    """Get the latest MODFLOW binary executables from a github site
    (https://github.com/MODFLOW-USGS/executables) for the specified operating
    system and put them in the specified path.

    Source: USGS

    Parameters
    ----------
    pth : str
        Location to put the executables (default is current working directory)

    version : str
        Version of the MODFLOW-USGS/executables release to use.

    pltfrm : str
        Platform that will run the executables.  Valid values include mac,
        linux, win32 and win64.  If platform is None, then routine will
        download the latest appropriate zipfile from the github repository
        based on the platform running this script.
    """
    try:
        import pymake
    except ModuleNotFoundError as e:
        print("Install pymake with "
              "`pip install "
              "https://github.com/modflowpy/pymake/zipball/master`")
        raise e
    # Determine the platform in order to construct the zip file name
    pltfrm = get_platform(pltfrm)
    zipname = '{}.zip'.format(pltfrm)

    # Determine path for file download and then download and unzip
    url = ('https://github.com/MODFLOW-USGS/executables/'
           'releases/download/{}/'.format(version))
    assets = {p: url + p for p in ['mac.zip', 'linux.zip',
                                   'win32.zip', 'win64.zip']}
    download_url = assets[zipname]
    pymake.download_and_unzip(download_url, pth)

    return

def add_heads_to_model_ds(model_ds, fname_hds=None):
    """reads the heads from a modflow .hds file and returns an xarray
    DataArray.

    Parameters
    ----------
    model_ds : TYPE
        DESCRIPTION.
    fname_hds : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    head_ar : TYPE
        DESCRIPTION.
    """

    if fname_hds is None:
        fname_hds = os.path.join(model_ds.model_ws, model_ds.model_name + '.hds')
    
    head_filled = get_heads_array(fname_hds, gridtype=model_ds.gridtype)
    
    

    if model_ds.gridtype == 'unstructured':
        head_ar = xr.DataArray(data=head_filled[:, :, :],
                               dims=('time', 'layer', 'cid'),
                               coords={'cid': model_ds.cid,
                                       'layer': model_ds.layer,
                                       'time': model_ds.time})
    elif model_ds.gridtype =='structured':
        head_ar = xr.DataArray(data=head_filled,
                           dims=('time', 'layer', 'y', 'x'),
                           coords={'x': model_ds.x,
                                   'y': model_ds.y,
                                   'layer': model_ds.layer,
                                   'time': model_ds.time})

    return head_ar

def get_heads_array(fname_hds, gridtype='structured',
                    fill_nans=True):
    """reads the heads from a modflow .hds file and returns a numpy array.

    assumes the dimensions of the heads file are:
        structured: time, layer, cid
        unstructured: time, layer, nrow, ncol


    Parameters
    ----------
    fname_hds : TYPE, optional
        DESCRIPTION. The default is None.
    gridtype : str, optional
        DESCRIPTION. The default is 'structured'.
    fill_nans : bool, optional
        if True the nan values are filled with the heads in the cells below

    Returns
    -------
    head_ar : np.ndarray
        heads array.
    """
    hdobj = flopy.utils.HeadFile(fname_hds)
    head = hdobj.get_alldata()
    head[head == head.max()] = np.nan

    if gridtype == 'unstructured':
        head_filled = np.ones((head.shape[0], head.shape[1], head.shape[3])) * np.nan
        
        for t in range(head.shape[0]):
            for lay in range(head.shape[1] - 1, -1, -1):
                head_filled[t][lay] = head[t][lay][0]
                if lay < (head.shape[1] - 1):
                    if fill_nans:
                        head_filled[t][lay] = np.where(np.isnan(head_filled[t][lay]),
                                                       head_filled[t][lay + 1],
                                                       head_filled[t][lay])

    elif gridtype =='structured':
        head_filled = np.zeros_like(head)
        for t in range(head.shape[0]):
            for lay in range(head.shape[1] - 1, -1, -1):
                head_filled[t][lay] = head[t][lay]
                if lay < (head.shape[1] - 1):
                    if fill_nans:
                        head_filled[t][lay] = np.where(np.isnan(head_filled[t][lay]),
                                                       head_filled[t][lay + 1],
                                                       head_filled[t][lay])
    else:
        raise ValueError('wrong gridtype')
        
    return head_filled


def download_mfbinaries(binpath=None, version='6.0'):
    """Download and unpack platform-specific modflow binaries.

    Source: USGS

    Parameters
    ----------
    binpath : str, optional
        path to directory to download binaries to, if it doesnt exist it
        is created. Default is None which sets dir to nlmod/bin.
    version : str, optional
        version string, by default '6.0'
    """
    if binpath is None:
        binpath = os.path.join(os.path.dirname(__file__), "..", "bin")
    pltfrm = get_platform(None)
    # Download and unpack mf6 exes
    getmfexes(pth=binpath, version=version, pltfrm=pltfrm)
    
    
