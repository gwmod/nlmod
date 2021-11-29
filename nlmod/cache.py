# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:45:23 2021

@author: oebbe
"""
import os
import functools
import numpy as np
import xarray as xr
import pickle
import numbers

import logging
logger = logging.getLogger(__name__)


def _check_model_ds(model_ds, model_ds2, check_grid=True):
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

    Raises
    ------
    ValueError
        if the gridtype of model_ds is not structured or unstructured.

    Returns
    -------
    bool
        True if the two datasets have the same grid and time discretization.
    """
    if 'time' in model_ds.dims:
        check_time = True

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
                    check_time = (model_ds['time'].data
                                  == model_ds2['time'].data).all()
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
                    check_time = (model_ds['time'].data
                                  == model_ds2['time'].data).all()
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


def _is_valid_cache(func_args_dic, func_args_dic_cache):
    """ checks if two dictionaries with function arguments are identical by
    checking:
        1. if they have the same keys
        2. if the items have the same dtype
        3. if the items have the same values (only if they have the type int, 
                                              float, bool, str, bytes, list, 
                                              tuple or np.ndarray)

    Parameters
    ----------
    func_args_dic : dictionary
        dictionary with all the args and kwargs of a function call.
    func_args_dic_cache : dictionary
        dictionary with all the args and kwargs of a previous function call of
        which the results are cached.

    Returns
    -------
    bool
        if True the dictionaries are identical which means that the cached
        data was created using the same function arguments as the requested 
        data.

    """
    for key, item in func_args_dic.items():
        # check if cache and function call have same argument names
        if not key in func_args_dic_cache.keys():
            logger.info('cache was created using different function arguments, do not use cached data')
            return False

        # check if cache and function call have same argument types
        if type(item) != type(func_args_dic_cache[key]):
            logger.info('cache was created using different function argument types, do not use cached data')
            return False

        # check if cache and function call have same argument values
        if item is None:
            # Value of None type is always None so the check happened in previous if statement
            pass
        elif isinstance(item, (numbers.Number, bool, str, bytes, list, tuple)):
            if item != func_args_dic_cache[key]:
                logger.info('cache was created using different function argument values, do not use cached data')
                return False
        elif isinstance(item, np.ndarray):
            if not np.array_equal(item, func_args_dic_cache[key]):
                logger.info('cache was created using different numpy array values, do not use cached data')
                return False
        elif isinstance(item, xr.DataArray):
            if not item.equals(func_args_dic_cache[key]):
                logger.info('cache was created using different DataArrays, do not use cached data')
                return False
        elif isinstance(item, dict):
            if not _is_valid_cache(item, func_args_dic_cache[key]):
                logger.info('cache was created using different dictionaries, do not use cached data')
                return False
        else:
            logger.info('cannot check if cache is valid, assuming invalid cache')
            logger.info(f'function argument of type {type(item)}')
            return False

    return True


def cache_netcdf(func):

    @functools.wraps(func)
    def decorator(*args, cachedir=None, cachename=None, **kwargs):

        if cachedir is None or cachename is None:
            return func(*args, **kwargs)

        if not cachename.endswith('.nc'):
            cachename += '.nc'

        fname_cache = os.path.join(cachedir, cachename)

        # check if cache is valid
        fname_pickle_cache = fname_cache.replace('.nc', '.pklz')
        func_args_dic = {f'arg{i}': args[i] for i in range(len(args))}
        func_args_dic.update(kwargs)

        # remove xarray dataset from function arguments
        dataset = None
        for key in list(func_args_dic.keys()):
            if isinstance(func_args_dic[key], xr.Dataset):
                if dataset is not None:
                    raise TypeError('function was called with multiple xarray dataset arguments')
                dataset = func_args_dic.pop(key)

        if os.path.exists(fname_cache) and os.path.exists(fname_pickle_cache):
            with open(fname_pickle_cache, "rb") as f:
                func_args_dic_cache = pickle.load(f)

            valid_cache = _is_valid_cache(func_args_dic, func_args_dic_cache)

            if valid_cache:
                cached_ds = xr.open_dataset(fname_cache)
                if dataset is None:
                    logger.info(f'using cached data -> {cachename}')
                    return cached_ds

                # check dataset
                if _check_model_ds(dataset, cached_ds):
                    logger.info(f'using cached data -> {cachename}')
                    return cached_ds

        # create cache if cache is invalid
        result = func(*args, **kwargs)

        logger.info(f'caching data -> {cachename}')

        if isinstance(result, xr.Dataset):
            result.to_netcdf(fname_cache)

            with open(fname_pickle_cache, 'wb') as fpklz:
                pickle.dump(func_args_dic, fpklz)
        else:
            raise TypeError(f'expected xarray Dataset, got {type(result)} instead')

        return result

    return decorator
