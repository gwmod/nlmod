# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:58:49 2021

@author: oebbe
"""

import xarray as xr
import numpy as np

from .. import util, cache

@cache.cache_netcdf
def get_chd_at_model_edge(model_ds, idomain):
    """get data array which is 1 at every active cell (defined by idomain) at
    the boundaries of the model (xmin, xmax, ymin, ymax). Other cells are 0.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    idomain : xarray.DataArray
        idomain used to get active cells and shape of DataArray

    Returns
    -------
    model_ds_out : xarray.Dataset
        dataset with chd array
    """
    
    # add constant head cells at model boundaries

    # get mask with grid edges
    xmin = model_ds['x'] == model_ds['x'].min()
    xmax = model_ds['x'] == model_ds['x'].max()
    ymin = model_ds['y'] == model_ds['y'].min()
    ymax = model_ds['y'] == model_ds['y'].max()
    
    model_ds_out = util.get_model_ds_empty(model_ds)

    if model_ds.gridtype == 'structured':
        mask2d = (ymin | ymax | xmin | xmax)

        # assign 1 to cells that are on the edge and have an active idomain
        model_ds_out['chd'] = xr.zeros_like(idomain)
        for lay in model_ds.layer:
            model_ds_out['chd'].loc[lay] = np.where(
                mask2d & (idomain.loc[lay] == 1), 1, 0)
        
    elif model_ds.gridtype == 'vertex':
        mask = np.where([xmin | xmax | ymin | ymax])[1]

        # assign 1 to cells that are on the edge, have an active idomain
        model_ds_out['chd'] = xr.zeros_like(idomain)
        model_ds_out['chd'].loc[:, mask] = 1
        model_ds_out['chd'] = xr.where(idomain == 1,
                                       model_ds_out['chd'], 0)
        

    return model_ds_out