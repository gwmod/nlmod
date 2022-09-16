# -*- coding: utf-8 -*-
"""Created on Fri Dec  3 11:58:49 2021.

@author: oebbe
"""

import numpy as np
import xarray as xr

from .. import cache, util


@cache.cache_netcdf
def chd_at_model_edge(ds, idomain):
    """get data array which is 1 at every active cell (defined by idomain) at
    the boundaries of the model (xmin, xmax, ymin, ymax). Other cells are 0.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    idomain : xarray.DataArray
        idomain used to get active cells and shape of DataArray

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with chd array
    """
    # add constant head cells at model boundaries
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        raise NotImplementedError("model edge not yet calculated for rotated grids")

    # get mask with grid edges
    xmin = ds["x"] == ds["x"].min()
    xmax = ds["x"] == ds["x"].max()
    ymin = ds["y"] == ds["y"].min()
    ymax = ds["y"] == ds["y"].max()

    ds_out = util.get_ds_empty(ds)

    if ds.gridtype == "structured":
        mask2d = ymin | ymax | xmin | xmax

        # assign 1 to cells that are on the edge and have an active idomain
        ds_out["chd"] = xr.zeros_like(idomain)
        for lay in ds.layer:
            ds_out["chd"].loc[lay] = np.where(mask2d & (idomain.loc[lay] == 1), 1, 0)

    elif ds.gridtype == "vertex":
        mask = np.where([xmin | xmax | ymin | ymax])[1]

        # assign 1 to cells that are on the edge, have an active idomain
        ds_out["chd"] = xr.zeros_like(idomain)
        ds_out["chd"].loc[:, mask] = 1
        ds_out["chd"] = xr.where(idomain == 1, ds_out["chd"], 0)

    return ds_out
