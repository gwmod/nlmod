# -*- coding: utf-8 -*-
"""
add knmi precipitation and evaporation to a modflow model

"""


import numpy as np
import xarray as xr

from tqdm import tqdm

import flopy
from .. import mdims
from . import mfpackages


def model_datasets_to_rch(gwf, model_ds, print_input=False):
    """ convert the recharge data in the model dataset to a recharge package 
    with time series.

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    model_ds : xr.DataSet
        dataset containing relevant model grid information
    print_input : bool, optional
        value is passed to flopy.mf6.ModflowGwfrch() to determine if input
        should be printed to the lst file. Default is False

    Returns
    -------
    rch : flopy.mf6.modflow.mfgwfrch.ModflowGwfrch
        recharge package

    """
    # check for nan values
    if model_ds['recharge'].isnull().any():
        raise ValueError('please remove nan values in recharge data array')

    # get stress period data
    if model_ds.steady_state:
        mask = model_ds['recharge'] != 0
        if model_ds.gridtype == 'structured':
            rch_spd_data = mdims.data_array_2d_to_rec_list(
                model_ds, mask, col1='recharge',
                first_active_layer=True,
                only_active_cells=False)
        elif model_ds.gridtype == 'unstructured':
            rch_spd_data = mdims.data_array_1d_unstr_to_rec_list(
                model_ds, mask, col1='recharge',
                first_active_layer=True,
                only_active_cells=False)

        # create rch package
        rch = flopy.mf6.ModflowGwfrch(gwf,
                                      filename=f'{gwf.name}.rch',
                                      pname=f'{gwf.name}',
                                      fixed_cell=False,
                                      maxbound=len(rch_spd_data),
                                      print_input=True,
                                      stress_period_data={0: rch_spd_data})

        return rch

    # transient recharge
    if model_ds.gridtype == 'structured':
        empty_str_array = np.zeros_like(model_ds['idomain'][0], dtype="S13")
        model_ds['rch_name'] = xr.DataArray(empty_str_array,
                                            dims=('y', 'x'),
                                            coords={'y': model_ds.y,
                                                    'x': model_ds.x})
        model_ds['rch_name'] = model_ds['rch_name'].astype(str)
        # dimension check
        if model_ds['recharge'].dims == ('time', 'y', 'x'):
            axis = 0
            rch_2d_arr = model_ds['recharge'].data.reshape(
                (model_ds.dims['time'], model_ds.dims['x'] * model_ds.dims['y'])).T

            # check if reshaping is correct
            if not (model_ds['recharge'].values[:, 0, 0] == rch_2d_arr[0]).all():
                raise ValueError(
                    'reshaping recharge to calculate unique time series did not work out as expected')

        elif model_ds['recharge'].dims == ('y', 'x', 'time'):
            axis = 2
            rch_2d_arr = model_ds['recharge'].data.reshape(
                (model_ds.dims['x'] * model_ds.dims['y'], model_ds.dims['time']))

            # check if reshaping is correct
            if not (model_ds['recharge'].values[0, 0, :] == rch_2d_arr[0]).all():
                raise ValueError(
                    'reshaping recharge to calculate unique time series did not work out as expected')

        else:
            raise ValueError('expected dataarray with 3 dimensions'
                             f'(time, y and x) or (y, x and time), not {model_ds["recharge"].dims}')

        rch_unique_arr = np.unique(rch_2d_arr, axis=0)
        rch_unique_dic = {}
        for i, unique_rch in enumerate(rch_unique_arr):
            model_ds['rch_name'].data[np.isin(
                model_ds['recharge'].values, unique_rch).all(axis=axis)] = f'rch_{i}'
            rch_unique_dic[f'rch_{i}'] = unique_rch

        mask = model_ds['rch_name'] != ''
        rch_spd_data = mdims.data_array_2d_to_rec_list(model_ds, mask,
                                                       col1='rch_name',
                                                       first_active_layer=True,
                                                       only_active_cells=False)

    elif model_ds.gridtype == 'unstructured':
        empty_str_array = np.zeros_like(model_ds['idomain'][0], dtype="S13")
        model_ds['rch_name'] = xr.DataArray(empty_str_array,
                                            dims=('cid'),
                                            coords={'cid': model_ds.cid})
        model_ds['rch_name'] = model_ds['rch_name'].astype(str)

        # dimension check
        if model_ds['recharge'].dims == ('cid', 'time'):
            rch_2d_arr = model_ds['recharge'].values
        elif model_ds['recharge'].dims == ('time', 'cid'):
            rch_2d_arr = model_ds['recharge'].values.T
        else:
            raise ValueError('expected dataarray with 2 dimensions'
                             f'(time, cid) or (cid, time), not {model_ds["recharge"].dims}')

        rch_unique_arr = np.unique(rch_2d_arr, axis=0)
        rch_unique_dic = {}
        for i, unique_rch in enumerate(rch_unique_arr):
            model_ds['rch_name'][(rch_2d_arr == unique_rch).all(
                axis=1)] = f'rch_{i}'
            rch_unique_dic[f'rch_{i}'] = unique_rch

        mask = model_ds['rch_name'] != ''
        rch_spd_data = mdims.data_array_1d_unstr_to_rec_list(model_ds, mask,
                                                             col1='rch_name',
                                                             first_active_layer=True,
                                                             only_active_cells=False)

    # create rch package
    rch = flopy.mf6.ModflowGwfrch(gwf, filename=f'{gwf.name}.rch',
                                  pname='rch',
                                  fixed_cell=False,
                                  maxbound=len(rch_spd_data),
                                  print_input=print_input,
                                  stress_period_data={0: rch_spd_data})

    # get timesteps
    tdis_perioddata = mfpackages.get_tdis_perioddata(model_ds)
    perlen_arr = [t[0] for t in tdis_perioddata]
    time_steps_rch = [0.0] + np.array(perlen_arr).cumsum().tolist()

    # create timeseries packages
    for i, key in tqdm(enumerate(rch_unique_dic.keys()),
                       total=len(rch_unique_dic.keys()),
                       desc="Building ts packages rch"):
        # add extra time step to the time series object (otherwise flopy fails)
        recharge_val = list(rch_unique_dic[key]) + [0.0]

        recharge = list(zip(time_steps_rch, recharge_val))
        if i == 0:
            rch.ts.initialize(filename=f'{key}.ts',
                              timeseries=recharge,
                              time_series_namerecord=key,
                              interpolation_methodrecord='stepwise')
        else:
            rch.ts.append_package(filename=f'{key}.ts',
                                  timeseries=recharge,
                                  time_series_namerecord=key,
                                  interpolation_methodrecord='stepwise')

    return rch
