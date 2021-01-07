# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:28:21 2020

@author: oebbe
"""
import os
import pandas as pd
import xarray as xr
import flopy
import datetime as dt
import numpy as np

from . import util
from . import mgrid


def get_well(model_ds,
             df_locaties_pwn,
             df_debiet_pwn,
             gwf,
             name='well_pwn',
             cachedir=None,
             use_cache=False,
             verbose=False):
    """ Get well dataset from dataframes with locations and extraction rates.

    Parameters
    ----------
    model_ds : xr.DataSet
        dataset containing relevant model information
    df_locaties_pwn : pandas DataFrame
        Dataframe with locations of the extraction wells.
    df_debiet_pwn :  pandas DataFrame
        Dataframe with extraction rates of the extraction wells.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached well data is used. The default is False.
    verbose : bool, optional
        print additional information to the screen. The default is False.

    Returns
    -------
    model_ds : xr.DataSet
        dataset with spatial model data including the well raster and 
        timeseries

    """

    model_ds = util.get_cache_netcdf(use_cache, cachedir, 'well_model_ds.nc',
                                     add_pwn_onttrekking_to_model_ds, 
                                     model_ds, name=name,
                                     df_locaties_pwn=df_locaties_pwn, 
                                     df_debiet_pwn=df_debiet_pwn,
                                     gwf=gwf, verbose=verbose)

    return model_ds


def read_excel_locaties_pwn(datadir, excel_name='OntledingNHDmodel_.xlsx',
                            sheet_name='Putclusters_model2016'):
    """ read excel with locations of extraction wells from PWN


    Parameters
    ----------
    datadir : str
        directory where the excel file is located in the onttrekkingen directory.
    excel_name : str, optional
        Name of the excel file with the PWN onttrekking locations. 
        The default is 'OntledingNHDmodel_.xlsx'.
    sheet_name : str, optional
        sheetname in the excel file with the locations. 
        The default is 'Putclusters_model2016'.

    Returns
    -------
    df_locaties_pwn : pandas DataFrame
        Dataframe with locations of the extraction wells.

    """
    df_locaties_pwn = pd.read_excel(os.path.join(datadir, 'onttrekkingen', excel_name),
                                    sheet_name=sheet_name,
                                    skiprows=2, usecols=[2, 3, 4, 5],
                                    engine="openpyxl")
    df_locaties_pwn.putcluster = df_locaties_pwn.putcluster.astype(str)

    return df_locaties_pwn


def read_excel_debieten_pwn(datadir, excel_name='OntledingNHDmodel_.xlsx',
                            sheet_name='Putclusters_model2016'):
    """ read excel with extraction rates of the PWN extraction wells    

    datadir : str
        directory where the excel file is located in the onttrekkingen directory.
    excel_name : str, optional
        Name of the excel file with the PWN onttrekking locations. 
        The default is 'OntledingNHDmodel_.xlsx'.
    sheet_name : str, optional
        sheetname in the excel file with the extraction rates. The default is 
        'Putclusters_model2016'.

    Returns
    -------
    df_debiet_pwn : pandas DataFrame
        Dataframe with extraction rates of the extraction wells.

    """
    df_debiet_pwn = pd.read_excel(os.path.join(datadir, 'onttrekkingen', excel_name),
                                  sheet_name=sheet_name,
                                  skiprows=2, index_col=0,
                                  engine="openpyxl",
                                  usecols=[8] + list(range(25, 210)),
                                  skipfooter=1053).T

    df_debiet_pwn.columns = df_debiet_pwn.columns.astype(int).astype(str)

    df_debiet_pwn.index = [pd.to_datetime(
        i[1:], format='%Y%m') for i in df_debiet_pwn.index]

    return df_debiet_pwn


def add_pwn_onttrekking_to_model_ds(model_ds, df_locaties_pwn, df_debiet_pwn,
                                    gwf, name='well_pwn'):
    """ add the extraction wells to the model dataset.


    Parameters
    ----------
    df_locaties_pwn : pandas DataFrame
        Dataframe with locations of the extraction wells.
    df_debiet_pwn :  pandas DataFrame
        Dataframe with extraction rates of the extraction wells.
    model_ds : xr.DataSet
        dataset containing relevant model information
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.

    Raises
    ------
    ValueError
        if the model period and the extraction rate period do not overlap a
        ValueError is raised.

    Returns
    -------
    model_ds_out : xr.DataSet
        dataset with new variables:
            - well_pwn with the names of all the extraction rates timeseries
            - well_pwn_{i} with the extraction rate time series.
    """

    model_ds_out = util.get_model_ds_empty(model_ds)
    # get well data array
    if (model_ds.gridtype == 'structured') and model_ds.steady_state:
        empty_time_array = np.zeros((model_ds.dims['y'],
                                     model_ds.dims['x']))
        model_ds_out[name] = xr.DataArray(empty_time_array,
                                          dims=('y', 'x'),
                                          coords={'x': model_ds.x,
                                                  'y': model_ds.y})

    elif (model_ds.gridtype == 'structured') and (not model_ds.steady_state):
        empty_time_array = np.zeros((model_ds.dims['y'],
                                     model_ds.dims['x'],
                                     model_ds.dims['time']))
        model_ds_out[name] = xr.DataArray(empty_time_array,
                                          dims=('y', 'x', 'time'),
                                          coords={'time': model_ds.time,
                                                  'x': model_ds.x,
                                                  'y': model_ds_out.y})

    elif (model_ds.gridtype == 'unstructured') and model_ds.steady_state:
        empty_time_array = np.zeros((model_ds.dims['cid']))
        model_ds_out[name] = xr.DataArray(empty_time_array,
                                          dims=('cid'),
                                          coords={'cid': model_ds.cid})
    elif (model_ds.gridtype == 'unstructured') and (not model_ds.steady_state):
        empty_time_array = np.zeros((model_ds.dims['cid'],
                                     model_ds.dims['time']))
        model_ds_out[name] = xr.DataArray(empty_time_array,
                                          dims=('cid', 'time'),
                                          coords={'time': model_ds.time,
                                                  'cid': model_ds.cid})

    # start en eindtijd
    start_ts = pd.Timestamp(model_ds.time.data[0])
    end_ts = pd.Timestamp(model_ds.time.data[-1])

    if model_ds.steady_state or model_ds.steady_start:
        start = dt.datetime(start_ts.year - 1, start_ts.month, start_ts.day)
        end = end_ts + pd.Timedelta(1, unit='D')
    else:
        start = start_ts - pd.Timedelta(1, unit='D')
        end = end_ts + pd.Timedelta(1, unit='D')

    # locatie put in model
    df_locaties_pwn['idx'] = [gwf.modelgrid.intersect(
        row['X'], row['Y']) for i, row in df_locaties_pwn.iterrows()]

    # voeg debiet toe aan model dataset
    for putcluster in df_debiet_pwn.columns:
        df_putcluster = df_locaties_pwn[df_locaties_pwn.putcluster == putcluster]

        # aantal putten per modelcel (idx) wordt gebruikt om het debiet te verdelen
        # iedere put evenveel debiet
        idx_count = df_putcluster.groupby('idx').count()

        for idx in idx_count.index:
            idx_factor = idx_count.loc[idx, 'putcluster'] / len(df_putcluster)
            debiet_ts = df_debiet_pwn.loc[:, putcluster] * idx_factor
            if debiet_ts.index[-1] < end:
                raise ValueError(
                    f'no pumping rate available at putcluster {putcluster} for date {end}')
            elif debiet_ts.index[0] > start:
                raise ValueError(
                    f'no pumping rate available at putcluster {putcluster} for date {start}')
            if model_ds.steady_state:
                debiet_average = debiet_ts[start:end].mean()
                if model_ds.gridtype == 'structured':
                    raise NotImplementedError
                elif model_ds.gridtype == 'unstructured':
                    model_ds_out[name].loc[idx] += debiet_average
            else:
                if model_ds.steady_start:
                    debiet_average = debiet_ts[start:end].mean()
                    debiet_ts = debiet_ts.reindex(
                        model_ds.time.data, method='bfill')
                    debiet_ts.loc[model_ds.time.data[0]] = debiet_average
                else:
                    debiet_ts = debiet_ts.reindex(
                        model_ds.time.data, method='bfill')

                if model_ds.gridtype == 'structured':
                    raise NotImplementedError
                elif model_ds.gridtype == 'unstructured':
                    model_ds_out[name].loc[idx, :] += debiet_ts.values

    return model_ds_out


def model_dataset_to_well(gwf, model_ds, name='well_pwn',
                          layer=2):
    """ create the well packages from the data in model_ds
    
    Note
    ----
    the layer in which the wells are placed should be active otherwise
    an error will be raised.


    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    model_ds : xr.DataSet
        dataset with well data
    name : str, optional
        name of the model dataset to use. The default is 'well_pwn'.

    Raises
    ------
    NotImplementedError
        raised is gridtype is structured.

    Returns
    -------
    well : flopy.mf6.ModflowGwfwel
        well package.

    """

    if model_ds.gridtype != 'unstructured':
        raise NotImplementedError(
            'this function is not yet available for structured grids')

    if model_ds.steady_state:
        well_spd = mgrid.data_array_unstructured_to_rec_list(model_ds,
                                                             model_ds[name] != 0,
                                                             col1=name,
                                                             layer=layer,
                                                             first_active_layer=False,
                                                             only_active_cells=False)

        well_pkg = flopy.mf6.ModflowGwfwel(gwf, filename=f'{gwf.name}.wel',
                                           pname='well',
                                           maxbound=len(well_spd),
                                           print_input=True,
                                           stress_period_data={0: well_spd})

        return well_pkg

    # transient well
    empty_str_array = np.zeros_like(model_ds['idomain'][0], dtype="S13")
    model_ds[f'{name}_name'] = xr.DataArray(empty_str_array,
                                            dims=('cid'),
                                            coords={'cid': model_ds.cid})
    model_ds[f'{name}_name'] = model_ds[f'{name}_name'].astype(str)

    # vind unieke debietreeksen
    well_unique_arr = np.unique(model_ds[name].data, axis=0)

    # verwijder reeksen waar debiet altijd 0 is
    well_unique_arr = well_unique_arr[(well_unique_arr != 0).all(axis=1)]

    # maak raster met unieke naam per debietreeks
    well_unique_dic = {}
    for i, unique_wel in enumerate(well_unique_arr):
        model_ds[f'{name}_name'][(model_ds[name].data == unique_wel).all(
            axis=1)] = f'{name}_{i}'
        well_unique_dic[f'{name}_{i}'] = unique_wel

    # create well package
    mask = model_ds[f'{name}_name'] != ''
    well_spd_data = mgrid.data_array_unstructured_to_rec_list(model_ds, mask,
                                                              col1=f'{name}_name',
                                                              layer=layer,
                                                              first_active_layer=False,
                                                              only_active_cells=False)

    well_pkg = flopy.mf6.ModflowGwfwel(gwf, filename=f'{gwf.name}.wel',
                                       pname='well',
                                       maxbound=len(well_spd_data),
                                       print_input=True,
                                       stress_period_data={0: well_spd_data})

    # create timeseries packages
    for i, key in enumerate(well_unique_dic.keys()):
        well_val = list(well_unique_dic[key].data) + [0.0]
        time_steps_wel = list(model_ds['time_steps'].data) + [model_ds.nper]
        well_rate = list(zip(time_steps_wel, well_val))
        if i == 0:
            well_pkg.ts.initialize(filename=f'{key}.ts',
                                   timeseries=well_rate,
                                   time_series_namerecord=key,
                                   interpolation_methodrecord='stepwise')
        else:
            well_pkg.ts.append_package(filename=f'{key}.ts',
                                       timeseries=well_rate,
                                       time_series_namerecord=key,
                                       interpolation_methodrecord='stepwise')

    return well_pkg
