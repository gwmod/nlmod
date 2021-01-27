# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:50:48 2020

@author: oebbe
"""

import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr


def get_model_ds_time(model_name, model_ws, start_time,
                      steady_state, steady_start=False,
                      mfversion='mf6',
                      time_units='DAYS', transient_timesteps=0,
                      perlen=1.0, steady_perlen=3650,
                      nstp=1, tsmult=1.0):
    """ Get a model dataset with the time variant data

    Parameters
    ----------
    model_name : str
        name of the model. Cannot have more than 16 characters (modflow)
    model_ws : str
        workspace of the model
    start_time : str or datetime
        start time of the model. This is the start_time of the transient time
        steps. 
    steady_state : bool
        if True the model is steady state with one time step.
    steady_start : bool
        if True the model is transient with a steady state start time step.
    mfversion : str, optional
        modflow version, can be mf2005 of mf6. Default is mf6
    time_units : str, optional
        time unit of the model. The default is 'DAYS'.
    transient_timesteps : int, optional
        number of transient time steps. The default is 0.
    perlen : float, int, list or np.array, optional
        length of each timestep depending on the type:
            - float or int: this is the length of all the time steps. If 
            steady_start is True the length of the first time step is defined
            by steady_perlen
            - list or array: the items are the length per timestep.
            the length of perlen should match the number of transient 
            timesteps (or transient timesteps +1 if steady_start is True) 
        The default is 1.0.
    steady_perlen : float, optional
        time step length of the first steady state timestep, this is used to 
        Only used if steady_start is True. Default is 3650 (~10 years)
    nstp : int, optional
        DESCRIPTION. The default is 1.
    tsmult : float, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with time variant model data

    """
    # checks
    if len(model_name) > 16 and mfversion == 'mf6':
        raise ValueError('model_name can not have more than 16 characters')
    elif time_units!='DAYS':
        raise NotImplementedError()
    elif steady_start and (transient_timesteps==0):
        raise ValueError('illegal combination of steady_start and transient'\
                         'timesteps please use steady_state=True')
    

    start_time_dt = pd.to_datetime(start_time)

    if steady_state:
        nper = 1
        start_time_dt = pd.to_datetime(start_time)
        time_dt =[start_time_dt]
    elif steady_start:   
        nper = 1 + transient_timesteps

        if isinstance(perlen, float) or isinstance(perlen, int):
            start_tran = pd.to_datetime(start_time) 
            start_time_dt = start_tran - dt.timedelta(days=steady_perlen)
            
            time_dt = pd.date_range(start_tran - pd.to_timedelta(perlen, unit=time_units), 
                                    start_tran + pd.to_timedelta((transient_timesteps - 1) * perlen,
                                    unit=time_units), periods=nper)
            time_dt.values[0] = start_time_dt
            
        elif isinstance(perlen, list) or isinstance(perlen, np.array):
            assert len(perlen)==nper
            start_tran = pd.to_datetime(start_time) 
            start_time_dt = start_tran - dt.timedelta(days=perlen[0])
            
            time_dt = [start_time_dt]
            for i, p in enumerate(perlen[:-1]):
                time_dt += [time_dt[i] + pd.to_timedelta(p, unit=time_units)] 
        else:
            raise ValueError(f'did not recognise perlen data type {perlen}')
            
    else:
        nper = transient_timesteps
        if isinstance(perlen, float) or isinstance(perlen, int):
            start_time_dt = pd.to_datetime(start_time) 
            time_dt = pd.date_range(start_time_dt, 
                                    start_time_dt + pd.to_timedelta((transient_timesteps - 1) * perlen,
                                    unit=time_units), periods=nper)
        elif isinstance(perlen, list) or isinstance(perlen, np.array):
            assert len(perlen)==nper
            start_time_dt = pd.to_datetime(start_time) 
            time_dt = [start_time_dt]
            for i, p in enumerate(perlen[:-1]):
                time_dt += [time_dt[i] + pd.to_timedelta(p, unit=time_units)]  
        else:
            raise ValueError(f'did not recognise perlen data type {perlen}')

    time_steps = list(range(nper))

    model_ds = xr.Dataset(coords={'time': time_dt})

    model_ds['time_steps'] = xr.DataArray(time_steps, dims=('time'),
                                          coords={'time': model_ds.time})
    model_ds.attrs['model_name'] = model_name
    model_ds.attrs['model_ws'] = model_ws
    model_ds.attrs['time_units'] = time_units
    model_ds.attrs['start_time'] = str(start_time_dt)
    model_ds.attrs['nper'] = nper
    model_ds.attrs['perlen'] = perlen
    model_ds.attrs['nstp'] = nstp
    model_ds.attrs['tsmult'] = tsmult
    model_ds.attrs['mfversion'] = mfversion

    # netcdf files cannot handle booleans
    model_ds.attrs['steady_start'] = int(steady_start)
    model_ds.attrs['steady_state'] = int(steady_state)

    return model_ds
