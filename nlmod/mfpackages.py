# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:20:34 2021

@author: oebbe
"""

import flopy
from nlmod import mtime

def sim_tdis_gwf_ims_from_model_ds(model_ds, verbose=False):
    
    # start creating model
    if verbose:
        print('creating modflow SIM, TDIS, GWF and IMS')

     # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(sim_name=model_ds.model_name, 
                                 exe_name=model_ds.mfversion,
                                 version=model_ds.mfversion, 
                                 sim_ws=model_ds.model_ws)
    
    tdis_perioddata = get_tdis_perioddata(model_ds)
    
    # Create the Flopy temporal discretization object
    flopy.mf6.modflow.mftdis.ModflowTdis(sim, pname='tdis',
                                         time_units=model_ds.time_units,
                                         nper=len(model_ds.time),
                                         start_date_time=model_ds.start_time,
                                         perioddata=tdis_perioddata)

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = '{}.nam'.format(model_ds.model_name)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_ds.model_name,
                               model_nam_file=model_nam_file)

    # Create the Flopy iterative model solver (ims) Package object
    flopy.mf6.modflow.mfims.ModflowIms(sim, pname='ims',
                                       complexity='MODERATE')
    
    return sim, gwf

def get_tdis_perioddata(model_ds):
    """ Get tdis_perioddata from model_ds

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with time variant model data

    Raises
    ------
    NotImplementedError
        cannot handle timesteps with variable step length.

    Returns
    -------
    tdis_perioddata : [perlen, nstp, tsmult]
        * perlen (double) is the length of a stress period.
        * nstp (integer) is the number of time steps in a stress period.
        * tsmult (double) is the multiplier for the length of successive time
          steps. The length of a time step is calculated by multiplying the
          length of the previous time step by TSMULT. The length of the first
          time step, :math:`\Delta t_1`, is related to PERLEN, NSTP, and
          TSMULT by the relation :math:`\Delta t_1= perlen \frac{tsmult -
          1}{tsmult^{nstp}-1}`.

    """
    
    try:
        float(model_ds.perlen)
        tdis_perioddata = [(model_ds.perlen, model_ds.nstp, model_ds.tsmult)] * int(model_ds.nper)
    except:
        raise NotImplementedError('variable time step length not yet implemented')
    
    # netcdf does not support multi-dimensional array attributes
    #model_ds.attrs['tdis_perioddata'] = tdis_perioddata
    
    return tdis_perioddata