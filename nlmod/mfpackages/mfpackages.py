# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 17:20:34 2021.

@author: oebbe
"""
import logging
import numbers
import os
import sys

import flopy
import numpy as np
import xarray as xr

from .. import mdims
from . import recharge

logger = logging.getLogger(__name__)


def sim_tdis_gwf_ims_from_model_ds(model_ds,
                                   complexity='MODERATE',
                                   exe_name=None):
    """create sim, tdis, gwf and ims package from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start_time,
        perlen, nstp, tsmult
    exe_name: str, optional
        path to modflow executable, default is None, which assumes binaries
        are available in nlmod/bin directory. Binaries can be downloaded
        using `nlmod.util.download_mfbinaries()`.

    Returns
    -------
    sim : flopy MFSimulation
        simulation object.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    """

    # start creating model
    logger.info('creating modflow SIM, TDIS, GWF and IMS')

    if exe_name is None:
        exe_name = os.path.join(os.path.dirname(__file__),
                                '..', '..', 'bin', model_ds.mfversion)
        if sys.platform.startswith('win'):
            exe_name += ".exe"

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(sim_name=model_ds.model_name,
                                 exe_name=exe_name,
                                 version=model_ds.mfversion,
                                 sim_ws=model_ds.model_ws)

    tdis_perioddata = get_tdis_perioddata(model_ds)

    # Create the Flopy temporal discretization object
    flopy.mf6.modflow.mftdis.ModflowTdis(sim,
                                         pname='tdis',
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
                                       complexity=complexity)

    return sim, gwf


def dis_from_model_ds(model_ds, gwf, length_units='METERS',
                      angrot=0):
    """get discretisation package from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    angrot : int or float, optional
        rotation angle. The default is 0.

    Returns
    -------
    dis : TYPE
        discretisation package.
    """

    if model_ds.gridtype != 'structured':
        raise ValueError(
            f'cannot create dis package for gridtype -> {model_ds.gridtype}')

    # check attributes
    for att in ['delr', 'delc']:
        if isinstance(model_ds.attrs[att], np.float32):
            model_ds.attrs[att] = float(model_ds.attrs[att])

    dis = flopy.mf6.ModflowGwfdis(gwf,
                                  pname='dis',
                                  length_units=length_units,
                                  xorigin=model_ds.extent[0],
                                  yorigin=model_ds.extent[2],
                                  angrot=angrot,
                                  nlay=model_ds.dims['layer'],
                                  nrow=model_ds.dims['y'],
                                  ncol=model_ds.dims['x'],
                                  delr=model_ds.delr,
                                  delc=model_ds.delc,
                                  top=model_ds['top'].data,
                                  botm=model_ds['bot'].data,
                                  idomain=model_ds['idomain'].data,
                                  filename=f'{model_ds.model_name}.dis')

    return dis


def disv_from_model_ds(model_ds, gwf, gridprops,
                       length_units='METERS',
                       angrot=0):
    """get discretisation vertices package from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    gridprops : dictionary
        dictionary with grid properties output from gridgen.
    length_units : str, optional
        length unit. The default is 'METERS'.
    angrot : int or float, optional
        rotation angle. The default is 0.

    Returns
    -------
    disv : flopy ModflowGwfdisv
        disv package
    """

    disv = flopy.mf6.ModflowGwfdisv(gwf,
                                    idomain=model_ds['idomain'].data,
                                    xorigin=model_ds.extent[0],
                                    yorigin=model_ds.extent[2],
                                    length_units=length_units,
                                    angrot=angrot,
                                    **gridprops)

    return disv


def npf_from_model_ds(model_ds, gwf, icelltype=0,
                      save_flows=False,
                      **kwargs):
    """get node property flow package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    icelltype : int, optional
        celltype. The default is 0.
    save_flows : bool, optional
        value is passed to flopy.mf6.ModflowGwfnpf() to determine if cell by
        cell flows should be saved to the cbb file. Default is False

    Raises
    ------
    NotImplementedError
        only icelltype 0 is implemented.

    Returns
    -------
    npf : flopy ModflowGwfnpf
        npf package.
    """

    npf = flopy.mf6.ModflowGwfnpf(gwf,
                                  pname='npf',
                                  icelltype=icelltype,
                                  k=model_ds['kh'].data,
                                  k33=model_ds['kv'].data,
                                  save_flows=save_flows,
                                  **kwargs)

    return npf


def ghb_from_model_ds(model_ds, gwf, da_name):
    """get general head boundary from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    da_name : str
        name of the ghb files in the model dataset.

    Raises
    ------
    ValueError
        raised if gridtype is not structured or unstructured.

    Returns
    -------
    ghb : flopy ModflowGwfghb
        ghb package
    """

    if model_ds.gridtype == 'structured':
        ghb_rec = mdims.data_array_2d_to_rec_list(model_ds,
                                                  model_ds[f'{da_name}_cond'] != 0,
                                                  col1=f'{da_name}_peil',
                                                  col2=f'{da_name}_cond',
                                                  first_active_layer=True,
                                                  only_active_cells=False,
                                                  layer=0)
    elif model_ds.gridtype == 'unstructured':
        ghb_rec = mdims.data_array_1d_unstr_to_rec_list(model_ds,
                                                        model_ds[f'{da_name}_cond'] != 0,
                                                        col1=f'{da_name}_peil',
                                                        col2=f'{da_name}_cond',
                                                        first_active_layer=True,
                                                        only_active_cells=False,
                                                        layer=0)
    else:
        raise ValueError(f'did not recognise gridtype {model_ds.gridtype}')

    if len(ghb_rec) > 0:
        ghb = flopy.mf6.ModflowGwfghb(gwf, print_input=True,
                                      maxbound=len(ghb_rec),
                                      stress_period_data=ghb_rec,
                                      save_flows=True)
        return ghb

    else:
        print('no ghb cells added')

        return None


def ic_from_model_ds(model_ds, gwf,
                     starting_head='starting_head'):
    """get initial condictions package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    starting_head : str, float or int, optional
        if type is int or float this is the starting head for all cells
        If the type is str the data variable from model_ds is used as starting
        head. The default is 'starting_head'.

    Returns
    -------
    ic : flopy ModflowGwfic
        ic package
    """
    if isinstance(starting_head, str):
        pass
    elif isinstance(starting_head, numbers.Number):
        model_ds['starting_head'] = starting_head * \
            xr.ones_like(model_ds['idomain'])
        model_ds['starting_head'].attrs['units'] = 'mNAP'    
        starting_head = 'starting_head'

    ic = flopy.mf6.ModflowGwfic(gwf, pname='ic',
                                strt=model_ds[starting_head].data)

    return ic


def sto_from_model_ds(model_ds, gwf,
                      sy=0.2, ss=0.000001,
                      iconvert=1, save_flows=False):
    """get storage package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    sy : float, optional
        DESCRIPTION. The default is 0.2.
    ss : float, optional
        specific storage. The default is 0.000001.
    iconvert : int, optional
        DESCRIPTION. The default is 1.
    save_flows : bool, optional
        value is passed to flopy.mf6.ModflowGwfsto() to determine if flows
        should be saved to the cbb file. Default is False

    Returns
    -------
    sto : flopy ModflowGwfsto
        sto package
    """

    if model_ds.steady_state:
        return None
    else:
        if model_ds.steady_start:
            sts_spd = {0: True}
            trn_spd = {1: True}
        else:
            sts_spd = None
            trn_spd = {0: True}

        sto = flopy.mf6.ModflowGwfsto(gwf, pname='sto',
                                      save_flows=save_flows,
                                      iconvert=iconvert,
                                      ss=ss, sy=sy, steady_state=sts_spd,
                                      transient=trn_spd)
        return sto


def chd_at_model_edge_from_model_ds(model_ds, gwf, head='starting_head'):
    """get constant head boundary at the model's edges from the model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    head : str, optional
        name of data variable in model_ds that is used as the head in the chd
        cells. The default is 'starting_head'.

    Returns
    -------
    chd : flopy ModflowGwfchd
        chd package
    """
    # add constant head cells at model boundaries

    # get mask with grid edges
    xmin = model_ds['x'] == model_ds['x'].min()
    xmax = model_ds['x'] == model_ds['x'].max()
    ymin = model_ds['y'] == model_ds['y'].min()
    ymax = model_ds['y'] == model_ds['y'].max()

    if model_ds.gridtype == 'structured':
        mask2d = (ymin | ymax | xmin | xmax)

        # assign 1 to cells that are on the edge and have an active idomain
        model_ds['chd'] = xr.zeros_like(model_ds['idomain'])
        for lay in model_ds.layer:
            model_ds['chd'].loc[lay] = np.where(
                mask2d & (model_ds['idomain'].loc[lay] == 1), 1, 0)

        # get the stress_period_data
        chd_rec = mdims.data_array_3d_to_rec_list(model_ds,
                                                  model_ds['chd'] != 0,
                                                  col1=head)
    elif model_ds.gridtype == 'unstructured':
        mask = np.where([xmin | xmax | ymin | ymax])[1]

        # assign 1 to cells that are on the edge, have an active idomain
        model_ds['chd'] = xr.zeros_like(model_ds['idomain'])
        model_ds['chd'].loc[:, mask] = 1
        model_ds['chd'] = xr.where(model_ds['idomain'] == 1,
                                   model_ds['chd'], 0)

        # get the stress_period_data
        cellids = np.where(model_ds['chd'])
        chd_rec = list(zip(zip(cellids[0],
                               cellids[1]),
                           [1.0] * len(cellids[0])))

    chd = flopy.mf6.ModflowGwfchd(gwf, pname='chd',
                                  maxbound=len(chd_rec),
                                  stress_period_data=chd_rec,
                                  save_flows=True)

    return chd


def surface_drain_from_model_ds(model_ds, gwf, surface_drn_cond=1000):
    """get surface level drain (maaivelddrainage in Dutch) from the model
    dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    surface_drn_cond : int or float, optional
        conductivity of the surface drain. The default is 1000.

    Returns
    -------
    drn : flopy ModflowGwfdrn
        drn package
    """

    model_ds.attrs['surface_drn_cond'] = surface_drn_cond
    mask = model_ds['ahn'].notnull()
    if model_ds.gridtype == 'structured':
        drn_rec = mdims.data_array_2d_to_rec_list(model_ds, mask, col1='ahn',
                                                  first_active_layer=True,
                                                  only_active_cells=False,
                                                  col2=model_ds.surface_drn_cond)
    elif model_ds.gridtype == 'unstructured':
        drn_rec = mdims.data_array_1d_unstr_to_rec_list(model_ds, mask,
                                                        col1='ahn',
                                                        col2=model_ds.surface_drn_cond,
                                                        first_active_layer=True,
                                                        only_active_cells=False)

    drn = flopy.mf6.ModflowGwfdrn(gwf, print_input=True,
                                  maxbound=len(drn_rec),
                                  stress_period_data={0: drn_rec},
                                  save_flows=True)

    return drn


def rch_from_model_ds(model_ds, gwf):
    """get recharge package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.

    Returns
    -------
    rch : flopy ModflowGwfrch
        rch package
    """

    # create recharge package
    rch = recharge.model_datasets_to_rch(gwf, model_ds)

    return rch


def oc_from_model_ds(model_ds, gwf, save_budget=True,
                     print_head=True):
    """get output control package from model dataset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.

    Returns
    -------
    oc : flopy ModflowGwfoc
        oc package
    """
    # Create the output control package
    headfile = '{}.hds'.format(model_ds.model_name)
    head_filerecord = [headfile]
    budgetfile = '{}.cbb'.format(model_ds.model_name)
    budget_filerecord = [budgetfile]
    saverecord = [('HEAD', 'LAST')]
    if save_budget:
        saverecord.append(('BUDGET', 'ALL'))
    if print_head:
        printrecord = [('HEAD', 'LAST')]
    else:
        printrecord = None

    oc = flopy.mf6.ModflowGwfoc(gwf, pname='oc',
                                saverecord=saverecord,
                                head_filerecord=head_filerecord,
                                budget_filerecord=budget_filerecord,
                                printrecord=printrecord)

    return oc


def get_tdis_perioddata(model_ds):
    """Get tdis_perioddata from model_ds.

    Parameters
    ----------
    model_ds : xarray.Dataset
        dataset with time variant model data

    Returns
    -------
    tdis_perioddata : [perlen, nstp, tsmult]
        - perlen (double) is the length of a stress period.
        - nstp (integer) is the number of time steps in a stress period.
        - tsmult (double) is the multiplier for the length of successive time
          steps. The length of a time step is calculated by multiplying the
          length of the previous time step by TSMULT. The length of the first
          time step, :math:`\Delta t_1`, is related to PERLEN, NSTP, and
          TSMULT by the relation :math:`\Delta t_1= perlen \frac{tsmult -
          1}{tsmult^{nstp}-1}`.
    """
    perlen = model_ds.perlen
    if isinstance(perlen, numbers.Number):
        tdis_perioddata = [(float(perlen), model_ds.nstp,
                            model_ds.tsmult)] * int(model_ds.nper)
    elif isinstance(perlen, (list, tuple, np.ndarray)):
        if model_ds.steady_start:
            assert len(perlen) == model_ds.dims['time']
        else:
            assert len(perlen) == model_ds.dims['time']
        tdis_perioddata = [(p, model_ds.nstp, model_ds.tsmult) for p in perlen]
    else:
        raise TypeError('did not recognise perlen type')

    # netcdf does not support multi-dimensional array attributes
    #model_ds.attrs['tdis_perioddata'] = tdis_perioddata

    return tdis_perioddata
