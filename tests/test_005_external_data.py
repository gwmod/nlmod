# -*- coding: utf-8 -*-

import test_001_model
import test_002_regis_geotop
import pandas as pd

import os
import nlmod
import pytest


def test_get_recharge():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # add knmi recharge to the model datasets
    model_ds = nlmod.read.knmi.add_knmi_to_model_dataset(model_ds,
                                                         verbose=True)

    return model_ds


def test_get_recharge_perlen14():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # modify mtime
    model_ds = model_ds.drop_dims('time')
    for model_ds_key in ['perlen', 'start_time', 'nper', 'nstp', 'tsmult',
                         'steady_start', 'steady_state']:
        model_ds.attrs.pop(model_ds_key)
    model_ds_time = nlmod.mdims.get_model_ds_time(model_ds.model_name,
                                                  model_ds.model_ws,
                                                  '2015-1-1',
                                                  False,
                                                  False,
                                                  transient_timesteps=10,
                                                  perlen=14)

    # merge new mtime with model_ds
    model_ds.update(model_ds_time)
    _ = [model_ds.attrs.update({key: item})
             for key, item in model_ds_time.attrs.items()]

    # add knmi recharge to the model datasets
    model_ds = nlmod.read.knmi.add_knmi_to_model_dataset(model_ds,
                                                         verbose=True)

    return model_ds


def test_get_recharge_perlen_list():

    perlen = [3650, 12, 15, 14, 16, 20]

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # modify mtime
    model_ds = model_ds.drop_dims('time')
    for model_ds_key in ['perlen', 'start_time', 'nper', 'nstp', 'tsmult',
                         'steady_start', 'steady_state']:
        model_ds.attrs.pop(model_ds_key)
    model_ds_time = nlmod.mdims.get_model_ds_time(model_ds.model_name,
                                                  model_ds.model_ws,
                                                  '2015-1-1',
                                                  False,
                                                  True,
                                                  transient_timesteps=5,
                                                  perlen=perlen)

    # merge new mtime with model_ds
    model_ds.update(model_ds_time)
    _ = [model_ds.attrs.update({key: item})
             for key, item in model_ds_time.attrs.items()]

    # add knmi recharge to the model datasets
    model_ds = nlmod.read.knmi.add_knmi_to_model_dataset(model_ds,
                                                  verbose=True)

    return model_ds


def test_get_recharge_steady_state():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # modify mtime
    model_ds = model_ds.drop_dims('time')
    for model_ds_key in ['perlen', 'start_time', 'nper', 'nstp', 'tsmult',
                         'steady_start', 'steady_state']:
        model_ds.attrs.pop(model_ds_key)
    model_ds_time = nlmod.mdims.get_model_ds_time(model_ds.model_name,
                                                  model_ds.model_ws,
                                                  '2000-1-1',
                                                  True,
                                                  False,
                                                  transient_timesteps=0,
                                                  perlen=3650)

    # merge new mtime with model_ds
    model_ds.update(model_ds_time)
    _ = [model_ds.attrs.update({key: item})
             for key, item in model_ds_time.attrs.items()]

    # add knmi recharge to the model datasets
    model_ds = nlmod.read.knmi.add_knmi_to_model_dataset(model_ds,
                                                  verbose=True)

    return model_ds


def test_get_ahn():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # add ahn data to the model datasets
    model_ds = nlmod.read.ahn.get_ahn_at_grid(model_ds)

    return model_ds


def test_get_surface_water_ghb():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)

    # add knmi recharge to the model datasets
    model_ds = nlmod.mfpackages.surface_water.surface_water_to_model_dataset(model_ds,
                                                                             gwf.modelgrid,
                                                                             'surface_water')

    return model_ds
