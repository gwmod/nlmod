# -*- coding: utf-8 -*-

import os

import nlmod
import pandas as pd
import pytest

import test_001_model
import test_002_regis_geotop


def test_get_recharge():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # add knmi recharge to the model dataset
    model_ds.update(nlmod.read.knmi.add_knmi_to_model_dataset(model_ds))

    return model_ds


def test_get_recharge_steady_state():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # modify mtime
    model_ds = model_ds.drop_dims('time')
    for model_ds_key in ['perlen', 'start_time', 'nper', 'nstp', 'tsmult',
                         'steady_start', 'steady_state']:
        model_ds.attrs.pop(model_ds_key)
    model_ds = nlmod.mdims.set_model_ds_time(model_ds,
                                             '2000-1-1',
                                              True,
                                              False,
                                              perlen=3650)

    # add knmi recharge to the model dataset
    model_ds.update(nlmod.read.knmi.add_knmi_to_model_dataset(model_ds))

    return model_ds


def test_get_ahn():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # add ahn data to the model dataset
    model_ds.update(nlmod.read.ahn.get_ahn_at_grid(model_ds))

    return model_ds


def test_get_surface_water_ghb():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)

    # add surface water levels to the model dataset
    model_ds.update(nlmod.read.rws.surface_water_to_model_dataset(model_ds,
                                                            gwf.modelgrid,
                                                            'surface_water'))

    return model_ds
