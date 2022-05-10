import nlmod

import test_001_model


def test_get_recharge():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # add knmi recharge to the model dataset
    model_ds.update(nlmod.read.knmi.get_recharge(model_ds))

    return model_ds


def test_get_recharge_steady_state():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # modify mtime
    model_ds = model_ds.drop_dims('time')
    model_ds = nlmod.mdims.set_model_ds_time(model_ds,
                                             start_time='2000-1-1',
                                             perlen=3650)

    # add knmi recharge to the model dataset
    model_ds.update(nlmod.read.knmi.get_recharge(model_ds))

    return model_ds


def test_get_ahn():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # add ahn data to the model dataset
    model_ds.update(nlmod.read.ahn.get_ahn(model_ds))

    return model_ds


def test_get_surface_water_ghb():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)

    # add surface water levels to the model dataset
    model_ds.update(nlmod.read.rws.get_surface_water(model_ds,
                                                     'surface_water'))

    return model_ds
