import matplotlib.pyplot as plt

import nlmod


def test_get_regis(extent=[98600.0, 99000.0, 489400.0, 489700.0]):
    regis_ds = nlmod.read.regis.get_regis(extent)

    assert regis_ds.sizes["layer"] == 20


def test_get_regis_botm_layer_BEk1(
    extent=[98700.0, 99000.0, 489500.0, 489700.0],
    botm_layer="MSc",
):
    regis_ds = nlmod.read.regis.get_regis(extent, botm_layer)
    assert regis_ds.sizes["layer"] == 15
    assert regis_ds.layer.values[-1] == botm_layer


def test_get_geotop(extent=[98600.0, 99000.0, 489400.0, 489700.0]):
    geotop_ds = nlmod.read.geotop.get_geotop(extent)
    line = [(extent[0], extent[2]), (extent[1], extent[3])]

    # also test the plot-methods
    f, ax = plt.subplots()
    nlmod.plot.geotop_lithok_in_cross_section(line, geotop_ds, ax=ax)

    f, ax = plt.subplots()
    nlmod.plot.geotop_lithok_on_map(geotop_ds, z=-20.2, ax=ax)


def test_get_regis_geotop(extent=[98600.0, 99000.0, 489400.0, 489700.0]):
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    regis_geotop_ds = nlmod.base.to_model_ds(regis_geotop_ds)
    assert regis_geotop_ds.sizes["layer"] == 24


def test_get_regis_geotop_keep_all_layers(
    extent=[98600.0, 99000.0, 489400.0, 489700.0],
):
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True, remove_nan_layers=False
    )
    assert regis_geotop_ds.sizes["layer"] == 137


def test_add_kh_and_kv():
    gt = nlmod.read.geotop.get_geotop(
        [118200, 118300, 439800, 439900], probabilities=True
    )

    # test with a value for kh for each lithoclass
    df = nlmod.read.geotop.get_lithok_props()
    # stochastic = None/False is allready tested in methods above, so we onlt test stochastic=True
    gt = nlmod.read.geotop.add_kh_and_kv(gt, df, stochastic=True)

    # test with a value for kh and kv for each combination of lithoclass and stratigraphic unit
    df = nlmod.read.geotop.get_kh_kv_table()
    gt = nlmod.read.geotop.add_kh_and_kv(gt, df)
    # again, but using the stochastic method
    gt = nlmod.read.geotop.add_kh_and_kv(gt, df, stochastic=True)
