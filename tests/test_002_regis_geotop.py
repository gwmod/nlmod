# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 16:23:35 2021.

@author: oebbe
"""

import nlmod


# @pytest.mark.skip(reason="too slow")
def test_get_regis(extent=[98600.0, 99000.0, 489400.0, 489700.0]):
    regis_ds = nlmod.read.regis.get_regis(extent)

    assert regis_ds.dims["layer"] == 20


# @pytest.mark.skip(reason="too slow")
def test_get_regis_botm_layer_BEk1(
    extent=[98700.0, 99000.0, 489500.0, 489700.0],
    botm_layer="MSc",
):
    regis_ds = nlmod.read.regis.get_regis(extent, botm_layer)
    assert regis_ds.dims["layer"] == 15
    assert regis_ds.layer.values[-1] == botm_layer


def test_get_geotop_raw(extent=[98600.0, 99000.0, 489400.0, 489700.0]):
    geotop_ds = nlmod.read.geotop.get_geotop_raw_within_extent(extent)
    line = [(extent[0], extent[2]), (extent[1], extent[3])]
    # also test the plot-method
    nlmod.plot.geotop_lithok_in_cross_section(line, geotop_ds)


# @pytest.mark.skip(reason="too slow")
def test_get_geotop(extent=[98600.0, 99000.0, 489400.0, 489700.0]):
    nlmod.read.geotop.get_geotop(extent)


# @pytest.mark.skip(reason="too slow")
def test_get_regis_geotop(extent=[98600.0, 99000.0, 489400.0, 489700.0]):
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    regis_geotop_ds = nlmod.base.to_model_ds(regis_geotop_ds)
    assert regis_geotop_ds.dims["layer"] == 24


# @pytest.mark.skip(reason="too slow")
def test_get_regis_geotop_keep_all_layers(
    extent=[98600.0, 99000.0, 489400.0, 489700.0],
):
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True, remove_nan_layers=False
    )
    assert regis_geotop_ds.dims["layer"] == 137
