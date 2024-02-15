import tempfile
import os
import numpy as np
import nlmod
import matplotlib.pyplot as plt

tempdir = tempfile.gettempdir()


def test_get_ds_rotated():
    extent = [98000.0, 99000.0, 489000.0, 490000.0]
    ds0 = nlmod.get_ds(extent, angrot=15)
    assert ds0.extent[0] == 0 and ds0.extent[2] == 0
    assert ds0.xorigin == extent[0] and ds0.yorigin == extent[2]

    # test refine method, by refining in all cells that contain surface water polygons
    bgt = nlmod.read.bgt.get_bgt(extent)
    model_ws = os.path.join(tempdir, "grid")
    ds = nlmod.grid.refine(ds0, model_ws=model_ws, refinement_features=[(bgt, 1)])
    assert len(ds.area) > np.prod(ds0.area.shape)
    assert ds.extent[0] == 0 and ds.extent[2] == 0
    assert ds.xorigin == extent[0] and ds.yorigin == extent[2]

    f0, ax0 = plt.subplots()
    nlmod.plot.modelgrid(ds0, ax=ax0)
    f, ax = plt.subplots()
    nlmod.plot.modelgrid(ds, ax=ax)
    assert (np.array(ax.axis()) == np.array(ax0.axis())).all()
