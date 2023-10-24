import os
import numpy as np
import tempfile
import nlmod
import pytest

tmpdir = tempfile.gettempdir()


@pytest.mark.slow
def test_buidrainage():
    model_ws = os.path.join(tmpdir, "buidrain")
    ds = nlmod.get_ds([110_000, 130_000, 435_000, 445_000], model_ws=model_ws)
    ds = nlmod.read.nhi.add_buisdrainage(ds)

    # assert that all locations with a specified depth also have a positive conductance
    mask = ~ds["buisdrain_depth"].isnull()
    assert np.all(ds["buisdrain_cond"].data[mask] > 0)

    # assert that all locations with a positive conductance also have a specified depth
    mask = ds["buisdrain_cond"] > 0
    assert np.all(~np.isnan(ds["buisdrain_depth"].data[mask]))
