# %%
import numpy as np
import pytest
from pandas import DataFrame

import nlmod


@pytest.fixture(scope="module")
def head_structured():
    fname_hds = "./tests/data/mf6output/structured/test.hds"
    grb_file = "./tests/data/mf6output/structured/test.dis.grb"
    head = nlmod.gwf.output.get_heads_da(fname=fname_hds, grb_file=grb_file)
    head.loc[{"x": head.x[0]}] = 3.0
    head.loc[{"x": head.x[-1]}] = 0.0
    return head


@pytest.fixture(scope="module")
def head_vertex():
    fname_hds = "./tests/data/mf6output/vertex/test.hds"
    grb_file = "./tests/data/mf6output/vertex/test.disv.grb"
    head = nlmod.gwf.output.get_heads_da(fname=fname_hds, grb_file=grb_file)
    head.loc[{"icell2d": head.icell2d[0]}] = 3.0
    head.loc[{"icell2d": head.icell2d[3]}] = 3.0
    head.loc[{"icell2d": head.icell2d[2]}] = 0.0
    head.loc[{"icell2d": head.icell2d[-1]}] = 0.0
    return head


@pytest.mark.parametrize("h", ["head_structured", "head_vertex"])
def test_interpolate_points(h, request):
    h = request.getfixturevalue(h)
    xi = [98_775, 98_900, 98_950, 98_950]
    yi = [489_600, 489_600, 489_650, 489_650]
    hi = nlmod.observations.interpolate_to_points_2d(h.isel(time=0, layer=0), xi, yi)
    assert (hi == np.array([2.5, 0.5, 0.0, 0.0])).all()

    # error when data is not 2D/1D
    with pytest.raises(AssertionError):
        hi = nlmod.observations.interpolate_to_points_2d(h.isel(layer=0), xi, yi)


def test_interpolate_to_points_structured(head_structured):
    head = head_structured
    head.loc[{"x": head.x[0]}] = 3.0
    head.loc[{"x": head.x[-1]}] = 0.0

    grb_file = "./tests/data/mf6output/structured/test.dis.grb"
    ds = nlmod.grid.modelgrid_to_ds(grbfile=grb_file)

    data = {
        "x": [98_775, 98_900, 98_950, 98_950],
        "y": [489_600, 489_600, 489_650, 489_650],
        "screen_top": [-1, -1, -35, 1000],
        "screen_bottom": [-20, -20, -100, -1000],
    }
    df = DataFrame(data)

    idx = nlmod.layers.get_modellayers_indexer(ds, df, full_output=True)

    # planar
    hi = nlmod.observations.interpolate_to_points(
        head.isel(time=0, layer=0),  # planar
        idx,
        xi="x_obs",
        yi="y_obs",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi == np.array([2.5, 0.5, 0.0, 0.0])).all()

    # planar with time
    hi = nlmod.observations.interpolate_to_points(
        head.isel(layer=0),  # planar with time
        idx,
        xi="x_obs",
        yi="y_obs",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi.isel(time=-1) == np.array([2.5, 0.5, 0.0, 0.0])).all()

    # layered
    hi = nlmod.observations.interpolate_to_points(
        head.isel(time=0),  # layered
        idx,
        xi="x_obs",
        yi="y_obs",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi == np.array([2.5, 0.5, 0.0, 0.0])).all()
    # layered with time
    hi = nlmod.observations.interpolate_to_points(
        head,
        idx,
        xi="x_obs",
        yi="y_obs",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi.isel(time=-1) == np.array([2.5, 0.5, 0.0, 0.0])).all()


def test_interpolate_to_points_vertex(head_vertex):
    head = head_vertex
    grb_file = "./tests/data/mf6output/vertex/test.disv.grb"
    ds = nlmod.grid.modelgrid_to_ds(grbfile=grb_file)

    data = {
        "x": [98_775, 98_900, 98_950, 98_950],
        "y": [489_600, 489_600, 489_650, 489_650],
        "screen_top": [-1, -1, -35, 1000],
        "screen_bottom": [-20, -20, -100, -1000],
    }
    df = DataFrame(data)

    idx = nlmod.layers.get_modellayers_indexer(ds, df, full_output=True)
    # planar
    hi = nlmod.observations.interpolate_to_points(
        head.isel(time=0, layer=0),  # planar
        idx,
        xi="x",
        yi="y",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi == np.array([2.5, 0.5, 0.0, 0.0])).all()
    # planar with time
    hi = nlmod.observations.interpolate_to_points(
        head.isel(layer=0),  # planar with time
        idx,
        xi="x",
        yi="y",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi.isel(time=-1) == np.array([2.5, 0.5, 0.0, 0.0])).all()
    # layered
    hi = nlmod.observations.interpolate_to_points(
        head.isel(time=0),  # layered
        idx,
        xi="x",
        yi="y",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi == np.array([2.5, 0.5, 0.0, 0.0])).all()
    # layered with time
    hi = nlmod.observations.interpolate_to_points(
        head,
        idx,
        xi="x",
        yi="y",
        x="x",
        y="y",
        layer="layer",
        full_output=False,
    )
    assert (hi.isel(time=-1) == np.array([2.5, 0.5, 0.0, 0.0])).all()
