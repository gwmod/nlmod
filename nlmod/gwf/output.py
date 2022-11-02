import logging
import os

import flopy
import numpy as np
import pandas as pd
import xarray as xr

from ..dims.resample import get_affine, get_xy_mid_structured

logger = logging.getLogger(__name__)


def get_heads_da(ds=None, gwf=None, fname_hds=None):
    """Reads heads file given either a dataset or a groundwater flow object.

    Note: Calling this function with ds is currently prevered over calling it
    with gwf, because the layer and time coordinates can not be fully
    reconstructed from gwf.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset with model data.
    gwf : flopy ModflowGwf
        Flopy groundwaterflow object.
    fname_hds : path, optional
        Instead of loading the binary heads file corresponding to ds or gwf
        load the heads from


    Returns
    -------
    head_ar : xarray.DataArray
        heads array.
    """
    assert (
        (ds is not None) + (gwf is not None)
    ) == 1, "Load the heads using either the ds or the gwf"

    if fname_hds is not None and ds is not None:
        headobj = flopy.utils.HeadFile(fname_hds)
        hdry = -1e30
        hnoflo = 1e30

    elif fname_hds is not None and gwf is not None:
        headobj = flopy.utils.HeadFile(fname_hds)
        hdry = gwf.hdry
        hnoflo = gwf.hnoflo

    elif fname_hds is None and gwf is not None:
        headobj = gwf.output.head()
        hdry = gwf.hdry
        hnoflo = gwf.hnoflo

    elif fname_hds is None and ds is not None:
        fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")
        headobj = flopy.utils.HeadFile(fname_hds)
        hdry = -1e30
        hnoflo = 1e30
    else:
        pass

    heads = headobj.get_alldata()
    heads[heads == hdry] = np.nan
    heads[heads == hnoflo] = np.nan

    if gwf is not None:
        gridtype = gwf.modelgrid.grid_type
    else:
        gridtype = ds.gridtype

    if gridtype == "vertex":
        head_ar = xr.DataArray(
            data=heads[:, :, 0],
            dims=("time", "layer", "icell2d"),
            coords={},
            attrs={"units": "mNAP"},
        )

    elif gridtype == "structured":
        if gwf is not None:
            delr = np.unique(gwf.modelgrid.delr).item()
            delc = np.unique(gwf.modelgrid.delc).item()
            extent = gwf.modelgrid.extent
            x, y = get_xy_mid_structured(extent, delr, delc)

        else:
            x = ds.x
            y = ds.y

        head_ar = xr.DataArray(
            data=heads,
            dims=("time", "layer", "y", "x"),
            coords={
                "x": x,
                "y": y,
            },
            attrs={"units": "mNAP"},
        )
    else:
        assert 0, "Gridtype not supported"

    if ds is not None:
        head_ar.coords["layer"] = ds.layer

        # TODO: temporarily only add time for when ds is passed because unable to
        # exactly recreate ds.time from gwf.
        times = np.array(
            [
                pd.Timestamp(ds.time.start)
                + pd.Timedelta(t, unit=ds.time.time_units[0])
                for t in headobj.get_times()
            ],
            dtype=np.datetime64,
        )
        head_ar.coords["time"] = times

    if ds is not None and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        affine = get_affine(ds)
        head_ar.rio.write_transform(affine, inplace=True)

    elif gwf is not None and gwf.modelgrid.angrot != 0.0:
        attrs = dict(
            delr=np.unique(gwf.modelgrid.delr).item(),
            delc=np.unique(gwf.modelgrid.delc).item(),
            xorigin=gwf.modelgrid.xoffset,
            yorigin=gwf.modelgrid.yoffset,
            angrot=gwf.modelgrid.angrot,
            extent=gwf.modelgrid.extent,
        )
        affine = get_affine(attrs)
        head_ar.rio.write_transform(affine, inplace=True)

    else:
        pass

    head_ar.rio.write_crs("EPSG:28992", inplace=True)

    return head_ar
