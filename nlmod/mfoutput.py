import logging

import numpy as np
import xarray as xr
from flopy.utils import CellBudgetFile, HeadFile

from .dims.resample import get_affine_mod_to_world, get_xy_mid_structured
from .dims.time import ds_time_idx

logger = logging.getLogger(__name__)


def _get_output_da(reader_func, ds=None, gwf_or_gwt=None, fname=None, **kwargs):
    """Reads mf6 output file given either a dataset or a gwf or gwt object.

    Note: Calling this function with ds is currently preferred over calling it
    with gwf/gwt, because the layer and time coordinates can not be fully
    reconstructed from gwf/gwt.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray dataset with model data.
    gwf_or_gwt : flopy ModflowGwt
        flopy groundwater flow or transport object.
    fname : path, optional
        instead of loading the binary concentration file corresponding to ds or
        gwf/gwt load the concentration from this file.


    Returns
    -------
    da : xarray.DataArray
        output data array.
    """
    out_obj = reader_func(ds, gwf_or_gwt, fname)

    if gwf_or_gwt is not None:
        hdry = gwf_or_gwt.hdry
        hnoflo = gwf_or_gwt.hnoflo
    else:
        hdry = -1e30
        hnoflo = 1e30

    # check whether out_obj is BudgetFile or HeadFile based on passed kwargs
    if isinstance(out_obj, CellBudgetFile):
        arr = out_obj.get_data(**kwargs)
    elif isinstance(out_obj, HeadFile):
        arr = out_obj.get_alldata(**kwargs)
    else:
        raise TypeError(f"Don't know how to deal with {type(out_obj)}!")

    if isinstance(arr, list):
        arr = np.stack(arr)

    arr[arr == hdry] = np.nan
    arr[arr == hnoflo] = np.nan

    if gwf_or_gwt is not None:
        gridtype = gwf_or_gwt.modelgrid.grid_type
    else:
        gridtype = ds.gridtype

    if gridtype == "vertex":
        da = xr.DataArray(
            data=arr[:, :, 0],
            dims=("time", "layer", "icell2d"),
            coords={},
        )

    elif gridtype == "structured":
        if gwf_or_gwt is not None:
            try:
                delr = np.unique(gwf_or_gwt.modelgrid.delr).item()
                delc = np.unique(gwf_or_gwt.modelgrid.delc).item()
                extent = gwf_or_gwt.modelgrid.extent
                x, y = get_xy_mid_structured(extent, delr, delc)
            except ValueError:  # delr/delc are variable
                # x, y in local coords
                x, y = gwf_or_gwt.modelgrid.xycenters
        else:
            x = ds.x
            y = ds.y

        da = xr.DataArray(
            data=arr,
            dims=("time", "layer", "y", "x"),
            coords={
                "x": x,
                "y": y,
            },
        )
    else:
        raise TypeError("Gridtype not supported")

    # set layer and time coordinates
    if gwf_or_gwt is not None:
        da.coords["layer"] = np.arange(gwf_or_gwt.modelgrid.nlay)
        da.coords["time"] = ds_time_idx(
            out_obj.get_times(),
            start_datetime=gwf_or_gwt.modeltime.start_datetime,
            time_units=gwf_or_gwt.modeltime.time_units,
        )
    else:
        da.coords["layer"] = ds.layer
        da.coords["time"] = ds_time_idx(
            out_obj.get_times(),
            start_datetime=ds.time.attrs["start"],
            time_units=ds.time.attrs["time_units"],
        )

    if ds is not None and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # affine = get_affine(ds)
        affine = get_affine_mod_to_world(ds)
        da.rio.write_transform(affine, inplace=True)

    elif gwf_or_gwt is not None and gwf_or_gwt.modelgrid.angrot != 0.0:
        attrs = {
            # "delr": np.unique(gwf_or_gwt.modelgrid.delr).item(),
            # "delc": np.unique(gwf_or_gwt.modelgrid.delc).item(),
            "xorigin": gwf_or_gwt.modelgrid.xoffset,
            "yorigin": gwf_or_gwt.modelgrid.yoffset,
            "angrot": gwf_or_gwt.modelgrid.angrot,
            "extent": gwf_or_gwt.modelgrid.extent,
        }
        affine = get_affine_mod_to_world(attrs)
        da.rio.write_transform(affine, inplace=True)

    da.rio.write_crs("EPSG:28992", inplace=True)

    return da
