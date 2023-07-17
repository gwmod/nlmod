import logging

import dask
import numpy as np
import xarray as xr

from .dims.resample import get_affine_mod_to_world, get_xy_mid_structured
from .dims.time import ds_time_idx

logger = logging.getLogger(__name__)


def _get_output_da(
    reader_func,
    ds=None,
    gwf_or_gwt=None,
    fname=None,
    delayed=False,
    chunked=False,
    **kwargs,
):
    """Reads mf6 output file given either a dataset or a gwf or gwt object.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray dataset with model data.
    gwf_or_gwt : flopy ModflowGwt
        flopy groundwater flow or transport object.
    fname : path, optional
        instead of loading the binary concentration file corresponding to ds or
        gwf/gwt load the concentration from this file.
    delayed : bool, optional
        if delayed is True, do not load output data into memory, default is False.
    chunked : bool, optional
        chunk data array containing output, default is False.


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

    if "kstpkper" in kwargs:
        kstpkper = kwargs.pop("kstpkper")
    else:
        kstpkper = out_obj.get_kstpkper()

    result = []
    for ki in kstpkper:
        d = dask.delayed(out_obj.get_data)(kstpkper=ki, **kwargs)
        arr = dask.array.from_delayed(d, shape=out_obj.mg.shape, dtype=float)
        result.append(arr)

    stacked_arr = dask.array.stack(result)

    if gwf_or_gwt is not None:
        gridtype = gwf_or_gwt.modelgrid.grid_type
    else:
        gridtype = ds.gridtype

    if gridtype == "vertex":
        if ds is not None:
            layers = ds["layer"].values
        else:
            layers = np.arange(stacked_arr.shape[1])
        
        if gwf_or_gwt is not None:
            x = gwf_or_gwt.modelgrid.xcellcenters
            y = gwf_or_gwt.modelgrid.ycellcenters
        else:
            x = ds.x
            y = ds.y
        coords = {"layer": layers, "y": y, "x": x}

        # stacked arr is 4d
        da = xr.DataArray(
            data=stacked_arr[:, :, 0, :],
            dims=("time", "layer", "icell2d"),
            coords=coords,
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
            data=stacked_arr,
            dims=("time", "layer", "y", "x"),
            coords={
                "x": x,
                "y": y,
            },
        )
    else:
        raise TypeError("Gridtype not supported")

    if chunked:
        da = da.chunk("auto")

    # replace dry/noflow with NaN
    da = da.where((da != hdry) | (da != hnoflo))

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

    # load into memory if indicated
    if not delayed:
        da.compute()

    return da
