import logging
import os

import flopy
import numpy as np
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
    headobj = _get_hds(ds=ds, gwf=gwf, fname_hds=fname_hds)

    if gwf is not None:
        hdry = gwf.hdry
        hnoflo = gwf.hnoflo
    else:
        hdry = -1e30
        hnoflo = 1e30

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
        head_ar.coords["time"] = ds.time

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

    head_ar.rio.write_crs("EPSG:28992", inplace=True)

    return head_ar


def _get_hds(ds=None, gwf=None, fname_hds=None):
    msg = "Load the heads using either the ds, gwf or fname_hds"
    assert ((ds is not None) + (gwf is not None) +
            (fname_hds is not None)) >= 1, msg

    if fname_hds is None:
        if ds is None:
            return gwf.output.head()
        else:
            fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")

    headobj = flopy.utils.HeadFile(fname_hds)

    return headobj


def _get_cbc(ds=None, gwf=None, fname_cbc=None):
    msg = "Load the budgets using either the ds or the gwf"
    assert ((ds is not None) + (gwf is not None)) == 1, msg

    if fname_cbc is None:
        if ds is None:
            cbf = gwf.output.budget()
        else:
            fname_cbc = os.path.join(ds.model_ws, ds.model_name + ".cbc")
    if fname_cbc is not None:
        cbf = flopy.utils.CellBudgetFile(fname_cbc)
    return cbf


def get_gwl_from_wet_cells(head, layer="layer", botm=None):
    """
    Get the groundwater level from a multi-dimensional head array where dry
    cells are NaN. This methods finds the most upper non-nan-value of each cell
    or timestep.

    Parameters
    ----------
    head : xarray.DataArray or numpy array
        A multi-dimensional array of head values. NaN-values represent inactive
        or dry cells.
    layer : string or int, optional
        The name of the layer dimension of head (if head is a DataArray) or the integer
        of the layer dimension of head (if head is a numpy array). The default is
        'layer'.
    botm : xarray.DataArray, optional
        A DataArray with the botm of each model-cell. It can be used to set heads below
        the botm of the cells to NaN. botm is only used when head is a DataArray.

    Returns
    -------
    gwl : numpy-array
        An array of the groundwater-level, without the layer-dimension.

    """
    if isinstance(head, xr.DataArray):
        head_da = head
        if botm is not None:
            head_da = head_da.where(head_da > botm)
        head = head_da.data
        if isinstance(layer, str):
            layer = head_da.dims.index(layer)
    else:
        head_da = None
    # take the first non-nan value along the layer dimension (1)
    top_layer = np.expand_dims(np.isnan(head).argmin(layer), layer)
    gwl = np.take_along_axis(head, top_layer, axis=layer)
    gwl = np.take(gwl, 0, axis=layer)
    if head_da is not None:
        dims = list(head_da.dims)
        dims.pop(layer)
        coords = dict(head_da.coords)
        # store the layer in which the groundwater level is of each cell and time
        top_layer = np.take(top_layer, 0, axis=layer)
        coords["layer"] = (dims, head_da.layer.data[top_layer])
        gwl = xr.DataArray(gwl, dims=dims, coords=coords)
    return gwl
