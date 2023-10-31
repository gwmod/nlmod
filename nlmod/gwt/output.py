import logging

import numpy as np
import xarray as xr

from ..dims.layers import calculate_thickness
from ..mfoutput.mfoutput import _get_heads_da, _get_time_index, _get_flopy_data_object

logger = logging.getLogger(__name__)


def get_concentration_obj(ds=None, gwt=None, fname=None, grbfile=None):
    """Get flopy HeadFile object connected to the file with the concetration of cells.

    Provide one of ds, gwf or fname.

    Parameters
    ----------
    ds : xarray.Dataset, optional
        model dataset, by default None
    gwt : flopy.mf6.ModflowGwt, optional
        groundwater transport model, by default None
    fname : str, optional
        path to heads file, by default None
    grbfile : str
        path to file containing binary grid information

    Returns
    -------
    headobj : flopy.utils.HeadFile
        HeadFile object handle
    """
    concobj = _get_flopy_data_object("concentration", ds, gwt, fname, grbfile)
    return concobj


def get_concentration_da(
    ds=None,
    gwt=None,
    fname=None,
    grbfile=None,
    delayed=False,
    chunked=False,
    **kwargs,
):
    """Reads binary concentration file.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset with model data.
    gwt : flopy ModflowGwt
        Flopy groundwater transport object.
    fname : path, optional
        Instead of loading the binary concentration file corresponding to ds or gwf
        load the concentration from this file.
    grbfile : str
        path to file containing binary grid information, only needed if reading
        output from file using fname
    delayed : bool, optional
        if delayed is True, do not load output data into memory, default is False.
    chunked : bool, optional
        chunk data array containing output, default is False.

    Returns
    -------
    conc_da : xarray.DataArray
        concentration data array.
    """
    cobj = get_concentration_obj(ds=ds, gwt=gwt, fname=fname, grbfile=grbfile)
    # gwt.output.concentration() defaults to a structured grid
    if gwt is not None and ds is None and fname is None:
        kwargs["modelgrid"] = gwt.modelgrid
    da = _get_heads_da(cobj, **kwargs)
    da.attrs["units"] = "concentration"

    # set time index if ds/gwt are provided
    if ds is not None or gwt is not None:
        da["time"] = _get_time_index(cobj, ds=ds, gwf_or_gwt=gwt)
        if ds is not None:
            da["layer"] = ds.layer

    if chunked:
        # chunk data array
        da = da.chunk("auto")

    if not delayed:
        # load into memory
        da = da.compute()

    return da


def get_concentration_at_gw_surface(conc, layer="layer"):
    """Get the concentration level from a multi-dimensional concentration array
    where dry or inactive cells are NaN. This methods finds the most upper non-
    nan-value of each cell or timestep.

    Parameters
    ----------
    conc : xarray.DataArray or numpy array
        A multi-dimensional array of conc values. NaN-values represent inactive
        or dry cells.
    layer : string or int, optional
        The name of the layer dimension of conc (if conc is a DataArray) or the integer
        of the layer dimension of conc (if conc is a numpy array). The default is
        'layer'.

    Returns
    -------
    ctop : numpy-array or xr.DataArray
        an array of the top level concentration, without the layer-dimension.
    """
    if isinstance(conc, xr.DataArray):
        conc_da = conc
        conc = conc_da.data
        if isinstance(layer, str):
            layer = conc_da.dims.index(layer)
    else:
        conc_da = None
    # take the first non-nan value along the layer dimension (1)
    top_layer = np.expand_dims(np.isnan(conc).argmin(layer), layer)
    ctop = np.take_along_axis(conc, top_layer, axis=layer)
    ctop = np.take(ctop, 0, axis=layer)
    if conc_da is not None:
        dims = list(conc_da.dims)
        dims.pop(layer)
        coords = dict(conc_da.coords)
        # store the layer in which the groundwater level is of each cell and time
        top_layer = np.take(top_layer, 0, axis=layer)
        coords["layer"] = (dims, conc_da.layer.data[top_layer])
        ctop = xr.DataArray(ctop, dims=dims, coords=coords)
        # to not confuse this coordinate with the default layer coord in nlmod
        # this source_layer has dims (time, cellid) or (time, y, x)
        # indicating the source layer of the concentration value for each time step
        ctop = ctop.rename({"layer": "source_layer"})
    return ctop


def freshwater_head(ds, hp, conc, denseref=None, drhodc=None):
    """Calculate equivalent freshwater head from point water heads.
    Heads file produced by mf6 contains point water heads.

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset containing layer elevation/thickness data, and
        reference density (denseref) relationship between concentration
        and density (drhodc) if not provided separately
    hp : xarray.DataArray
        data array containing point water heads
    conc : xarray.DataArray
        data array containing concentration
    denseref : float, optional
        reference density, by default None, which will use denseref attribute in
        model dataset.
    drhodc : float, optional
        density-concentration gradient, by default None, which will use drhodc
        attribute in model dataset.

    Returns
    -------
    hf : xarray.DataArray
        data array containing equivalent freshwater heads.
    """
    if denseref is None:
        denseref = ds.denseref
    if drhodc is None:
        drhodc = ds.drhodc
    density = denseref + drhodc * conc
    if "z" not in ds:
        if "thickness" not in ds:
            thickness = calculate_thickness(ds)
        else:
            thickness = ds.thickness
        z = ds["botm"] + thickness / 2.0
    else:
        z = ds["z"]
    hf = density / denseref * hp - (density - denseref) / denseref * z
    return hf


def pointwater_head(ds, hf, conc, denseref=None, drhodc=None):
    """Calculate point water head from freshwater heads.
    Heads file produced by mf6 contains point water heads.

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset containing layer elevation/thickness data, and
        reference density (denseref) relationship between concentration
        and density (drhodc) if not provided separately
    hf : xarray.DataArray
        data array containing freshwater heads
    conc : xarray.DataArray
        data array containing concentration
    denseref : float, optional
        reference density, by default None, which will use denseref attribute in
        model dataset.
    drhodc : float, optional
        density-concentration gradient, by default None, which will use drhodc
        attribute in model dataset.

    Returns
    -------
    hf : xarray.DataArray
        data array containing point water heads.
    """
    if denseref is None:
        denseref = ds.denseref
    if drhodc is None:
        drhodc = ds.drhodc
    density = denseref + drhodc * conc
    if "z" not in ds:
        if "thickness" not in ds:
            thickness = calculate_thickness(ds)
        z = ds["botm"] + thickness / 2.0
    else:
        z = ds["z"]
    hp = denseref / density * hf + (density - denseref) / density * z
    return hp
