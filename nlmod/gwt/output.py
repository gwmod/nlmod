import logging
import os

import flopy
import numpy as np
import xarray as xr

from ..dims.layers import calculate_thickness
from ..mfoutput import _get_output_da

logger = logging.getLogger(__name__)


def _get_concentration(ds=None, gwt=None, fname_conc=None):
    msg = "Load the concentration using either the ds or the gwt"
    assert ((ds is not None) + (gwt is not None)) == 1, msg

    if fname_conc is None:
        if ds is None:
            concobj = gwt.output.concentration()
        else:
            fname_conc = os.path.join(ds.model_ws, f"{ds.model_name}_gwt.ucn")
    if fname_conc is not None:
        concobj = flopy.utils.HeadFile(fname_conc, text="concentration")
    return concobj


def get_concentration_da(
    ds=None, gwt=None, fname_conc=None, delayed=False, chunked=False
):
    """Reads concentration file given either a dataset or a groundwater flow object.

    Note: Calling this function with ds is currently preferred over calling it
    with gwt, because the layer and time coordinates can not be fully
    reconstructed from gwt.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset with model data.
    gwt : flopy ModflowGwt
        Flopy groundwater transport object.
    fname_conc : path, optional
        Instead of loading the binary concentration file corresponding to ds or gwf
        load the concentration from this file.
    delayed : bool, optional
        if delayed is True, do not load output data into memory, default is False.
    chunked : bool, optional
        chunk data array containing output, default is False.

    Returns
    -------
    conc_da : xarray.DataArray
        concentration data array.
    """
    conc_da = _get_output_da(
        _get_concentration,
        ds=ds,
        gwf_or_gwt=gwt,
        fname=fname_conc,
        delayed=delayed,
        chunked=chunked,
    )
    conc_da.attrs["units"] = "concentration"
    return conc_da


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
    return ctop


def freshwater_head(ds, hp, conc, denseref=None, drhodc=None):
    """Calculate equivalent freshwater head from point water heads.

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
        z = ds["botm"] + thickness / 2.0
    else:
        z = ds["z"]
    hf = density / denseref * hp - (density - denseref) / denseref * z
    return hf


def pointwater_head(ds, hf, conc, denseref=None, drhodc=None):
    """Calculate point water head from freshwater heads.

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
