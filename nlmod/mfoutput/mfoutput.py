import logging
import os
import warnings

import dask
import flopy
import xarray as xr

from ..dims.grid import (
    get_affine_mod_to_world,
    get_dims_coords_from_modelgrid,
    modelgrid_from_ds,
)
from ..dims.time import ds_time_idx
from .binaryfile import _get_binary_budget_data, _get_binary_head_data

logger = logging.getLogger(__name__)


def _get_dask_array(func, kstpkper, shape, **kwargs):
    """Get stacked dask array for given timesteps.

    Parameters
    ----------
    func : function
        function that returns a numpy array
    kstpkper : list of tuples
        list of tuples containing (timestep, stressperiod) indices.
    shape : tuple
        shape of array that is returned by func
    **kwargs
        additional kwargs passed to func

    Returns
    -------
    dask.array
        stacked dask array
    """
    result = []
    for ki in kstpkper:
        d = dask.delayed(func)(kstpkper=ki, **kwargs)
        arr = dask.array.from_delayed(d, shape=shape, dtype=float)
        result.append(arr)
    return dask.array.stack(result)


def _get_time_index(fobj, ds=None, gwf_or_gwt=None):
    """Get time index based on flopy binaryfile object.

    Binary files objects are e.g. flopy.utils.HeadFile, flopy.utils.CellBudgetFile.

    Parameters
    ----------
    fobj : flopy.utils.HeadFile or flopy.utils.CellBudgetFile
        flopy binary file object
    ds : xarray.Dataset, optional
        model dataset, by default None
    gwf_or_gwt : flopy.mf6.ModflowGwf or flopy.mf6.ModflowGwt, optional
        flopy groundwater flow or transport model, by default None

    Returns
    -------
    tindex : xarray.IndexVariable
        index variable with time converted to timestamps
    """
    # set layer and time coordinates
    if gwf_or_gwt is not None:
        tindex = ds_time_idx(
            fobj.get_times(),
            start_datetime=gwf_or_gwt.modeltime.start_datetime,
            time_units=gwf_or_gwt.modeltime.time_units,
        )
    elif ds is not None:
        if "time" in ds:
            dtype = "float" if ds.time.dtype.kind in ["i", "f"] else "datetime"
        else:
            dtype = "float"
        tindex = ds_time_idx(
            fobj.get_times(),
            start_datetime=(ds.time.attrs["start"] if "time" in ds else None),
            time_units=(ds.time.attrs["time_units"] if "time" in ds else None),
            dtype=dtype,
        )
    else:
        raise ValueError("Provide either ds or gwf_or_gwt")
    return tindex


def _create_da(arr, modelgrid, times, hdry=-1e30, hnoflo=1e30):
    """Create data array based on array, modelgrid, and time array.

    Parameters
    ----------
    arr : dask.array or numpy.array
        array containing data
    modelgrid : flopy.discretization.Grid
        flopy modelgrid object
    times : list or array
        list or array containing times as floats (usually in days)
    hdry : float, optional
        The value of dry cells, which will be replaced by NaNs. If hdry is None, the
        values of dry cells will not be replaced by NaNs. The default is -1e30.
    hnoflo : float, optional
        The value of no-flow cells, which will be replaced by NaNs. If hnoflo is None,
        the values of no-flow cells will not be replaced by NaNs. The default is 1e30.

    Returns
    -------
    da : xarray.DataArray
        data array with spatial dimensions based on modelgrid and
        time dimension based on times
    """
    # create data array
    dims, coords = get_dims_coords_from_modelgrid(modelgrid)
    da = xr.DataArray(data=arr, dims=("time",) + dims, coords=coords)

    if hdry is not None or hnoflo is not None:
        # set dry/no-flow to nan
        if hdry is None:
            mask = da != hnoflo
        elif hnoflo is None:
            mask = da != hdry
        else:
            mask = (da != hdry) & (da != hnoflo)
        da = da.where(mask)

    # set local time coordinates
    da.coords["time"] = ds_time_idx(times)

    # set affine if angrot != 0.0
    if modelgrid.angrot != 0.0:
        attrs = {
            "xorigin": modelgrid.xoffset,
            "yorigin": modelgrid.yoffset,
            "angrot": modelgrid.angrot,
            "extent": modelgrid.extent,
        }
        affine = get_affine_mod_to_world(attrs)
        da.rio.write_transform(affine, inplace=True)

    # write CRS
    da.rio.write_crs("EPSG:28992", inplace=True)
    return da


def _get_heads_da(
    hobj,
    modelgrid=None,
    **kwargs,
):
    """Get heads data array based on HeadFile object.

    Optionally provide modelgrid separately if HeadFile object does not contain
    correct grid definition.

    Parameters
    ----------
    hobj : flopy.utils.HeadFile
        flopy HeadFile object for binary heads
    modelgrid : flopy.discretization.Grid, optional
        flopy modelgrid object, default is None, in which case the modelgrid
        is derived from `hobj.mg`

    Returns
    -------
    da : xarray.DataArray
        output data array
    """
    if "kstpkper" in kwargs:
        kstpkper = kwargs.pop("kstpkper")
    else:
        kstpkper = hobj.get_kstpkper()

    if modelgrid is None:
        modelgrid = hobj.mg
    # shape is derived from hobj, not modelgrid as array read from
    # binary file always has 3 dimensions
    shape = (hobj.nlay, hobj.nrow, hobj.ncol)

    # load data from binary file
    stacked_arr = _get_dask_array(
        _get_binary_head_data, kstpkper=kstpkper, shape=shape, fobj=hobj
    )

    # check for vertex grids
    if modelgrid.grid_type == "vertex":
        if stacked_arr.ndim == 4:
            stacked_arr = stacked_arr[:, :, 0, :]

    # create data array
    da = _create_da(stacked_arr, modelgrid, hobj.get_times(), **kwargs)

    return da


def _get_budget_da(
    cbcobj,
    text,
    modelgrid=None,
    column="q",
    **kwargs,
):
    """Get budget data array based on CellBudgetFile and text string.

    Optionally provide modelgrid separately if CellBudgetFile object does not contain
    correct grid definition.

    Parameters
    ----------
    cbcobj : flopy.utils.CellBudgetFile
        flopy HeadFile object for binary heads
    text: str
        string indicating which dataset to load from budget file
    modelgrid : flopy.discretization.Grid, optional
        flopy modelgrid object, default is None, in which case the modelgrid
        is derived from `cbcobj.modelgrid`
    column : str
        name of column in rec-array to read, default is 'q' which contains the fluxes
        for most budget datasets.

    Returns
    -------
    da : xarray.DataArray
        output data array.
    """
    if "kstpkper" in kwargs:
        kstpkper = kwargs.pop("kstpkper")
    else:
        kstpkper = cbcobj.get_kstpkper()

    if modelgrid is None:
        modelgrid = cbcobj.modelgrid

    # load data from binary file
    shape = modelgrid.shape
    stacked_arr = _get_dask_array(
        _get_binary_budget_data,
        kstpkper=kstpkper,
        shape=shape,
        fobj=cbcobj,
        text=text,
        column=column,
    )

    # create data array
    da = _create_da(stacked_arr, modelgrid, cbcobj.get_times())

    return da


def _get_flopy_data_object(
    var, ds=None, gwml=None, fname=None, grb_file=None, **kwargs
):
    """Get modflow HeadFile or CellBudgetFile object, containg heads, budgets or
    concentrations.

    Provide one of ds, gwf or fname.

    Parameters
    ----------
    var : str
        The name of the variable. Can be 'head', 'budget' or 'concentration'.
    ds : xarray.Dataset, optional
        model dataset, by default None
    gwml : flopy.mf6.ModflowGwf or flopy.mf6.ModflowGwt, optional
        groundwater flow or transport model, by default None
    fname : str, optional
        path to Head- or CellBudgetFile, by default None
    grb_file : str, optional
        path to file containing binary grid information, if None modelgrid
        information is obtained from ds. By default None

    Returns
    -------
    flopy.utils.HeadFile or flopy.utils.CellBudgetFile
    """
    if var == "head":
        ml_name = "gwf"
        extension = ".hds"
    elif var == "budget":
        ml_name = "gwf"
        extension = ".cbc"
    elif var == "concentration":
        ml_name = "gwt"
        extension = "_gwt.ucn"
    else:
        raise (ValueError(f"Unknown variable {var}"))

    if fname is None:
        if ds is None:
            if gwml is None:
                msg = f"Load the {var}s using either ds, {ml_name} or fname"
                raise (ValueError(msg))
            # return gwf.output.head(), gwf.output.budget() or gwt.output.concentration()
            return getattr(gwml.output, var)()
        fname = os.path.join(ds.model_ws, ds.model_name + extension)
    if grb_file is None and ds is not None:
        # get grb file
        grb_file = _get_grb_file(ds)
    if grb_file is not None and os.path.exists(grb_file):
        modelgrid = flopy.mf6.utils.MfGrdFile(grb_file).modelgrid
    elif ds is not None:
        modelgrid = modelgrid_from_ds(ds)
    else:
        modelgrid = None

    msg = f"Cannot create {var} data-array without grid information."
    if var == "budget":
        if modelgrid is None:
            logger.error(msg)
            raise ValueError(msg)
        return flopy.utils.CellBudgetFile(fname, modelgrid=modelgrid, **kwargs)
    else:
        if modelgrid is None:
            logger.warning(msg)
            warnings.warn(msg)
        return flopy.utils.HeadFile(fname, text=var, modelgrid=modelgrid, **kwargs)


def _get_grb_file(ds):
    if ds.gridtype == "vertex":
        grb_file = os.path.join(ds.model_ws, ds.model_name + ".disv.grb")
    elif ds.gridtype == "structured":
        grb_file = os.path.join(ds.model_ws, ds.model_name + ".dis.grb")
    return grb_file
