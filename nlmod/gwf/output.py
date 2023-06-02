import logging
import os
import warnings

import flopy
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from ..dims.grid import modelgrid_from_ds
from ..dims.resample import get_affine_mod_to_world, get_xy_mid_structured
from ..dims.time import ds_time_idx

logger = logging.getLogger(__name__)


def get_heads_da(ds=None, gwf=None, fname_hds=None):
    """Reads heads file given either a dataset or a groundwater flow object.

    Note: Calling this function with ds is currently preferred over calling it
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
            try:
                delr = np.unique(gwf.modelgrid.delr).item()
                delc = np.unique(gwf.modelgrid.delc).item()
                extent = gwf.modelgrid.extent
                x, y = get_xy_mid_structured(extent, delr, delc)
            except ValueError:
                # x, y in local coords
                x, y = gwf.modelgrid.xycenters

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

    # set layer and time coordinates
    if gwf is not None:
        head_ar.coords["layer"] = np.arange(gwf.modelgrid.nlay)
        head_ar.coords["time"] = ds_time_idx(
            headobj.get_times(),
            start_datetime=gwf.modeltime.start_datetime,
            time_units=gwf.modeltime.time_units,
        )
    else:
        head_ar.coords["layer"] = ds.layer
        head_ar.coords["time"] = ds_time_idx(
            headobj.get_times(),
            start_datetime=ds.time.attrs["start"],
            time_units=ds.time.attrs["time_units"],
        )

    if ds is not None and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # affine = get_affine(ds)
        affine = get_affine_mod_to_world(ds)
        head_ar.rio.write_transform(affine, inplace=True)

    elif gwf is not None and gwf.modelgrid.angrot != 0.0:
        attrs = dict(
            # delr=np.unique(gwf.modelgrid.delr).item(),
            # delc=np.unique(gwf.modelgrid.delc).item(),
            xorigin=gwf.modelgrid.xoffset,
            yorigin=gwf.modelgrid.yoffset,
            angrot=gwf.modelgrid.angrot,
            extent=gwf.modelgrid.extent,
        )
        affine = get_affine_mod_to_world(attrs)
        head_ar.rio.write_transform(affine, inplace=True)

    head_ar.rio.write_crs("EPSG:28992", inplace=True)

    return head_ar


def get_budget_da(text, ds=None, gwf=None, fname_cbc=None, kstpkper=None):
    """Reads budget file given either a dataset or a groundwater flow object.

    Parameters
    ----------
    text : str
        record to get from budget file
    ds : xarray.Dataset, optional
        xarray dataset with model data. One of ds or gwf must be provided.
    gwf : flopy ModflowGwf, optional
        Flopy groundwaterflow object. One of ds or gwf must be provided.
    fname_cbc : path, optional
        specify the budget file to load, if not provided budget file will
        be obtained from ds or gwf.

    Returns
    -------
    q_ar : xarray.DataArray
        budget data array.
    """
    cbcobj = _get_cbc(ds=ds, gwf=gwf, fname_cbc=fname_cbc)

    q = cbcobj.get_data(text=text, kstpkper=kstpkper, full3D=True)
    q = np.stack(q)

    if gwf is not None:
        gridtype = gwf.modelgrid.grid_type
    else:
        gridtype = ds.gridtype

    if gridtype == "vertex":
        q_ar = xr.DataArray(
            data=q,
            dims=("time", "layer", "icell2d"),
            coords={},
            attrs={"units": "m3/d"},
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

        q_ar = xr.DataArray(
            data=q,
            dims=("time", "layer", "y", "x"),
            coords={
                "x": x,
                "y": y,
            },
            attrs={"units": "m3/d"},
        )
    else:
        assert 0, "Gridtype not supported"

    # set layer and time coordinates
    if gwf is not None:
        q_ar.coords["layer"] = np.arange(gwf.modelgrid.nlay)
        # q_ar.coords["time"] = ds_time_from_model(gwf)
        q_ar.coords["time"] = ds_time_idx(
            cbcobj.get_times(),
            start_datetime=gwf.modeltime.start_datetime,
            time_units=gwf.modeltime.time_units,
        )
    else:
        q_ar.coords["layer"] = ds.layer
        q_ar.coords["time"] = ds_time_idx(
            cbcobj.get_times(),
            start_datetime=ds.time.attrs["start_datetime"],
            time_units=ds.time.attrs["time_units"],
        )
    if ds is not None and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # affine = get_affine(ds)  # TODO: determine which is needed?
        affine = get_affine_mod_to_world(ds)
        q_ar.rio.write_transform(affine, inplace=True)

    elif gwf is not None and gwf.modelgrid.angrot != 0.0:
        attrs = dict(
            delr=np.unique(gwf.modelgrid.delr).item(),
            delc=np.unique(gwf.modelgrid.delc).item(),
            xorigin=gwf.modelgrid.xoffset,
            yorigin=gwf.modelgrid.yoffset,
            angrot=gwf.modelgrid.angrot,
            extent=gwf.modelgrid.extent,
        )
        affine = get_affine_mod_to_world(attrs)
        q_ar.rio.write_transform(affine, inplace=True)

    q_ar.rio.write_crs("EPSG:28992", inplace=True)

    return q_ar


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
            cbc = gwf.output.budget()
        else:
            fname_cbc = os.path.join(ds.model_ws, ds.model_name + ".cbc")
    if fname_cbc is not None:
        cbc = flopy.utils.CellBudgetFile(fname_cbc)
    return cbc


def get_gwl_from_wet_cells(head, layer="layer", botm=None):
    """Get the groundwater level from a multi-dimensional head array where dry
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


def get_head_at_point(head, x, y, ds=None, gi=None, drop_nan_layers=True):
    """Get the head at a certain point from a head DataArray for all cells.

    Parameters
    ----------
    head : xarray.DataArray
        A DataArray of heads, with dimensions (time, layer, y, x) or (time, layer,
        icell2d).
    x : float
        The x-coordinate of the requested head.
    y : float
        The y-coordinate of the requested head.
    ds : xarray.Dataset, optional
        Xarray dataset with model data. Only used when a Vertex grid is used, and gi is
        not supplied. The default is None.
    gi : flopy.utils.GridIntersect, optional
        A GridIntersect class, to determine the cell at point x,y. Only used when a
        Vertex grid is used, and it is determined from ds when None. The default is
        None.
    drop_nan_layers : bool, optional
        Drop layers that are NaN at all timesteps. The default is True.

    Returns
    -------
    head_point : xarray.DataArray
        A DataArray with dimensions (time, layer).
    """
    if "icell2d" in head.dims:
        if gi is None:
            if ds is None:
                raise (Exception("Please supply either gi or ds for a vertex grid"))
            gi = flopy.utils.GridIntersect(
                modelgrid_from_ds(ds), method="vertex")
        icelld2 = gi.intersect(Point(x, y))["cellids"][0]
        head_point = head[:, :, icelld2]
    else:
        head_point = head.interp(x=x, y=y, method="nearest")
    if drop_nan_layers:
        # only keep layers that are active at this location
        head_point = head_point[:, ~head_point.isnull().all("time")]
    return head_point


def _calculate_gxg(
    head_bimonthly: xr.DataArray, below_surfacelevel: bool = False
) -> xr.DataArray:
    import bottleneck as bn

    # Most efficient way of finding the three highest and three lowest is via a
    # partition. See:
    # https://bottleneck.readthedocs.io/en/latest/reference.html#bottleneck.partition

    def lowest3_mean(da: xr.DataArray):
        a = bn.partition(da.values, kth=2, axis=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmean(a[..., :3], axis=-1)

        template = da.isel(bimonth=0)
        return template.copy(data=result)

    def highest3_mean(da: xr.DataArray):
        a = bn.partition(-da.values, kth=2, axis=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmean(-a[..., :3], axis=-1)

        template = da.isel(bimonth=0)
        return template.copy(data=result)

    timesize = head_bimonthly["time"].size
    if timesize % 24 != 0:
        raise ValueError("head is not bimonthly for a full set of years")
    n_year = int(timesize / 24)

    # First and second date of March: 4, 5; first date of April: 6.
    month_index = np.array([4, 5, 6])
    # Repeat this for every year in dataset, and increment by 24 per repetition.
    yearly_increments = (np.arange(n_year) * 24)[:, np.newaxis]
    # Broadcast to a full set
    gvg_index = xr.DataArray(
        data=(month_index + yearly_increments), dims=("hydroyear", "bimonth")
    )
    gvg_data = head_bimonthly.isel(time=gvg_index)
    # Filters years without 3 available measurments.
    gvg_years = gvg_data.count("bimonth") == 3
    gvg_data = gvg_data.where(gvg_years)

    # Hydrological years: running from 1 April to 1 April in the Netherlands.
    # Increment run from April (6th date) to April (30th date) for every year.
    # Broadcast to a full set
    newdims = ("hydroyear", "bimonth")
    gxg_index = xr.DataArray(
        data=(np.arange(6, 30) + yearly_increments[:-1]),
        dims=newdims,
    )
    gxg_data = head_bimonthly.isel(time=gxg_index)
    dims = [dim for dim in gxg_data.dims if dim not in newdims]
    dims.extend(newdims)
    gxg_data = gxg_data.transpose(*dims)

    # Filter years without 24 measurements.
    gxg_years = gxg_data.count("bimonth") == 24
    gxg_data = gxg_data.where(gxg_years)

    # First compute LG3 and HG3 per hydrological year, then compute the mean over the total.
    if gxg_data.chunks is not None:
        # If data is lazily loaded/chunked, process data of one year at a time.
        gxg_data = gxg_data.chunk({"hydroyear": 1})
        lg3 = xr.map_blocks(lowest3_mean, gxg_data,
                            template=gxg_data.isel(bimonth=0))
        hg3 = xr.map_blocks(highest3_mean, gxg_data,
                            template=gxg_data.isel(bimonth=0))
    else:
        # Otherwise, just compute it in a single go.
        lg3 = lowest3_mean(gxg_data)
        hg3 = highest3_mean(gxg_data)

    gxg = xr.Dataset()
    gxg["gvg"] = gvg_data.mean(("hydroyear", "bimonth"))

    ghg = hg3.mean("hydroyear")
    glg = lg3.mean("hydroyear")
    if below_surfacelevel:
        gxg["glg"] = ghg
        gxg["ghg"] = glg
    else:
        gxg["glg"] = glg
        gxg["ghg"] = ghg

    # Add the numbers of years used in the calculation
    gxg["n_years_gvg"] = gvg_years.sum("hydroyear")
    gxg["n_years_gxg"] = gxg_years.sum("hydroyear")
    return gxg


def calculate_gxg(
    head: xr.DataArray,
    below_surfacelevel: bool = False,
    tolerance: pd.Timedelta = pd.Timedelta(days=7),
) -> xr.DataArray:
    """Calculate GxG groundwater characteristics from head time series.

    GLG and GHG (average lowest and average highest groundwater level respectively) are
    calculated as the average of the three lowest (GLG) or highest (GHG) head values per
    Dutch hydrological year (april - april), for head values measured at a semi-monthly
    frequency (14th and 28th of every month). GVG (average spring groundwater level) is
    calculated as the average of groundwater level on 14th and 28th of March, and 14th
    of April. Supplied head values are resampled (nearest) to the 14/28 frequency.

    Hydrological years without all 24 14/28 dates present are discarded for glg and ghg.
    Years without the 3 dates for gvg are discarded.

    This method is copied from imod-python, and edited so that head-DataArray does not
    need to contain dimensions 'x' and 'y', so this method also works for refined grids.
    The original method can be found in:
    https://gitlab.com/deltares/imod/imod-python/-/blob/master/imod/evaluate/head.py

    Parameters
    ----------
    head : xr.DataArray of floats
        Head relative to sea level, in m, or m below surface level if
        `below_surfacelevel` is set to True. Must have dimenstion 'time'.
    below_surfacelevel : boolean, optional, default: False.
        False (default) if heads are relative to a datum (e.g. sea level). If
        True, heads are taken as m below surface level.
    tolerance: pd.Timedelta, default: 7 days.
        Maximum time window allowed when searching for dates around the 14th
        and 28th of every month.

    Returns
    -------
    gxg : xr.Dataset
        Dataset containing ``glg``: average lowest head, ``ghg``: average
        highest head, ``gvg``: average spring head, ``n_years_gvg``: numbers of
        years used for gvg, ``n_years_gxg``: numbers of years used for glg and
        ghg.

    Examples
    --------
    Load the heads, and calculate groundwater characteristics for the simulation period:

    >>> import nlmod
    >>> head = nlmod.gwf.get_heads_da(ds)
    >>> gxg = nlmod.evaluate.calculate_gxg(head)
    """
    # if not head.dims == ("time", "y", "x"):
    #    raise ValueError('Dimensions must be ("time", "y", "x")')
    if not np.issubdtype(head["time"].dtype, np.datetime64):
        raise ValueError("Time must have dtype numpy datetime64")

    # Reindex to GxG frequency date_range: every 14th and 28th of the month.
    start = f"{int(head['time'][0].dt.year)}-01-01"
    end = f"{int(head['time'][-1].dt.year)}-12-31"
    dates = pd.date_range(start=start, end=end,
                          freq="SMS") + pd.DateOffset(days=13)
    head_bimonthly = head.reindex(
        time=dates, method="nearest", tolerance=tolerance)

    gxg = _calculate_gxg(head_bimonthly, below_surfacelevel)
    return gxg
