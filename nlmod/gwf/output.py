import logging
import warnings

import flopy
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from ..dims.grid import modelgrid_from_ds
from ..dims.resample import get_affine_world_to_mod
from ..mfoutput.mfoutput import (
    _get_budget_da,
    _get_heads_da,
    _get_time_index,
    _get_flopy_data_object,
    _get_grb_file,
)

logger = logging.getLogger(__name__)


def get_headfile(ds=None, gwf=None, fname=None, grb_file=None):
    """Get flopy HeadFile object.

    Provide one of ds, gwf or fname.

    Parameters
    ----------
    ds : xarray.Dataset, optional
        model dataset, by default None
    gwf : flopy.mf6.ModflowGwf, optional
        groundwater flow model, by default None
    fname : str, optional
        path to heads file, by default None
    grb_file : str
        path to file containing binary grid information

    Returns
    -------
    flopy.utils.HeadFile
        HeadFile object handle
    """
    return _get_flopy_data_object("head", ds, gwf, fname, grb_file)


def get_heads_da(
    ds=None,
    gwf=None,
    fname=None,
    grb_file=None,
    delayed=False,
    chunked=False,
    **kwargs,
):
    """Read binary heads file.


    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset with model data.
    gwf : flopy ModflowGwf
        Flopy groundwaterflow object.
    fname : path, optional
        path to a binary heads file
    grb_file : str, optional
        path to file containing binary grid information, only needed if reading
        output from file using fname
    delayed : bool, optional
        if delayed is True, do not load output data into memory, default is False.
    chunked : bool, optional
        chunk data array containing output, default is False.

    Returns
    -------
    da : xarray.DataArray
        heads data array.
    """
    hobj = get_headfile(ds=ds, gwf=gwf, fname=fname, grb_file=grb_file)
    # gwf.output.head() defaults to a structured grid
    if gwf is not None and ds is None and fname is None:
        kwargs["modelgrid"] = gwf.modelgrid
    da = _get_heads_da(hobj, **kwargs)
    da.attrs["units"] = "m NAP"

    # set time index if ds/gwf are provided
    if ds is not None or gwf is not None:
        da["time"] = _get_time_index(hobj, ds=ds, gwf_or_gwt=gwf)
        if ds is not None:
            da["layer"] = ds.layer

    if chunked:
        # chunk data array
        da = da.chunk("auto")

    if not delayed:
        # load into memory
        da = da.compute()

    return da


def get_cellbudgetfile(ds=None, gwf=None, fname=None, grb_file=None):
    """Get flopy CellBudgetFile object.

    Provide one of ds, gwf or fname.

    Parameters
    ----------
    ds : xarray.Dataset, optional
        model dataset, by default None
    gwf : flopy.mf6.ModflowGwf, optional
        groundwater flow model, by default None
    fname_cbc : str, optional
        path to cell budget file, by default None
    grb_file : str, optional
        path to file containing binary grid information, only needed if
        fname_cbc is passed as only argument.

    Returns
    -------
    flopy.utils.CellBudgetFile
        CellBudgetFile object handle
    """
    return _get_flopy_data_object("budget", ds, gwf, fname, grb_file)


def get_budget_da(
    text,
    ds=None,
    gwf=None,
    fname=None,
    grb_file=None,
    column="q",
    delayed=False,
    chunked=False,
    **kwargs,
):
    """Read binary budget file.

    Parameters
    ----------
    text : str
        record to get from budget file
    ds : xarray.Dataset, optional
        xarray dataset with model data. One of ds or gwf must be provided.
    gwf : flopy ModflowGwf, optional
        Flopy groundwaterflow object. One of ds or gwf must be provided.
    fname : path, optional
        specify the budget file to load, if not provided budget file will
        be obtained from ds or gwf.
    grb_file : str
        path to file containing binary grid information, only needed if reading
        output from file using fname
    column : str
        name of column in rec-array to read, default is 'q' which contains the fluxes
        for most budget datasets.
    delayed : bool, optional
        if delayed is True, do not load output data into memory, default is False.
    chunked : bool, optional
        chunk data array containing output, default is False.

    Returns
    -------
    da : xarray.DataArray
        budget data array.
    """
    cbcobj = get_cellbudgetfile(ds=ds, gwf=gwf, fname=fname, grb_file=grb_file)
    da = _get_budget_da(cbcobj, text, column=column, **kwargs)
    da.attrs["units"] = "m3/d"

    # set time index if ds/gwt are provided
    if ds is not None or gwf is not None:
        da["time"] = _get_time_index(cbcobj, ds=ds, gwf_or_gwt=gwf)
        if ds is not None:
            da["layer"] = ds.layer

    if chunked:
        # chunk data array
        da = da.chunk("auto")

    if not delayed:
        # load into memory
        da = da.compute()

    return da


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


def get_flow_residuals(ds, gwf=None, fname=None, grb_file=None, kstpkper=None):
    """
    Get the flow residuals of a MODFLOW 6 simulation.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset with model data.
    gwf : flopy ModflowGwf, optional
        Flopy groundwaterflow object. One of ds or gwf must be provided.
    fname : path, optional
        specify the budget file to load, if not provided budget file will
        be obtained from ds or gwf.
    grb_file : str
        The location of the grb-file. grb_file is determied from ds when None. The
        default is None.
    kstpkper : tuple of 2 ints, optional
        The index of the timestep and the stress period to include in the result. Include
        all data in the budget-file when None. The default is None.

    Returns
    -------
    da : xr.DataArray
        The flow residual in each cell, in m3/d.

    """
    if grb_file is None:
        grb_file = _get_grb_file(ds)
    grb = flopy.mf6.utils.MfGrdFile(grb_file)
    cbf = get_cellbudgetfile(ds=ds, gwf=gwf, fname=fname, grb_file=grb_file)
    dims = ds["botm"].dims
    coords = ds["botm"].coords
    flowja = cbf.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)
    mask_active = np.diff(grb.ia) > 0
    flowja_index = grb.ia[:-1][mask_active]
    if kstpkper is None:
        # loop over all timesteps/stress-periods
        residuals = []
        for iflowja in flowja:
            # residuals.append(flopy.mf6.utils.get_residuals(iflowja, grb_file))
            # use our own faster method instead of a for loop:
            residual = np.full(grb.shape, np.NaN)
            residual.ravel()[mask_active] = iflowja.flatten()[flowja_index]
            residuals.append(residual)
        dims = ("time",) + dims
        coords = dict(coords) | {"time": _get_time_index(cbf, ds)}
    else:
        # residuals = flopy.mf6.utils.get_residuals(flowja[0], grb_file)
        # use our own faster method instead of a for loop:
        residuals = np.full(grb.shape, np.NaN)
        residuals.ravel()[mask_active] = flowja[0].flatten()[flowja_index]
    da = xr.DataArray(residuals, dims=dims, coords=coords)
    return da


def get_flow_lower_face(
    ds, gwf=None, fname=None, grb_file=None, kstpkper=None, lays=None
):
    """
    Get the flow over the lower face of all model cells

    The flow Lower Face (flf) used to be written to the budget file in previous versions
    of MODFLOW. In MODFLOW 6 we determine these flows from the flow-ja-face-records.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset with model data.
    gwf : flopy ModflowGwf, optional
        Flopy groundwaterflow object. One of ds or gwf must be provided.
    fname : path, optional
        specify the budget file to load, if not provided budget file will
        be obtained from ds or gwf.
    grb_file : str, optional
        The location of the grb-file. grb_file is determied from ds when None. The
        default is None.
    kstpkper : tuple of 2 ints, optional
        The index of the timestep and the stress period to include in the result. Include
        all data in the budget-file when None. The default is None.
    lays : int or list of ints, optional
        The layers to include in the result. When lays is None, all layers are included.
        The default is None.

    Returns
    -------
    da : xr.DataArray
        The flow over the lower face of each cell, in m3/d.

    """
    if grb_file is None:
        grb_file = _get_grb_file(ds)
    cbf = get_cellbudgetfile(ds=ds, gwf=gwf, fname=fname, grb_file=grb_file)
    flowja = cbf.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)

    if ds.gridtype == "vertex":
        # determine flf_index first
        grb = flopy.mf6.utils.MfGrdFile(grb_file)

        if lays is None:
            lays = range(grb.nlay)
        if isinstance(lays, int):
            lays = [lays]
        shape = (len(lays), len(ds.icell2d))

        flf_index = np.full(shape, -1)
        # get these properties outside of the for loop to increase speed
        grb_ia = grb.ia
        grb_ja = grb.ja
        for ilay, lay in enumerate(lays):
            ja_start_next_layer = (lay + 1) * grb.ncpl
            for icell2d in range(grb.ncpl):
                node = lay * grb.ncpl + icell2d
                ia = np.arange(grb_ia[node], grb_ia[node + 1])
                mask = grb_ja[ia] >= ja_start_next_layer
                if mask.any():
                    # assert mask.sum() == 1
                    flf_index[ilay, icell2d] = int(ia[mask])
        coords = ds["botm"][lays].coords
    else:
        coords = ds["botm"].coords
    dims = ds["botm"].dims

    if kstpkper is None:
        # loop over all tiesteps/stress-periods
        flfs = []
        for iflowja in flowja:
            if ds.gridtype == "vertex":
                flf = np.full(shape, np.NaN)
                mask = flf_index >= 0
                flf[mask] = iflowja[0, 0, flf_index[mask]]
            else:
                _, _, flf = flopy.mf6.utils.get_structured_faceflows(iflowja, grb_file)
            flfs.append(flf)
        dims = ("time",) + dims
        coords = dict(coords) | {"time": _get_time_index(cbf, ds)}
    else:
        if ds.gridtype == "vertex":
            flfs = np.full(shape, np.NaN)
            mask = flf_index >= 0
            flfs[mask] = flowja[0][0, 0, flf_index[mask]]
        else:
            _, _, flfs = flopy.mf6.utils.get_structured_faceflows(flowja[0], grb_file)
    da = xr.DataArray(flfs, dims=dims, coords=coords)
    if ds.gridtype != "vertex" and lays is not None:
        da = da.isel(layer=lays)
    return da


def get_head_at_point(
    head, x, y, ds=None, gi=None, drop_nan_layers=True, rotated=False
):
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
    rotated : bool, optional
        If the model grid has a rotation, and rotated is False, x and y are in model
        coordinates. Otherwise x and y are in real world coordinates. The defaults is
        False.

    Returns
    -------
    head_point : xarray.DataArray
        A DataArray with dimensions (time, layer).
    """
    if rotated and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # calculate model coordinates from the specified real-world coordinates
        x, y = get_affine_world_to_mod(ds) * (x, y)

    if "icell2d" in head.dims:
        if gi is None:
            if ds is None:
                raise (
                    ValueError(
                        "Please supply either gi (GridIntersect) or ds for a vertex grid"
                    )
                )
            gi = flopy.utils.GridIntersect(modelgrid_from_ds(ds), method="vertex")
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
        lg3 = xr.map_blocks(lowest3_mean, gxg_data, template=gxg_data.isel(bimonth=0))
        hg3 = xr.map_blocks(highest3_mean, gxg_data, template=gxg_data.isel(bimonth=0))
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
    >>> gxg = nlmod.gwf.output.calculate_gxg(head)
    """
    # if not head.dims == ("time", "y", "x"):
    #    raise ValueError('Dimensions must be ("time", "y", "x")')
    if not np.issubdtype(head["time"].dtype, np.datetime64):
        raise ValueError("Time must have dtype numpy datetime64")

    # Reindex to GxG frequency date_range: every 14th and 28th of the month.
    start = f"{int(head['time'][0].dt.year)}-01-01"
    end = f"{int(head['time'][-1].dt.year)}-12-31"
    dates = pd.date_range(start=start, end=end, freq="SMS") + pd.DateOffset(days=13)
    head_bimonthly = head.reindex(time=dates, method="nearest", tolerance=tolerance)

    gxg = _calculate_gxg(head_bimonthly, below_surfacelevel)
    return gxg
