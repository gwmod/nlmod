import logging
import numbers

import flopy
import numpy as np
import pandas as pd
import xarray as xr

from ..dims.grid import cols_to_reclist, da_to_reclist
from ..dims.layers import (
    calculate_thickness,
    get_first_active_layer,
    get_first_active_layer_from_idomain,
    get_idomain,
)
from ..dims.time import dataframe_to_flopy_timeseries
from ..util import _get_value_from_ds_datavar

logger = logging.getLogger(__name__)


def ds_to_rch(
    gwf, ds, mask=None, pname="rch", recharge="recharge", auxiliary=None, **kwargs
):
    """Convert recharge data in the model dataset to a rch package with time series.

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    ds : xr.DataSet
        dataset containing relevant model grid information
    mask : xr.DataArray
        data array containing mask, recharge is only added where mask is True
    pname : str, optional
        package name. The default is 'rch'.
    recharge : str, xr.DataArray or float, optional
        When recharge is a string, it is the name of the variable in ds that contains
        the recharge flux rate. When recharge is a float, it is interpreted as the
        constant recharge rate that is applied in all active cells (within mask if mask
        is supplied). The default is "recharge".
    auxiliary : str or list of str
        name(s) of data arrays to include as auxiliary data to reclist

    Returns
    -------
    rch : flopy.mf6.ModflowGwfrch
        recharge package
    """
    stn_var = f"{recharge}_stn" if isinstance(recharge, str) else "recharge_stn"
    recharge, mask_recharge, rch_unique_df = _get_meteo_da_from_input(
        recharge, ds, pname, stn_var=stn_var
    )
    if mask is not None:
        mask_recharge = mask & mask_recharge

    spd = da_to_reclist(
        ds,
        mask_recharge,
        col1=recharge,
        first_active_layer=True,
        aux=auxiliary,
    )

    # create rch package
    rch = flopy.mf6.ModflowGwfrch(
        gwf,
        filename=f"{gwf.name}.rch",
        pname=pname,
        fixed_cell=False,
        auxiliary="CONCENTRATION" if auxiliary is not None else None,
        maxbound=len(spd),
        stress_period_data={0: spd},
        **kwargs,
    )
    if (auxiliary is not None) and (ds.transport == 1):
        logger.info("-> adding RCH to SSM sources list")
        ssm_sources = list(ds.attrs["ssm_sources"])
        if rch.package_name not in ssm_sources:
            ssm_sources += [rch.package_name]
            ds.attrs["ssm_sources"] = ssm_sources

    if rch_unique_df is not None:
        # create timeseries packages
        _add_time_series(rch, rch_unique_df, ds)

    return rch


def ds_to_evt(
    gwf,
    ds,
    mask=None,
    pname="evt",
    rate="evaporation",
    nseg=1,
    surface=None,
    depth=None,
    auxiliary=None,
    **kwargs,
):
    """Convert evaporation data in the model dataset to a evt package with time series.

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    ds : xr.DataSet
        dataset containing relevant model grid information
    mask : xr.DataArray
        data array containing mask, evt is only added where mask is True
    pname : str, optional
        package name. The default is 'evt'.
    rate : str, xr.DataArray or float, optional
        When rate is a string, it is the name of the variable in ds that contains the
        maximum ET flux rate. When rate is a float, it is interpreted as the constant
        evaporation rate that is applied in all active cells (within mask if mask is
        supplied). The default is "evaporation".
    nseg : int, optional
        number of ET segments. Only 1 is supported for now. The default is 1.
    surface : str, float or xr.DataArray, optional
        The elevation of the ET surface. Set to 1 meter below top when None. The default
        is None.
    depth : str, float or xr.DataArray, optional
        The ET extinction depth. Set to 1 meter (below surface) when None. The default
        is None.
    auxiliary : str or list of str
        name(s) of data arrays to include as auxiliary data to reclist
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    evt : flopy.mf6.ModflowGwfevt
        evapotranspiiration package.
    """
    assert nseg == 1, "More than one evaporation segment not yet supported"
    if "surf_rate_specified" in kwargs:
        raise (NotImplementedError("surf_rate_specified not yet supported"))
    if surface is None:
        logger.info("Setting evaporation surface to 1 meter below top")
        surface = ds["top"] - 1.0
    if depth is None:
        logger.info("Setting extinction depth to 1 meter below surface")
        depth = 1.0

    stn_var = f"{rate}_stn" if isinstance(rate, str) else "evaporation_stn"
    rate, mask_rate, evt_unique_df = _get_meteo_da_from_input(
        rate, ds, pname, stn_var=stn_var
    )

    if mask is not None:
        mask_rate = mask & mask_rate

    spd = da_to_reclist(
        ds,
        mask_rate,
        col1=surface,
        col2=rate,
        col3=depth,
        first_active_layer=True,
        only_active_cells=False,
        aux=auxiliary,
    )

    # create rch package
    evt = flopy.mf6.ModflowGwfevt(
        gwf,
        filename=f"{gwf.name}.evt",
        pname=pname,
        fixed_cell=False,
        auxiliary="CONCENTRATION" if auxiliary is not None else None,
        maxbound=len(spd),
        nseg=nseg,
        stress_period_data={0: spd},
        **kwargs,
    )

    if (auxiliary is not None) and (ds.transport == 1):
        logger.info("-> adding EVT to SSM sources list")
        ssm_sources = list(ds.attrs["ssm_sources"])
        if evt.package_name not in ssm_sources:
            ssm_sources += [evt.package_name]
            ds.attrs["ssm_sources"] = ssm_sources

    if evt_unique_df is not None:
        # create timeseries packages
        _add_time_series(evt, evt_unique_df, ds)

    return evt


def ds_to_uzf(
    gwf,
    ds,
    mask=None,
    pname="uzf",
    surfdep=0.05,
    vks="kv",
    thtr=0.1,
    thts=0.3,
    thti=0.1,
    eps=3.5,
    landflag=None,
    finf="recharge",
    pet="evaporation",
    extdp=None,
    extwc=None,
    ha=None,
    hroot=None,
    rootact=None,
    simulate_et=True,
    linear_gwet=True,
    unsat_etwc=False,
    unsat_etae=False,
    obs_depth_interval=None,
    obs_z=None,
    mask_obs=None,
    **kwargs,
):
    """Create a unsaturated zone flow package for modflow 6.

    This method adds uzf-cells to all active Modflow cells (unless mask is specified).

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    ds : xr.DataSet
        dataset containing relevant model grid information
    mask : xr.DataArray
        data array containing mask, uzf is only added where mask is True
    pname : str, optional
        package name. The default is 'uzf'.
    surfdep : float, str or array-like
        surfdep is the surface depression depth of the UZF cell. When passed as string,
        the array is obtained from ds. The default is 0.05 m.
    vks : float, str or array-like
        vks is the saturated vertical hydraulic conductivity of the UZF cell. This value
        is used with the Brooks-Corey function and the simulated water content to
        calculate the partially saturated hydraulic conductivity. When passed as string,
        the array is obtained from ds. The default is 'kv'.
    thtr : float, str or array-like
        thtr is the residual (irreducible) water content of the UZF cell. This residual
        water is not available to plants and will not drain into underlying aquifer
        cells. When passed as string, the array is obtained from ds. The default is
        0.1.
    thts : float, str or array-like
        thts is the saturated water content of the UZF cell. The values for saturated
        and residual water content should be set in a manner that is consistent with the
        specific yield value specified in the Storage Package. The saturated water
        content must be greater than the residual content. When passed as string, the
        array is obtained from ds. The default is 0.3.
    thti : float, str or array-like
        thti is the initial water content of the UZF cell. The value must be greater
        than or equal to the residual water content and less than or equal to the
        saturated water content. When passed as string, the array is obtained from ds.
        The default is 0.1.
    eps : float, str or array-like
        eps is the exponent used in the Brooks-Corey function. The Brooks-Corey function
        is used by UZF to calculated hydraulic conductivity under partially saturated
        conditions as a function of water content and the user-specified saturated
        hydraulic conductivity. Values must be between 3.5 and 14.0. When passed as
        string, the array is obtained from ds. The default is 3.5.
    landflag : xr.DataArray, optional
        A DataArray with integer values, set to one for land surface cells indicating
        that boundary conditions can be applied and data can be specified in the PERIOD
        block. A value of 0 specifies a non-land surface cell. Landflag is set to one
        for the most upper active layer in each vertical column (determined form ds)
        when it is None. The default is None.
    finf :float, str or array-like
        The applied infiltration rate of the UZF cell (:math:`LT^{-1}`). When passed as
        string, the array is obtained from ds. The default is "recharge".
    pet : float, str or array-like
        The potential evapotranspiration rate of the UZF cell and specified GWF cell.
        Evapotranspiration is first removed from the unsaturated zone and any remaining
        potential evapotranspiration is applied to the saturated zone. only
        used if simulate_et=True. When passed as string, the array is obtained from ds.
        The default is "evaporation".
    extdp : float, optional
        Value that defines the evapotranspiration extinction depth of the UZF cell, in
        meters below the top of the model. Set to 2.0 meter when None. The default is
        None.
    extwc : float, optional
        The evapotranspiration extinction water content of the UZF cell. Only used if
        simulate_et=True and unsat_etwc=True. Set to thtr when None. The default is
        None.
    ha : float, optional
        The air entry potential (head) of the UZF cell. Only used if simulate_et=True
        and unsat_etae=True. Set to 0.0 when None. The default is None.
    hroot : float, optional
        The root potential (head) of the UZF cell. Only used if simulate_et=True and
        unsat_etae=True. Set to 0.0 when None. The default is None.
    rootact : float, optional
        the root activity function of the UZF cell. ROOTACT is the length of roots in
        a given volume of soil divided by that volume. Values range from 0 to about 3
        :math:`cm^{-2}`, depending on the plant community and its stage of development.
        Only used if simulate_et=True and unsat_etae=True. Set to 0.0 when None. The
        default is None.
    simulate_et : bool, optional
        If True, ET in the unsaturated (UZF) and saturated zones (GWF) will be
        simulated. The default is True.
    linear_gwet : bool, optional
        If True, groundwater ET will be simulated using the original ET formulation of
        MODFLOW-2005. When False, and square_gwet is True as an extra argument, no
        evaporation from the saturated zone will be simulated. The default is True.
    unsat_etwc : bool, optional
        If True, ET in the unsaturated zone will be simulated as a function of the
        specified PET rate while the water content (THETA) is greater than the ET
        extinction water content (EXTWC). The default is False.
    unsat_etae : bool, optional
        If True, ET in the unsaturated zone will be simulated using a capillary pressure
        based formulation. Capillary pressure is calculated using the Brooks-Corey
        retention function. The default is False.
    obs_depth_interval : float, optional
        The depths at which observations of the water content in each cell are added.
        When not None, this creates a CSV output file with water content at different
        z-coordinates in each UZF cell. The default is None.
    obs_z : array-like, optional
        The z-coordinate at which observations of the water content in each cell are
        added. When not None, this creates a CSV output file with water content at fixes
        z-coordinates in each UZF cell. The default is None.
    mask_obs : xr.DataArray, optional
        Mask with the cells where an observations is added. If None all cells will get
        an observation. The default is None.
    ** kwargs : dict
        Kwargs are passed onto flopy.mf6.ModflowGwfuzf


    Returns
    -------
    uzf : flopy.mf6.ModflowGwfuzf
        Unsaturated zone flow package for Modflow 6.
    """
    if mask is None:
        mask = ds["area"] > 0

    if "layer" not in mask.dims:
        mask = mask.expand_dims(dim={"layer": ds.layer})

    # only add uzf-cells in active cells
    idomain = get_idomain(ds)
    mask = mask & (idomain > 0)

    # generate packagedata
    surfdep = _get_value_from_ds_datavar(ds, "surfdep", surfdep, return_da=True)
    vks = _get_value_from_ds_datavar(ds, "vks", vks, return_da=True)
    thtr = _get_value_from_ds_datavar(ds, "thtr", thtr, return_da=True)
    thts = _get_value_from_ds_datavar(ds, "thts", thts, return_da=True)
    thti = _get_value_from_ds_datavar(ds, "thti", thti, return_da=True)
    eps = _get_value_from_ds_datavar(ds, "eps", eps, return_da=True)

    nuzfcells = int(mask.sum())
    cellids = np.where(mask)

    iuzno = xr.full_like(ds["botm"], -1, dtype=int)
    iuzno.data[mask] = np.arange(nuzfcells)

    if landflag is None:
        landflag = xr.full_like(ds["botm"], 0, dtype=int)
        # set the landflag in the top layer to 1
        fal = get_first_active_layer_from_idomain(idomain)
        # for the inactive domain set fal to 0 (setting nodata to 0 gives problems)
        fal.data[fal == fal.nodata] = 0
        landflag[fal] = 1
        # set landflag to 0 in inactivate domain (where we set fal to 0 before)
        landflag = xr.where(idomain > 0, landflag, 0)

    # determine ivertcon, by setting its value to iuzno of the layer below
    ivertcon = xr.full_like(ds["botm"], -1, dtype=int)
    ivertcon.data[:-1] = iuzno.data[1:]
    # then use bfill to accont for inactive cells in the layer below, and set nans to -1
    ivertcon = ivertcon.where(ivertcon >= 0).bfill("layer").fillna(-1).astype(int)

    # packagedata : [iuzno, cellid, landflag, ivertcon, surfdep, vks, thtr, thts, thti, eps, boundname]
    packagedata = cols_to_reclist(
        ds,
        cellids,
        iuzno,
        landflag,
        ivertcon,
        surfdep,
        vks,
        thtr,
        thts,
        thti,
        eps,
        cellid_column=1,
    )

    # add perioddata for all uzf cells that are at the surface
    mask_surface = (landflag == 1) & mask

    # perioddata : [iuzno, finf, pet, extdp, extwc, ha, hroot, rootact, aux]
    stn_var = f"{finf}_stn" if isinstance(finf, str) else "recharge_stn"
    finf, mask_finf, uzf_unique_df = _get_meteo_da_from_input(
        finf, ds, "finf", stn_var=stn_var
    )
    finf = finf.expand_dims(dim={"layer": ds.layer})
    mask_surface = mask_surface & mask_finf

    stn_var = f"{pet}_stn" if isinstance(pet, str) else "evaporation_stn"
    pet, mask_pet, pet_unique_df = _get_meteo_da_from_input(
        pet, ds, "pet", stn_var=stn_var
    )
    pet = pet.expand_dims(dim={"layer": ds.layer})
    mask_surface = mask_surface & mask_pet

    # combine the time series of finf and pet
    if uzf_unique_df is not None and pet_unique_df is not None:
        uzf_unique_df = pd.concat((uzf_unique_df, pet_unique_df), axis=1)

    if extdp is None:
        extdp = 2.0
        # EXTDP is always specified, but is only used if SIMULATE_ET
        if simulate_et:
            logger.info(f"Setting extinction depth (extdp) to {extdp} meter below top")
    if extwc is None:
        extwc = thtr
        if simulate_et and unsat_etwc:
            logger.info(
                f"Setting evapotranspiration extinction water content (extwc) to {extwc}"
            )
    if ha is None:
        ha = 0.0
        if simulate_et and unsat_etae:
            logger.info(f"Setting air entry potential (ha) to {ha}")
    if hroot is None:
        hroot = 0.0
        if simulate_et and unsat_etae:
            logger.info(f"Setting root potential (hroot) to {hroot}")
    if rootact is None:
        rootact = 0.0
        if simulate_et and unsat_etae:
            logger.info(f"Setting root activity function (rootact) to {rootact}")

    cellids_land = np.where(mask_surface)

    perioddata = cols_to_reclist(
        ds,
        cellids_land,
        iuzno,
        finf,
        pet,
        extdp,
        extwc,
        ha,
        hroot,
        rootact,
        cellid_column=None,
    )

    observations = None
    # observation nodes uzf
    if obs_depth_interval is not None or obs_z is not None:
        if mask_obs is None:
            mask_obs = mask

        if (mask_obs & ~mask).any():
            raise ValueError("can only have observations in active uzf cells")

        # get iuzno numbers where an observation is required
        iuzno_obs = xr.where(mask_obs, iuzno, np.nan).values
        iuzno_obs_vals = np.unique(iuzno_obs[~np.isnan(iuzno_obs)]).astype(int)

        # get cell ids of observations
        cellids_obs = list(zip(*np.where(mask_obs)))
        cellid_str = ["_".join(map(str, x)) for x in cellids_obs]

        # account for surfdep, as this decreases the height of the top of the upper cell
        # otherwise modflow may return an error
        thickness = calculate_thickness(ds)
        if isinstance(surfdep, numbers.Number):
            surfdep = xr.ones_like(thickness) * surfdep
        thickness = [thickness[x] - landflag[x] * surfdep[x] / 2 for x in cellids_obs]

        # create observations list
        obsdepths = []
        if obs_depth_interval is not None:
            for i, iuzno_o in enumerate(iuzno_obs_vals):
                depths = np.arange(obs_depth_interval, thickness[i], obs_depth_interval)
                for depth in depths:
                    name = f"wc_{cellid_str[i]}_{depth:0.2f}"
                    obsdepths.append((name, "water-content", iuzno_o + 1, depth))

        if obs_z is not None:
            botm = np.asarray([ds["botm"][x] for x in cellids_obs])
            top = botm + np.asarray(thickness)

            for i, iuzno_o in enumerate(iuzno_obs_vals):
                within_cell = (obs_z > botm[i]) & (obs_z <= top[i])
                for z in obs_z[within_cell]:
                    depth = top[i] - z
                    name = f"wc_{cellid_str[iuzno_o]}_{z:0.2f}"
                    obsdepths.append((name, "water-content", iuzno_o + 1, depth))

        observations = {ds.model_name + ".uzf.obs.csv": obsdepths}

    uzf = flopy.mf6.ModflowGwfuzf(
        gwf,
        filename=f"{gwf.name}.uzf",
        pname=pname,
        nuzfcells=nuzfcells,
        packagedata=packagedata,
        perioddata={0: perioddata},
        simulate_et=simulate_et,
        linear_gwet=linear_gwet,
        unsat_etwc=unsat_etwc,
        unsat_etae=unsat_etae,
        observations=observations,
        **kwargs,
    )

    if uzf_unique_df is not None:
        # create timeseries packages
        _add_time_series(uzf, uzf_unique_df, ds)


def _get_meteo_da_from_input(recharge, ds, pname, stn_var):
    """
    Normalize meteorological input for use in package stress-period data.

    This helper accepts several input forms for a meteorological variable
    (e.g. recharge, evaporation, finf/pet for UZF) and returns:
      - a spatial DataArray that describes either a per-cell scalar value or a
        per-cell reference to a timeseries name,
      - a boolean mask DataArray indicating which cells should receive the
        meteorological input,
      - a pandas DataFrame of unique timeseries values when time series are used
        (or None otherwise).

    Supported input types and behavior
    - str:
        Treated as the name of a variable in `ds` (i.e. `recharge = ds[recharge]`).
        Further processing follows the xr.DataArray rules below.
    - xr.DataArray:
        * 1-D time series (dims == ("time",)):
            Interpreted as a single time series applied to all active cells.
            The returned `recharge` DataArray contains the timeseries name for
            every cell (string values), `mask_recharge` marks active cells, and
            `rch_unique_df` is a DataFrame with that single series.
        * 2-D time-by-station (dims == ("time", "stn_*")):
            Interpreted as a set of timeseries, one per station. `ds[stn_var]`
            is expected to map each spatial cell to a station index. The
            returned `recharge` contains per-cell timeseries names constructed
            from the station index, `mask_recharge` marks cells with a valid
            station mapping, and `rch_unique_df` is the DataFrame of station
            timeseries (column names are prefixed with `pname_`).
        * Per-cell (spatial) array with or without time dimension:
            If the array contains a time dimension (transient, `time` in dims
            and more than one period), unique time series are identified using
            `_get_unique_series`. The returned `recharge` will then be a
            per-cell DataArray of timeseries names and `rch_unique_df` contains
            the corresponding series. If no time dimension (or single time
            index), `recharge` is reduced to a per-cell scalar array and
            `rch_unique_df` is None. NaN values in the active domain are not
            allowed (ValueError).
    - float:
        Treated as a spatially-constant value applied to all active cells;
        `mask_recharge` marks active cells and no timeseries DataFrame is
        returned.

    Parameters
    ----------
    recharge : float, str, or xr.DataArray
        The input meteorological data. See "Supported input types and behavior"
        above for interpretation rules.
    ds : xr.Dataset
        Model dataset. Used to determine active cells and grid dimensions.
    pname : str
        Package name used as prefix when constructing timeseries names.
    stn_var : str
        Name of the DataArray in `ds` that maps cells to station indices. Used
        only when `recharge` is a time-by-station DataArray.

    Returns
    -------
    recharge : xr.DataArray
        If timeseries are used, a spatial DataArray of dtype string containing
        the timeseries name for each cell (e.g. "rch_0", "rch_1", ...).
        Otherwise a per-cell scalar DataArray (no time dimension) with the
        meteorological value(s).
    mask_recharge : xr.DataArray (bool)
        Boolean mask indicating which cells should receive the input (True
        means apply input). Typically this is based on the first active layer.
    rch_unique_df : pd.DataFrame or None
        When one or more unique timeseries are detected, a DataFrame indexed by
        model `ds.time` containing each unique series is returned. If no
        timeseries are used, returns None.

    Raises
    ------
    ValueError
        If per-cell data contains NaN values in the active model domain.
    NotImplementedError
        If `recharge` is of an unsupported type.

    Notes
    -----
    - Timeseries names are constructed using `pname` as a prefix (e.g.
      "evt_0", "finf_1").
    """
    fal = get_first_active_layer(ds)
    # get stress period data
    if isinstance(recharge, str):
        recharge = ds[recharge]
    rch_unique_df = None
    if isinstance(recharge, xr.DataArray):
        if recharge.dims == ("time",):
            # recharge only consists of the dimension time, so no spatial variation
            use_ts = True

            ts_name = f"{pname}_0"
            rch_unique_df = pd.DataFrame(recharge, columns=[ts_name])
            dims = ds["top"].dims
            coords = ds["top"].coords
            shape = [ds.sizes[dim] for dim in dims]
            recharge = xr.DataArray(np.full(shape, ts_name), dims=dims, coords=coords)
            mask_recharge = fal != fal.attrs["nodata"]
        elif (
            len(recharge.dims) == 2
            and recharge.dims[0] == "time"
            and recharge.dims[1].startswith("stn_")
        ):
            # recharge is a DataArray with time series for every station
            use_ts = True
            rch_unique_df = recharge.to_pandas()
            recharge = ds[stn_var].copy()
            mask_recharge = recharge != recharge.attrs["nodata"]

            # make sure the name of the time-series are strings
            def get_ts_name(stn):
                return f"{pname}_{stn}"

            rch_unique_df.columns = [get_ts_name(x) for x in rch_unique_df.columns]
            recharge = xr.apply_ufunc(get_ts_name, recharge.astype(int), vectorize=True)
        else:
            # recharge is a DataArray with a value for every cell and possibly time
            use_ts = "time" in recharge.dims and len(ds["time"]) > 1
            # check for nan values in active model domain
            if recharge.where(fal != fal.attrs["nodata"], 0.0).isnull().any():
                raise ValueError("please remove nan values in recharge data array")

            if use_ts:
                recharge, rch_unique_df = _get_unique_series(ds, recharge, pname)
                mask_recharge = recharge != ""
            else:
                if "time" in recharge.dims:
                    recharge = recharge.isel(time=0)
                mask_recharge = recharge != 0
    elif isinstance(recharge, float):
        mask_recharge = fal != fal.attrs["nodata"]
        use_ts = False
    else:
        raise NotImplementedError("Type {type(recharge)} not supported for recharge")

    return recharge, mask_recharge, rch_unique_df


def _get_unique_series(ds, da, pname):
    """Get the location and values of unique time series from a variable var in ds.

    Parameters
    ----------
    ds : xr.Dataset
        The model Dataset.
    da : xr.DataArray
        The 3d (structured) or 2d (vertext) DataArray that contains the timeseries.
    pname : str
        Package name, which is used for the name of the time series.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    rch_name_da : xr.DataArray
        The name of the recharge series for each of the cells.
    rch_unique_df : pd.DataFrame
        The values of each of the time series.
    """
    rch_name_arr = np.empty_like(ds["top"].values, dtype="U13")

    # transient
    if ds.gridtype == "structured":
        if len(da.dims) != 3:
            raise ValueError(
                "expected dataarray with 3 dimensions"
                f"(time, y and x) or (y, x and time), not {da.dims}"
            )
        recharge = da.transpose("y", "x", "time").data
        shape = (ds.sizes["y"] * ds.sizes["x"], ds.sizes["time"])
        rch_2d_arr = recharge.reshape(shape)

    elif ds.gridtype == "vertex":
        # dimension check
        if len(da.dims) != 2:
            raise ValueError(
                "expected dataarray with 2 dimensions"
                f"(time, icell2d) or (icell2d, time), not {da.dims}"
            )
        rch_2d_arr = da.transpose("icell2d", "time").data

    rch_unique_arr = np.unique(rch_2d_arr, axis=0)
    rch_unique_dic = {}
    for i, unique_rch in enumerate(rch_unique_arr):
        mask = (rch_2d_arr == unique_rch).all(axis=1)
        if len(rch_name_arr.shape) > 1:
            mask = mask.reshape(rch_name_arr.shape)
        rch_name_arr[mask] = f"{pname}_{i}"
        rch_unique_dic[f"{pname}_{i}"] = unique_rch

    rch_name_da = xr.DataArray(
        rch_name_arr, dims=ds["top"].dims, coords=ds["top"].coords
    )
    rch_unique_df = pd.DataFrame(rch_unique_dic, index=ds.time)

    return rch_name_da, rch_unique_df


def _add_time_series(package, df, ds):
    """Add time series to a package.

    Parameters
    ----------
    rch : mfpackage.MFPackage
        The Flopy package to which to add the timeseries.
    df : pd.DataFrame
        A pandas DataFrane that contains the time series values.
    ds : xr.Dataset
        The model Dataset. It is used to get the time of the time series.

    Returns
    -------
    None.
    """
    if df.isna().any(axis=None):
        # make sure there are no NaN's, as otherwise they will be filled by zeros later
        raise (ValueError("There cannot be nan's in the DataFrame"))
    # set the first value for the start-time as well
    df.loc[pd.to_datetime(ds.time.start)] = df.iloc[0]
    # combine the values with the start of each period, and set the last value to 0.0
    df = df.sort_index().shift(-1).fillna(0.0)

    dataframe_to_flopy_timeseries(
        df, ds=ds, package=package, interpolation_methodrecord="stepwise"
    )
