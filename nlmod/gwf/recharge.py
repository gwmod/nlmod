import logging

import flopy
import numpy as np
import pandas as pd
import xarray as xr

from ..dims.grid import da_to_reclist, cols_to_reclist
from ..dims.layers import (
    get_idomain,
    get_first_active_layer_from_idomain,
    calculate_thickness,
)
from ..dims.time import dataframe_to_flopy_timeseries
from ..util import _get_value_from_ds_datavar

logger = logging.getLogger(__name__)


def ds_to_rch(
    gwf, ds, mask=None, pname="rch", recharge="recharge", auxiliary=None, **kwargs
):
    """Convert the recharge data in the model dataset to a rch package with
    time series.

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
    auxiliary : str or list of str
        name(s) of data arrays to include as auxiliary data to reclist
    recharge : str, optional
        The name of the variable in ds that contains the recharge flux rate. The default
        is "recharge".

    Returns
    -------
    rch : flopy.mf6.ModflowGwfrch
        recharge package
    """
    # check for nan values
    if ds["recharge"].isnull().any():
        raise ValueError("please remove nan values in recharge data array")

    # get stress period data
    use_ts = "time" in ds[recharge].dims and len(ds["time"]) > 1
    if not use_ts:
        recharge = ds[recharge]
        if "time" in recharge.dims:
            recharge = recharge.isel(time=0)
        mask_recharge = recharge != 0
    else:
        rch_name_arr, rch_unique_dic = _get_unique_series(ds, recharge, pname)
        ds["rch_name"] = ds["top"].dims, rch_name_arr
        recharge = ds["rch_name"]
        mask_recharge = recharge != ""

    if mask is not None:
        mask_recharge = mask & mask_recharge

    spd = da_to_reclist(
        ds,
        mask_recharge,
        col1=recharge,
        first_active_layer=True,
        only_active_cells=False,
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
        logger.info("-> adding GHB to SSM sources list")
        ssm_sources = list(ds.attrs["ssm_sources"])
        if rch.package_name not in ssm_sources:
            ssm_sources += [rch.package_name]
            ds.attrs["ssm_sources"] = ssm_sources

    if use_ts:
        # create timeseries packages
        _add_time_series(rch, rch_unique_dic, ds)

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
    **kwargs,
):
    """Convert the evaporation data in the model dataset to a evt package with
    time series.

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
    rate : str, optional
        The name of the variable in ds that contains the maximum ET flux rate. The
        default is "evaporation".
    nseg : int, optional
        number of ET segments. Only 1 is supported for now. The default is 1.
    surface : str, float or xr.DataArray, optional
        The elevation of the ET surface. Set to 1 meter below top when None. The default
        is None.
    depth : str, float or xr.DataArray, optional
        The ET extinction depth. Set to 1 meter (below surface) when None. The default
        is None.
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

    # check for nan values
    if ds[rate].isnull().any():
        raise ValueError("please remove nan values in evaporation data array")

    # get stress period data
    use_ts = "time" in ds[rate].dims and len(ds["time"]) > 1
    if not use_ts:
        rate = ds[rate]
        if "time" in rate.dims:
            rate = rate.isel(time=0)
        mask_rate = rate != 0
    else:
        evt_name_arr, evt_unique_dic = _get_unique_series(ds, rate, pname)
        ds["evt_name"] = ds["top"].dims, evt_name_arr
        rate = ds["evt_name"]
        mask_rate = rate != ""

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
    )

    # create rch package
    evt = flopy.mf6.ModflowGwfevt(
        gwf,
        filename=f"{gwf.name}.evt",
        pname=pname,
        fixed_cell=False,
        maxbound=len(spd),
        nseg=nseg,
        stress_period_data={0: spd},
        **kwargs,
    )

    if use_ts:
        # create timeseries packages
        _add_time_series(evt, evt_unique_dic, ds)

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
    **kwargs,
):
    """Create a unsaturated zone flow package for modflow 6. This method adds uzf-cells
    to all active Modflow cells (unless mask is specified).

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
        block. A value of 0 specifies a non-land surface cell. Landflag is determined
        from ds when it is None. The default is None.
    finf :float, str or array-like
        The applied infiltration rate of the UZF cell (:math:`LT^{-1}`). When passed as
        string, the array is obtained from ds. The default is "recharge".
    pet : float, str or array-like
        The potential evapotranspiration rate of the UZF cell and specified GWF cell.
        Evapotranspiration is first removed from the unsaturated zone and any remaining
        potential evapotranspiration is applied to the saturated zone. only
        used if SIMULATE_ET. When passed as string, the array is obtained from ds. The
        default is "evaporation".
    extdp : float, optional
        Value that defines the evapotranspiration extinction depth of the UZF cell, in
        meters below the top of the model. Set to 2.0 meter when None. The default is
        None.
    extwc : float, optional
        The evapotranspiration extinction water content of the UZF cell. Only used if
        SIMULATE_ET and UNSAT_ETWC. Set to thtr when None. The default is None.
    ha : float, optional
        The air entry potential (head) of the UZF cell. Only used if SIMULATE_ET and
        UNSAT_ETAE. Set to 0.0 when None. The default is None.
    hroot : float, optional
        The root potential (head) of the UZF cell. Only used if SIMULATE_ET and
        UNSAT_ETAE. Set to 0.0 when None. The default is None.
    rootact : float, optional
        the root activity function of the UZF cell. ROOTACT is the length of roots in
        a given volume of soil divided by that volume. Values range from 0 to about 3
        :math:`cm^{-2}`, depending on the plant community and its stage of development.
        Only used if SIMULATE_ET and UNSAT_ETAE. Set to 0.0 when None. The default is
        None.
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
        The depths at which observations of the water depth in each cell are added. The
        user-specified depth must be greater than or equal to zero and less than the
        thickness of GWF cellid (TOP - BOT).
        The
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
    vks = _get_value_from_ds_datavar(ds, "vk", vks, return_da=True)
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
        landflag[get_first_active_layer_from_idomain(idomain)] = 1

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
    mask = landflag == 1

    # perioddata : [iuzno, finf, pet, extdp, extwc, ha, hroot, rootact, aux]
    finf_name_arr, uzf_unique_dic = _get_unique_series(ds, finf, "finf")
    finf = "rch_name"
    ds[finf] = ds["top"].dims, finf_name_arr
    ds[finf] = ds[finf].expand_dims(dim={"layer": ds.layer})
    if mask is not None:
        mask = (ds[finf] != "") & mask
    else:
        mask = ds[finf] != ""

    pet_name_arr, pet_unique_dic = _get_unique_series(ds, pet, "pet")
    pet = "evt_name"
    ds[pet] = ds["top"].dims, pet_name_arr
    ds[pet] = ds[pet].expand_dims(dim={"layer": ds.layer})
    if mask is not None:
        mask = (ds[pet] != "") & mask
    else:
        mask = ds[pet] != ""

    # combine the time series of finf and pet
    uzf_unique_dic.update(pet_unique_dic)

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

    cellids_land = np.where(mask)

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
        cellid_per_iuzno = list(zip(*cellids))
        cellid_str = [
            str(x).replace("(", "").replace(")", "").replace(", ", "_")
            for x in cellid_per_iuzno
        ]
        thickness = calculate_thickness(ds).data[iuzno >= 0]
        obsdepths = []
        if obs_depth_interval is not None:
            for i in range(nuzfcells):
                depths = np.arange(obs_depth_interval, thickness[i], obs_depth_interval)
                for depth in depths:
                    name = f"wc_{cellid_str[i]}_{depth:0.2f}"
                    obsdepths.append((name, "water-content", i + 1, depth))
        if obs_z is not None:
            botm = ds["botm"].data[iuzno >= 0]
            top = botm + thickness - landflag.data[iuzno >= 0] * surfdep / 2
            for i in range(nuzfcells):
                mask = (obs_z > botm[i]) & (obs_z <= top[i])
                for z in obs_z[mask]:
                    depth = top[i] - z
                    name = f"wc_{cellid_str[i]}_{z:0.2f}"
                    obsdepths.append((name, "water-content", i + 1, depth))

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

    # create timeseries packages
    _add_time_series(uzf, uzf_unique_dic, ds)


def _get_unique_series(ds, var, pname):
    """Get the location and values of unique time series from a variable var in
    ds.

    Parameters
    ----------
    ds : xr.Dataset
        The model Dataset.
    var : str
        The 3d (structured) or 2d (vertext) variable in ds that contains the timeseries.
    pname : str
        Package name, which is used for the name of the time series.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    rch_name : np.ndarray
        The name of the recharge series for each of the cells.
    rch_unique_dic : dict
        The values of each of the time series.
    """
    rch_name_arr = np.empty_like(ds["top"].values, dtype="U13")

    # transient
    if ds.gridtype == "structured":
        if len(ds[var].dims) != 3:
            raise ValueError(
                "expected dataarray with 3 dimensions"
                f"(time, y and x) or (y, x and time), not {ds[var].dims}"
            )
        recharge = ds[var].transpose("y", "x", "time").data
        shape = (ds.dims["y"] * ds.dims["x"], ds.dims["time"])
        rch_2d_arr = recharge.reshape(shape)

    elif ds.gridtype == "vertex":
        # dimension check
        if len(ds[var].dims) != 2:
            raise ValueError(
                "expected dataarray with 2 dimensions"
                f"(time, icell2d) or (icell2d, time), not {ds[var].dims}"
            )
        rch_2d_arr = ds[var].transpose("icell2d", "time").data

    rch_unique_arr = np.unique(rch_2d_arr, axis=0)
    rch_unique_dic = {}
    for i, unique_rch in enumerate(rch_unique_arr):
        mask = (rch_2d_arr == unique_rch).all(axis=1)
        if len(rch_name_arr.shape) > 1:
            mask = mask.reshape(rch_name_arr.shape)
        rch_name_arr[mask] = f"{pname}_{i}"
        rch_unique_dic[f"{pname}_{i}"] = unique_rch

    return rch_name_arr, rch_unique_dic


def _add_time_series(package, rch_unique_dic, ds):
    """Add time series to a package.

    Parameters
    ----------
    rch : mfpackage.MFPackage
        The Flopy package to which to add the timeseries.
    rch_unique_dic : dict
        A dictionary whch contains the time series values.
    ds : xr.Dataset
        The model Dataset. It is used to get the time of the time series.

    Returns
    -------
    None.
    """
    # generate a DataFrame
    df = pd.DataFrame(rch_unique_dic, index=ds.time)
    if df.isna().any(axis=None):
        # make sure there are no NaN's, as otherwise they will be filled by zeros later
        raise (ValueError("There cannot be nan's in the DataFrame"))
    # set the first value for the start-time as well
    df.loc[pd.to_datetime(ds.time.start)] = df.iloc[0]
    # combine the values with the start of each period, and ste the last value to 0.0
    df = df.sort_index().shift(-1).fillna(0.0)

    dataframe_to_flopy_timeseries(
        df, ds=ds, package=package, interpolation_methodrecord="stepwise"
    )
