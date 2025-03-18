import logging
import numbers
import warnings

import flopy
import xarray as xr

from ..dims import grid
from ..dims.grid import get_delc, get_delr
from ..dims.layers import get_idomain
from ..dims.shared import get_area
from ..sim import ims, sim, tdis
from ..util import _get_value_from_ds_attr, _get_value_from_ds_datavar
from . import recharge

logger = logging.getLogger(__name__)


def gwf(ds, sim, under_relaxation=False, **kwargs):
    """Create groundwater flow model from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start,
        perlen, nstp, tsmult
    sim : flopy MFSimulation
        simulation object.

    Returns
    -------
    gwf : flopy ModflowGwf
        groundwaterflow object.
    """
    # start creating model
    logger.info("creating mf6 GWF")

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{ds.model_name}.nam"

    if "newtonoptions" in kwargs:
        newtonoptions = kwargs.pop("newtonoptions")
    elif under_relaxation:
        newtonoptions = "under_relaxation"
    else:
        newtonoptions = None

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=ds.model_name,
        model_nam_file=model_nam_file,
        newtonoptions=newtonoptions,
        **kwargs,
    )

    return gwf


def dis(ds, gwf, length_units="METERS", pname="dis", **kwargs):
    """Create discretisation package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    dis : flopy ModflowGwfdis
        discretisation package.
    """
    return _dis(ds, gwf, length_units, pname, **kwargs)


def _dis(ds, model, length_units="METERS", pname="dis", **kwargs):
    """Create discretisation package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    model : flopy ModflowGwf or flopy ModflowGwt
        groundwaterflow or groundwater transport object
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    dis : flopy ModflowGwfdis or flopy ModflowGwtdis
        discretisation package.
    """
    logger.info("creating mf6 DIS")

    if ds.gridtype == "vertex":
        return disv(ds, model, length_units=length_units)

    # check attributes
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        xorigin = ds.attrs["xorigin"]
        yorigin = ds.attrs["yorigin"]
        angrot = ds.attrs["angrot"]
    else:
        xorigin = ds.extent[0]
        yorigin = ds.extent[2]
        angrot = 0.0

    idomain = get_idomain(ds).data
    if model.model_type == "gwf6":
        dis = flopy.mf6.ModflowGwfdis(
            model,
            pname=pname,
            length_units=length_units,
            xorigin=xorigin,
            yorigin=yorigin,
            angrot=angrot,
            nlay=ds.sizes["layer"],
            nrow=ds.sizes["y"],
            ncol=ds.sizes["x"],
            delr=get_delr(ds),
            delc=get_delc(ds),
            top=ds["top"].data,
            botm=ds["botm"].data,
            idomain=idomain,
            filename=f"{ds.model_name}.dis",
            **kwargs,
        )
    elif model.model_type == "gwt6":
        dis = flopy.mf6.ModflowGwtdis(
            model,
            pname=pname,
            length_units=length_units,
            xorigin=xorigin,
            yorigin=yorigin,
            angrot=angrot,
            nlay=ds.sizes["layer"],
            nrow=ds.sizes["y"],
            ncol=ds.sizes["x"],
            delr=get_delr(ds),
            delc=get_delc(ds),
            top=ds["top"].data,
            botm=ds["botm"].data,
            idomain=idomain,
            filename=f"{ds.model_name}_gwt.dis",
            **kwargs,
        )
    else:
        raise ValueError("Unknown model type.")

    return dis


def disv(ds, gwf, length_units="METERS", pname="disv", **kwargs):
    """Create discretisation vertices package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    model : flopy ModflowGwf
        groundwater flow object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    disv : flopy ModflowGwfdisv
        disv package
    """
    return _disv(ds, gwf, length_units, pname, **kwargs)


def _disv(ds, model, length_units="METERS", pname="disv", **kwargs):
    """Create discretisation vertices package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    model : flopy ModflowGwf or flopy ModflowGwt
        groundwater flow or groundwater transport object.
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    disv : flopy ModflowGwfdisv or flopy ModflowGwtdisv
        disv package
    """
    logger.info("creating mf6 DISV")

    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        xorigin = ds.attrs["xorigin"]
        yorigin = ds.attrs["yorigin"]
        angrot = ds.attrs["angrot"]
    else:
        xorigin = 0.0
        yorigin = 0.0
        angrot = 0.0

    vertices = grid.get_vertices_from_ds(ds)
    cell2d = grid.get_cell2d_from_ds(ds)
    idomain = get_idomain(ds).data
    if model.model_type == "gwf6":
        disv = flopy.mf6.ModflowGwfdisv(
            model,
            idomain=idomain,
            xorigin=xorigin,
            yorigin=yorigin,
            length_units=length_units,
            angrot=angrot,
            nlay=len(ds.layer),
            ncpl=len(ds.icell2d),
            nvert=len(ds.iv),
            top=ds["top"].data,
            botm=ds["botm"].data,
            vertices=vertices,
            cell2d=cell2d,
            pname=pname,
            **kwargs,
        )
    elif model.model_type == "gwt6":
        disv = flopy.mf6.ModflowGwtdisv(
            model,
            idomain=idomain,
            xorigin=xorigin,
            yorigin=yorigin,
            length_units=length_units,
            angrot=angrot,
            nlay=len(ds.layer),
            ncpl=len(ds.icell2d),
            nvert=len(ds.iv),
            top=ds["top"].data,
            botm=ds["botm"].data,
            vertices=vertices,
            cell2d=cell2d,
            pname=pname,
            filename=f"{ds.model_name}_gwt.disv",
            **kwargs,
        )
    else:
        raise ValueError("Unknown model type.")

    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        model.modelgrid.set_coord_info(xoff=xorigin, yoff=yorigin, angrot=angrot)

    return disv


def npf(
    ds, gwf, k="kh", k33="kv", icelltype=0, save_flows=False, pname="npf", **kwargs
):
    """Create node property flow package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    icelltype : int or str, optional
        celltype, if int the icelltype for all layer, if str the icelltype from
        the model ds is used. The default is 0.
    k : str or array-like
        horizontal hydraulic conductivity, when passed as string, the array
        is obtained from ds. By default assumes data is stored as "kh".
    k33 : str or array-like
        vertical hydraulic conductivity, when passed as string, the array
        is obtained from ds. By default assumes data is stored as "kv".
    save_flows : bool, optional
        value is passed to flopy.mf6.ModflowGwfnpf() to determine if cell by
        cell flows should be saved to the cbb file. Default is False
    pname : str, optional
        package name

    Raises
    ------
    NotImplementedError
        only icelltype 0 is implemented.

    Returns
    -------
    npf : flopy ModflowGwfnpf
        npf package.
    """
    logger.info("creating mf6 NPF")

    if isinstance(icelltype, str):
        icelltype = ds[icelltype]

    k = _get_value_from_ds_datavar(ds, "k", k)
    k33 = _get_value_from_ds_datavar(ds, "k33", k33)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        pname=pname,
        icelltype=icelltype,
        k=k,
        k33=k33,
        save_flows=save_flows,
        **kwargs,
    )

    return npf


def ghb(
    ds,
    gwf,
    bhead=None,
    cond=None,
    pname="ghb",
    auxiliary=None,
    layer=None,
    **kwargs,
):
    """Create general head boundary from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    bhead : str or xarray.DataArray, optional
        ghb boundary head, either as string pointing to data array in ds or as data
        array. By default None, which assumes data array is stored under "ghb_bhead".
    cond : str or xarray.DataArray, optional
        ghb conductance, either as string pointing to data array in ds or as data array.
        By default None, which assumes data array is stored under "ghb_cond".
    pname : str, optional
        package name
    auxiliary : str or list of str
        name(s) of data arrays to include as auxiliary data to reclist
    layer : int or None
        The layer in which the boundary is added. It is added to the first active layer
        when layer is None. The default is None.

    Raises
    ------
    ValueError
        raised if gridtype is not structured or vertex.

    Returns
    -------
    ghb : flopy ModflowGwfghb
        ghb package
    """
    logger.info("creating mf6 GHB")

    mask_arr = _get_value_from_ds_datavar(ds, "ghb_cond", cond, return_da=True)
    mask = mask_arr > 0

    first_active_layer = layer is None
    ghb_rec = grid.da_to_reclist(
        ds,
        mask,
        col1=bhead,
        col2=cond,
        layer=layer,
        aux=auxiliary,
        first_active_layer=first_active_layer,
        only_active_cells=False,
    )

    if len(ghb_rec) > 0:
        ghb = flopy.mf6.ModflowGwfghb(
            gwf,
            auxiliary="CONCENTRATION" if auxiliary is not None else None,
            maxbound=len(ghb_rec),
            stress_period_data=ghb_rec,
            save_flows=True,
            pname=pname,
            **kwargs,
        )
        if (auxiliary is not None) and (ds.transport == 1):
            logger.info("-> adding GHB to SSM sources list")
            ssm_sources = list(ds.attrs["ssm_sources"])
            if ghb.package_name not in ssm_sources:
                ssm_sources += [ghb.package_name]
                ds.attrs["ssm_sources"] = ssm_sources
        return ghb

    else:
        logger.warning("no ghb pkg added")
        return None


def drn(
    ds,
    gwf,
    elev="drn_elev",
    cond="drn_cond",
    pname="drn",
    layer=None,
    **kwargs,
):
    """Create drain from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    elev : str or xarray.DataArray, optional
        drain elevation, either as string pointing to data
        array in ds or as data array. By default assumes
        data array is stored under "drn_elev".
    cond : str or xarray.DataArray, optional
        drain conductance, either as string pointing to data
        array in ds or as data array. By default assumes
        data array is stored under "drn_cond".
    da_name : str, deprecated
        this is deprecated, name of the drn files in the model dataset
    pname : str, optional
        package name
    layer : int or None
        The layer in which the boundary is added. It is added to the first active layer
        when layer is None. The default is None.

    Returns
    -------
    drn : flopy ModflowGwfdrn
        drn package
    """
    logger.info("creating mf6 DRN")

    mask_arr = _get_value_from_ds_datavar(ds, "cond", cond, return_da=True)
    mask = mask_arr > 0

    first_active_layer = layer is None
    drn_rec = grid.da_to_reclist(
        ds,
        mask=mask,
        col1=elev,
        col2=cond,
        layer=layer,
        first_active_layer=first_active_layer,
        only_active_cells=False,
    )

    if len(drn_rec) > 0:
        drn = flopy.mf6.ModflowGwfdrn(
            gwf,
            maxbound=len(drn_rec),
            stress_period_data=drn_rec,
            save_flows=True,
            pname=pname,
            **kwargs,
        )
        return drn

    else:
        logger.warning("no drn pkg added")

        return None


def riv(
    ds,
    gwf,
    stage="riv_stage",
    cond="riv_cond",
    rbot="riv_rbot",
    pname="riv",
    auxiliary=None,
    layer=None,
    **kwargs,
):
    """Create river package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    stage : str or xarray.DataArray, optional
        river stage, either as string pointing to data
        array in ds or as data array. By default assumes
        data array is stored under "riv_stage".
    cond : str or xarray.DataArray, optional
        river conductance, either as string pointing to data
        array in ds or as data array. By default assumes
        data array is stored under "riv_cond".
    rbot : str or xarray.DataArray, optional
        river bottom elevation, either as string pointing to data
        array in ds or as data array. By default assumes
        data array is stored under "riv_rbot".
    pname : str, optional
        package name
    auxiliary : str or list of str
        name(s) of data arrays to include as auxiliary data to reclist
    layer : int or None
        The layer in which the boundary is added. It is added to the first active layer
        when layer is None. The default is None.

    Returns
    -------
    riv : flopy ModflowGwfriv
        riv package
    """
    logger.info("creating mf6 RIV")

    mask_arr = _get_value_from_ds_datavar(ds, "cond", cond, return_da=True)
    mask = mask_arr > 0

    first_active_layer = layer is None
    riv_rec = grid.da_to_reclist(
        ds,
        mask=mask,
        col1=stage,
        col2=cond,
        col3=rbot,
        layer=layer,
        aux=auxiliary,
        first_active_layer=first_active_layer,
        only_active_cells=False,
    )

    if len(riv_rec) > 0:
        riv = flopy.mf6.ModflowGwfriv(
            gwf,
            maxbound=len(riv_rec),
            stress_period_data=riv_rec,
            auxiliary="CONCENTRATION" if auxiliary is not None else None,
            save_flows=True,
            pname=pname,
            **kwargs,
        )
        if (auxiliary is not None) and (ds.transport == 1):
            logger.info("-> adding RIV to SSM sources list")
            ssm_sources = list(ds.attrs["ssm_sources"])
            if riv.package_name not in ssm_sources:
                ssm_sources += [riv.package_name]
                ds.attrs["ssm_sources"] = ssm_sources
        return riv
    else:
        logger.warning("no riv pkg added")
        return None


def chd(
    ds,
    gwf,
    mask="chd_mask",
    head="chd_head",
    pname="chd",
    auxiliary=None,
    layer=0,
    **kwargs,
):
    """Create constant head package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    mask : str, optional
        name of data variable in ds that is 1 for cells with a constant
        head and zero for all other cells. The default is 'chd_mask'.
    head : str, optional
        name of data variable in ds that is used as the head in the chd
        cells. By default, assumes head data is stored as 'chd_head'.
    pname : str, optional
        package name
    auxiliary : str or list of str
        name(s) of data arrays to include as auxiliary data to reclist
    layer : int or None
        The layer in which the boundary is added. It is added to the first active layer
        when layer is None. The default is 0.
    chd : str, optional
        deprecated, the new argument is 'mask'

    Returns
    -------
    chd : flopy ModflowGwfchd
        chd package
    """
    logger.info("creating mf6 CHD")

    if "chd" in kwargs:
        warnings.warn(
            "the 'chd' kwarg has been renamed to 'mask'!",
            DeprecationWarning,
        )
        mask = kwargs.pop("chd")

    maskarr = _get_value_from_ds_datavar(ds, "mask", mask, return_da=True)
    mask = maskarr > 0

    # get the stress_period_data
    first_active_layer = layer is None
    chd_rec = grid.da_to_reclist(
        ds,
        mask,
        col1=head,
        layer=layer,
        aux=auxiliary,
        first_active_layer=first_active_layer,
    )

    if len(chd_rec) > 0:
        chd = flopy.mf6.ModflowGwfchd(
            gwf,
            auxiliary="CONCENTRATION" if auxiliary is not None else None,
            pname=pname,
            maxbound=len(chd_rec),
            stress_period_data=chd_rec,
            save_flows=True,
            **kwargs,
        )
        if (auxiliary is not None) and (ds.transport == 1):
            logger.info("-> adding CHD to SSM sources list")
            ssm_sources = list(ds.attrs["ssm_sources"])
            if chd.package_name not in ssm_sources:
                ssm_sources += [chd.package_name]
                ds.attrs["ssm_sources"] = ssm_sources
        return chd
    else:
        logger.warning("no chd pkg added")
        return None


def ic(ds, gwf, starting_head="starting_head", pname="ic", **kwargs):
    """Create initial condictions package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    starting_head : str, float or int, optional
        if type is int or float this is the starting head for all cells
        If the type is str the data variable from ds is used as starting
        head. The default is 'starting_head'.
    pname : str, optional
        package name

    Returns
    -------
    ic : flopy ModflowGwfic
        ic package
    """
    logger.info("creating mf6 IC")

    if isinstance(starting_head, numbers.Number):
        logger.info("adding 'starting_head' data array to ds")
        ds["starting_head"] = starting_head * xr.ones_like(ds["botm"])
        ds["starting_head"].attrs["units"] = "mNAP"
        starting_head = "starting_head"

    strt = _get_value_from_ds_datavar(ds, "starting_head", starting_head)
    ic = flopy.mf6.ModflowGwfic(gwf, pname=pname, strt=strt, **kwargs)

    return ic


def sto(
    ds,
    gwf,
    sy="sy",
    ss="ss",
    iconvert=1,
    save_flows=False,
    pname="sto",
    **kwargs,
):
    """Create storage package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    sy : str or float, optional
        specific yield. The default is "sy", or 0.2 if "sy" is not in ds.
    ss : str or float, optional
        specific storage. The default is "ss", or 0.000001 if "ss" is not in ds.
    iconvert : int, optional
        See description in ModflowGwfsto. The default is 1 (differs from FloPY).
    save_flows : bool, optional
        value is passed to flopy.mf6.ModflowGwfsto() to determine if flows
        should be saved to the cbb file. Default is False
    pname : str, optional
        package name

    Returns
    -------
    sto : flopy ModflowGwfsto
        sto package
    """
    logger.info("creating mf6 STO")

    if "time" not in ds or ds["steady"].all():
        logger.warning("Model is steady-state, no STO package created.")
        return None
    else:
        sts_spd = {iper: bool(b) for iper, b in enumerate(ds["steady"])}
        trn_spd = {iper: not bool(b) for iper, b in enumerate(ds["steady"])}

        sy = _get_value_from_ds_datavar(ds, "sy", sy, default=0.2)
        ss = _get_value_from_ds_datavar(ds, "ss", ss, default=1e-5)

        sto = flopy.mf6.ModflowGwfsto(
            gwf,
            pname=pname,
            save_flows=save_flows,
            iconvert=iconvert,
            ss=ss,
            sy=sy,
            steady_state=sts_spd,
            transient=trn_spd,
            **kwargs,
        )
        return sto


def surface_drain_from_ds(ds, gwf, resistance, elev="ahn", pname="drn", **kwargs):
    """Create surface level drain (maaivelddrainage in Dutch) from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    resistance : int or float
        resistance of the surface drain. This value is used to
        calculate drain conductance by scaling with cell area.
    elev : str or xarray.DataArray
        name pointing to the data array containing surface drain elevation
        data, or pass the data array directly. By default assumes
        the elevation data is stored under "ahn".
    pname : str, optional
        package name

    Returns
    -------
    drn : flopy ModflowGwfdrn
        drn package
    """
    ds.attrs["surface_drn_resistance"] = resistance

    maskarr = _get_value_from_ds_datavar(ds, "elev", elev, return_da=True)
    mask = maskarr.notnull()

    area = ds["area"] if "area" in ds else get_area(ds)

    drn_rec = grid.da_to_reclist(
        ds,
        mask,
        col1=elev,
        col2=area / ds.surface_drn_resistance,
        first_active_layer=True,
        only_active_cells=False,
    )

    drn = flopy.mf6.ModflowGwfdrn(
        gwf,
        pname=pname,
        maxbound=len(drn_rec),
        stress_period_data={0: drn_rec},
        save_flows=True,
        **kwargs,
    )

    return drn


def rch(ds, gwf, pname="rch", **kwargs):
    """Create recharge package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    pname : str, optional
        package name

    Returns
    -------
    rch : flopy ModflowGwfrch
        rch package
    """
    logger.info("creating mf6 RCH")
    # create recharge package
    rch = recharge.ds_to_rch(gwf, ds, pname=pname, **kwargs)

    return rch


def evt(ds, gwf, pname="evt", **kwargs):
    """Create evapotranspiration package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    pname : str, optional
        package name

    Returns
    -------
    evt : flopy ModflowGwfevt
        rch package
    """
    logger.info("creating mf6 EVT")

    # create recharge package
    evt = recharge.ds_to_evt(gwf, ds, pname=pname, **kwargs)

    return evt


def uzf(ds, gwf, pname="uzf", **kwargs):
    """Create unsaturated zone flow package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    pname : str, optional
        package name

    Returns
    -------
    uzf : flopy ModflowGwfuzf
        uzf package
    """
    logger.info("creating mf6 UZF")

    # create uzf package
    uzf = recharge.ds_to_uzf(gwf, ds, pname=pname, **kwargs)

    return uzf


def _set_record(out, budget, output="head"):
    record = []
    if isinstance(out, bool):
        if out:
            out = "LAST"
        else:
            out = None
    if out is not None:
        record.append((output.upper(), out))
    if isinstance(budget, bool):
        if budget:
            budget = "LAST"
        else:
            budget = None
    if budget is not None:
        record.append(("BUDGET", budget))
    return record


def buy(ds, gwf, pname="buy", **kwargs):
    """Create buoyancy package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    pname : str, optional
        package name, by default "buy"

    Returns
    -------
    buy : flopy ModflowGwfbuy
        buy package

    Raises
    ------
    ValueError
        if transport is not
    """
    if not ds.transport:
        logger.error("BUY package requires a groundwater transport model")
        raise ValueError(
            "BUY package requires a groundwater transport model. "
            "Set 'transport' to True in model dataset."
        )

    drhodc = _get_value_from_ds_attr(
        ds, "drhodc", attr="drhodc", value=kwargs.pop("drhodc", None)
    )
    crhoref = _get_value_from_ds_attr(
        ds, "crhoref", attr="crhoref", value=kwargs.pop("crhoref", None)
    )
    denseref = _get_value_from_ds_attr(
        ds, "denseref", attr="denseref", value=kwargs.pop("denseref", None)
    )

    pdata = [(0, drhodc, crhoref, f"{ds.model_name}_gwt", "CONCENTRATION")]

    buy = flopy.mf6.ModflowGwfbuy(
        gwf,
        denseref=denseref,
        nrhospecies=len(pdata),
        packagedata=pdata,
        pname=pname,
        **kwargs,
    )
    return buy


def oc(
    ds,
    gwf,
    save_head=True,
    save_budget=True,
    print_head=False,
    print_budget=False,
    pname="oc",
    **kwargs,
):
    """Create output control package from model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    save_head : bool or str
        Saves the head to the output-file. If save_head is a string, it needs to be
        "all", "first" or "last". If save_head is True, it is set to "last". The default
        is True.
    save_budget : bool or str
        Saves the budgets to the output-file. If save_budget is a string, it needs to be
        "all", "first" or "last". If save_budget is True, it is set to "last". The
        default is True.
    print_head : bool or str
        Prints the head to the list-file. If print_head is a string, it needs to be
        "all", "first" or "last". If print_head is True, it is set to "last". The default
        is False.
    print_budget : bool or str
        Prints the budgets to the list-file. If print_budget is a string, it needs to be
        "all", "first" or "last". If print_budget is True, it is set to "last". The
        default is False.
    pname : str, optional
        package name

    Returns
    -------
    oc : flopy ModflowGwfoc
        oc package
    """
    logger.info("creating mf6 OC")

    # Create the output control package
    headfile = f"{ds.model_name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{ds.model_name}.cbc"
    budget_filerecord = [budgetfile]
    saverecord = _set_record(save_head, save_budget, output="head")
    printrecord = _set_record(print_head, print_budget, output="head")

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname=pname,
        saverecord=saverecord,
        printrecord=printrecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        **kwargs,
    )

    return oc


def ds_to_gwf(ds, complexity="SIMPLE", icelltype=0, under_relaxation=False):
    """Generate Simulation and GWF model from model DataSet.

    Builds the following packages:
    - sim
    - tdis
    - ims
    - gwf
      - dis
      - npf
      - ic
      - oc
      - rch if "recharge" is present in DataSet
      - evt if "evaporation" is present in DataSet

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset

    Returns
    -------
    flopy.mf6.ModflowGwf
        MODFLOW6 GroundwaterFlow model object.
    """
    # create simulation
    mf_sim = sim(ds)

    # create time discretisation
    tdis(ds, mf_sim)

    # create ims
    ims(mf_sim, complexity=complexity)

    # create groundwater flow model
    mf_gwf = gwf(ds, mf_sim, under_relaxation=under_relaxation)

    # Create discretization
    if ds.gridtype == "structured":
        dis(ds, mf_gwf)
    elif ds.gridtype == "vertex":
        disv(ds, mf_gwf)
    else:
        raise TypeError("gridtype not recognized.")

    # create node property flow
    npf(ds, mf_gwf, icelltype=icelltype)

    # Create the initial conditions package
    starting_head = "starting_head"
    if starting_head not in ds:
        starting_head = 0.0
    ic(ds, mf_gwf, starting_head=starting_head)

    # Create the output control package
    oc(ds, mf_gwf)

    if "recharge" in ds:
        rch(ds, mf_gwf)

    if "evaporation" in ds:
        evt(ds, mf_gwf)

    return mf_gwf
