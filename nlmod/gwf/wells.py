import logging

import flopy as fp
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..dims.grid import gdf_to_grid

logger = logging.getLogger(__name__)


def wel_from_df(
    df,
    gwf,
    x="x",
    y="y",
    top="top",
    botm="botm",
    Q="Q",
    aux=None,
    boundnames=None,
    ds=None,
    auxmultname="multiplier",
    **kwargs,
):
    """Add a Well (WEL) package based on input from a (Geo)DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        A (Geo)DataFrame containing the properties of the wells.
    gwf : flopy ModflowGwf
        Groundwaterflow object to add the wel-package to.
    x : str, optional
        The column in df that contains the x-coordinate of the well. Only used when df
        is a DataFrame. The default is 'x'.
    y : str, optional
        The column in df that contains the y-coordinate of the well. Only used when df
        is a DataFrame. The default is 'y'.
    top : str
        The column in df that contains the z-coordinate of the top of the well screen.
        The defaults is 'top'.
    botm : str
        The column in df that contains the z-coordinate of the bottom of the well
        screen. The defaults is 'botm'.
    Q : str, optional
        The column in df that contains the volumetric well rate. This column can contain
        floats, or strings belonging to timeseries added later. A positive value
        indicates recharge (injection) and a negative value indicates discharge
        (extraction)  The default is "Q".
    aux : str of list of str, optional
        The column(s) in df that contain auxiliary variables. The default is None.
    boundnames : str, optional
        THe column in df thet . The default is None.
    ds : xarray.Dataset
        Dataset with model data. Needed to determine cellid when grid-rotation is used.
        The default is None.
    auxmultname : str, optional
        The name of the auxiliary varibale that contains the multiplication factors to
        distribute the well discharge over different layers. When auxmultname is None,
        this auxiliary variable will not be added, and Q is multiplied by these factors
        directly. auxmultname cannot be None when df[Q] contains strings (the names of
        timeseries). The default is "multiplier".
    **kwargs : dict
        kwargs are passed to flopy.mf6.ModflowGwfwel.

    Returns
    -------
    wel : flopy.mf6.ModflowGwfwel
        wel package.
    """
    if aux is None:
        aux = []
    if not isinstance(aux, list):
        aux = [aux]

    df = _add_cellid(df, ds=ds, gwf=gwf, x=x, y=y)
    multipliers = _get_layer_multiplier_for_wells(df, top, botm, ds=ds, gwf=gwf)

    # collect data
    well_lrcd = []
    for index, irow in df.iterrows():
        wlayers = np.where(multipliers[index] > 0)[0]
        for k in wlayers:
            multiplier = multipliers[index][k]
            q = irow[Q]
            if auxmultname is None:
                q = q * multiplier
            if isinstance(irow["cellid"], int):
                # vertex grid
                cellid = (k, irow["cellid"])
            else:
                # structured grid
                cellid = (k, irow["cellid"][0], irow["cellid"][1])
            wdata = [cellid, q]
            for iaux in aux:
                wdata.append(irow[iaux])
            if auxmultname is not None:
                wdata.append(multiplier)
            if boundnames is not None:
                wdata.append(irow[boundnames])
            well_lrcd.append(wdata)

    if auxmultname is not None:
        aux.append(auxmultname)

    wel_spd = {0: well_lrcd}

    if len(aux) == 0:
        aux = None

    wel = fp.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_spd,
        auxiliary=aux,
        boundnames=boundnames is not None,
        auxmultname=auxmultname,
        **kwargs,
    )

    return wel


def maw_from_df(
    df,
    gwf,
    x="x",
    y="y",
    top="top",
    botm="botm",
    Q="Q",
    rw="rw",
    radius_skin=None,
    hk_skin=None,
    condeqn="THIEM",
    strt=None,
    group=None,
    aux=None,
    boundnames=None,
    ds=None,
    silent=False,
    **kwargs,
):
    """Add a Multi-Aquifer Well (MAW) package based on input from a (Geo)DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        A (Geo)DataFrame containing the properties of the wells.
    gwf : flopy ModflowGwf
        Groundwaterflow object to add the wel-package to.
    x : str, optional
        The column in df that contains the x-coordinate of the well. Only used when df
        is a DataFrame. The default is 'x'.
    y : str, optional
        The column in df that contains the y-coordinate of the well. Only used when df
        is a DataFrame. The default is 'y'.
    top : str
        The column in df that contains the z-coordinate of the top of the well screen.
        The defaults is 'top'.
    botm : str
        The column in df that contains the z-coordinate of the bottom of the well
        screen. The defaults is 'botm'.
    Q : str, optional
        The column in df that contains the volumetric well rate. This column can contain
        floats, or strings belonging to timeseries added later. A positive value
        indicates recharge (injection) and a negative value indicates discharge
        (extraction). If wells are grouped, the values refer to the rates of all the
        wells in the group combined and thus must be the same for all wells in the
        group. The default is "Q".
    rw : str, optional
        The column in df that contains the radius for the multi-aquifer well. The
        default is "rw".
    radius_skin : str, optional
        The column in df that contains the radius of the skin around the well; the
        distance between the center of the well and the outside of the filter pack. Is
        larger than `rw`. Only used if `condeqn` is SKIN, CUMULATIVE, or MEAN. The
        default is None, which means that the skin is not used.
    hk_skin : str, optional
        The column in df that contains the horizontal hydraulic conductivity of the skin
        around the well. Only used if `condeqn` is SKIN, CUMULATIVE, or MEAN. The
        defaultis None, which means that the skin is not used.
    condeqn : str, optional
        String that defines the conductance equation that is used to calculate the
        saturated conductance for the multi-aquifer well. The default is "THIEM".
    strt : float, optional
        The starting head for the multi-aquifer well. The default is None, which uses
        model surface level as the strt value.
    group : str, optional
        The column in df that contains the group name for the wells. If this is not
        None, wells with the same group name are grouped together such that the rate is
        divided over the wells in the group. Note that empty strings are treated as
        unique group names, so wells with an empty string in the group column are
        treated as a separate group. The default is None, which means that each well is
        treated as a separate well.
    aux : str of list of str, optional
        The column(s) in df that contain auxiliary variables. The default is None.
    boundnames : str, optional
        The column in df that contains the boundary names. The default is None.
    ds : xarray.Dataset
        Dataset with model data. Needed to determine cellid when grid-rotation is used.
        The default is None.
    silent : bool, optional
        Hide progressbar when silent is True. Default is False.
    **kwargs : TYPE
        Kwargs are passed onto ModflowGwfmaw.

    Returns
    -------
    wel : flopy.mf6.ModflowGwfmaw
        maw package.
    """
    if aux is None:
        aux = []
    if not isinstance(aux, list):
        aux = [aux]

    df = _add_cellid(df, ds=ds, gwf=gwf, x=x, y=y, silent=silent)
    multipliers = _get_layer_multiplier_for_wells(df, top, botm, ds=ds, gwf=gwf)

    # configure groups
    if df.index.has_duplicates:
        raise ValueError(
            "The index of the DataFrame must be unique. Indexing `multipliers` would "
            "go wrong"
        )

    if group is not None:
        mask = df[group].notna() & df[group].ne("")
        group_by = df[group].where(mask, other=df.index.astype(str))
    else:
        group_by = df.index.astype(str)

    packagedata = []
    connectiondata = []
    perioddata = []

    iw = 0  # grouped well index
    for well_group_name, well_group in tqdm(
        df.groupby(group_by),
        total=group_by.nunique(),
        desc="Adding MAW wells",
        disable=silent,
    ):
        # [wellno, radius, bottom, strt, condeqn, ngwfnodes]
        if strt is None:
            if pd.api.types.is_integer_dtype(well_group["cellid"]):
                # vertex grid
                if ds is None:
                    wstrt = gwf.dis.top[well_group["cellid"]].mean()
                else:
                    wstrt = ds.top.values[well_group["cellid"]].mean()
            else:
                # structured grid
                idx, idy = np.stack(well_group["cellid"]).T
                if ds is None:
                    wstrt = gwf.dis.top[idx, idy].mean()
                else:
                    wstrt = ds.top.values[idx, idy].mean()
        else:
            wstrt = strt

        number_of_well_sections = (multipliers[well_group.index] > 0.0).values.sum()
        group_rw = well_group[rw].mean()
        # The bottom elevation defines the lowest well head that will be simulated when
        # the NEWTON UNDER_RELAXATION option is specified in the GWF model name file.
        # Should actually be ~10m below the elevation of the pump.
        group_botm = well_group[botm].min()
        pakdata = [iw, group_rw, group_botm, wstrt, condeqn, number_of_well_sections]
        for iaux in aux:
            if well_group[iaux].nunique() == 1:
                pakdata.append(well_group[iaux].iloc[0])
            else:
                raise ValueError(
                    f"Auxiliary variable {iaux} cannot be used for grouped wells, "
                    "because the values of wells are different among group "
                    f"{well_group_name}: {well_group[iaux].unique()}."
                )
        if boundnames is not None:
            if well_group[boundnames].nunique() == 1:
                pakdata.append(well_group[boundnames].iloc[0])
            else:
                raise ValueError(
                    f"Boundary name {boundnames} cannot be used for grouped wells, "
                    "because the values of wells are different among group "
                    f"{well_group_name}: {well_group[boundnames].unique()}."
                )
        packagedata.append(pakdata)

        # [wellno, mawsetting]
        # Flow rate for the well_group
        if well_group[Q].nunique() > 1:
            raise ValueError(
                f"Group flow rate {Q} cannot be different among wells in "
                f"{well_group_name}."
            )
        perioddata.append([iw, "RATE", well_group[Q].iloc[0]])

        # [wellno, icon, cellid, scrn_top, scrn_bot, hk_skin, radius_skin]
        iwellpart = 0  # index of well part in the well_group
        for index, irow in well_group.iterrows():
            wlayers = np.where(multipliers[index] > 0)[0]

            for k in wlayers:
                if k == 0:
                    laytop = gwf.modelgrid.top if ds is None else ds.top.values
                else:
                    laytop = (
                        gwf.modelgrid.botm[k - 1]
                        if ds is None
                        else ds.botm.values[k - 1]
                    )
                laybot = gwf.modelgrid.botm[k] if ds is None else ds.botm.values[k]

                if isinstance(irow["cellid"], int):
                    # vertex grid
                    cellid = (k, irow["cellid"])
                    laytop = laytop[irow["cellid"]]
                    laybot = laybot[irow["cellid"]]
                else:
                    # structured grid
                    cellid = (k, irow["cellid"][0], irow["cellid"][1])
                    laytop = laytop[irow["cellid"][0], irow["cellid"][1]]
                    laybot = laybot[irow["cellid"][0], irow["cellid"][1]]

                scrn_top = np.min([irow[top], laytop])
                scrn_bot = np.max([irow[botm], laybot])

                hk_skin_part = 0.0 if hk_skin is None else irow[hk_skin]
                radius_skin_part = 0.0 if radius_skin is None else irow[radius_skin]

                condata = [
                    iw,
                    iwellpart,
                    cellid,
                    scrn_top,
                    scrn_bot,
                    hk_skin_part,
                    radius_skin_part,
                ]
                connectiondata.append(condata)
                iwellpart += 1
        iw += 1

    if len(aux) == 0:
        aux = None
    maw = fp.mf6.ModflowGwfmaw(
        gwf,
        nmawwells=iw,
        auxiliary=aux,
        boundnames=boundnames is not None,
        packagedata=packagedata,
        connectiondata=connectiondata,
        perioddata=perioddata,
        **kwargs,
    )

    return maw


def _add_cellid(df, ds=None, gwf=None, x="x", y="y", silent=False):
    """Intersect a DataFrame of point Data with the model grid, and add cellid-column.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        A (Geo)DataFrame containing the properties of the wells.
    ds : xarray.Dataset
        Dataset with model data. Either supply ds or gwf. The default is None.
    gwf : flopy ModflowGwf
        Groundwaterflow object. Only used when ds is None. The default is None.
    x : str, optional
        The column in df that contains the x-coordinate of the well. Only used when df
        is a DataFrame. The default is 'x'.
    y : str, optional
        The column in df that contains the y-coordinate of the well. Only used when df
        is a DataFrame. The default is 'y'.
    silent : bool, optional
        Hide progressbar when silent is True. Default is False.

    Returns
    -------
    df : gpd.GeoDataFrame
        A GeoDataFrame with a column named cellid that contains the icell2d-number
        (vertex-grid) or (row, column) (structured grid).
    """
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))
    if "cellid" not in df.columns:
        df = gdf_to_grid(df, gwf if ds is None else ds, silent=silent)
    return df


def _get_layer_multiplier_for_wells(df, top, botm, ds=None, gwf=None):
    """Get factors (pandas.DataFrame) for each layer that well screens intersects with.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the properties of the wells.
    top : str
        The column in df that contains the z-coordinate of the top of the well screen.
    botm : str
        The column in df that contains the z-coordinate of the bottom of the well
        screen.
    ds : xarray.Dataset
        Dataset with model data. Either supply ds or gwf. The default is None.
    gwf : flopy ModflowGwf
        Groundwaterflow object. Only used when ds is None. The default is None.

    Returns
    -------
    multipliers : pd.DataFrame
        A DataFrame containg the multiplication factors, with the layers as the index
        and the name of the well screens (the index of df) as columns.
    """
    # get required data either from  gwf or ds
    if ds is not None:
        ml_top = ds["top"].data
        ml_bot = ds["botm"].data
        kh = ds["kh"].data
        layer = ds.layer
    elif gwf is not None:
        ml_top = gwf.dis.top.array
        ml_bot = gwf.dis.botm.array
        kh = gwf.npf.k.array
        layer = range(gwf.dis.nlay.array)
    else:
        raise (TypeError("Either supply ds or gwf to determine layer multiplyer"))

    multipliers = {}
    for index, irow in df.iterrows():
        multipliers[index] = _get_layer_multiplier_for_well(
            irow["cellid"], irow[top], irow[botm], ml_top, ml_bot, kh
        )

        if (multipliers[index] == 0).all():
            logger.warning(f"No layers found for well {index}")
    multipliers = pd.DataFrame(multipliers, index=layer, columns=df.index)
    return multipliers


def _get_layer_multiplier_for_well(cid, well_top, well_bot, ml_top, ml_bot, ml_kh):
    """Get a factor (numpy array) for each layer that a well screen intersects with.

    Parameters
    ----------
    cid : int or tuple of 2 ints
        THe cellid of the well (either icell2d or (row, column).
    well_top : float
        The z-coordinate of the top of the well screen.
    well_bot : float
        The z-coordinate of the top of the well screen.
    ml_top : numpy array
        The top of the upper layer of the model (1d or 2d)
    ml_bot : numpy array
        The bottom of all cells of the model (2d or 3d)
    ml_kh : numpy array
        The horizontal conductivity of all cells of the model (2d or 3d).

    Returns
    -------
    multiplier : numpy array
        An array with a factor (between 0 and 1) for each of the model layers.
    """
    # keep the tops and botms of the cell where the well is in
    ml_top_cid = ml_top[cid].copy()
    if isinstance(cid, int):
        ml_bot_cid = ml_bot[:, cid].copy()
        ml_kh_cid = ml_kh[:, cid].copy()
    else:
        ml_bot_cid = ml_bot[:, cid[0], cid[1]].copy()
        ml_kh_cid = ml_kh[:, cid[0], cid[1]].copy()
    ml_top_cid = np.array([ml_top_cid] + list(ml_bot_cid[:-1]))

    # only keep the part of layers along the well filter
    ml_top_cid[ml_top_cid > well_top] = well_top
    ml_top_cid[ml_top_cid < well_bot] = well_bot
    ml_bot_cid[ml_bot_cid > well_top] = well_top
    ml_bot_cid[ml_bot_cid < well_bot] = well_bot

    # calculate remaining kd along the well filter
    kd = ml_kh_cid * (ml_top_cid - ml_bot_cid)
    mask = kd < 0
    if np.any(mask):
        logger.warning("There are negative thicknesses at cellid {cid}")
        kd[mask] = 0
    if (kd == 0).all():
        # the well does not cross any of the layers. Just return an array of zeros.
        multiplier = kd
    else:
        # divide by the total kd to get a factor
        multiplier = kd / kd.sum()
    return multiplier
