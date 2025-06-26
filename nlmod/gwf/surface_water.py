import logging
import warnings

import flopy
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from tqdm import tqdm

from ..cache import cache_pickle
from ..dims.grid import (
    gdf_to_grid,
    get_delc,
    get_delr,
    get_extent_polygon,
)
from ..dims.layers import get_idomain
from ..read import bgt, waterboard
from ..util import extent_to_polygon, gdf_intersection_join, zonal_statistics

logger = logging.getLogger(__name__)


@cache_pickle
def aggregate(gdf, method, ds=None):
    """Aggregate surface water features.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing surfacewater polygons per grid cell.
        Must contain columns 'stage' (waterlevel),
        'c0' (bottom resistance), and 'botm' (bottom elevation)
    method : str, optional
        "area_weighted" for area-weighted params,
        "max_area" for max area params
        "de_lange" for De Lange formula for conductance
    ds : xr.Dataset, optional
        DataSet containing model layer information (only required for
        method='de_lange')

    Returns
    -------
    celldata : pd.DataFrame
        DataFrame with aggregated surface water parameters per grid cell
    """
    required_cols = {"stage", "c0", "botm"}
    missing_cols = required_cols.difference(gdf.columns)
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Post process intersection result
    gr = gdf.groupby(by="cellid")
    celldata = pd.DataFrame(index=gr.groups.keys())

    for cid, group in tqdm(gr, desc="Aggregate surface water data"):
        stage, cond, rbot = get_surfacewater_params(group, method, cid=cid, ds=ds)

        celldata.loc[cid, "stage"] = stage
        celldata.loc[cid, "cond"] = cond
        celldata.loc[cid, "rbot"] = rbot
        celldata.loc[cid, "area"] = group.area.sum()

    return celldata


def get_surfacewater_params(group, method, cid=None, ds=None, delange_params=None):
    if method == "area_weighted":
        # stage
        stage = agg_area_weighted(group, "stage")
        # cond
        c0 = agg_area_weighted(group, "c0")
        cond = group.area.sum() / c0
        # rbot
        rbot = group["botm"].min()

    elif method == "max_area":
        # stage
        stage = agg_max_area(group, "stage")
        # cond
        c0 = agg_max_area(group, "c0")
        cond = group.area.sum() / c0
        # rbot
        rbot = group["botm"].min()

    elif method == "de_lange":
        if ds is None:
            raise ValueError("Please supply model dataset (ds) when method=='de_lange'")
        # get additional requisite parameters
        if delange_params is None:
            delange_params = {}

        # defaults
        c1 = delange_params.pop("c1", 0.0)
        N = delange_params.pop("N", 1e-3)

        # stage
        stage = agg_area_weighted(group, "stage")

        # cond
        c0 = agg_area_weighted(group, "c0")
        _, _, cond = agg_de_lange(group, cid, ds, c1=c1, c0=c0, N=N)

        # rbot
        rbot = group["botm"].min()

    else:
        raise ValueError(f"Method '{method}' not recognized!")

    return stage, cond, rbot


def agg_max_area(gdf, col):
    return gdf.loc[gdf.area.idxmax(), col]


def agg_area_weighted(gdf, col):
    nanmask = gdf[col].isna()
    aw = (gdf.area * gdf[col]).sum(skipna=True) / gdf.loc[~nanmask].area.sum()
    return aw


def agg_de_lange(group, cid, ds, c1=0.0, c0=1.0, N=1e-3, crad_positive=True):
    (A, laytop, laybot, kh, kv) = get_subsurface_params_by_cellid(ds, cid)

    rbot = group["botm"].min()

    # select active layers
    thickness = -np.diff(np.hstack((laytop, laybot)))
    active = thickness > 0
    laybot = laybot[active]
    kh = kh[active]
    kv = kv[active]
    thickness = thickness[active]

    # layer thickn.
    if np.isnan(rbot):
        raise ValueError(f"rbot is NaN in cell {cid}")
    H0 = laytop - laybot[laybot < rbot][0]
    ilay = 0
    rlay = np.where(laybot < rbot)[0][0]

    # equivalent hydraulic conductivities
    H = thickness[ilay : rlay + 1]
    kv = kv[ilay : rlay + 1]
    kh = kh[ilay : rlay + 1]
    kveq = np.sum(H) / np.sum(H / kv)
    kheq = np.sum(H * kh) / np.sum(H)

    # length
    len_est = estimate_polygon_length(group)
    li = len_est.sum()
    # correction if group contains multiple shapes
    # but covers whole cell
    if group.area.sum() == A:
        delr = get_delr(ds)
        assert len(np.unique(delr)) == 1, "Variable grid size is not yet supported"
        delc = get_delc(ds)
        assert len(np.unique(delc)) == 1, "Variable grid size is not yet supported"
        li = A / np.max([delr[0], delc[0]])

    # width
    B = group.area.sum(skipna=True) / li

    # mean water level
    p = group.loc[group.area.idxmax(), "stage"]  # waterlevel

    # calculate params
    pstar, cstar, cond = de_lange_eqns(
        A, H0, kveq, kheq, c1, li, B, c0, p, N, crad_positive=crad_positive
    )

    return pstar, cstar, cond


def get_subsurface_params_by_cellid(ds, cid):
    r, c = cid
    A = ds.area.isel(x=c, y=r).data
    laytop = ds["top"].isel(x=c, y=r).data
    laybot = ds["botm"].isel(x=c, y=r).data
    kv = ds["kv"].isel(x=c, y=r).data
    kh = ds["kh"].isel(x=c, y=r).data
    return A, laytop, laybot, kh, kv


def de_lange_eqns(A, H0, kv, kh, c1, li, Bin, c0, p, N, crad_positive=True):
    """Calculates the conductance according to De Lange.

    Parameters
    ----------
    A : float
        celoppervlak (m2)
    H0 : float
        doorstroomde dikte (m)
    kv : float
        verticale doorlotendheid (m/d)
    kh : float
        horizontale doorlatendheid (m/d)
    c1 : float
        deklaagweerstand (d)
    li : float
        lengte van de waterlopen (m)
    Bin : float
        bodembreedte (m)
    c0 : float
        slootbodemweerstand (d)
    p : float
        water peil
    N : float
        grondwateraanvulling
    crad_positive: bool, optional
        whether to allow negative crad values. If True, crad will be set to 0
        if it is negative.

    Returns
    -------
    float
        Conductance (m2/d)
    """
    if li > 1e-3 and Bin > 1e-3 and A > 1e-3:
        Bcor = max(Bin, 1e-3)  # has no effect
        L = A / li - Bcor
        y = c1 + H0 / kv

        labdaL = np.sqrt(y * kh * H0)
        if L > 1e-3:
            xL = L / (2 * labdaL)
            FL = xL * coth(xL)
        else:
            FL = 0.0

        labdaB = np.sqrt(y * kh * H0 * c0 / (y + c0))
        xB = Bcor / (2 * labdaB)
        FB = xB * coth(xB)

        CL = (c0 + y) * FL + (c0 * L / Bcor) * FB
        if CL == 0.0:
            CB = 1.0
        else:
            CB = (c1 + c0 + H0 / kv) / (CL - c0 * L / Bcor) * CL

        # volgens Kees Maas mag deze ook < 0 zijn...
        # er miste ook een correctie in de log voor anisotropie
        # Crad = max(0., L / (np.pi * np.sqrt(kv * kh))
        #            * np.log(4 * H0 / (np.pi * Bcor)))
        crad = radial_resistance(L, Bcor, H0, kh, kv)
        if crad_positive:
            crad = max([0.0, crad])

        # Conductance
        pSl = Bcor * li / A
        if pSl >= 1.0 - 1e-10:
            Wp = 1 / (pSl / CB) + crad - c1
        else:
            Wp = 1 / ((1.0 - pSl) / CL + pSl / CB) + crad - c1
        cond = A / Wp

        # cstar, pstar
        cLstar = CL + crad

        pstar = p + N * (cLstar - y) * (y + c0) * L / (Bcor * cLstar + L * y)
        cstar = cLstar * (c0 + y) * (Bcor + L) / (Bcor * cLstar + L * y)

        return pstar, cstar, cond
    else:
        return 0.0, 0.0, 0.0


def radial_resistance(L, B, H, kh, kv):
    return (
        L
        / (np.pi * np.sqrt(kh * kv))
        * np.log(4 * H * np.sqrt(kh) / (np.pi * B * np.sqrt(kv)))
    )


def coth(x):
    return 1.0 / np.tanh(x)


def estimate_polygon_length(gdf):
    # estimate length from polygon (for shapefactor > 4)
    shape_factor = gdf.length / np.sqrt(gdf.area)

    len_est1 = (gdf.length - np.sqrt(gdf.length**2 - 16 * gdf.area)) / 4
    len_est2 = (gdf.length + np.sqrt(gdf.length**2 - 16 * gdf.area)) / 4
    len_est = pd.concat([len_est1, len_est2], axis=1).max(axis=1)

    # estimate length from minimum rotated rectangle (for shapefactor < 4)
    min_rect = gdf.geometry.apply(lambda g: g.minimum_rotated_rectangle)
    xy = min_rect.apply(
        lambda g: np.sqrt(
            (np.array(g.exterior.xy[0]) - np.array(g.exterior.xy[0][0])) ** 2
            + (np.array(g.exterior.xy[1]) - np.array(g.exterior.xy[1][0])) ** 2
        )
    )
    len_est3 = xy.apply(lambda a: np.partition(a.flatten(), -2)[-2])

    # update length estimate where shape factor is lower than 4
    len_est.loc[shape_factor < 4] = len_est3.loc[shape_factor < 4]

    return len_est


def distribute_cond_over_lays(
    cond, cellid, rivbot, laytop, laybot, idomain=None, kh=None, stage=None
):
    """Distribute the conductance in a cell over the layers in that cell, based on the
    the river-bottom and the layer bottoms, and optionally based on the stage and the
    hydraulic conductivity.
    """
    if isinstance(rivbot, (np.ndarray, xr.DataArray)):
        rivbot = float(rivbot[cellid])
    if len(laybot.shape) == 3:
        # the grid is structured grid
        laytop = laytop[cellid[0], cellid[1]]
        laybot = laybot[:, cellid[0], cellid[1]]
        if idomain is not None:
            idomain = idomain[:, cellid[0], cellid[1]]
        if kh is not None:
            kh = kh[:, cellid[0], cellid[1]]
    elif len(laybot.shape) == 2:
        # the grid is a vertex grid
        laytop = laytop[cellid]
        laybot = laybot[:, cellid]
        if idomain is not None:
            idomain = idomain[:, cellid]
        if kh is not None:
            kh = kh[:, cellid]

    if stage is None or isinstance(stage, str):
        lays = np.arange(int(np.sum(rivbot < laybot)) + 1)
    elif np.isfinite(stage):
        lays = np.arange(int(np.sum(stage < laybot)), int(np.sum(rivbot < laybot)) + 1)
    else:
        lays = np.arange(int(np.sum(rivbot < laybot)) + 1)
    if idomain is not None:
        # only distribute conductance over active layers
        lays = lays[idomain[lays] > 0]
    topbot = np.hstack((laytop, laybot))
    topbot[topbot < rivbot] = rivbot
    d = -1 * np.diff(topbot)
    if kh is not None:
        kd = kh * d
    else:
        kd = d
    if np.all(kd <= 0):
        # when for some reason the kd is 0 in all layers (for example when the
        # river bottom is above all the layers), add to the first active layer
        if idomain is not None:
            try:
                first_active = np.where(idomain > 0)[0][0]
            except IndexError:
                warnings.warn(f"No active layers in {cellid}, returning NaNs.")
                return np.nan, np.nan
        else:
            first_active = 0
        lays = [first_active]
        kd[first_active] = 1.0
    conds = cond * kd[lays] / np.sum(kd[lays])
    return np.array(lays), np.array(conds)


def build_spd(
    celldata,
    pkg,
    ds,
    layer_method="lay_of_rbot",
    desc=None,
    silent=False,
):
    """Build stress period data for package (RIV, DRN, GHB).

    Parameters
    ----------
    celldata : geopandas.GeoDataFrame
        GeoDataFrame containing data. Cellid must be the index,
        and must have columns "rbot", "stage" and "cond".
        Optional columns are 'boundname' and 'aux'. These columns should have type str.
    pkg : str
        Modflow package: RIV, DRN or GHB
    ds : xr.Dataset
        DataSet containing model layer information
    layer_method: layer_method : str, optional
        The method used to distribute the conductance over the layers. Possible
        values are 'lay_of_rbot' and 'distribute_cond_over_lays'. The default
        is "lay_of_rbot".
    desc : string, optional
        The description of the progressbar. The default is None, so desc will be
        "Building stress period data RIV/DRN/GHB".
    silent : bool, optional
        Do not show a progressbar when silent is True. The default is False.

    Returns
    -------
    spd : list
        list containing stress period data:
        - RIV: [(cellid), stage, cond, rbot]
        - DRN: [(cellid), elev, cond]
        - GHB: [(cellid), elev, cond]
    """
    spd = []

    top = ds.top.data
    botm = ds.botm.data
    idomain = get_idomain(ds).data
    kh = ds.kh.data

    # ignore records without a stage
    mask = celldata["stage"].isna()
    if mask.any():
        logger.warning(f"{mask.sum()} records without a stage ignored")
        celldata = celldata[~mask]

    if desc is None:
        desc = f"Building stress period data {pkg}"
    for cellid, row in tqdm(
        celldata.iterrows(),
        total=celldata.index.size,
        desc=desc,
        disable=silent,
    ):
        # check if there is an active layer for this cell
        if ds.gridtype == "vertex":
            idomain_cell = idomain[:, cellid]
            botm_cell = botm[:, cellid]
        elif ds.gridtype == "structured":
            idomain_cell = idomain[:, cellid[0], cellid[1]]
            botm_cell = botm[:, cellid[0], cellid[1]]
        if (idomain_cell <= 0).all():
            continue

        # rbot
        if "rbot" in row.index:
            rbot = row["rbot"]
            if np.isnan(rbot):
                raise ValueError(f"rbot is NaN in cell {cellid}")
        elif pkg == "RIV":
            raise ValueError("Column 'rbot' required for building RIV package!")
        else:
            rbot = np.nan

        # stage
        stage = row["stage"]

        if not isinstance(stage, str) and stage < rbot and np.isfinite(rbot):
            logger.warning(
                f"WARNING: stage below bottom elevation in {cellid}, "
                "stage reset to rbot!"
            )
            stage = rbot

        # conductance
        cond = row["cond"]

        # check value
        if np.isnan(cond):
            raise ValueError(
                f"Conductance is NaN in cell {cellid}. Info: area={row.area:.2f} "
                f"len={row.len_estimate:.2f}, BL={row['rbot']}"
            )

        if cond < 0:
            raise ValueError(
                f"Conductance is negative in cell {cellid}. Info: area={row.area:.2f} "
                f"len={row.len_estimate:.2f}, BL={row['rbot']}"
            )

        if layer_method == "distribute_cond_over_lays":
            # if surface water penetrates multiple layers:
            lays, conds = distribute_cond_over_lays(
                cond,
                cellid,
                rbot,
                top,
                botm,
                idomain,
                kh,
                stage,
            )
        elif layer_method == "lay_of_rbot":
            mask = (rbot > botm_cell) & (idomain_cell > 0)
            if not mask.any():
                # rbot is below the bottom of the model, maybe the stage is above it?
                mask = (stage > botm_cell) & (idomain_cell > 0)
                if not mask.any():
                    raise (
                        ValueError("rbot and stage are below the bottom of the model")
                    )
            lays = [np.where(mask)[0][0]]
            conds = [cond]
        else:
            raise (ValueError(f"Method {layer_method} unknown"))

        auxlist = []
        if "aux" in row:
            auxlist.append(row["aux"])
        if "boundname" in row:
            auxlist.append(row["boundname"])

        if ds.gridtype == "vertex":
            cellid = (cellid,)

        # write SPD
        for lay, cond in zip(lays, conds):
            cid = (lay,) + cellid
            if pkg == "RIV":
                spd.append([cid, stage, cond, rbot] + auxlist)
            elif pkg in ["DRN", "GHB"]:
                spd.append([cid, stage, cond] + auxlist)

    return spd


def add_info_to_gdf(*args, **kwargs):
    logger.warning(
        "nlmod.gwf.surface_water.add_info_to_gdf is deprecated. "
        "Use nlmod.util.gdf_intersection_join instead."
    )
    return gdf_intersection_join(*args, **kwargs)


def get_gdf_stage(gdf, season="winter"):
    """Get the stage from a GeoDataFrame for a specific season.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame of the polygons of the BGT with added information in the
        columns 'summer_stage', 'winter_stage', and 'ahn_min'.
    season : str, optional
        The season for which the stage needs to be determined. The default is
        "winter".

    Returns
    -------
    stage : pandas.Series
        The stage for each of the records in the GeoDataFrame.
    """
    stage = gdf[f"{season}_stage"].copy()
    if "ahn_min" in gdf:
        # when the minimum surface level is above the stage
        # or when no stage is available
        # use the minimum surface level
        stage = pd.concat((stage, gdf["ahn_min"]), axis=1).max(axis=1)
    return stage


def download_level_areas(
    gdf=None,
    extent=None,
    config=None,
    raise_exceptions=True,
    drop_duplicates=True,
    **kwargs,
):
    """Download level areas (peilgebieden) of bronhouders.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame with surface water features, containing the column "bronhouder".
    extent : list, tuple or np.array, optional
        Model extent (xmin, xmax, ymin, ymax). When extent is None, all data of the
        water boards in gdf are downloaded downloaded.
    config : dict, optional
        A dictionary with information about the webservices of the water boards. When
        config is None, it is created with nlmod.read.waterboard.get_configuration().
        The default is None.
    raise_exceptions : bool, optional
        Raises exceptions, mostly caused by a webservice that is offline. When
        raise_exceptions is False, the error is raised as a warning. The default is
        True.
    drop_duplicates : bool, optional
        Drop features with a duplicate index, keeping the first occurence. The default
        is True.

    Returns
    -------
    la : dict
        A dictionary with the name of the waterboards as keys and GeoDataFrames with
        level areas as values.
    """
    if config is None:
        config = waterboard.get_configuration()
    wbs = _get_waterboard_selection(gdf=gdf, extent=extent, config=config)

    la = {}
    data_kind = "level_areas"
    for wb in wbs:
        logger.info(f"Downloading {data_kind} for {wb}")
        try:
            lawb = waterboard.get_data(wb, data_kind, extent, **kwargs)
            if len(lawb) == 0:
                logger.info(f"No {data_kind} for {wb} found within model area")
                continue
            if drop_duplicates:
                mask = lawb.index.duplicated()
                if mask.any():
                    msg = "Dropping {} level area(s) of {} with duplicate indexes"
                    logger.warning(msg.format(mask.sum(), wb))
                    lawb = lawb.loc[~mask]

            la[wb] = lawb
            mask = ~la[wb].is_valid
            if mask.any():
                logger.warning(
                    f"{mask.sum()} geometries of level areas of {wb} are invalid. Thet are made valid by adding a buffer of 0.0."
                )
                # first copy to prevent ValueError: assignment destination is read-only
                la[wb] = la[wb].copy()
                la[wb].loc[mask, "geometry"] = la[wb][mask].buffer(0.0)
        except Exception as e:
            if str(e) == f"{data_kind} not available for {wb}":
                logger.warning(e)
            elif raise_exceptions:
                raise
            else:
                logger.warning(e)
    return la


def download_watercourses(
    gdf=None, extent=None, config=None, raise_exceptions=True, **kwargs
):
    """Download watercourses of bronhouders.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame with surface water features, containing the column "bronhouder".
        Determine the required waterboards for this gdf, when not None. The default is
        None.
    extent : list, tuple or np.array, optional
        Model extent (xmin, xmax, ymin, ymax). When extent is None, all data of the
        water boards in gdf are downloaded downloaded. The default is None.
    config : dict, optional
        A dictionary with information about the webservices of the water boards. When
        config is None, it is created with nlmod.read.waterboard.get_configuration().
        The default is None.
    raise_exceptions : bool, optional
        Raises exceptions, mostly caused by a webservice that is offline. When
        raise_exceptions is False, the error is raised as a warning. The default is
        True.

    Returns
    -------
    wc : dict
        A dictionary with the name of the waterboards as keys and GeoDataFrames with
        watercourses as values.
    """
    if config is None:
        config = waterboard.get_configuration()
    wbs = _get_waterboard_selection(gdf=gdf, extent=extent, config=config)
    wc = {}
    data_kind = "watercourses"
    for wb in wbs:
        logger.info(f"Downloading {data_kind} for {wb}")
        try:
            wcwb = waterboard.get_data(wb, data_kind, extent, **kwargs)
            if len(wcwb) == 0:
                logger.info(f"No {data_kind} for {wb} found within model area")
                continue
            wc[wb] = wcwb
        except Exception as e:
            if str(e) == f"{data_kind} not available for {wb}":
                logger.warning(e)
            elif raise_exceptions:
                raise
            else:
                logger.warning(e)
    return wc


def _get_waterboard_selection(gdf=None, extent=None, config=None):
    """Internal method to select waterboards to get data from."""
    if config is None:
        config = waterboard.get_configuration()
    if gdf is None and extent is None:
        raise (ValueError("Please specify either gdf or extent"))

    if gdf is not None:
        bronhouders = gdf["bronhouder"].unique()
        wbs = []
        for wb in config.keys():
            if config[wb]["bgt_code"] in bronhouders:
                wbs.append(wb)
    elif extent is not None:
        wb_gdf = waterboard.get_polygons()
        wbs = wb_gdf.index[wb_gdf.intersects(extent_to_polygon(extent))]
    return wbs


def add_stages_from_waterboards(
    gdf, la=None, extent=None, columns=None, config=None, min_total_overlap=0.0
):
    """Add information from level areas (peilgebieden) to bgt-polygons.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame with surface water features, containing the column "bronhouder".
    la : dict, optional
        A dictionary with the name of the waterboards as keys and GeoDataFrames with
        level areas as values. It is generated by download_level_areas when None.
        The default is None.
    extent : list, tuple or np.array
        Model extent (xmin, xmax, ymin, ymax). When extent is None, all data of the
        water boards in gdf are downloaded downloaded.
    columns : TYPE, optional
        The columns that are added to gdf. Columns defaults to 'summer_stage' and
        'winter_stage' when None. The default is None.
    config : dict, optional
        A dictionary with information about the webservices of the water boards. When
        config is None, it is created with nlmod.read.waterboard.get_configuration().
        The default is None.
    min_total_overlap : float, optional
        Only add data from waterboards to gdf when the total overlap between a feature
        in gdf with all the features from the waterboard is larger than the fraction
        min_total_overlap. The default is 0.0.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame with surface water features, with the added columns
    """
    if config is None:
        config = waterboard.get_configuration()
    if la is None:
        la = download_level_areas(gdf, extent=extent, config=config)
    if columns is None:
        columns = ["summer_stage", "winter_stage"]
    gdf[columns] = np.nan
    for wb in la.keys():
        if len(la[wb]) == 0:
            continue
        mask = gdf["bronhouder"] == config[wb]["bgt_code"]
        gdf.loc[mask, columns] = gdf_intersection_join(
            la[wb],
            gdf[mask],
            columns=columns,
            min_total_overlap=min_total_overlap,
            desc=f"Adding {columns} from {wb}",
        )[columns]
    return gdf


def add_bottom_height_from_waterboards(
    gdf, wc=None, extent=None, columns=None, config=None, min_total_overlap=0.0
):
    """Add information from watercourses to bgt-polygons.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame with surface water features, containing the column "bronhouder".
    wc : dict, optional
        A dictionary with the name of the waterboards as keys and GeoDataFrames with
        watercourses as values. It is generated by download_watercourses when None.
        The default is None.
    extent : list, tuple or np.array
        Model extent (xmin, xmax, ymin, ymax). When extent is None, all data of the
        water boards in gdf are downloaded downloaded.
    columns : TYPE, optional
        The columns that are added to gdf. Columns defaults to 'bottom_height' when
        None. The default is None.
    config : dict, optional
        A dictionary with information about the webservices of the water boards. When
        config is None, it is created with nlmod.read.waterboard.get_configuration().
        The default is None.
    min_total_overlap : float, optional
        Only add data from waterboards to gdf when the total overlap between a feature
        in gdf with all the features from the waterboard is larger than the fraction
        min_total_overlap. The default is 0.0.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame with surface water features, with the added columns
    """
    if config is None:
        config = waterboard.get_configuration()
    if wc is None:
        wc = download_watercourses(gdf, extent=extent, config=config)
    if columns is None:
        columns = ["bottom_height"]
    gdf[columns] = np.nan
    for wb in wc.keys():
        if len(wc[wb]) == 0:
            continue
        mask = gdf["bronhouder"] == config[wb]["bgt_code"]
        gdf.loc[mask, columns] = gdf_intersection_join(
            wc[wb],
            gdf[mask],
            columns=columns,
            min_total_overlap=min_total_overlap,
            desc=f"Adding {columns} from {wb}",
            geom_type="LineString",
        )[columns]
    return gdf


def get_gdf(ds=None, extent=None, fname_ahn=None, ahn=None, buffer=0.0):
    """Generate a GeoDataFrame based on BGT-data and data from waterboards.

    Parameters
    ----------
    ds : TYPE, optional
        The Model Dataset, used to determine the extent (when None) and to grid the
        surface level features. The default is None.
    extent : list, tuple or np.array
        Model extent (xmin, xmax, ymin, ymax). When extent is None, extent is extracted
        from ds
    fname_ahn : str, optional
        When not None, fname_ahn is the path to a tiff-file with ahn-data, to calculate
        the minimum height of the surface level near the surface water features. The
        default is None.
    ahn : xr.DataArray, optional
        When not None, ahn is a DataArray containing the height of the surface level and
        is used to calculate the minimum height of the surface level near the surface
        water features. The default is None.
    buffer : float, optional
        The buffer that is applied around surface water features to calculate the
        minimum surface level near these features. The default is 0.0.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame with surface water features, with added columns from waterboards
        and gridded to the model grid (when ds is aupplied)
    """
    if extent is None:
        if ds is None:
            raise (ValueError("Please supply either ds or extent to get_gdf"))
        extent = get_extent_polygon(ds)
    gdf = bgt.download_bgt(extent)
    if fname_ahn is not None:
        from rasterstats import zonal_stats

        stats = zonal_stats(gdf.geometry.buffer(buffer), fname_ahn, stats="min")
        gdf["ahn_min"] = [x["min"] for x in stats]
    if ahn is not None:
        if fname_ahn is not None:
            logger.warning("Data from {fname_ahn} is overwritten by data from ahn")
        gdf = add_min_ahn_to_gdf(gdf, ahn, buffer=buffer)
    if isinstance(extent, Polygon):
        bs = extent.bounds
        extent = [bs[0], bs[2], bs[1], bs[3]]
    gdf = add_stages_from_waterboards(gdf, extent=extent)
    gdf = add_bottom_height_from_waterboards(gdf, extent=extent)
    if ds is not None:
        return gdf_to_grid(gdf, ds).set_index("cellid")
    return gdf


def add_min_ahn_to_gdf(
    gdf, ahn, buffer=0.0, column="ahn_min", statistic="min", **kwargs
):
    """Add a column with the minimum surface level height near surface water features.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame with surface water features
    ahn : xr.DataArray
        A DataArray containing the height of the surface level.
    buffer : float, optional
        The buffer that is applied around surface water features to calculate the
        minimum surface level near these features. The default is 0.0.
    column : string, optional
        The name of the new column in gdf containing the minimum surface level height.
        The default is 'ahn_min'.
    statistic : string, optional
        The statistic to calculate at each surface water feature. The default is 'min'.
    progressbar : bool, optional
        Show a progressbar when True. The default is False.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame with surface water features, with an added column containing the
        minimum surface level height near the features.
    """
    gdf = zonal_statistics(
        gdf, ahn, columns=column, buffer=buffer, statistics=statistic, **kwargs
    )
    return gdf


def gdf_to_seasonal_pkg(
    gdf,
    gwf,
    ds,
    pkg="DRN",
    default_water_depth=0.5,
    boundname_column="identificatie",
    c0=1.0,
    summer_months=(4, 5, 6, 7, 8, 9),
    layer_method="lay_of_rbot",
    silent=False,
    start_summer=None,
    start_winter=None,
    **kwargs,
):
    """Add a surface water package to a groundwater-model, based on input from a
    GeoDataFrame. This method adds two boundary conditions for each record in the
    GeoDataFrame: one for the winter_stage and one for the summer_stage.
    The conductance of each record is a time-series called 'winter' or 'summer' with
    values of either 0 or 1. These conductance values are multiplied by an auxiliary
    variable that contains the actual conductance.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame with Polygon-data. Cellid must be the index and must have columns
        'winter_stage' and 'summer_stage'.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    ds : xr.Dataset
        Dataset with model data.
    pkg: str, optional
        The package to generate. Possible options are 'DRN', 'RIV' and 'GHB'.
        The default is 'DRN'.
    default_water_depth : float, optional
        The default water depth, only used when there is no 'rbot' column in
        gdf or when this column contains nans. The default is 0.5.
    boundname_column : str, optional
        The name of the column in gdf to use for the boundnames. The default is
        "identificatie", which is a unique identifier in the BGT.
    c0 : float, optional
        The resistance of the surface water, in days. Only used when there is
        no 'cond' column in gdf. The default is 1.0.
    summer_months : list or tuple, optional
        The months in which 'summer_stage' is active (one-based). The parameter
        summer_months is used to calculate start_summer or start_winter, when they are
        None. The default is (4, 5, 6, 7, 8, 9), which means summer is from april
        through september.
    layer_method : str, optional
        The method used to distribute the conductance over the layers. Possible
        values are 'lay_of_rbot' and 'distribute_cond_over_lays'. The default
        is "lay_of_rbot".
    silent : bool, optional
        Do not show a progressbar when silent is True. The default is False.
    start_summer : str, optional
        A string with the month and day of the start of summer (one-based), seperated by
        '-'. For example '4-1' for april 1st. When start_summer is None it is calculated
        from the parameter 'summer_months'. The default is None.
    start_winter : str, optional
        A string with the month and day of the start of winter (one-based), seperated by
        '-'. For example '10-1' for october 1st. When start_winter is None it is
        calculated from the parameter 'summer_months'. The default is None.
    **kwargs : dict
        Kwargs are passed onto ModflowGwfdrn, ModflowGwfriv or ModflowGwfghb.

    Returns
    -------
    package : ModflowGwfdrn, ModflowGwfriv or ModflowGwfghb
        The generated flopy-package
    """
    if gdf.index.name != "cellid":
        # if "cellid" not in gdf:
        #    gdf = gdf_to_grid(gdf, gwf)
        gdf = gdf.set_index("cellid")
    else:
        # make sure changes to the DataFrame are temporarily
        gdf = gdf.copy()

    stages = (
        get_gdf_stage(gdf, "winter"),
        get_gdf_stage(gdf, "summer"),
    )

    # make sure we have a bottom height
    if "rbot" not in gdf:
        gdf["rbot"] = np.nan
    mask = gdf["rbot"].isna()
    if mask.any():
        logger.info(
            f"Filling {mask.sum()} NaN's in rbot using a water depth of {default_water_depth} meter."
        )
        min_stage = pd.concat(stages, axis=1).min(axis=1)
        gdf.loc[mask, "rbot"] = min_stage[mask] - default_water_depth

    if "cond" not in gdf:
        logger.info(
            f"Calcluating {pkg}-conductance based on as resistance of {c0} days."
        )
        gdf["cond"] = gdf.geometry.area / c0

    if boundname_column is not None:
        gdf["boundname"] = gdf[boundname_column]

    spd = []
    seasons = ["winter", "summer"]
    for iseason, season in enumerate(seasons):
        # use  a winter and summer level
        gdf["stage"] = stages[iseason]

        mask = gdf["stage"] < gdf["rbot"]
        gdf.loc[mask, "stage"] = gdf.loc[mask, "rbot"]
        gdf["aux"] = season

        spd.extend(
            build_spd(
                gdf,
                pkg,
                ds,
                layer_method=layer_method,
                desc=f"Building stress period data for {season} {pkg}",
                silent=silent,
            )
        )
    # from the release notes (6.3.0):
    # When this AUXMULTNAME option is used, the multiplier value in the
    # AUXMULTNAME column should not be represented with a time series unless
    # the value to scale is also represented with a time series
    # So we switch the conductance (column 2) and the multiplier (column 3/4)
    spd = np.array(spd, dtype=object)
    if len(spd) > 0:
        if pkg == "RIV":
            spd[:, [2, 4]] = spd[:, [4, 2]]
        else:
            spd[:, [2, 3]] = spd[:, [3, 2]]
    spd = spd.tolist()

    if boundname_column is None:
        observations = None
    else:
        observations = []
        for boundname in np.unique(gdf[boundname_column]):
            observations.append((boundname, pkg, boundname))
        observations = {f"{pkg}_flows.csv": observations}
    if pkg == "DRN":
        cl = flopy.mf6.ModflowGwfdrn
    elif pkg == "RIV":
        cl = flopy.mf6.ModflowGwfriv
    elif pkg == "GHB":
        cl = flopy.mf6.ModflowGwfghb
    else:
        raise (ValueError(f"Unknown package: {pkg}"))
    package = cl(
        gwf,
        stress_period_data={0: spd},
        boundnames=boundname_column is not None,
        auxmultname="cond_fact",
        auxiliary=["cond_fact"],
        observations=observations,
        **kwargs,
    )
    # add timeseries for the seasons 'winter' and 'summer'
    add_season_timeseries(
        ds,
        package,
        summer_months=summer_months,
        winter_name="winter",
        summer_name="summer",
        start_summer=start_summer,
        start_winter=start_winter,
    )
    return package


def add_season_timeseries(
    ds,
    package,
    summer_months=(4, 5, 6, 7, 8, 9),
    filename="season.ts",
    summer_name="summer",
    winter_name="winter",
    start_summer=None,
    start_winter=None,
):
    """Add time series indicating which season is active (e.g. summer/winter).

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset used for time discretization.
    package : flopy.mf6 package
        Modflow 6 package to add time series to.
    summer_months : tuple, optional
        The summer months (one-based). The parameter summer_months is used to calculate
        start_summer or start_winter, when they are None. The default is
        (4, 5, 6, 7, 8, 9), so from april to september.
    filename : str, optional
        The name of time series file. The default is "season.ts".
    winter_name : str, optional
        The name of the time-series with ones in winter. The default is "winter".
    summer_name : str, optional
        The name of the time-series with ones in summer. The default is "summer".
    start_summer : str, optional
        A string with the month and day of the start of summer (one-based), seperated by
        '-'. For example '4-1' for april 1st. When start_summer is None it is calculated
        from the parameter 'summer_months'. The default is None.
    start_winter : str, optional
        A string with the month and day of the start of winter (one-based), seperated by
        '-'. For example '10-1' for october 1st. When start_winter is None it is
        calculated from the parameter 'summer_months'. The default is None.
    """
    if ds.time.dtype.kind != "M":
        raise TypeError("add_season_timeseries requires a datetime64[ns] time index")
    start_summer, start_winter = _get_start_summer_and_winter(
        start_summer, start_winter, summer_months
    )
    tmin = pd.to_datetime(ds.time.start)
    if _is_in_summer(tmin, start_summer, start_winter):
        ts_data = [(0.0, 0.0, 1.0)]
    else:
        ts_data = [(0.0, 1.0, 0.0)]
    tmax = pd.to_datetime(ds["time"].data[-1])
    years = range(tmin.year, tmax.year + 1)
    for year in years:
        # add a record for the start of summer
        time = pd.Timestamp(f"{year}-{start_summer}")
        time = (time - tmin) / pd.to_timedelta(1, "D")
        if time > 0:
            ts_data.append((time, 1.0, 0.0))
        # add a record for the start of winter
        time = pd.Timestamp(f"{year}-{start_winter}")
        time = (time - tmin) / pd.to_timedelta(1, "D")
        if time > 0:
            ts_data.append((time, 0.0, 1.0))

    return package.ts.initialize(
        filename=filename,
        timeseries=ts_data,
        time_series_namerecord=[summer_name, winter_name],
        interpolation_methodrecord=["stepwise", "stepwise"],
    )


def _get_start_summer_and_winter(start_summer, start_winter, summer_months):
    if start_summer is None:
        start_summer = f"{summer_months[0]}-1"
    if start_winter is None:
        start_winter = f"{summer_months[-1] + 1}-1"
    return start_summer, start_winter


def _is_in_summer(time, start_summer, start_winter):
    after_start_summer = time >= pd.Timestamp(f"{time.year}-{start_summer}")
    before_start_winter = time < pd.Timestamp(f"{time.year}-{start_winter}")
    return after_start_summer and before_start_winter


def get_seaonal_timeseries(
    ds,
    summer_value,
    winter_value,
    summer_months=(4, 5, 6, 7, 8, 9),
    start_summer=None,
    start_winter=None,
):
    """
    Get a time-series of winter-values and summer-values.

    The index of the returned series is set at the start of the period that the value
    describes. This is the way that flopy uses this value in a timeseries, when
    interpolation_methodrecord is set to 'stepwise'.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset used for time discretization.
    summer_value : float
        The value to be used in summer.
    winter_value : float
        The value to be used in winter.
    summer_months : tuple, optional
        summer months (one-based). The parameter summer_months is used to calculate
        start_summer or start_winter, when they are None. The default is
        (4, 5, 6, 7, 8, 9), so from april to september.
    start_summer : str, optional
        A string with the month and day of the start of summer (one-based), seperated by
        '-'. For example '4-1' for april 1st. When start_summer is None it is calculated
        from the parameter 'summer_months'. The default is None.
    start_winter : str, optional
        A string with the month and day of the start of winter (one-based), seperated by
        '-'. For example '10-1' for october 1st. When start_winter is None it is
        calculated from the parameter 'summer_months'. The default is None.

    Returns
    -------
    s : pd.Series
        A series with stages as the values and time as the index.

    """
    start_summer, start_winter = _get_start_summer_and_winter(
        start_summer, start_winter, summer_months
    )
    tmin = pd.to_datetime(ds.time.start)
    index = [tmin]
    if _is_in_summer(tmin, start_summer, start_winter):
        values = [summer_value]
    else:
        values = [winter_value]
    tmax = pd.to_datetime(ds["time"].data[-1])
    years = range(tmin.year, tmax.year + 1)
    for year in years:
        # add a record for the start of summer
        time = pd.Timestamp(f"{year}-{start_summer}")
        if time > tmin:
            index.append(time)
            values.append(summer_value)
        # add a record for the start of winter
        time = pd.Timestamp(f"{year}-{start_winter}")
        if time > tmin:
            index.append(time)
            values.append(winter_value)

    s = pd.Series(values, index=index)
    return s


def rivdata_from_xylist(gwf, xylist, layer, stage, cond, rbot, aux=None):
    gi = flopy.utils.GridIntersect(gwf.modelgrid, method="vertex")
    cellids = gi.intersect(xylist, shapetype="linestring")["cellids"]
    riv_data = []
    for cid in cellids:
        if isinstance(cid, (list, tuple)) and len(cid) == 2:
            idata = [(layer, cid[0], cid[1]), stage, cond, rbot]
            if aux is not None:
                idata.append(aux)
            riv_data.append(idata)
        else:
            idata = [(layer, cid), stage, cond, rbot]
            if aux is not None:
                idata.append(aux)
            riv_data.append(idata)
    return riv_data
