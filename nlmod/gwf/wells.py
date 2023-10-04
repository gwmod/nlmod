import logging

import flopy as fp
import numpy as np
import pandas as pd
import geopandas as gpd

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
    condeqn="THIEM",
    strt=0.0,
    aux=None,
    boundnames=None,
    ds=None,
    **kwargs,
):
    if aux is None:
        aux = []
    if not isinstance(aux, list):
        aux = [aux]

    df = _add_cellid(df, ds=ds, gwf=gwf, x=x, y=y)
    multipliers = _get_layer_multiplier_for_wells(df, top, botm, ds=ds, gwf=gwf)

    maw_pakdata = []
    maw_conndata = []
    maw_perdata = []

    iw = 0
    for index, irow in df.iterrows():
        try:
            cid1 = gwf.modelgrid.intersect(irow[x], irow[y], irow[top], forgive=False)
            cid2 = gwf.modelgrid.intersect(irow[x], irow[y], irow[botm], forgive=False)
        except Exception:
            logger.warning(
                f"Well {index} outside of model domain ({irow[x]}, {irow[y]})"
            )
            continue
        kb = cid2[0]
        if len(cid1) == 2:
            kt, icell2d = cid1
            idomain_mask = gwf.modelgrid.idomain[kt : kb + 1, icell2d] > 0
        elif len(cid1) == 3:
            kt, i, j = cid1
            idomain_mask = gwf.modelgrid.idomain[kt : kb + 1, i, j] > 0

        wlayers = np.arange(kt, kb + 1)[idomain_mask]

        wlayers = np.where(multipliers[index] > 0)[0]
        # <wellno> <radius> <bottom> <strt> <condeqn> <ngwfnodes>
        pakdata = [iw, irow[rw], irow[botm], strt, condeqn, len(wlayers)]
        for iaux in aux:
            pakdata.append(irow[iaux])
        if boundnames is not None:
            pakdata.append(irow[boundnames])
        maw_pakdata.append(pakdata)
        # <wellno> <mawsetting>
        maw_perdata.append([iw, "RATE", irow[Q]])

        for iwellpart, k in enumerate(wlayers):
            if k == 0:
                laytop = gwf.modelgrid.top
            else:
                laytop = gwf.modelgrid.botm[k - 1]
            laybot = gwf.modelgrid.botm[k]

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

            mawdata = [
                iw,
                iwellpart,
                cellid,
                scrn_top,
                scrn_bot,
                0.0,
                0.0,
            ]
            maw_conndata.append(mawdata)
        iw += 1

    if len(aux) == 0:
        aux = None
    maw = fp.mf6.ModflowGwfmaw(
        gwf,
        nmawwells=iw,
        auxiliary=aux,
        boundnames=boundnames is not None,
        packagedata=maw_pakdata,
        connectiondata=maw_conndata,
        perioddata=maw_perdata,
        **kwargs,
    )

    return maw


def _add_cellid(df, ds=None, gwf=None, x=None, y=None):
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))
    if "cellid" not in df.columns:
        df = gdf_to_grid(df, gwf if ds is None else ds)
    return df


def _get_layer_multiplier_for_wells(df, top, botm, ds=None, gwf=None):
    # get required data either from  gwf or ds
    if ds is None:
        ml_top = gwf.dis.top.array
        ml_bot = gwf.dis.botm.array
        kh = gwf.npf.k.array
        layer = range(gwf.dis.nlay.array)
    else:
        ml_top = ds["top"].data
        ml_bot = ds["botm"].data
        kh = ds["kh"].data
        layer = ds.layer

    multipliers = dict()
    for index, irow in df.iterrows():
        multipliers[index] = _get_layer_multiplier_for_well(
            irow["cellid"], irow[top], irow[botm], ml_top, ml_bot, kh
        )

        if (multipliers[index] == 0).all():
            logger.warning(f"No layers found for well {index}")
    multipliers = pd.DataFrame(multipliers, index=layer, columns=df.index)
    return multipliers


def _get_layer_multiplier_for_well(cid, well_top, well_bot, ml_top, ml_bot, ml_kh):
    """Get a factor for each layer that a well"""
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
        # devide by the total kd to get a factor
        multiplier = kd / kd.sum()
    return multiplier
