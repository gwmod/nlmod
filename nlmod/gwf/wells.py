import flopy as fp
import numpy as np


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
    **kwargs,
):

    # collect data
    well_lrcd = []

    for _, irow in df.iterrows():
        cid1 = gwf.modelgrid.intersect(irow[x], irow[y], irow[top])
        cid2 = gwf.modelgrid.intersect(irow[x], irow[y], irow[botm])
        if len(cid1) == 2:
            kt, icell2d = cid1
        elif len(cid1) == 3:
            kt, i, j = cid1
        kb = cid2[0]
        wlayers = np.arange(kt, kb + 1)
        for k in wlayers:
            if len(cid1) == 2:
                wdata = [(k, icell2d), irow[Q] / len(wlayers)]
            elif len(cid1) == 3:
                wdata = [(k, i, j), irow[Q] / len(wlayers)]

            if aux is not None:
                wdata.append(irow[aux])
            if boundnames is not None:
                wdata.append(irow[boundnames])
            well_lrcd.append(wdata)

    wel_spd = {0: well_lrcd}

    if aux is not None:
        aux = True
    wel = fp.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_spd,
        auxiliary=aux,
        boundnames=boundnames is not None,
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
    boundnames=None,
    **kwargs,
):

    maw_pakdata = []
    maw_conndata = []
    maw_perdata = []

    for iw, irow in df.iterrows():
        cid1 = gwf.modelgrid.intersect(irow[x], irow[y], irow[top])
        cid2 = gwf.modelgrid.intersect(irow[x], irow[y], irow[botm])
        if len(cid1) == 2:
            kt, icell2d = cid1
        elif len(cid1) == 3:
            kt, i, j = cid1
        kb = cid2[0]
        wlayers = np.arange(kt, kb + 1)

        # <wellno> <radius> <bottom> <strt> <condeqn> <ngwfnodes>
        pakdata = [iw, irow[rw], irow[top], strt, condeqn, len(wlayers)]
        if boundnames is not None:
            pakdata.append(irow[boundnames])
        maw_pakdata.append(pakdata)
        # <wellno> <mawsetting>
        maw_perdata.append([iw, "RATE", irow[Q]])

        for iwellpart, k in enumerate(wlayers):
            if gwf.modelgrid.grid_type == "vertex":
                laytop = gwf.modelgrid.botm[k - 1, icell2d]
                laybot = gwf.modelgrid.botm[k, icell2d]
                # <wellno> <icon> <cellid(ncelldim)> <scrn_top> <scrn_bot> <hk_skin> <radius_skin>
                mawdata = [
                    iw,
                    iwellpart,
                    (k, icell2d),
                    np.min([irow[top], laytop]),
                    laybot,
                    0.0,
                    0.0,
                ]
            elif gwf.modelgrid.grid_type == "structured":
                laytop = gwf.modelgrid.botm[k - 1, i, j]
                laybot = gwf.modelgrid.botm[k, i, j]
                # <wellno> <icon> <cellid(ncelldim)> <scrn_top> <scrn_bot> <hk_skin> <radius_skin>
                mawdata = [
                    iw,
                    iwellpart,
                    (k, i, j),
                    np.min([irow[top], laytop]),
                    laybot,
                    0.0,
                    0.0,
                ]
            maw_conndata.append(mawdata)

    maw = fp.mf6.ModflowGwfmaw(
        gwf,
        nmawwells=df.index.size,
        boundnames=boundnames is not None,
        packagedata=maw_pakdata,
        connectiondata=maw_conndata,
        perioddata=maw_perdata,
        **kwargs,
    )

    return maw