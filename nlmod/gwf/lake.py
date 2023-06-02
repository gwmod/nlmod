import logging

import flopy
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAKE_KWDS = [
    "STATUS",
    "STAGE",
    "RAINFALL",
    "EVAPORATION",
    "RUNOFF",
    "INFLOW",
    "WITHDRAWAL",
    "AUXILIARY",
    "RATE",
    "INVERT",
    "WIDTH",
    "SLOPE",
    "ROUGH",
]

# order of dictionary matters!
OUTLET_DEFAULT = {
    "couttype": "WEIR",
    "outlet_invert": "use_elevation",
    "outlet_width": 1.0,
    "outlet_rough": 0.0,
    "outlet_slope": 0.0,
}


def lake_from_gdf(
    gwf,
    gdf,
    ds,
    recharge=True,
    claktype="VERTICAL",
    boundname_column="identificatie",
    obs_type="STAGE",
    surfdep=0.05,
    pname="lak",
    **kwargs,
):
    """Add a lake from a geodataframe.

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    gdf : gpd.GeoDataframe
        geodataframe with the cellids as the index and the columns:
            lakeno : with the number of the lake
            strt : with the starting head of the lake
            clake : with the bed resistance of the lake
            optional columns are 'STATUS', 'STAGE', 'RAINFALL', 'EVAPORATION',
            'RUNOFF', 'INFLOW', 'WITHDRAWAL', 'AUXILIARY', 'RATE', 'INVERT',
            'WIDTH', 'SLOPE', 'ROUGH'. These columns should contain the name
            of a dataarray in ds with the dimension time.
        if the lake has any outlets they should be specified in the columns
            lakeout : the lake number of the outlet, if this is -1 the water
            is removed from the model.
            optinal columns are 'couttype', 'outlet_invert', 'outlet_width',
            'outlet_rough' and 'outlet_slope'. These columns should contain a
            unique value for each outlet.
    ds : xr.DataSet
        dataset containing relevant model grid and time information
    recharge : bool, optional
        if True recharge will be added to the lake and removed from the
        recharge package. The recharge
    claktype : str, optional
        defines the lake-GWF connection type. For now only VERTICAL is
        supported. The default is 'VERTICAL'.
    boundname_column : str, optional
        THe name of the column in gdf to use for the boundnames. The default is
        "identificatie", which is a unique identifier in the BGT.
    surfdep : float, optional
        Defines the surface depression depth for VERTICAL lake-GWF connections.
        The default is 0.05.
    pname : str, optional
        name of the lake package. The default is 'lak'.
    **kwargs :
        passed to flopy.mf6.ModflowGwflak.

    Raises
    ------
    NotImplementedError

    Returns
    -------
    lak : flopy lake package
    """
    if claktype != "VERTICAL":
        raise NotImplementedError("function only tested for claktype=VERTICAL")

    if ds.gridtype != "vertex":
        raise NotImplementedError("only works with a vertex grid")

    assert ds.time.time_units.lower() == "days", "expected time unit days"
    time_conversion = 86400.0
    # length unit is always meters in nlmod
    length_conversion = 1.0

    packagedata = []
    connectiondata = []
    perioddata = {}
    for iper in range(ds.dims["time"]):
        perioddata[iper] = []

    lake_settings = [setting for setting in LAKE_KWDS if setting in gdf.columns]

    if "lakeout" in gdf.columns:
        outlets = []
        outlet_no = 0
        use_outlets = True
        logger.debug("using lake outlets")
    else:
        use_outlets = False
        noutlets = None
        outlets = None

    for lakeno, lake_gdf in gdf.groupby("lakeno"):
        nlakeconn = lake_gdf.shape[0]
        strt = lake_gdf["strt"].iloc[0]
        assert (lake_gdf["strt"] == strt).all(
        ), "a single lake should have single strt"

        if boundname_column is not None:
            boundname = lake_gdf[boundname_column].iloc[0]
            assert (
                lake_gdf[boundname_column] == boundname
            ).all(), f"a single lake should have a single {boundname_column}"
            packagedata.append([lakeno, strt, nlakeconn, boundname])
        else:
            packagedata.append([lakeno, strt, nlakeconn])

        iconn = 0
        for icell2d, row in lake_gdf.iterrows():
            cellid = (0, icell2d)  # assuming lake in the top layer

            # If BEDLEAK is specified to be NONE, the lake-GWF connection
            # conductance is solely a function of aquifer properties in the
            # connected GWF cell and lakebed sediments are assumed to be absent.
            clake = row["clake"]
            bedleak = 1 / clake
            belev = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            telev = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            connlen = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            connwidth = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            connectiondata.append(
                [
                    lakeno,
                    iconn,
                    cellid,
                    claktype,
                    bedleak,
                    belev,
                    telev,
                    connlen,
                    connwidth,
                ]
            )
            iconn += 1

        # add outlets to lake
        if use_outlets and (not lake_gdf["lakeout"].isna().all()):
            lakeout = lake_gdf["lakeout"].iloc[0]
            if not (lake_gdf["lakeout"] == lakeout).all():
                raise ValueError(
                    f'expected single value for lakeout and lake number {lakeno}, got {lake_gdf["lakeout"]}'
                )

            assert lakeno != lakeout, "lakein and lakeout cannot be the same"

            outsettings = []
            for outset, default_value in OUTLET_DEFAULT.items():
                if outset not in lake_gdf.columns:
                    logger.debug(
                        f"no value specified for {outset} and lake no {lakeno}, using default value {default_value}"
                    )
                    setval = default_value
                else:
                    setval = lake_gdf[outset].iloc[0]
                    if pd.notna(setval):
                        if not (lake_gdf[outset] == setval).all():
                            raise ValueError(
                                f"expected single data variable for {outset} and lake number {lakeno}, got {lake_gdf[outset]}"
                            )
                    else:  # setval is nan or None
                        setval = default_value
                        logger.debug(
                            f"no value specified for {outset} and lake no {lakeno}, using default value {default_value}"
                        )
                if outset == "outlet_invert" and isinstance(setval, str):
                    if setval == "use_elevation":
                        setval = strt
                    else:
                        raise NotImplementedError(
                            "outlet_invert should not be a string"
                        )
                outsettings.append(setval)
            outlets.append([outlet_no, lakeno, lakeout] + outsettings)
            outlet_no += 1
        for iper in range(ds.dims["time"]):
            if recharge:
                # add recharge to lake
                cellids = [row[2][1] for row in connectiondata]
                rech = ds["recharge"][iper, cellids].values.mean()
                if rech >= 0:
                    perioddata[iper].append([lakeno, "RAINFALL", rech])
                    perioddata[iper].append([lakeno, "EVAPORATION", 0])
                else:
                    perioddata[iper].append([lakeno, "RAINFALL", 0])
                    perioddata[iper].append([lakeno, "EVAPORATION", -rech])
                # set recharge to zero in dataset
                ds["recharge"][iper, cellids] = 0

            # add other time variant settings to lake
            for lake_setting in lake_settings:
                datavar = lake_gdf[lake_setting].iloc[0]
                if not pd.notna(datavar):  # None or nan
                    logger.debug(
                        f"no {lake_setting} given for lake no {lakeno}")
                    continue
                if not (lake_gdf[lake_setting] == datavar).all():
                    raise ValueError(
                        f"expected single data variable for {lake_setting} and lake number {lakeno}, got {lake_gdf[lake_setting]}"
                    )
                perioddata[iper].append(
                    [lakeno, lake_setting, ds[datavar].values[iper]]
                )

    if use_outlets:
        noutlets = len(outlets)

    if boundname_column is not None:
        observations = []
        for boundname in np.unique(gdf[boundname_column]):
            observations.append((boundname, obs_type, boundname))
        observations = {f"{pname}_{obs_type}.csv": observations}
    else:
        observations = None

    lak = flopy.mf6.ModflowGwflak(
        gwf,
        surfdep=surfdep,
        time_conversion=time_conversion,
        length_conversion=length_conversion,
        nlakes=len(packagedata),
        packagedata=packagedata,
        connectiondata=connectiondata,
        perioddata=perioddata,
        boundnames=boundname_column is not None,
        observations=observations,
        budget_filerecord=f"{pname}.bgt",
        stage_filerecord=f"{pname}.hds",
        noutlets=noutlets,
        outlets=outlets,
        **kwargs,
    )

    return lak
