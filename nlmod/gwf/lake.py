import logging

import flopy
import numpy as np
import pandas as pd

from ..dims.layers import get_first_active_layer

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
    rainfall=None,
    evaporation=None,
    claktype="VERTICAL",
    boundname_column="identificatie",
    obs_type="STAGE",
    surfdep=0.05,
    pname="lak",
    gwt=None,
    obs_type_gwt="CONCENTRATION",
    rainfall_concentration=0.0,
    evaporation_concentration=0.0,
    **kwargs,
):
    """Add a lake from a geodataframe.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
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
        if the lake has any outlets they should be specified in the column
            lakeout : the lake number of the outlet, if this is -1 the water
            is removed from the model.
            optinal columns are 'couttype', 'outlet_invert', 'outlet_width',
            'outlet_rough' and 'outlet_slope'. These columns should contain a
            unique value for each outlet.
    ds : xr.Dataset
        dataset containing relevant model grid and time information
    rainfall : int, float, str, np.array or pd.DataFrame, optional
        The rainfall to be applied on the lakes. If rainfall is a DataFrame, there
        should be one column for each lake, with lakeno as the column index, or the
        boundnames if boundname_column is specified. There should be one row for each
        stress period. To generate rainfall and evaporation from a model dataset, the
        method `clip_meteorological_data_from_ds` can be used.
        If rainfall is not a DataFrame, all lakes have the same rainfall-values.
        When rainfall is a pandas Series or a numpy array, a value for rainfall for each
        stress period is taken from this series/array.
        When rainfall is a float, it is directly used as the rainfall-value.
        The default is None.
    evaporation : int, float, str, np.array or pd.DataFrame, optional
        The evaporation to be applied on the lakes. If evaporation is a DataFrame, there
        should be one column for each lake, with lakeno as the column index, or the
        boundnames if boundname_column is specified. There should be one row for each
        stress period. To generate rainfall and evaporation from a model dataset, the
        method `clip_meteorological_data_from_ds` can be used.
        If evaporation is not a DataFrame, all lakes have the same evaporation-values.
        When evaporation is a pandas Series or a numpy array, a value for evaporation
        for each stress period is taken from this series/array.
        When evaporation is a float, it is directly used as the evaporation-value.
        The default is None.
    claktype : str, optional
        defines the lake-GWF connection type. For now only VERTICAL is supported. The
        default is 'VERTICAL'.
    boundname_column : str, optional
        The name of the column in gdf to use for the boundnames. The default is
        "identificatie", which is a unique identifier in the BGT.
    surfdep : float, optional
        Defines the surface depression depth for VERTICAL lake-GWF connections. The
        default is 0.05.
    pname : str, optional
        name of the lake package. The default is 'lak'.
    obs_type : str or list of str, optional
        observation or list of observations to add to the lake package. The default
        is 'STAGE'.
    obs_type_gwt : str or list of str, optional
        observation or list of observations to add to the lake transport package.
        The default is 'CONCENTRATION'.
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
    # TODO: Let's add a check for a length unit of meters if we ever add a length unit
    # to ds
    length_conversion = 1.0

    packagedata = []
    connectiondata = []
    perioddata = {}
    for iper in range(ds.sizes["time"]):
        perioddata[iper] = []

    if gwt is not None:
        packagedata_gwt = []
        perioddata_gwt = {0: []}

    lake_settings = [setting for setting in LAKE_KWDS if setting in gdf.columns]

    outlets = []
    outlet_no = 0

    fal = get_first_active_layer(ds).data

    if "lakeno" not in gdf.columns:
        gdf = add_lakeno_to_gdf(gdf, boundname_column)

    for lakeno, lake_gdf in gdf.groupby("lakeno"):
        nlakeconn = lake_gdf.shape[0]
        if "strt" in lake_gdf:
            strt = _get_and_check_single_value(lake_gdf, "strt")
        else:
            # take the mean of the starting concentrations of the connected cells
            head = ds["starting_head"].data[fal[lake_gdf.index], lake_gdf.index]
            area = ds["area"].data[lake_gdf.index]
            strt = (head * area).sum() / area.sum()
        if boundname_column is not None:
            boundname = _get_and_check_single_value(lake_gdf, f"{boundname_column}")
            packagedata.append([lakeno, strt, nlakeconn, boundname])
        else:
            packagedata.append([lakeno, strt, nlakeconn])

        iconn = 0
        for icell2d, row in lake_gdf.iterrows():
            cellid = (fal[icell2d], icell2d)  # assuming lake in the top layer

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
        if (
            "lakeout" in lake_gdf.columns
            and not lake_gdf["lakeout"].isna().all()
            and not lake_gdf["lakeout"].eq("").all()
        ):
            lakeout = _get_and_check_single_value(lake_gdf, "lakeout")
            if isinstance(lakeout, str):
                # when lakeout is a string, it represents the boundname
                # we need to find the lakeno that belongs to this boundname
                boundnameout = lakeout
                if boundname_column not in gdf.columns:
                    raise KeyError(
                        f"Make sure column {boundname_column} is present in gdf"
                    )
                mask = gdf[boundname_column] == boundnameout
                lakeout = gdf.loc[mask, "lakeno"].iloc[0]
                if not (gdf.loc[mask, "lakeno"] == lakeout).all():
                    raise ValueError(
                        f"expected single value of lakeno for lakeout {boundnameout}, got {gdf.loc[mask, 'lakeno']}"
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
                    # setval can be the name of a timeseries
                    # only when it is equal to "use_elevation" we set the invert to strt
                    if setval == "use_elevation":
                        setval = strt

                outsettings.append(setval)
            outlets.append([outlet_no, lakeno, lakeout] + outsettings)
            outlet_no += 1

        if boundname_column is None:
            key = lakeno
        else:
            key = boundname

        for iper in range(ds.sizes["time"]):
            if rainfall is not None:
                value = _parse_laksetting_value(rainfall, ds, key, iper)
                perioddata[iper].append([lakeno, "RAINFALL", value])
            if evaporation is not None:
                value = _parse_laksetting_value(evaporation, ds, key, iper)
                perioddata[iper].append([lakeno, "EVAPORATION", value])

            # add other time variant settings to lake
            for lake_setting in lake_settings:
                datavar = _get_and_check_single_value(lake_gdf, lake_setting)
                if pd.isna(datavar) or datavar == "":  # None or nan or ""
                    logger.debug(f"no {lake_setting} given for lake no {lakeno}")
                    continue
                perioddata[iper].append(
                    [lakeno, lake_setting, ds[datavar].values[iper]]
                )
        if gwt is not None:
            if "strt_concentration" in lake_gdf.columns:
                strt = _get_and_check_single_value(lake_gdf, "strt_concentration")
            else:
                # take the mean of the starting concentrations of the connected cells
                conc = ds["chloride"].data[fal[lake_gdf.index], lake_gdf.index]
                area = ds["area"].data[lake_gdf.index]
                strt = (conc * area).sum() / area.sum()
            if boundname_column is not None:
                packagedata_gwt.append([lakeno, strt, boundname])
            else:
                packagedata_gwt.append([lakeno, strt])
            if rainfall is not None:
                perioddata_gwt[0].append([lakeno, "rainfall", rainfall_concentration])
            if evaporation is not None:
                perioddata_gwt[0].append(
                    [lakeno, "evaporation", evaporation_concentration]
                )

    if boundname_column is not None:
        observations = {}
        if isinstance(obs_type, str):
            obs_type = [obs_type]
        for otype in obs_type:
            obs_list = [(x, otype, x) for x in np.unique(gdf[boundname_column])]
            observations[f"{pname}_{otype}.csv"] = obs_list
    else:
        observations = None

    boundnames = boundname_column is not None
    lak = flopy.mf6.ModflowGwflak(
        gwf,
        surfdep=surfdep,
        time_conversion=time_conversion,
        length_conversion=length_conversion,
        nlakes=len(packagedata),
        packagedata=packagedata,
        connectiondata=connectiondata,
        perioddata=perioddata,
        boundnames=boundnames,
        observations=observations,
        budget_filerecord=f"{pname}.bgt",
        stage_filerecord=f"{pname}.hds",
        noutlets=len(outlets) if outlets else None,
        outlets=outlets if outlets else None,
        pname=pname,
        **kwargs,
    )

    if gwt is not None:
        if boundname_column is not None:
            observations_gwt = {}
            if isinstance(obs_type_gwt, str):
                obs_type_gwt = [obs_type_gwt]
            for otype in obs_type_gwt:
                obs_list_gwt = [(x, otype, x) for x in np.unique(gdf[boundname_column])]
                observations_gwt[f"{pname}_{otype}.gwt.csv"] = obs_list_gwt
        else:
            observations_gwt = None

        lkt = flopy.mf6.ModflowGwtlkt(
            gwt,
            pname=lak.package_name,
            packagedata=packagedata_gwt,
            lakeperioddata=perioddata_gwt,
            boundnames=boundnames,
            observations=observations_gwt,
            budget_filerecord=f"{pname}_gwt.bgt",
            concentration_filerecord=f"{pname}_gwt.unc",
        )
        return lak, lkt

    return lak


def _get_and_check_single_value(lake_gdf, column):
    value = lake_gdf[column].iloc[0]
    if lake_gdf[column].isna().all() or lake_gdf[column].eq("").all():
        return value
    if not (lake_gdf[column] == value).all():
        raise (AssertionError(f"A single lake should have a single {column}"))
    return value


def _parse_laksetting_value(value, ds, key, iper):
    if isinstance(value, (float, int, str)):
        return value
    elif isinstance(value, pd.Series):
        assert len(value.index) == len(ds.time) and (value.index == ds.time).all()
        return value.iloc[iper]
    elif isinstance(value, pd.DataFrame):
        assert len(value.index) == len(ds.time) and (value.index == ds.time).all()
        return value[key].iloc[iper]
    else:
        assert len(value) == len(ds.time)
        return value[iper]


def add_lakeno_to_gdf(gdf, boundname_column):
    if boundname_column not in gdf.columns:
        raise (KeyError(f"Cannot find column {boundname_column} in gdf"))
    names = gdf[boundname_column].unique()
    gdf["lakeno"] = None
    for lakeno, name in enumerate(names):
        mask = gdf[boundname_column] == name
        gdf.loc[mask, "lakeno"] = lakeno
    return gdf


def _copy_da_from_ds(gdf, ds, variable, boundname_column=None, set_to_0_in_ds=False):
    if boundname_column is None:
        columns = gdf["lakeno"].unique()
    else:
        columns = gdf[boundname_column].unique()
    df = pd.DataFrame(index=ds.time, columns=columns)
    for column in columns:
        if boundname_column is None:
            mask = gdf["lakeno"] == column
        else:
            mask = gdf[boundname_column] == column
        cellids = gdf.index[mask]
        area = ds["area"].loc[cellids]
        if "time" in ds[variable].dims:
            da_cells = ds[variable].loc[:, cellids].copy()
            if set_to_0_in_ds:
                ds[variable][:, cellids] = 0.0
            # calculate thea area-weighted mean
            df[column] = (da_cells * area).sum("icell2d") / area.sum()
        else:
            da_cells = ds[variable].loc[cellids].copy()
            if set_to_0_in_ds:
                ds[variable][:, cellids] = 0.0
            # calculate thea area-weighted mean
            df[column] = float((da_cells * area).sum("icell2d") / area.sum())
    return df


def copy_meteorological_data_from_ds(
    gdf, ds, boundname_column=None, set_to_0_in_ds=False
):
    """
    Copy meteorlogical data from the model dataset, and return rainfall and evaporation.
    This method retrieves the values of rainfall and evaporation from a model Dataset.
    It uses the 'recharge'variable, and optionally the 'evaporation'-variable, and
    returns a rainfall- and evaporation-DataFrame. These dataframes contain input for
    each of the lakes. The columns of this DataFrame are either the boundnames (when
    boundname_column is specified) or the lake-number (lakeno).

    Parameters
    ----------
    gdf : gpd.GeoDataframe
        geodataframe with the cellids as the index
    ds : xr.Dataset
        dataset containing relevant model grid and time information
    boundname_column : str, optional
        The name of the column in gdf to use for the boundnames. When boundname_column
        is None, the lake-number (lakeno) is used to determine which rows in gdf belong
        to each lake, and the columns of rainfall and evaporation are set by the
        lake-number. If boundname_column is not None, the boundnames are used instead of
        the lake-number. The default is None.
    set_to_0_in_ds : bool, optional
        If True, sets the meteorological data to 0 in ds, which is not recommended. The
        default is False.

    Returns
    -------
    rainfall : pd.DataFrame
        The rainfall of each lake (columns) in time (index).
    evaporation : pd.DataFrame
        The evaporation of each lake (columns) in time (index).

    """
    logger.warning("clip_meteorological_data_from_ds ")
    if "evaporation" in ds:
        rainfall = _copy_da_from_ds(
            gdf,
            ds,
            "recharge",
            boundname_column=boundname_column,
            set_to_0_in_ds=set_to_0_in_ds,
        )
        evaporation = _copy_da_from_ds(
            gdf,
            ds,
            "evaporation",
            boundname_column=boundname_column,
            set_to_0_in_ds=set_to_0_in_ds,
        )
    else:
        recharge = _copy_da_from_ds(
            gdf,
            ds,
            "recharge",
            boundname_column=boundname_column,
            set_to_0_in_ds=set_to_0_in_ds,
        )
        rainfall = recharge.where(recharge > 0.0, 0.0)
        evaporation = -recharge.where(recharge < 0.0, 0.0)
    return rainfall, evaporation


def clip_meteorological_data_from_ds(
    gdf,
    ds,
    boundname_column=None,
):
    """
    Clip meteorlogical data from the model dataset, and return rainfall and evaporation.
    This method retrieves the values of rainfall and evaporation from a model Dataset.
    It uses the 'recharge'variable, and optionally the 'evaporation'-variable, and
    returns a rainfall- and evaporation-DataFrame. These dataframes contain input for
    each of the lakes. The columns of this DataFrame are either the boundnames (when
    boundname_column is specified) or the lake-number (lakeno).

    clip_meteorological_data_from_ds sets the meteorological data in the model detaset
    to 0. It turns out MODFLOW 6 already does this, and so this can create problemsm,
    see https://github.com/gwmod/nlmod/issues/497. Therefore, this method is deprected,
    and replaced by copy_meteorological_data_from_ds.

    Parameters
    ----------
    gdf : gpd.GeoDataframe
        geodataframe with the cellids as the index
    ds : xr.Dataset
        dataset containing relevant model grid and time information
    boundname_column : str, optional
        The name of the column in gdf to use for the boundnames. When boundname_column
        is None, the lake-number (lakeno) is used to determine which rows in gdf belong
        to each lake, and the columns of rainfall and evaporation are set by the
        lake-number. If boundname_column is not None, the boundnames are used instead of
        the lake-number. The default is None.

    Returns
    -------
    rainfall : pd.DataFrame
        The rainfall of each lake (columns) in time (index).
    evaporation : pd.DataFrame
        The evaporation of each lake (columns) in time (index).

    """
    logger.warning(
        "`clip_meteorological_data_from_ds` is deprecated and replaced by "
        "`copy_meteorological_data_from_ds`. See "
        "https://github.com/gwmod/nlmod/issues/497."
    )
    return copy_meteorological_data_from_ds(
        gdf, ds, boundname_column=boundname_column, set_to_0_in_ds=True
    )
