import logging

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd

from ..dims.grid import gdf_to_grid
from ..dims.layers import get_idomain
from ..util import tqdm
from .surface_water import build_spd

logger = logging.getLogger(__name__)

LINE_GEOM_TYPES = {"LineString", "MultiLineString"}
POLYGON_GEOM_TYPES = {"Polygon", "MultiPolygon"}
POINT_GEOM_TYPES = {"Point"}


def drain_from_df(
    df,
    gwf,
    ds,
    elev="elevation",
    cond="cond",
    conductance_per_length="conductance_per_meter",
    conductance_per_area="conductance_per_squared_meter",
    x="x",
    y="y",
    boundnames=None,
    mover_destinations=None,
    layer_method="lay_of_rbot",
    pname="drn",
    silent=False,
    return_provider_mapping=False,
    **kwargs,
):
    """Add a Drain (DRN) package based on input from a (Geo)DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        A (Geo)DataFrame containing the drain properties. Line and polygon
        geometries are intersected with the model grid and converted to
        conductance using ``conductance_per_length`` and ``conductance_per_area``.
        Point geometries and non-geometric data require ``cond`` to contain the
        integrated MF6 drain conductance.
    gwf : flopy ModflowGwf
        Groundwaterflow object to add the DRN package to.
    ds : xarray.Dataset
        Dataset with model data. Used for grid intersection and layer placement.
    elev : str, optional
        Column in ``df`` that contains the drain elevation. The default is
        "elevation".
    cond : str, optional
        Column in ``df`` that contains the integrated drain conductance. Required
        for point geometries and direct cellid input. The default is "cond".
    conductance_per_length : str, optional
        Column in ``df`` that contains conductance per metre for line geometries.
        The default is "conductance_per_meter".
    conductance_per_area : str, optional
        Column in ``df`` that contains conductance per square metre for polygon
        geometries. The default is "conductance_per_squared_meter".
    x : str, optional
        Column in ``df`` that contains the x-coordinate for point drains when
        ``df`` is not a GeoDataFrame and no ``cellid`` column is present. The
        default is "x".
    y : str, optional
        Column in ``df`` that contains the y-coordinate for point drains when
        ``df`` is not a GeoDataFrame and no ``cellid`` column is present. The
        default is "y".
    boundnames : str, optional
        Column in ``df`` that contains boundary names. These are written to the
        DRN package and included in the provider mapping. The default is None.
    mover_destinations : str, optional
        Column in ``df`` that identifies the intended MVR receiver for each drain.
        This function does not create MVR routes, but stores this column in the
        provider mapping. The default is None.
    layer_method : str, optional
        Method used by ``nlmod.gwf.surface_water.build_spd`` for layer placement.
        The default is "lay_of_rbot".
    pname : str, optional
        Package name. The default is "drn".
    silent : bool, optional
        Do not show progress bars when silent is True. The default is False.
    return_provider_mapping : bool, optional
        Return ``(drn, provider_mapping)`` when True. The mapping is always added
        to the returned package as ``drn.mvr_provider_mapping``. Provider IDs are
        zero-based FloPy numeric indices for use in MVR period data. The default
        is False.
    **kwargs : dict
        Kwargs are passed to ``flopy.mf6.ModflowGwfdrn``. Use ``mover=True`` to
        make the drain package available to an MVR package.

    Returns
    -------
    drn : flopy.mf6.ModflowGwfdrn
        DRN package. When ``return_provider_mapping`` is True, returns a tuple of
        ``(drn, provider_mapping)``.

    Notes
    -----
    This function overlaps with ``nlmod.gwf.surface_water.gdf_to_seasonal_pkg``
    for polygon-to-DRN conversion. Use ``gdf_to_seasonal_pkg`` for surface-water
    polygons with winter and summer stages and seasonal conductance timeseries.
    Use ``drain_from_df`` for fixed drain features such as pipes, basins, point
    drains, or direct cellid input, and when deterministic MVR provider IDs are
    needed.

    For vector geometries and 2D cellids, layer placement is delegated to
    ``nlmod.gwf.surface_water.build_spd``. That helper uses
    ``nlmod.dims.layers.get_idomain`` to skip columns without active cells and to
    place the drain in a suitable active layer (``idomain > 0``), not in inactive
    (``idomain == 0``) or vertical pass-through (``idomain < 0``) cells. FloPy
    receives explicit 3D DRN cellids and does not relocate boundaries. Therefore
    explicit 3D cellids passed to this function that target ``idomain <= 0`` are
    remapped to the nearest active layer in the same vertical column, based on the
    drain elevation. If no active layer exists, or if remapping is required for a
    nonnumeric drain elevation, a ``ValueError`` is raised.
    """
    logger.info("creating mf6 DRN from dataframe")

    celldata = _drain_celldata_from_df(
        df,
        gwf=gwf,
        ds=ds,
        elev=elev,
        cond=cond,
        conductance_per_length=conductance_per_length,
        conductance_per_area=conductance_per_area,
        x=x,
        y=y,
        boundnames=boundnames,
        mover_destinations=mover_destinations,
        silent=silent,
    )
    spd, provider_mapping = _build_spd_with_provider_mapping(
        celldata,
        ds=ds,
        layer_method=layer_method,
        silent=silent,
    )

    if len(spd) == 0:
        logger.warning("no drn pkg added")
        if return_provider_mapping:
            return None, provider_mapping
        return None

    save_flows = kwargs.pop("save_flows", True)
    drn = flopy.mf6.ModflowGwfdrn(
        gwf,
        maxbound=len(spd),
        stress_period_data={0: spd},
        save_flows=save_flows,
        boundnames=boundnames is not None,
        pname=pname,
        **kwargs,
    )

    provider_mapping["package"] = drn.package_name
    drn.mvr_provider_mapping = provider_mapping
    if return_provider_mapping:
        return drn, provider_mapping
    return drn


def mvr_perioddata_from_provider_mapping(
    provider_mapping,
    receiver_package,
    receiver_id_map,
    receiver_column="mover_destination",
    provider_package=None,
    mvrtype="FACTOR",
    value=1.0,
):
    """Build MVR perioddata for drain outflow using a provider mapping.

    Parameters
    ----------
    provider_mapping : pd.DataFrame
        Provider mapping returned by ``drain_from_df``. Must contain
        ``mvr_provider_id`` and the receiver column.
    receiver_package : str
        Name of the receiving package in the MVR package.
    receiver_id_map : dict or pd.Series
        Mapping from values in ``receiver_column`` to zero-based receiver IDs.
    receiver_column : str, optional
        Column in ``provider_mapping`` that identifies the receiver. The default is
        "mover_destination".
    provider_package : str, optional
        Name of the provider package. When None, the ``package`` column in
        ``provider_mapping`` is used. The default is None.
    mvrtype : str, optional
        MVR rule type. The default is "FACTOR".
    value : float, optional
        MVR rule value. The default is 1.0.

    Returns
    -------
    perioddata : list
        Perioddata records that can be passed to ``flopy.mf6.ModflowGwfmvr``.
    """
    _validate_columns(
        provider_mapping, ["mvr_provider_id", receiver_column], "provider_mapping"
    )
    if provider_package is None:
        _validate_columns(provider_mapping, ["package"], "provider_mapping")

    perioddata = []
    for _, row in provider_mapping.dropna(subset=[receiver_column]).iterrows():
        receiver_name = row[receiver_column]
        if receiver_name not in receiver_id_map:
            raise KeyError(f"Receiver {receiver_name!r} not found in receiver_id_map")
        perioddata.append(
            (
                row["package"] if provider_package is None else provider_package,
                row["mvr_provider_id"],
                receiver_package,
                receiver_id_map[receiver_name],
                mvrtype,
                value,
            )
        )
    return perioddata


def _drain_celldata_from_df(
    df,
    gwf,
    ds,
    elev,
    cond,
    conductance_per_length,
    conductance_per_area,
    x,
    y,
    boundnames,
    mover_destinations,
    silent,
):
    if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("df must be a pandas DataFrame or geopandas GeoDataFrame")

    _validate_columns(df, [elev], "df")
    if boundnames is not None:
        _validate_columns(df, [boundnames], "df")
    if mover_destinations is not None:
        _validate_columns(df, [mover_destinations], "df")

    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
    elif "cellid" not in df.columns and {x, y}.issubset(df.columns):
        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[x], df[y]))
    else:
        return _cellid_celldata_from_df(
            df.copy(),
            elev=elev,
            cond=cond,
            boundnames=boundnames,
            mover_destinations=mover_destinations,
        )

    gdf["_source_index"] = gdf.index.to_numpy()
    parts = []
    for geom_types, conductance_column, measure in (
        (LINE_GEOM_TYPES, conductance_per_length, "length"),
        (POLYGON_GEOM_TYPES, conductance_per_area, "area"),
        (POINT_GEOM_TYPES, cond, None),
    ):
        subset = gdf[gdf.geom_type.isin(geom_types)]
        if subset.empty:
            continue
        if measure is None:
            _validate_columns(subset, [cond], "df")
        else:
            _validate_columns(subset, [conductance_column], "df")
        parts.append(
            _geodataframe_celldata(
                subset,
                gwf=gwf,
                ds=ds,
                elev=elev,
                cond=cond,
                conductance_column=conductance_column,
                measure=measure,
                boundnames=boundnames,
                mover_destinations=mover_destinations,
                silent=silent,
            )
        )

    unsupported_geom_types = set(gdf.geom_type.unique()).difference(
        LINE_GEOM_TYPES | POLYGON_GEOM_TYPES | POINT_GEOM_TYPES
    )
    if unsupported_geom_types:
        raise TypeError(f"Unsupported drain geometry types: {unsupported_geom_types}")

    if len(parts) == 0:
        return pd.DataFrame(
            columns=[
                "stage",
                "rbot",
                "cond",
                "area",
                "len_estimate",
                "source_index",
            ]
        )
    return pd.concat(parts, axis=0)


def _geodataframe_celldata(
    gdf,
    gwf,
    ds,
    elev,
    cond,
    conductance_column,
    measure,
    boundnames,
    mover_destinations,
    silent,
):
    if "cellid" not in gdf.columns:
        gdf = gdf_to_grid(gdf, ds if ds is not None else gwf, silent=silent)

    celldata = _base_celldata(gdf, elev, boundnames, mover_destinations)
    if measure == "length":
        celldata["cond"] = gdf.geometry.length.to_numpy() * gdf[conductance_column]
        celldata["len_estimate"] = gdf.geometry.length.to_numpy()
        celldata["area"] = np.nan
    elif measure == "area":
        celldata["cond"] = gdf.geometry.area.to_numpy() * gdf[conductance_column]
        celldata["area"] = gdf.geometry.area.to_numpy()
        celldata["len_estimate"] = np.nan
    elif measure is None:
        celldata["cond"] = gdf[cond].to_numpy()
        celldata["area"] = np.nan
        celldata["len_estimate"] = np.nan
    else:
        raise ValueError(f"Unknown measure: {measure}")
    return celldata.set_index("cellid")


def _cellid_celldata_from_df(df, elev, cond, boundnames, mover_destinations):
    _validate_columns(df, ["cellid", cond], "df")
    celldata = _base_celldata(df, elev, boundnames, mover_destinations)
    celldata["cond"] = df[cond].to_numpy()
    celldata["area"] = np.nan
    celldata["len_estimate"] = np.nan
    return celldata.set_index("cellid")


def _base_celldata(df, elev, boundnames, mover_destinations):
    celldata = pd.DataFrame(index=df.index)
    celldata["cellid"] = df["cellid"].to_numpy()
    celldata["stage"] = df[elev].to_numpy()
    celldata["rbot"] = df[elev].to_numpy()
    celldata["source_index"] = (
        df["_source_index"].to_numpy()
        if "_source_index" in df.columns
        else df.index.to_numpy()
    )
    if boundnames is not None:
        celldata["boundname"] = df[boundnames].to_numpy()
    if mover_destinations is not None:
        celldata["mover_destination"] = df[mover_destinations].to_numpy()
    return celldata


def _build_spd_with_provider_mapping(celldata, ds, layer_method, silent):
    spd = []
    provider_mapping = []
    idomain = None
    for cellid, row in tqdm(
        celldata.iterrows(),
        total=celldata.index.size,
        desc="Building stress period data DRN",
        disable=silent,
    ):
        if _is_3d_cellid(cellid, ds):
            if idomain is None:
                idomain = get_idomain(ds)
            row_spd = [_record_from_3d_cellid(cellid, row, ds, idomain)]
        else:
            index = np.empty(1, dtype=object)
            index[0] = cellid
            row_df = pd.DataFrame(
                [row],
                index=pd.Index(index, name=celldata.index.name),
            )
            row_spd = build_spd(
                row_df,
                "DRN",
                ds,
                layer_method=layer_method,
                silent=True,
            )
        for record in row_spd:
            provider_mapping.append(
                {
                    "mvr_provider_id": len(spd),
                    "cellid": record[0],
                    "elev": record[1],
                    "cond": record[2],
                    "source_index": row["source_index"],
                    "boundname": row.get("boundname"),
                    "mover_destination": row.get("mover_destination"),
                }
            )
            spd.append(record)
    return spd, pd.DataFrame(provider_mapping)


def _is_3d_cellid(cellid, ds):
    if not isinstance(cellid, tuple):
        return False
    if ds.gridtype == "vertex":
        return len(cellid) == 2
    if ds.gridtype == "structured":
        return len(cellid) == 3
    raise ValueError(f"Unsupported gridtype: {ds.gridtype}")


def _record_from_3d_cellid(cellid, row, ds, idomain):
    cellid = _get_active_cellid_for_3d_cellid(cellid, row, ds, idomain)

    if np.isnan(row["cond"]):
        raise ValueError(f"Conductance is NaN in cell {cellid}")
    if row["cond"] < 0:
        raise ValueError(f"Conductance is negative in cell {cellid}")

    auxlist = []
    if "boundname" in row:
        auxlist.append(row["boundname"])
    return [cellid, row["stage"], row["cond"]] + auxlist


def _get_active_cellid_for_3d_cellid(cellid, row, ds, idomain):
    if idomain.data[cellid] > 0:
        return cellid

    try:
        drain_elevation = float(row["stage"])
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Cannot remap DRN cellid {cellid} to the nearest active layer "
            "because the drain elevation is not numeric."
        ) from err
    if not np.isfinite(drain_elevation):
        raise ValueError(
            f"Cannot remap DRN cellid {cellid} to the nearest active layer "
            "because the drain elevation is not finite."
        )

    column_cellid = cellid[1:]
    column_indexer = dict(zip(ds["botm"].dims[1:], column_cellid, strict=True))
    idomain_column = idomain.isel(column_indexer).data
    layer_botms = ds["botm"].isel(column_indexer).data
    top_indexer = {
        dim: index for dim, index in column_indexer.items() if dim in ds["top"].dims
    }
    if "layer" in ds["top"].dims:
        layer_tops = ds["top"].isel(top_indexer).data
    else:
        layer_tops = np.r_[ds["top"].isel(top_indexer).data, layer_botms[:-1]]

    active_layers = np.where(idomain_column > 0)[0]
    if len(active_layers) == 0:
        raise ValueError(
            f"Cannot remap DRN cellid {cellid}; the vertical column has no active "
            "layers."
        )

    nearest_layer = min(
        active_layers,
        key=lambda active_layer: (
            _distance_to_layer_interval(
                drain_elevation,
                layer_top=layer_tops[active_layer],
                layer_botm=layer_botms[active_layer],
            ),
            abs(active_layer - cellid[0]),
            active_layer,
        ),
    )
    return (nearest_layer,) + column_cellid


def _distance_to_layer_interval(elevation, layer_top, layer_botm):
    upper = max(layer_top, layer_botm)
    lower = min(layer_top, layer_botm)
    return abs(elevation - np.clip(elevation, lower, upper))


def _validate_columns(df, columns, name):
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
