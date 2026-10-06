import logging

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd

from ..dims.grid import gdf_to_grid
from .surface_water import build_spd

logger = logging.getLogger(__name__)

__all__ = ["drain_from_df"]

LINE_GEOM_TYPES = {"LineString", "MultiLineString"}
POLYGON_GEOM_TYPES = {"Polygon", "MultiPolygon"}
POINT_GEOM_TYPES = {"Point"}


def drain_from_df(
    df,
    gwf,
    ds,
    *,
    elev="elevation",
    cond="cond",
    conductance_per_length="conductance_per_meter",
    conductance_per_area="conductance_per_squared_meter",
    x="x",
    y="y",
    boundnames=None,
    pname="drn",
    save_flows=True,
    silent=False,
    **kwargs,
):
    """Add a Drain (DRN) package based on input from a (Geo)DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        A (Geo)DataFrame containing the drain properties. Line and polygon
        geometries are intersected with the model grid and converted to
        conductance using ``conductance_per_length`` and ``conductance_per_area``.
        Point geometries, or a DataFrame with ``x`` and ``y`` columns, require
        ``cond`` to contain the integrated MF6 drain conductance.
    gwf : flopy ModflowGwf
        Groundwaterflow object to add the DRN package to.
    ds : xarray.Dataset
        Dataset with model data. Used for grid intersection and layer placement.
    elev : str, optional
        Column in ``df`` that contains the drain elevation. The default is
        "elevation".
    cond : str, optional
        Column in ``df`` that contains the integrated drain conductance. Required
        for point geometries. The default is "cond".
    conductance_per_length : str, optional
        Column in ``df`` that contains conductance per metre for line geometries.
        The default is "conductance_per_meter".
    conductance_per_area : str, optional
        Column in ``df`` that contains conductance per square metre for polygon
        geometries. The default is "conductance_per_squared_meter".
    x : str, optional
        Column in ``df`` that contains the x-coordinate for point drains when
        ``df`` is not a GeoDataFrame. The default is "x".
    y : str, optional
        Column in ``df`` that contains the y-coordinate for point drains when
        ``df`` is not a GeoDataFrame. The default is "y".
    boundnames : str, optional
        Column in ``df`` that contains boundary names. These are written to the
        DRN package. The default is None.
    pname : str, optional
        Package name. The default is "drn".
    save_flows : bool, optional
        Save the drain flows to the budget file. The default is True.
    silent : bool, optional
        Do not show progress bars when silent is True. The default is False.
    **kwargs : dict
        Kwargs are passed to ``flopy.mf6.ModflowGwfdrn``.

    Returns
    -------
    drn : flopy.mf6.ModflowGwfdrn or None
        DRN package. Returns None when no stress-period data are generated.

    Notes
    -----
    This function overlaps with ``nlmod.gwf.surface_water.gdf_to_seasonal_pkg``
    for polygon-to-DRN conversion. Use ``gdf_to_seasonal_pkg`` for surface-water
    polygons with winter and summer stages and seasonal conductance timeseries.
    Use ``drain_from_df`` for fixed drain features such as pipes, basins, or point
    drains.

    Each feature keeps its own elevation, also when several features share a cell.
    Layer placement is delegated to ``nlmod.gwf.surface_water.build_spd``, which
    skips columns without active cells and places each drain in the active layer
    that contains its elevation.
    """
    logger.info("creating mf6 DRN from dataframe")

    if not isinstance(df, gpd.GeoDataFrame):
        if not {x, y}.issubset(df.columns):
            raise ValueError(
                f"df must be a GeoDataFrame or contain the columns {x!r} and {y!r}"
            )
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))

    geom_type = df.geom_type
    is_line = geom_type.isin(LINE_GEOM_TYPES)
    is_polygon = geom_type.isin(POLYGON_GEOM_TYPES)
    is_point = geom_type.isin(POINT_GEOM_TYPES)
    unsupported = set(geom_type[~(is_line | is_polygon | is_point)])
    if unsupported:
        raise TypeError(f"Unsupported drain geometry types: {unsupported}")

    required = {elev}
    if boundnames is not None:
        required.add(boundnames)
    if is_line.any():
        required.add(conductance_per_length)
    if is_polygon.any():
        required.add(conductance_per_area)
    if is_point.any():
        required.add(cond)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    if df.empty:
        logger.warning("no drn pkg added")
        return None

    # conductance per unit of measure; the measure follows from the clipped geometry
    df = df.assign(
        _measure=np.select([is_line, is_polygon], ["length", "area"], "point"),
        _cond=np.select(
            [is_line, is_polygon],
            [
                df.get(conductance_per_length, np.nan),
                df.get(conductance_per_area, np.nan),
            ],
            default=df.get(cond, np.nan),
        ),
    )
    gdf = gdf_to_grid(df, ds, silent=silent)
    measure = np.select(
        [gdf["_measure"] == "length", gdf["_measure"] == "area"],
        [gdf.geometry.length, gdf.geometry.area],
        default=1.0,
    )
    celldata = pd.DataFrame(
        {
            "cellid": gdf["cellid"].to_numpy(),
            "stage": gdf[elev].to_numpy(),
            # build_spd places the drain in the layer of rbot
            "rbot": gdf[elev].to_numpy(),
            "cond": measure * gdf["_cond"].to_numpy(),
            "area": gdf.geometry.area.to_numpy(),
        }
    ).set_index("cellid")
    if boundnames is not None:
        celldata["boundname"] = gdf[boundnames].to_numpy()

    spd = build_spd(celldata, "DRN", ds, silent=silent)
    if len(spd) == 0:
        logger.warning("no drn pkg added")
        return None

    return flopy.mf6.ModflowGwfdrn(
        gwf,
        maxbound=len(spd),
        stress_period_data={0: spd},
        save_flows=save_flows,
        boundnames=boundnames is not None,
        pname=pname,
        **kwargs,
    )
