import datetime as dt
import logging
import os
import warnings
from typing import Callable, Literal, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from rioxarray.merge import merge_arrays
from tqdm import tqdm

from nlmod import NLMOD_DATADIR, cache, dims, util
from nlmod.read.webservices import arcrest

logger = logging.getLogger(__name__)


@cache.cache_pickle
def get_gdf_surface_water(ds=None, extent=None):
    """Read a shapefile with surface water as a geodataframe, cut by the extent of the
    model.

    Parameters
    ----------
    ds : xr.DataSet, None, optional
        dataset containing relevant model information
    extent : list, tuple or np.array, optional
        desired model extent (xmin, xmax, ymin, ymax)

    Returns
    -------
    gdf_opp_water : GeoDataframe
        surface water geodataframe.
    """
    if ds is None and extent is None:
        raise ValueError("At least one of 'ds' or 'extent' must be provided.")

    # laad bestanden in
    fname = os.path.join(NLMOD_DATADIR, "shapes", "opp_water.shp")
    gdf_swater = gpd.read_file(fname)
    if ds is not None:
        extent = dims.get_extent(ds)

    if (extent[0] < 93000) & (extent[2] < 480000):
        logger.warning(
            "This function does not yield good results for the North Sea, see https://github.com/gwmod/nlmod/issues/225"
        )

    gdf_swater = util.gdf_within_extent(gdf_swater, extent)

    return gdf_swater


@cache.cache_netcdf(coords_3d=True)
def get_surface_water(ds, gdf=None, da_basename="rws_oppwater"):
    """Create 3 data-arrays from the shapefile with surface water:

    .. deprecated:: 0.10.0
        `get_surface_water` will be removed in nlmod 1.0.0, it is replaced by
        `discretize_surface_water` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    - area: area of the shape in the cell
    - cond: conductance based on the area and "bweerstand" column in shapefile
    - stage: surface water level based on the "peil" column in the shapefile

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    gdf : gpd.GeoDataFrame or None, optional
        geometries of the surface water, if None the geometries are obtained using
        get_gdf_surface_water. The default is None.
    da_basename : str
        name of the polygon shapes, name is used as a prefix
        to store data arrays in ds

    Returns
    -------
    ds : xarray.Dataset
        dataset with modelgrid data.
    """

    warnings.warn(
        "'get_surface_water' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.rws.discretize_surface_water' to project the surface water on the model grid",
        DeprecationWarning,
    )

    if gdf is None:
        gdf = get_gdf_surface_water(ds)

    return discretize_surface_water(ds, gdf, da_basename)


@cache.cache_netcdf(coords_3d=True)
def discretize_surface_water(ds, gdf, da_basename="rws_oppwater"):
    """Create 3 data-arrays from the shapefile with surface water:

    - area: area of the shape in the cell
    - cond: conductance based on the area and "bweerstand" column in shapefile
    - stage: surface water level based on the "peil" column in the shapefile

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    gdf : gpd.GeoDataFrame or None, optional
        geometries of the surface water and the columns 'bweerstand' and 'peil'.
    da_basename : str
        name of the polygon shapes, name is used as a prefix
        to store data arrays in ds

    Returns
    -------
    ds : xarray.Dataset
        dataset with modelgrid data.
    """
    modelgrid = dims.modelgrid_from_ds(ds)

    area = xr.zeros_like(ds["top"])
    cond = xr.zeros_like(ds["top"])
    peil = xr.zeros_like(ds["top"])
    for _, row in gdf.iterrows():
        area_pol = dims.polygon_to_area(
            modelgrid,
            row["geometry"],
            xr.ones_like(ds["top"]),
            ds.gridtype,
        )
        cond = xr.where(area_pol > area, area_pol / row["bweerstand"], cond)
        peil = xr.where(area_pol > area, row["peil"], peil)
        area = xr.where(area_pol > area, area_pol, area)

    ds_out = util.get_ds_empty(ds, keep_coords=("y", "x"))
    ds_out[f"{da_basename}_area"] = area
    ds_out[f"{da_basename}_area"].attrs["units"] = "m2"
    ds_out[f"{da_basename}_cond"] = cond
    ds_out[f"{da_basename}_cond"].attrs["units"] = "m2/day"
    ds_out[f"{da_basename}_stage"] = peil
    ds_out[f"{da_basename}_stage"].attrs["units"] = "mNAP"

    for datavar in ds_out:
        ds_out[datavar].attrs["source"] = "RWS"
        ds_out[datavar].attrs["date"] = dt.datetime.now().strftime("%Y%m%d")

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def get_northsea(ds, gdf=None, da_name="northsea"):
    """Get Dataset which is 1 at the northsea and 0 everywhere else. Sea is defined by
    rws surface water shapefile.

    .. deprecated:: 0.10.0
        `get_northsea` will be removed in nlmod 1.0.0, it is replaced by
        `discretize_northsea` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    ds : xr.DataSet, None, optional
        dataset containing relevant model information
    gdf : gpd.GeoDataFrame or None, optional
        geometries of the surface water, if None the geometries are obtained using
        get_gdf_surface_water. The default is None.
    da_name : str, optional
        name of the datavar that identifies sea cells

    Returns
    -------
    ds_out : xr.DataSet
        Dataset with a single DataArray, this DataArray is 1 at sea and 0
        everywhere else. Grid dimensions according to ds.
    """
    if gdf is None:
        gdf = get_gdf_surface_water(ds=ds)

    warnings.warn(
        "'get_northsea' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.rws.discretize_northsea' to project the northsea on the model grid",
        DeprecationWarning,
    )

    return discretize_northsea(ds, gdf, da_name)


@cache.cache_netcdf(coords_2d=True)
def discretize_northsea(ds, gdf, da_name="northsea"):
    """Get Dataset which is 1 at the northsea and 0 everywhere else. Sea is defined by
    rws surface water shapefile.

    Parameters
    ----------
    ds : xr.DataSet, None, optional
        dataset containing relevant model information
    gdf : gpd.GeoDataFrame or None, optional
        geometries of the surface water, if None the geometries are obtained using
        get_gdf_surface_water. The default is None.
    da_name : str, optional
        name of the datavar that identifies sea cells

    Returns
    -------
    ds_out : xr.DataSet
        Dataset with a single DataArray, this DataArray is 1 at sea and 0
        everywhere else. Grid dimensions according to ds.
    """
    if gdf is None:
        gdf = get_gdf_surface_water(ds=ds)

    # find grid cells with sea
    swater_zee = gdf[
        gdf["OWMNAAM"].isin(
            [
                "Rijn territoriaal water",
                "Waddenzee",
                "Waddenzee vastelandskust",
                "Hollandse kust (kustwater)",
                "Waddenkust (kustwater)",
            ]
        )
    ]

    ds_out = dims.gdf_to_bool_ds(swater_zee, ds, da_name, keep_coords=("y", "x"))

    return ds_out


def calculate_sea_coverage(
    dtm,
    ds=None,
    zmax=0.0,
    xy_sea=None,
    diagonal=False,
    method="mode",
    nodata=-1,
    return_filled_dtm=False,
):
    """Determine where the sea is by interpreting the digital terrain model.

    This method assumes the pixel defined in xy_sea (by default top-left) of the
    DTM-DataArray is sea. It then determines the height of the sea that is required for
    other pixels to become sea as well, taking into account the pixels in between.

    Parameters
    ----------
    dtm : xr.DataArray
        The digital terrain data, which can be of higher resolution than ds, Nans are
        filled by the minial value of dtm.
    ds : xr.Dataset, optional
        Dataset with model information. When ds is not None, the sea DataArray is
        transformed to the model grid. THe default is None.
    zmax : float, optional
        Locations thet become sea when the sea level reaches a level of zmax will get a
        value of 1 in the resulting DataArray. The default is 0.0.
    xy_sea : tuble of 2 floats
        The x- and y-coordinate of a location within the dtm that is sea. From this
        point, calculate_sea determines at what level each cell becomes wet. When
        xy_cell is None, the most northwest grid cell is sea, which is appropriate for
        the Netherlands. The default is None.
    diagonal : bool, optional
        When true, dtm-values are connected diagonally as well (to determine the level
        the sea will reach). The default is False.
    method : str, optional
        The method used to scale the dtm to ds. The default is "mode" (mode means that
        if more than half of the (not-nan) cells are wet, the cell is classified as
        sea).
    nodata : int or float, optional
        The value for model cells outside the coverage of the dtm.
        Only used internally. The default is -1.
    return_filled_dtm : bool, optional
        When True, return the filled dtm. The default is False.

    Returns
    -------
    sea : xr.DataArray
        A DataArray with value of 1 where the sea is and 0 where it is not.
    """
    from skimage.morphology import reconstruction

    if not (dtm < zmax).any():
        logger.warning(
            f"There are no values in dtm below {zmax}. The provided dtm "
            "probably is not appropriate to calculate the sea boundary."
        )
    # fill nans by the minimum value of dtm
    dtm = dtm.where(~np.isnan(dtm), dtm.min())
    seed = xr.full_like(dtm, dtm.max())
    if xy_sea is None:
        xy_sea = (dtm.x.data.min(), dtm.y.data.max())
    # determine the closest x and y in the dtm grid
    x_sea = dtm.x.sel(x=xy_sea[0], method="nearest")
    y_sea = dtm.y.sel(y=xy_sea[1], method="nearest")
    dtm.loc[{"x": x_sea, "y": y_sea}] = dtm.min()
    seed.loc[{"x": x_sea, "y": y_sea}] = dtm.min()
    seed = seed.data

    footprint = np.ones((3, 3), dtype="bool")
    if not diagonal:
        footprint[[0, 0, 2, 2], [0, 2, 2, 0]] = False  # no diagonal connections
    filled = reconstruction(seed, dtm.data, method="erosion", footprint=footprint)
    dtm.data = filled
    if return_filled_dtm:
        return dtm

    sea_dtm = dtm < zmax
    if method == "mode":
        sea_dtm = sea_dtm.astype(int)
    else:
        sea_dtm = sea_dtm.astype(float)
    if ds is not None:
        sea = dims.structured_da_to_ds(sea_dtm, ds, method=method, nodata=nodata)
        if (sea == nodata).any():
            logger.info(
                "The dtm data does not cover the entire model domain."
                " Assuming cells outside dtm-cover to be sea."
            )
            sea = sea.where(sea != nodata, 1)
        return sea
    return sea_dtm


def get_gdr_configuration() -> dict:
    """Get configuration for GDR data.

    Note: Currently only includes configuration for bathymetry data. Other datasets
    can be added in the future. See
    https://geo.rijkswaterstaat.nl/arcgis/rest/services/GDR/ for available data.

    Returns
    -------
    config : dict
        configuration dictionary containing urls and layer numbers for GDR data.
    """
    config = {}
    config["bodemhoogte"] = {
        "index": {
            "url": (
                "https://geo.rijkswaterstaat.nl/arcgis/rest/services/GDR/"
                "bodemhoogte_index/FeatureServer"
            )
        },
        "20m": {"layer": 0},
        "1m": {"layer": 2},
    }
    return config


def get_bathymetry_gdf(
    resolution: Literal["20m", "1m"] = "20m",
    extent: Optional[list[float]] = None,
    config: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """Get bathymetry dataframe from RWS.

    .. deprecated:: 0.10.0
        `get_bathymetry_gdf` will be removed in nlmod 1.0.0, it is replaced by
        `download_bathymetry_gdf` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Note that the 20m resolution does not contain bathymetry data for the major rivers.
    If you need the bathymetry of the major rivers, use the 1m resolution.

    Parameters
    ----------
    resolution : str, optional
        resolution of the bathymetry data, "1m" or "20m". The default is "20m".
    extent : tuple, optional
        extent of the model domain. The default is None.
    config : dict, optional
        configuration dictionary containing urls and layer numbers for GDR data. The
        default is None, which uses the default configuration provided by
        the function `get_gdr_configuration()`.
    """
    warnings.warn(
        "'get_bathymetry_gdf' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.rws.download_bathymetry_gdf' to project the buisdrainage on the model grid",
        DeprecationWarning,
    )

    return download_bathymetry_gdf(resolution, extent, config)


def download_bathymetry_gdf(
    resolution: Literal["20m", "1m"] = "20m",
    extent: Optional[list[float]] = None,
    config: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """Get bathymetry dataframe from RWS.

    Note that the 20m resolution does not contain bathymetry data for the major rivers.
    If you need the bathymetry of the major rivers, use the 1m resolution.

    Parameters
    ----------
    resolution : str, optional
        resolution of the bathymetry data, "1m" or "20m". The default is "20m".
    extent : tuple, optional
        extent of the model domain. The default is None.
    config : dict, optional
        configuration dictionary containing urls and layer numbers for GDR data. The
        default is None, which uses the default configuration provided by
        the function `get_gdr_configuration()`.
    """
    if config is None:
        config = get_gdr_configuration()
    url = config["bodemhoogte"]["index"]["url"]
    layer = config["bodemhoogte"][resolution]["layer"]
    return arcrest(url, layer, extent=extent)


@cache.cache_netcdf()
def get_bathymetry(
    extent: list[float],
    resolution: Literal["20m", "1m"] = "20m",
    res: Optional[float] = None,
    method: Union[str, Callable, None] = None,
    chunks: Optional[Union[str, dict[str, int]]] = "auto",
    config: Optional[dict] = None,
) -> xr.DataArray:
    """Get bathymetry data from RWS.

    .. deprecated:: 0.10.0
        `get_bathymetry` will be removed in nlmod 1.0.0, it is replaced by
        `download_bathymetry` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Bathymetry is available at 20m resolution and at 1m resolution. The 20m
    resolution is available for large water bodies, but not in the major rivers.
    The 1m dataset covers the major waterbodies across all of the Netherlands.

    Parameters
    ----------
    extent : tuple
        extent of the model domain
    resolution : str, optional
        resolution of the bathymetry data, "1m" or "20m". The default is "20m".
    res : float, optional
        resolution of the output data array. The default is None, which uses
        resolution of the input datasets. Resampling method is provided by the method
        kwarg.
    method : str, optional
        Rasterio resampling method. The default is None. Pre-defined method are
        "first", "last", "min", "nearest", "sum" or "count". But custom callables
        are also supported. See the rasterio documentation for more information.
    chunks : dict, optional
        chunks for the output data array. The default is "auto", which lets xarray/dask
        pick the chunksize. Set to None to avoid chunking.
    config : dict, optional
        configuration dictionary containing urls and layer numbers for GDR data. The
        default is None, which uses the default configuration provided by
        the function `get_gdr_configuration()`.

    Returns
    -------
    bathymetry : xr.DataArray
        bathymetry data
    """
    warnings.warn(
        "'get_bathymetry' is deprecated and will be removed in a future version. "
        "Use 'nlmod.read.rws.download_bathymetry' to download bathymetry data",
        DeprecationWarning,
    )

    return download_bathymetry(extent, resolution, res, method, chunks, config)


@cache.cache_netcdf()
def download_bathymetry(
    extent: list[float],
    resolution: Literal["20m", "1m"] = "20m",
    res: Optional[float] = None,
    method: Union[str, Callable, None] = None,
    chunks: Optional[Union[str, dict[str, int]]] = "auto",
    config: Optional[dict] = None,
) -> xr.DataArray:
    """Download bathymetry data from RWS.

    Bathymetry is available at 20m resolution and at 1m resolution. The 20m
    resolution is available for large water bodies, but not in the major rivers.
    The 1m dataset covers the major waterbodies across all of the Netherlands.

    Parameters
    ----------
    extent : tuple
        extent of the model domain
    resolution : str, optional
        resolution of the bathymetry data, "1m" or "20m". The default is "20m".
    res : float, optional
        resolution of the output data array. The default is None, which uses
        resolution of the input datasets. Resampling method is provided by the method
        kwarg.
    method : str, optional
        Rasterio resampling method. The default is None. Pre-defined method are
        "first", "last", "min", "nearest", "sum" or "count". But custom callables
        are also supported. See the rasterio documentation for more information.
    chunks : dict, optional
        chunks for the output data array. The default is "auto", which lets xarray/dask
        pick the chunksize. Set to None to avoid chunking.
    config : dict, optional
        configuration dictionary containing urls and layer numbers for GDR data. The
        default is None, which uses the default configuration provided by
        the function `get_gdr_configuration()`.

    Returns
    -------
    bathymetry : xr.DataArray
        bathymetry data
    """
    gdf = download_bathymetry_gdf(resolution=resolution, extent=extent, config=config)

    xmin, xmax, ymin, ymax = extent
    dataarrays = []

    for _, row in tqdm(
        gdf.iterrows(), desc="Downloading bathymetry", total=gdf.index.size
    ):
        url = row["geotiff"]
        # NOTE: link to 20m dataset is incorrect in the index
        if resolution == "20m":
            url = url.replace("Noordzee_20_LAT", "bodemhoogte_20mtr")
        ds = xr.open_dataset(url, engine="rasterio")
        ds = ds.assign_coords({"y": ds["y"].round(0), "x": ds["x"].round(0)})
        da = (
            ds["band_data"]
            .sel(band=1, x=slice(xmin, xmax), y=slice(ymax, ymin))
            .drop_vars("band")
        )
        if chunks:
            da = da.chunk(chunks)
        dataarrays.append(da)

    if len(dataarrays) > 1:
        da = merge_arrays(
            dataarrays,
            bounds=[xmin, ymin, xmax, ymax],
            res=res,
            method=method,
        )
    else:
        da = dataarrays[0]
        if res is not None:
            da = da.rio.reproject(
                da.rio.crs,
                res=res,
                resampling=method,
            )
    return da
