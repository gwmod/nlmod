import datetime as dt
import logging
import os
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from rioxarray.merge import merge_arrays
from tqdm.auto import tqdm

from nlmod import NLMOD_DATADIR, cache, dims, util
from nlmod.read import jarkus
from nlmod.read.webservices import arcrest

logger = logging.getLogger(__name__)


def get_gdf_surface_water(ds):
    """Read a shapefile with surface water as a geodataframe, cut by the extent of the
    model.

    Parameters
    ----------
    ds : xr.DataSet
        dataset containing relevant model information

    Returns
    -------
    gdf_opp_water : GeoDataframe
        surface water geodataframe.
    """
    # laad bestanden in
    fname = os.path.join(NLMOD_DATADIR, "shapes", "opp_water.shp")
    gdf_swater = gpd.read_file(fname)
    extent = dims.get_extent(ds)
    gdf_swater = util.gdf_within_extent(gdf_swater, extent)

    return gdf_swater


@cache.cache_netcdf(coords_3d=True)
def get_surface_water(ds, da_basename):
    """Create 3 data-arrays from the shapefile with surface water:

    - area: area of the shape in the cell
    - cond: conductance based on the area and "bweerstand" column in shapefile
    - stage: surface water level based on the "peil" column in the shapefile

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    da_basename : str
        name of the polygon shapes, name is used as a prefix
        to store data arrays in ds

    Returns
    -------
    ds : xarray.Dataset
        dataset with modelgrid data.
    """
    modelgrid = dims.modelgrid_from_ds(ds)
    gdf = get_gdf_surface_water(ds)

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
def get_northsea(ds, da_name="northsea"):
    """Get Dataset which is 1 at the northsea and 0 everywhere else. Sea is defined by
    rws surface water shapefile.

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    da_name : str, optional
        name of the datavar that identifies sea cells

    Returns
    -------
    ds_out : xr.DataSet
        Dataset with a single DataArray, this DataArray is 1 at sea and 0
        everywhere else. Grid dimensions according to ds.
    """
    gdf_surf_water = get_gdf_surface_water(ds)

    # find grid cells with sea
    swater_zee = gdf_surf_water[
        gdf_surf_water["OWMNAAM"].isin(
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


def add_northsea(ds, cachedir=None):
    """Add datavariable bathymetry to model dataset.

    Performs the following steps:

    a) get cells from modelgrid that are within the northsea, add data
       variable 'northsea' to ds
    b) fill top, bot, kh and kv add northsea cell by extrapolation
    c) get bathymetry (northsea depth) from jarkus.
    """
    logger.info(
        "Filling NaN values in top/botm and kh/kv in "
        "North Sea using bathymetry data from jarkus"
    )

    # find grid cells with northsea
    ds.update(get_northsea(ds, cachedir=cachedir, cachename="sea_ds.nc"))

    # fill top, bot, kh, kv at sea cells
    fal = dims.get_first_active_layer(ds)
    fill_mask = (fal == fal.attrs["nodata"]) * ds["northsea"]
    ds = dims.fill_top_bot_kh_kv_at_mask(ds, fill_mask)

    # add bathymetry noordzee
    ds.update(
        jarkus.get_bathymetry(
            ds,
            ds["northsea"],
            cachedir=cachedir,
            cachename="bathymetry_ds.nc",
        )
    )

    ds = jarkus.add_bathymetry_to_top_bot_kh_kv(ds, ds["bathymetry"], fill_mask)

    # remove inactive layers
    ds = dims.remove_inactive_layers(ds)
    return ds


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
    resolution: str = "1m",
    extent: Optional[list[float]] = None,
) -> gpd.GeoDataFrame:
    """Get bathymetry dataframe from RWS.

    Parameters
    ----------
    resolution : str, optional
        resolution of the bathymetry data, "1m" or "20m". The default is "1m".
    extent : tuple, optional
        extent of the model domain. The default is None.
    """
    config = get_gdr_configuration()
    url = config["bodemhoogte"]["index"]["url"]
    layer = config["bodemhoogte"][resolution]["layer"]
    return arcrest(url, layer, extent=extent)


@cache.cache_netcdf()
def get_bathymetry(
    extent: list[float],
    resolution: str = "1m",
    res: Optional[float] = None,
    method: Optional[str] = None,
    chunks: Optional[Union[str, dict[str, int]]] = "default",
) -> xr.DataArray:
    """Get bathymetry data from RWS.

    Bathymetry is available at 20m resolution and at 1m resolution. The 20m
    resolution is available for large water bodies, but not in the rivers. The
    1m dataset covers the whole Netherlands.

    Parameters
    ----------
    extent : tuple
        extent of the model domain
    resolution : str, optional
        resolution of the bathymetry data, "1m" or "20m". The default is "1m".
    res : float, optional
        resolution of the output data array. The default is None, which uses
        resolution of the input datasets. Resampling method is provided by the method
        kwarg.
    method : str, optional
        resampling method. The default is None. See rasterio.enums.Resampling for
        supported methods. Examples are ["min", "max", "mean", "nearest"].
    chunks : dict, optional
        chunks for the output data array. The default is "default", which uses
        {"x": 10_000, "y": 10_000}. Set to None to avoid chunking.

    Returns
    -------
    bathymetry : xr.DataArray
        bathymetry data
    """
    gdf = get_bathymetry_gdf(resolution=resolution, extent=extent)

    xmin, xmax, ymin, ymax = extent
    dataarrays = []

    if chunks == "default":
        chunks = {"x": 10_000, "y": 10_000}

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
