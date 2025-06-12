import io
import logging
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray

from ..dims.resample import structured_da_to_ds

logger = logging.getLogger(__name__)


def _download_file(url, pathname, filename=None, overwrite=False, timeout=120.0):
    """Download a file from the NHI website.

    Parameters
    ----------
    url : str
        The url of the file to download.
    pathname : str
        The pathname to which the file is downloaded.
    filename : str, optional
        The name of the file to contain the downloadded data. When filename is None, it
        is derived from url. The default is None.
    overwrite : bool, optional
        Overwrite the file if it allready exists. The default is False.
    timeout : float, optional
        How many seconds to wait for the server to send data before giving up. The
        default is 120.

    Returns
    -------
    fname : str
        The full path of the downloaded file.
    """
    if filename is None:
        filename = url.split("/")[-1]
    fname = os.path.join(pathname, filename)
    if overwrite or not os.path.isfile(fname):
        logger.info(f"Downloading {filename}")
        r = requests.get(url, allow_redirects=True, timeout=timeout)
        with open(fname, "wb") as file:
            file.write(r.content)
    return fname


def download_buisdrainage(pathname, overwrite=False):
    """Download resistance and depth of buisdrainage from the NHI website.

    Parameters
    ----------
    pathname : str
        The pathname to which the files are downloaded.
    overwrite : bool, optional
        Overwrite the files if they allready exists. The default is False.

    Returns
    -------
    fname_c : str
        The full path of the downloaded file containing the resistance of buisdrainage.
    fname_d : str
        The full path of the downloaded file containing the depth of buisdrainage.
    """
    url_bas = "https://thredds.data.nhi.nu/thredds/fileServer/opendap/models/nhi3_2/25m"

    # download resistance
    url = f"{url_bas}/buisdrain_c_ras25/buisdrain_c_ras25.nc"
    fname_c = _download_file(url, pathname, overwrite=overwrite)

    # download drain depth
    url = f"{url_bas}/buisdrain_d_ras25/buisdrain_d_ras25.nc"
    fname_d = _download_file(url, pathname, overwrite=overwrite)

    return fname_c, fname_d


def add_buisdrainage(
    ds,
    pathname=None,
    cond_var="buisdrain_cond",
    depth_var="buisdrain_depth",
    cond_method="average",
    depth_method="mode",
):
    """Add data about the buisdrainage to the model Dataset.

    This data consists of the conductance of buisdrainage (m2/d) and the depth of
    buisdrainage (m to surface level). With the default settings for `cond_method` and
    `depth_method`, the conductance is calculated from the area weighted average of the
    resistance in a cell, while the depth is set to the value which appears most often
    in a cell. Cells without data (or 0 in the case of the resistance) are ignored in
    these calculations.

    The original data of the resistance and the depth of buisdrainage at a 25x25 m scale
    is downloaded to `pathname` when not found.

    Parameters
    ----------
    ds : xr.Dataset
        The model Dataset.
    pathname : str, optional
        The pathname containing the downloaded files or the pathname to which the files
        are downloaded. When pathname is None, it is set the the cachedir. The default
        is None.
    cond_var : str, optional
        The name of the variable in ds to contain the data about the conductance of
        buisdrainage. The default is "buisdrain_cond".
    depth_var : str, optional
        The name of the variable in ds to contain the data about the depth of
        buisdrainage. The default is "buisdrain_depth".
    cond_method : str, optional
        The method to transform the conductance of buisdrainage to the model Dataset.
        The default is "average".
    depth_method : str, optional
        The method to transform the depth of buisdrainage to the model Dataset. The
        default is "mode".

    Returns
    -------
    ds : xr.Dataset
        The model dataset with added variables with the names `cond_var` and
        `depth_var`.
    """
    
    warnings.warn(
    "'add_buisdrainage' is deprecated and will be removed in a future version. "
    "Use 'discretize_buisdrainage' to project the buisdrainage on the model grid",
    DeprecationWarning)


    return discretize_buisdrainage(ds,
                                    pathname,
                                    cond_var,
                                    depth_var,
                                    cond_method,
                                    depth_method,
                                )


def discretize_buisdrainage(
    ds,
    pathname=None,
    cond_var="buisdrain_cond",
    depth_var="buisdrain_depth",
    cond_method="average",
    depth_method="mode",
):
    """Add data about the buisdrainage to the model Dataset.

    This data consists of the conductance of buisdrainage (m2/d) and the depth of
    buisdrainage (m to surface level). With the default settings for `cond_method` and
    `depth_method`, the conductance is calculated from the area weighted average of the
    resistance in a cell, while the depth is set to the value which appears most often
    in a cell. Cells without data (or 0 in the case of the resistance) are ignored in
    these calculations.

    The original data of the resistance and the depth of buisdrainage at a 25x25 m scale
    is downloaded to `pathname` when not found.

    Parameters
    ----------
    ds : xr.Dataset
        The model Dataset.
    pathname : str, optional
        The pathname containing the downloaded files or the pathname to which the files
        are downloaded. When pathname is None, it is set the the cachedir. The default
        is None.
    cond_var : str, optional
        The name of the variable in ds to contain the data about the conductance of
        buisdrainage. The default is "buisdrain_cond".
    depth_var : str, optional
        The name of the variable in ds to contain the data about the depth of
        buisdrainage. The default is "buisdrain_depth".
    cond_method : str, optional
        The method to transform the conductance of buisdrainage to the model Dataset.
        The default is "average".
    depth_method : str, optional
        The method to transform the depth of buisdrainage to the model Dataset. The
        default is "mode".

    Returns
    -------
    ds : xr.Dataset
        The model dataset with added variables with the names `cond_var` and
        `depth_var`.
    """
    if pathname is None:
        pathname = ds.cachedir
    # download files if needed
    fname_c, fname_d = download_buisdrainage(pathname)

    # make sure crs is set on ds
    if ds.rio.crs is None:
        ds = ds.rio.write_crs(28992)

    # use cond_methd for conductance
    # (default is "average" to account for locations without pipe drainage, where the
    # conductance is 0)
    buisdrain_c = rioxarray.open_rasterio(fname_c, mask_and_scale=True)[0]
    # calculate a conductance (per m2) from a resistance
    cond = 1 / buisdrain_c
    # set conductance to 0 where resistance is infinite or 0
    cond = cond.where(~(np.isinf(cond) | np.isnan(cond)), 0.0)
    cond = cond.rio.write_crs(buisdrain_c.rio.crs)
    # resample to model grid
    ds[cond_var] = structured_da_to_ds(cond, ds, method=cond_method)
    # multiply by area to get a conductance
    ds[cond_var] = ds[cond_var] * ds["area"]

    # use depth_method to retrieve the depth
    # (default is "mode" for depth that occurs most in each cell)
    mask_and_scale = False
    buisdrain_d = rioxarray.open_rasterio(fname_d, mask_and_scale=mask_and_scale)[0]
    if mask_and_scale:
        nodata = np.nan
    else:
        nodata = buisdrain_d.attrs["_FillValue"]
    # set buisdrain_d to nodata where it is 0
    mask = buisdrain_d != 0
    buisdrain_d = buisdrain_d.where(mask, nodata).rio.write_crs(buisdrain_d.rio.crs)
    # resample to model grid
    ds[depth_var] = structured_da_to_ds(
        buisdrain_d, ds, method=depth_method, nodata=nodata
    )
    if not mask_and_scale:
        # set nodata values to NaN
        ds[depth_var] = ds[depth_var].where(ds[depth_var] != nodata)

    # from cm to m
    ds[depth_var] = ds[depth_var] / 100.0

    return ds



def get_gwo_wells(
    username,
    password,
    n_well_filters=1_000,
    well_site=None,
    organisation=None,
    status=None,
    well_index="Name",
    timeout=120,
    **kwargs,
):
    """Deprecated, use 'download_gwo_wells' instead
    """
    warnings.warn(
    "'get_gwo_wells' is deprecated and will be removed in a future version. "
    "Use 'download_gwo_wells' instead",
    DeprecationWarning)

    return download_gwo_wells(username,
                                password,
                                n_well_filters=n_well_filters,
                                well_site=well_site,
                                organisation=organisation,
                                status=status,
                                well_index=well_index,
                                timeout=timeout,
                                **kwargs,
                            )


def download_gwo_wells(
    username,
    password,
    n_well_filters=1_000,
    well_site=None,
    organisation=None,
    status=None,
    well_index="Name",
    timeout=120,
    **kwargs,
):
    """Get metadata of extraction wells from the NHI GWO database.

    Parameters
    ----------
    username : str
        The username of the NHI GWO database. To retrieve a username and password visit
        https://gwo.nhi.nu/register/.
    password : str
        The password of the NHI GWO database. To retrieve a username and password visit
        https://gwo.nhi.nu/register/.
    n_well_filters : int, optional
        The number of wells that are requested per page. This number determines in how
        many pieces the request is split. The default is 1000.
    organisation : str, optional
        The organisation that manages the wells. If not None, the organisation will be
        used to filter the wells. The default is None.
    well_site : str, optional
        The name of well site the wells belong to. If not None, the well site will be
        used to filter the wells. The default is None.
    status : str, optional
        The status of the wells. If not None, the status will be used to filter the
        wells. Possible values are "Active", "Inactive" or "Abandoned". The default is
        None.
    well_index : str, tuple or list, optional
        The column(s) in the resulting GeoDataFrame that is/are used as the index of
        this GeoDataFrame. The default is "Name".
    timeout : int, optional
        The timeout time (in seconds) for requests to the database. The default is
        120 seconds.
    **kwargs : dict
        Kwargs are passed as additional parameters in the request to the database. For
        available parameters see https://gwo.nhi.nu/api/v1/download/.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the properties of the wells and their filters.
    """
    # zie https://gwo.nhi.nu/api/v1/download/
    url = "https://gwo.nhi.nu/api/v1/well_filters/"

    page = 1
    properties = []
    while page is not None:
        params = {"format": "csv", "n_well_filters": n_well_filters, "page": page}
        if status is not None:
            params["well__status"] = status
        if organisation is not None:
            params["well__organization"] = organisation
        if well_site is not None:
            params["well__site"] = well_site
        params.update(kwargs)

        r = requests.get(url, auth=(username, password), params=params, timeout=timeout)
        if not r.ok:
            r.raise_for_status()
        content = r.content.decode("utf-8")
        if len(content) == 0:
            if page == 1:
                msg = "No extraction wells found for the requested parameters"
                raise ValueError(msg)
            else:
                # the number of wells is exactly a multiple of n_well_filters
                page = None
                continue
        lines = content.split("\n")
        empty_lines = np.where([set(line) == set(";") for line in lines])[0]
        assert len(empty_lines) == 1, "Returned extraction wells cannot be interpreted"
        skiprows = list(range(empty_lines[0] + 1)) + [empty_lines[0] + 2]
        df = pd.read_csv(io.StringIO(content), skiprows=skiprows, sep=";")
        properties.append(df)

        if len(df) == n_well_filters:
            page += 1
        else:
            page = None
    df = pd.concat(properties)
    geometry = gpd.points_from_xy(df.XCoordinate, df.YCoordinate)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=28992)
    if well_index is not None:
        gdf = gdf.set_index(well_index)
    return gdf


def get_gwo_measurements(
    username,
    password,
    n_measurements=10_000,
    well_site=None,
    well_index="Name",
    measurement_index=("Name", "DateTime"),
    timeout=120,
    **kwargs,
):
    """Deprecated, use 'download_gwo_measurements' instead
    """
    warnings.warn(
    "'get_gwo_measurements' is deprecated and will be removed in a future version. "
    "Use 'download_gwo_measurements' instead",
    DeprecationWarning)

    return download_gwo_measurements(username,
                                password,
                                n_measurements=n_measurements,
                                well_site=well_site,
                                well_index=well_index,
                                measurement_index=measurement_index,
                                timeout=timeout,
                                **kwargs,
                            )


def download_gwo_measurements(
    username,
    password,
    n_measurements=10_000,
    well_site=None,
    well_index="Name",
    measurement_index=("Name", "DateTime"),
    timeout=120,
    **kwargs,
):
    """Get extraction rates and metadata of wells from the NHI GWO database.

    Parameters
    ----------
    username : str
        The username of the NHI GWO database. To retrieve a username and password visit
        https://gwo.nhi.nu/register/.
    password : str
        The password of the NHI GWO database. To retrieve a username and password visit
        https://gwo.nhi.nu/register/.
    n_measurements : int, optional
        The number of measurements that are requested per page, with a maximum of
        200,000. This number determines in how many pieces the request is split. The
        default is 10,000.
    well_site : str, optional
        The name of well site the wells belong to. If not None, the well site will be
        used to filter the wells. The default is None.
    well_index : str, tuple or list, optional
        The column(s) in the resulting GeoDataFrame that is/are used as the index of
        this GeoDataFrame. The default is "Name".
    measurement_index :  str, tuple or list, optional, optional
        The column(s) in the resulting measurement-DataFrame that is/are used as the
        index of this DataFrame. The default is ("Name", "DateTime").
    timeout : int, optional
        The timeout time (in seconds) of requests to the database. The default is
        120 seconds.
    **kwargs : dict
        Kwargs are passed as additional parameters in the request to the database. For
        available parameters see https://gwo.nhi.nu/api/v1/download/.

    Returns
    -------
    measurements : pandas.DataFrame
        A DataFrame containing the extraction rates of the wells in the database.
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the properties of the wells and their filters.
    """
    url = "http://gwo.nhi.nu/api/v1/measurements/"
    properties = []
    measurements = []
    page = 1
    while page is not None:
        params = {
            "format": "csv",
            "n_measurements": n_measurements,
            "page": page,
        }
        if well_site is not None:
            params["filter__well__site"] = well_site
        params.update(kwargs)
        r = requests.get(url, auth=(username, password), params=params, timeout=timeout)

        content = r.content.decode("utf-8")
        if len(content) == 0:
            if page == 1:
                msg = "No extraction rates found for the requested parameters"
                raise (ValueError(msg))
            else:
                # the number of measurements is exactly a multiple of n_measurements
                page = None
                continue
        lines = content.split("\n")
        empty_lines = np.where([set(line) == set(";") for line in lines])[0]
        assert len(empty_lines) == 2, "Returned extraction rates cannot be interpreted"

        # read properties
        skiprows = list(range(empty_lines[0] + 1)) + [empty_lines[0] + 2]
        nrows = empty_lines[1] - empty_lines[0] - 3
        df = pd.read_csv(io.StringIO(content), sep=";", skiprows=skiprows, nrows=nrows)
        properties.append(df)

        # read measurements
        skiprows = list(range(empty_lines[1] + 1)) + [empty_lines[1] + 2]
        df = pd.read_csv(
            io.StringIO(content),
            skiprows=skiprows,
            sep=";",
            parse_dates=["DateTime"],
            dayfirst=True,
        )
        measurements.append(df)
        if len(df) == n_measurements:
            page += 1
        else:
            page = None
    measurements = pd.concat(measurements)
    # drop columns without measurements
    measurements = measurements.loc[:, ~measurements.isna().all()]
    if measurement_index is not None:
        if isinstance(measurement_index, tuple):
            measurement_index = list(measurement_index)
        measurements = measurements.set_index(["Name", "DateTime"])
    df = pd.concat(properties)
    geometry = gpd.points_from_xy(df.XCoordinate, df.YCoordinate)
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    if well_index is not None:
        gdf = gdf.set_index(well_index)
        # drop duplicate properties from multiple pages
        gdf = gdf[~gdf.index.duplicated()]
    return measurements, gdf
