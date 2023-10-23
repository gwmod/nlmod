import logging
import os

import numpy as np
import requests
import rioxarray

from ..dims.resample import structured_da_to_ds

logger = logging.getLogger(__name__)


def download_file(url, pathname, filename=None, overwrite=False, timeout=120.0):
    """
    Download a file from the NHI website.

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
    """
    Download resistance and depth of buisdrainage from the NHI website

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
    fname_c = download_file(url, pathname, overwrite=overwrite)

    # download drain depth
    url = f"{url_bas}/buisdrain_d_ras25/buisdrain_d_ras25.nc"
    fname_d = download_file(url, pathname, overwrite=overwrite)

    return fname_c, fname_d


def add_buisdrainage(
    ds,
    pathname=None,
    cond_var="buisdrain_cond",
    depth_var="buisdrain_depth",
    cond_method="average",
    depth_method="mode",
):
    """
    Add data about the buisdrainage to the model Dataset.

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
