import logging
import warnings
import os
import re
import sys

import flopy
import geopandas as gpd
import numpy as np
import requests
import xarray as xr
from shapely.geometry import box

logger = logging.getLogger(__name__)


def get_model_dirs(model_ws):
    """Creates a new model workspace directory, if it does not exists yet.
    Within the model workspace directory a few subdirectories are created (if
    they don't exist yet):

    - figure
    - cache

    Parameters
    ----------
    model_ws : str
        model workspace.

    Returns
    -------
    figdir : str
        figure directory inside model workspace.
    cachedir : str
        cache directory inside model workspace.
    """
    figdir = os.path.join(model_ws, "figure")
    cachedir = os.path.join(model_ws, "cache")
    if not os.path.exists(model_ws):
        os.makedirs(model_ws)

    if not os.path.exists(figdir):
        os.mkdir(figdir)

    if not os.path.exists(cachedir):
        os.mkdir(cachedir)

    return figdir, cachedir


def get_exe_path(exe_name='mf6'):
    """get the full path of the executable. Uses the bin directory in the
    nlmod package.
    

    Parameters
    ----------
    exe_name : str, optional
        name of the executable. The default is 'mf6'.

    Returns
    -------
    exe_path : str
        full path of the executable.

    """
    exe_path = os.path.join(
        os.path.dirname(__file__), "bin", exe_name
    )
    if sys.platform.startswith("win"):
        exe_path += ".exe"
        
    return exe_path


def get_model_ds_empty(model_ds):
    """get a copy of a model dataset with only grid and time information.

    Parameters
    ----------
    model_ds : xr.Dataset
        dataset with at least the variables layer, x, y and time

    Returns
    -------
    model_ds_out : xr.Dataset
        dataset with only model grid and time information
    """

    return model_ds[list(model_ds.coords)].copy()


def get_da_from_da_ds(da_ds, dims=("y", "x"), data=None):
    """get a dataarray from model_ds with certain dimensions.

    Parameters
    ----------
    da_ds : xr.Dataset or xr.DataArray
        Dataset or DataArray with at least the dimensions defined in dims
    dims : tuple of string, optional
        dimensions of the desired data array. Default is 'y', 'x'
    data : None, int, float, array_like, optional
        data to fill data array with. if None DataArray is filled with nans.
        Default is None.

    Returns
    -------
    da : xr.DataArray
        DataArray with coordinates from model_ds
    """
    if not isinstance(dims, tuple):
        raise TypeError(
            "keyword argument dims should be of type tuple not {type(dims)}"
        )

    coords = {dim: da_ds[dim] for dim in dims}
    da = xr.DataArray(data, dims=dims, coords=coords)
    for dim in dims:
        # remove the coordinate again, when it was not defined in da_ds
        if dim not in da_ds.coords:
            da = da.drop_vars(dim)

    return da


def find_most_recent_file(folder, name, extension=".pklz"):
    """find the most recent file in a folder. File must startwith name and
    end width extension. If you want to look for the most recent folder use
    extension = ''.

    Parameters
    ----------
    folder : str
        path of folder to look for files
    name : str
        find only files that start with this name
    extension : str
        find only files with this extension

    Returns
    -------
    newest_file : str
        name of the most recent file
    """

    i = 0
    for file in os.listdir(folder):
        if file.startswith(name) and file.endswith(extension):
            if i == 0:
                newest_file = os.path.join(folder, file)
                time_prev_file = os.stat(newest_file).st_mtime
            else:
                check_file = os.path.join(folder, file)
                if os.stat(check_file).st_mtime > time_prev_file:
                    newest_file = check_file
                    time_prev_file = os.stat(check_file).st_mtime
            i += 1

    if i == 0:
        return None

    return newest_file


def compare_model_extents(extent1, extent2):
    """check overlap between two model extents.

    Parameters
    ----------
    extent1 : list, tuple or array
        first extent [xmin, xmax, ymin, ymax]
    extent2 : xr.DataSet
        second extent

    Returns
    -------
    int
        several outcomes:
            1: extent1 is completely within extent2
            2: extent2 is completely within extent1
    """

    # option1 extent1 is completely within extent2
    check_xmin = extent1[0] >= extent2[0]
    check_xmax = extent1[1] <= extent2[1]
    check_ymin = extent1[2] >= extent2[2]
    check_ymax = extent1[3] <= extent2[3]

    if check_xmin and check_xmax and check_ymin and check_ymax:
        logger.info("extent1 is completely within extent2 ")
        return 1

    # option2 extent2 is completely within extent1
    if (not check_xmin) and (not check_xmax) and (not check_ymin) and (not check_ymax):
        logger.info("extent2 is completely within extent1")
        return 2

    # option 3 left bound
    if (not check_xmin) and check_xmax and check_ymin and check_ymax:
        logger.info(
            "extent1 is completely within extent2 except for the left bound (xmin)"
        )
        return 3

    # option 4 right bound
    if check_xmin and (not check_xmax) and check_ymin and check_ymax:
        logger.info(
            "extent1 is completely within extent2 except for the right bound (xmax)"
        )
        return 4

    # option 10
    if check_xmin and (not check_xmax) and (not check_ymin) and (not check_ymax):
        logger.info("only the left bound of extent 1 is within extent 2")
        return 10

    raise NotImplementedError("other options are not yet implemented")


def polygon_from_extent(extent):
    """create a shapely polygon from a given extent.

    Parameters
    ----------
    extent : tuple, list or array
        extent (xmin, xmax, ymin, ymax).

    Returns
    -------
    polygon_ext : shapely.geometry.polygon.Polygon
        polygon of the extent.
    """

    bbox = (extent[0], extent[2], extent[1], extent[3])
    polygon_ext = box(*tuple(bbox))

    return polygon_ext


def gdf_from_extent(extent, crs="EPSG:28992"):
    """create a geodataframe with a single polygon with the extent given.

    Parameters
    ----------
    extent : tuple, list or array
        extent.
    crs : str, optional
        coördinate reference system of the extent, default is EPSG:28992
        (RD new)

    Returns
    -------
    gdf_extent : GeoDataFrame
        geodataframe with extent.
    """

    geom_extent = polygon_from_extent(extent)
    gdf_extent = gpd.GeoDataFrame(geometry=[geom_extent], crs=crs)

    return gdf_extent


def gdf_within_extent(gdf, extent):
    """select only parts of the geodataframe within the extent. Only accepts
    Polygon and Linestring geometry types.

    Parameters
    ----------
    gdf : geopandas GeoDataFrame
        dataframe with polygon features.
    extent : list or tuple
        extent to slice gdf, (xmin, xmax, ymin, ymax).

    Returns
    -------
    gdf : geopandas GeoDataFrame
        dataframe with only polygon features within the extent.
    """
    # create geodataframe from the extent
    gdf_extent = gdf_from_extent(extent, crs=gdf.crs)

    # check type
    geom_types = gdf.geom_type.unique()
    if len(geom_types) > 1:
        # exception if geomtypes is a combination of Polygon and Multipolygon
        multipoly_check = ("Polygon" in geom_types) and ("MultiPolygon" in geom_types)
        if (len(geom_types) == 2) and multipoly_check:
            gdf = gpd.overlay(gdf, gdf_extent)
        else:
            raise TypeError(f"Only accepts single geometry type not {geom_types}")
    elif geom_types[0] == "Polygon":
        gdf = gpd.overlay(gdf, gdf_extent)
    elif geom_types[0] == "LineString":
        gdf = gpd.sjoin(gdf, gdf_extent)
    elif geom_types[0] == "Point":
        gdf = gdf.loc[gdf.within(gdf_extent.geometry.values[0])]
    else:
        raise TypeError("Function is not tested for geometry type: " f"{geom_types[0]}")

    return gdf


def get_google_drive_filename(fid):
    """get the filename of a google drive file.

    Parameters
    ----------
    fid : str
        google drive id name of a file.

    Returns
    -------
    file_name : str
        filename.
    """
    warnings.warn(
        "this function is no longer supported use the gdown package instead",
        DeprecationWarning,
    )

    if isinstance(id, requests.Response):
        response = id
    else:
        url = "https://drive.google.com/uc?export=download&id=" + fid
        response = requests.get(url)
    header = response.headers["Content-Disposition"]
    file_name = re.search(r'filename="(.*)"', header).group(1)
    return file_name


def download_file_from_google_drive(fid, destination=None):
    """download a file from google drive using it's id.

    Parameters
    ----------
    fid : str
        google drive id name of a file.
    destination : str, optional
        location to save the file to. If destination is None the file is
        written to the current working directory. The default is None.
    """
    warnings.warn(
        "this function is no longer supported use the gdown package instead",
        DeprecationWarning,
    )

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": fid}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": fid, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    if destination is None:
        destination = get_google_drive_filename(fid)
    else:
        if os.path.isdir(destination):
            filename = get_google_drive_filename(fid)
            destination = os.path.join(destination, filename)

    save_response_content(response, destination)


# %% helper functions (from USGS)


def get_platform(pltfrm):
    """Determine the platform in order to construct the zip file name.

    Source: USGS

    Parameters
    ----------
    pltfrm : str, optional
        check if platform string is correct for downloading binaries,
        default is None and will determine platform string based on system

    Returns
    -------
    pltfrm : str
        return platform string
    """
    if pltfrm is None:
        if sys.platform.lower() == "darwin":
            pltfrm = "mac"
        elif sys.platform.lower().startswith("linux"):
            pltfrm = "linux"
        elif "win" in sys.platform.lower():
            is_64bits = sys.maxsize > 2**32
            if is_64bits:
                pltfrm = "win64"
            else:
                pltfrm = "win32"
        else:
            errmsg = "Could not determine platform" f".  sys.platform is {sys.platform}"
            raise Exception(errmsg)
    else:
        assert pltfrm in ["mac", "linux", "win32", "win64"]
    return pltfrm


def getmfexes(pth=".", version="", pltfrm=None):
    """Get the latest MODFLOW binary executables from a github site
    (https://github.com/MODFLOW-USGS/executables) for the specified operating
    system and put them in the specified path.

    Source: USGS

    Parameters
    ----------
    pth : str
        Location to put the executables (default is current working directory)

    version : str
        Version of the MODFLOW-USGS/executables release to use.

    pltfrm : str
        Platform that will run the executables.  Valid values include mac,
        linux, win32 and win64.  If platform is None, then routine will
        download the latest appropriate zipfile from the github repository
        based on the platform running this script.
    """
    try:
        import pymake
    except ModuleNotFoundError as e:
        print(
            "Install pymake with "
            "`pip install "
            "https://github.com/modflowpy/pymake/zipball/master`"
        )
        raise e
    # Determine the platform in order to construct the zip file name
    pltfrm = get_platform(pltfrm)
    zipname = f"{pltfrm}.zip"

    # Determine path for file download and then download and unzip
    url = "https://github.com/MODFLOW-USGS/executables/" f"releases/download/{version}/"
    assets = {p: url + p for p in ["mac.zip", "linux.zip", "win32.zip", "win64.zip"]}
    download_url = assets[zipname]
    pymake.download_and_unzip(download_url, pth)


def get_heads_dataarray(model_ds, fill_nans=False, fname_hds=None):
    """reads the heads from a modflow .hds file and returns an xarray
    DataArray.

    Parameters
    ----------
    model_ds : TYPE
        DESCRIPTION.
    fill_nans : bool, optional
        if True the nan values are filled with the heads in the cells below
    fname_hds : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    head_ar : TYPE
        DESCRIPTION.
    """

    if fname_hds is None:
        fname_hds = os.path.join(model_ds.model_ws, model_ds.model_name + ".hds")

    head = get_heads_array(fname_hds, fill_nans=fill_nans)

    if model_ds.gridtype == "vertex":
        head_ar = xr.DataArray(
            data=head[:, :, 0],
            dims=("time", "layer", "icell2d"),
            coords={
                "icell2d": model_ds.icell2d,
                "layer": model_ds.layer,
                "time": model_ds.time,
            },
        )
    elif model_ds.gridtype == "structured":
        head_ar = xr.DataArray(
            data=head,
            dims=("time", "layer", "y", "x"),
            coords={
                "x": model_ds.x,
                "y": model_ds.y,
                "layer": model_ds.layer,
                "time": model_ds.time,
            },
        )

    return head_ar


def get_heads_array(fname_hds, fill_nans=False):
    """reads the heads from a modflow .hds file and returns a numpy array.

    assumes the dimensions of the heads file are:
        structured: time, layer, icell2d
        vertex: time, layer, nrow, ncol


    Parameters
    ----------
    fname_hds : TYPE, optional
        DESCRIPTION. The default is None.
    fill_nans : bool, optional
        if True the nan values are filled with the heads in the cells below

    Returns
    -------
    head_ar : np.ndarray
        heads array.
    """
    hdobj = flopy.utils.HeadFile(fname_hds)
    head = hdobj.get_alldata()
    head[head == 1e30] = np.nan

    if fill_nans:
        for lay in range(head.shape[1] - 2, -1, -1):
            head[:, lay] = np.where(
                np.isnan(head[:, lay]), head[:, lay + 1], head[:, lay]
            )
    return head


def download_mfbinaries(binpath=None, version="8.0"):
    """Download and unpack platform-specific modflow binaries.

    Source: USGS

    Parameters
    ----------
    binpath : str, optional
        path to directory to download binaries to, if it doesnt exist it
        is created. Default is None which sets dir to nlmod/bin.
    version : str, optional
        version string, by default 8.0
    """
    if binpath is None:
        binpath = os.path.join(os.path.dirname(__file__), "bin")
    pltfrm = get_platform(None)
    # Download and unpack mf6 exes
    getmfexes(pth=binpath, version=version, pltfrm=pltfrm)


def check_presence_mfbinaries(exe_name="mf6", binpath=None):
    """Check if exe_name is present in the binpath folder.

    Parameters
    ----------
    exe_name : str, optional
        the name of the file that is checked to be present, by default 'mf6'
    binpath : str, optional
        path to directory to download binaries to, if it doesnt exist it
        is created. Default is None which sets dir to nlmod/bin.
    """
    if binpath is None:
        binpath = os.path.join(os.path.dirname(__file__), "bin")
    if not os.path.isdir(binpath):
        return False
    files = [os.path.splitext(file)[0] for file in os.listdir(binpath)]
    return exe_name in files
