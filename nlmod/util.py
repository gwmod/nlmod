import logging
import os
import re
import sys
import warnings
from typing import Dict, Optional

import flopy
import geopandas as gpd
import numpy as np
import requests
import xarray as xr
from colorama import Back, Fore, Style
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


def get_exe_path(exe_name="mf6"):
    """get the full path of the executable. Uses the bin directory in the nlmod
    package.

    Parameters
    ----------
    exe_name : str, optional
        name of the executable. The default is 'mf6'.

    Returns
    -------
    exe_path : str
        full path of the executable.
    """
    exe_path = os.path.join(os.path.dirname(__file__), "bin", exe_name)
    if sys.platform.startswith("win"):
        exe_path += ".exe"

    if not os.path.exists(exe_path):
        logger.warning(
            f"executable {exe_path} not found, download the binaries using nlmod.util.download_mfbinaries"
        )

    return exe_path


def get_ds_empty(ds):
    """get a copy of a model dataset with only coordinate information.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with coordinates

    Returns
    -------
    empty_ds : xr.Dataset
        dataset with only model coordinate information
    """

    empty_ds = xr.Dataset()
    for coord in list(ds.coords):
        empty_ds = empty_ds.assign_coords(coords={coord: ds[coord]})

    return empty_ds


def get_da_from_da_ds(da_ds, dims=("y", "x"), data=None):
    """get a dataarray from ds with certain dimensions.

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
        DataArray with coordinates from ds
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


def get_google_drive_filename(fid, timeout=120):
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
        response = requests.get(url, timeout=timeout)
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


def get_heads_dataarray(ds, fill_nans=False, fname_hds=None):
    """reads the heads from a modflow .hds file and returns an xarray
    DataArray.

    Parameters
    ----------
    ds : TYPE
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
    logger.warning(
        "nlmod.util.get_heads_dataarray is deprecated. "
        "Please use nlmod.gwf.get_heads_da instead"
    )

    if fname_hds is None:
        fname_hds = os.path.join(ds.model_ws, ds.model_name + ".hds")

    head = get_heads_array(fname_hds, fill_nans=fill_nans)

    if ds.gridtype == "vertex":
        head_ar = xr.DataArray(
            data=head[:, :, 0],
            dims=("time", "layer", "icell2d"),
            coords={
                "icell2d": ds.icell2d,
                "layer": ds.layer,
                "time": ds.time,
            },
        )
    elif ds.gridtype == "structured":
        head_ar = xr.DataArray(
            data=head,
            dims=("time", "layer", "y", "x"),
            coords={
                "x": ds.x,
                "y": ds.y,
                "layer": ds.layer,
                "time": ds.time,
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
    logger.warning(
        "nlmod.util.get_heads_array is deprecated. "
        "Please use nlmod.gwf.get_heads_da instead"
    )
    hdobj = flopy.utils.HeadFile(fname_hds)
    head = hdobj.get_alldata()
    head[head == 1e30] = np.nan

    if fill_nans:
        for lay in range(head.shape[1] - 2, -1, -1):
            head[:, lay] = np.where(
                np.isnan(head[:, lay]), head[:, lay + 1], head[:, lay]
            )
    return head


def download_mfbinaries(bindir=None):
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

    if bindir is None:
        bindir = os.path.join(os.path.dirname(__file__), "bin")
    if not os.path.isdir(bindir):
        os.makedirs(bindir)
    flopy.utils.get_modflow(bindir)


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


class ColoredFormatter(logging.Formatter):
    """Colored log formatter.

    Taken from
    https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
    """

    def __init__(
        self, *args, colors: Optional[Dict[str, str]] = None, **kwargs
    ) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL

        return super().format(record)


def get_color_logger(level="INFO"):
    formatter = ColoredFormatter(
        "{color}{levelname}:{name}:{message}{reset}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        colors={
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
        },
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level))

    logging.captureWarnings(True)
    return logger
