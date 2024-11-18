import json
import logging
import os
import re
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import requests
import xarray as xr
from colorama import Back, Fore, Style
from flopy.utils import get_modflow
from flopy.utils.get_modflow import flopy_appdata_path, get_release
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from tqdm import tqdm

logger = logging.getLogger(__name__)

nlmod_bindir = Path(__file__).parent / "bin"


class LayerError(Exception):
    """Generic error when modifying layers."""


class MissingValueError(Exception):
    """Generic error when an expected value is not defined."""


def check_da_dims_coords(da, ds):
    """Check if DataArray dimensions and coordinates match those in Dataset.

    Only checks overlapping dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        dataarray to check
    ds : xarray.Dataset or xarray.DataArray
        dataset or dataarray to compare coords with

    Returns
    -------
    True
        if dimensions and coordinates match

    Raises
    ------
    AssertionError
        error that coordinates do not match
    """
    shared_dims = set(da.dims) & set(ds.dims)
    for dim in shared_dims:
        try:
            xr.testing.assert_identical(da[dim], ds[dim])
        except AssertionError as e:
            logger.error(f"da '{da.name}' coordinates do not match ds!")
            raise e
    return True


def get_model_dirs(model_ws):
    """Creates a new model workspace directory, if it does not exists yet. Within the
    model workspace directory a few subdirectories are created (if they don't exist
    yet):

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


def get_exe_path(
    exe_name="mf6",
    bindir=None,
    download_if_not_found=True,
    version_tag=None,
    repo="executables",
):
    """Get the full path of the executable.

    Searching for the executables is done in the following order:
    0. If exe_name is a full path, return the full path of the executable.
    1. The directory specified with `bindir`. Raises error if exe_name is provided
       and not found.
    2. The directory used by nlmod installed in this environment.
    3. If the executables were downloaded with flopy/nlmod from an other env,
       most recent installation location of MODFLOW is found in flopy metadata

    Else:
    4. Download the executables using `version_tag` and `repo`.

    The returned directory is checked to contain exe_name if it is provided.

    Parameters
    ----------
    exe_name : str, optional
        The name of the executable, by default "mf6".
    bindir : Path, optional
        The directory where the executables are stored, by default None
    download_if_not_found : bool, optional
        Download the executables if they are not found, by default True.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default), "modflow6",
        or "modflow6-nightly-build". If repo and version_tag are provided the most
        recent installation location of MODFLOW is found in flopy metadata that
        respects `version_tag` and `repo`. If not found, the executables are downloaded
        using repo and version_tag.
    version_tag : str, default None
        GitHub release ID: for example "18.0" or "latest". If repo and version_tag are
        provided the most recent installation location of MODFLOW is found in flopy
        metadata that respects `version_tag` and `repo`. If not found, the executables
        are downloaded using repo and version_tag.

    Returns
    -------
    exe_full_path : str
        full path of the executable.
    """
    if sys.platform.startswith("win") and not exe_name.endswith(".exe"):
        exe_name += ".exe"

    # If exe_name is a full path
    if Path(exe_name).exists():
        enable_version_check = version_tag is not None and repo is not None

        if enable_version_check:
            msg = (
                "Incompatible arguments. If exe_name is provided, unable to check "
                "the version."
            )
            raise ValueError(msg)
        exe_full_path = exe_name

    else:
        exe_full_path = str(
            get_bin_directory(
                exe_name=exe_name,
                bindir=bindir,
                download_if_not_found=download_if_not_found,
                version_tag=version_tag,
                repo=repo,
            )
            / exe_name
        )

    msg = f"Executable path: {exe_full_path}"
    logger.debug(msg)

    return exe_full_path


def get_bin_directory(
    exe_name="mf6",
    bindir=None,
    download_if_not_found=True,
    version_tag=None,
    repo="executables",
) -> Path:
    """Get the directory where the executables are stored.

    Searching for the executables is done in the following order:
    0. If exe_name is a full path, return the full path of the executable.
    1. The directory specified with `bindir`. Raises error if exe_name is provided
        and not found. Requires enable_version_check to be False.
    2. The directory used by nlmod installed in this environment.
    3. If the executables were downloaded with flopy/nlmod from an other env,
        most recent installation location of MODFLOW is found in flopy metadata

    Else:
    4. Download the executables using `version_tag` and `repo`.

    The returned directory is checked to contain exe_name if exe_name is provided. If
    exe_name is set to None only the existence of the directory is checked.

    Parameters
    ----------
    exe_name : str, optional
        The name of the executable, by default mf6.
    bindir : Path, optional
        The directory where the executables are stored, by default "mf6".
    download_if_not_found : bool, optional
        Download the executables if they are not found, by default True.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default), "modflow6",
        or "modflow6-nightly-build". If repo and version_tag are provided the most
        recent installation location of MODFLOW is found in flopy metadata that
        respects `version_tag` and `repo`. If not found, the executables are downloaded
        using repo and version_tag.
    version_tag : str, default None
        GitHub release ID: for example "18.0" or "latest". If repo and version_tag are
        provided the most recent installation location of MODFLOW is found in flopy
        metadata that respects `version_tag` and `repo`. If not found, the executables
        are downloaded using repo and version_tag.

    Returns
    -------
    Path
        The directory where the executables are stored.

    Raises
    ------
    FileNotFoundError
        If the executables are not found in the specified directories.
    """
    bindir = Path(bindir) if bindir is not None else None

    if sys.platform.startswith("win") and not exe_name.endswith(".exe"):
        exe_name += ".exe"

    enable_version_check = version_tag is not None

    # If exe_name is a full path
    if Path(exe_name).exists():
        if enable_version_check:
            msg = (
                "Incompatible arguments. If exe_name is provided, unable to check "
                "the version."
            )
            raise ValueError(msg)
        return Path(exe_name).parent

    # If bindir is provided
    if bindir is not None and enable_version_check:
        msg = (
            "Incompatible arguments. If bindir is provided, "
            "unable to check the version."
        )
        raise ValueError(msg)

    use_bindir = (
        bindir is not None and exe_name is not None and (bindir / exe_name).exists()
    )
    use_bindir |= bindir is not None and exe_name is None and bindir.exists()

    if use_bindir:
        return bindir

    # If the executables are in the flopy directory
    flopy_bindirs = get_flopy_bin_directories(version_tag=version_tag, repo=repo)

    if exe_name is not None:
        flopy_bindirs = [
            flopy_bindir
            for flopy_bindir in flopy_bindirs
            if Path(flopy_bindir / exe_name).exists()
        ]
    else:
        flopy_bindirs = [
            flopy_bindir
            for flopy_bindir in flopy_bindirs
            if Path(flopy_bindir).exists()
        ]

    if nlmod_bindir in flopy_bindirs:
        return nlmod_bindir

    if flopy_bindirs:
        # Get most recent directory
        return flopy_bindirs[-1]

    # Else download the executables
    if download_if_not_found:
        download_mfbinaries(
            bindir=bindir,
            version_tag=version_tag if version_tag is not None else "latest",
            repo=repo,
        )

        # Rerun this function
        return get_bin_directory(
            exe_name=exe_name,
            bindir=bindir,
            download_if_not_found=False,
            version_tag=version_tag,
            repo=repo,
        )

    else:
        msg = (
            f"Could not find {exe_name} in {bindir}, "
            f"{nlmod_bindir} and {flopy_bindirs}."
        )
        raise FileNotFoundError(msg)


def get_flopy_bin_directories(version_tag=None, repo="executables"):
    """Get the directories where the executables are stored.

    Obtain the bin directory installed with flopy. If enable_version_check is True,
    all installation location of MODFLOW are found in flopy metadata that respects
    `version_tag` and `repo`.

    Parameters
    ----------
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default),
        "modflow6", or "modflow6-nightly-build". If repo and version_tag are provided
        the most recent installation location of MODFLOW is found in flopy metadata
        that respects `version_tag` and `repo`. If not found, the executables are
        downloaded using repo and version_tag.
    version_tag : str, default None
        GitHub release ID: for example "18.0" or "latest". If repo and version_tag are
        provided the most recent installation location of MODFLOW is found in flopy
        metadata that respects `version_tag` and `repo`. If not found, the executables
        are downloaded using repo and version_tag.

    Returns
    -------
    list
        list of directories where the executables are stored.
    """
    flopy_metadata_fp = flopy_appdata_path / "get_modflow.json"

    if not flopy_metadata_fp.exists():
        return []

    meta_raw = flopy_metadata_fp.read_text()

    # Remove trailing characters that are not part of the JSON.
    while meta_raw[-3:] != "}\n]":
        meta_raw = meta_raw[:-1]

    # Get metadata of all flopy installations
    meta_list = json.loads(meta_raw)

    enable_version_check = version_tag is not None and repo is not None

    if enable_version_check:
        msg = (
            "The version of the executables will be checked, because the "
            f"`version_tag={version_tag}` is passed to `get_flopy_bin_directories()`."
        )

        # To convert latest into an explicit tag
        if version_tag == "latest":
            version_tag_pin = get_release(tag=version_tag, repo=repo, quiet=True)[
                "tag_name"
            ]
        else:
            version_tag_pin = version_tag

        # get path to the most recent installation. Appended to end of get_modflow.json
        meta_list_validversion = [
            meta
            for meta in meta_list
            if (meta["release_id"] == version_tag_pin) and (meta["repo"] == repo)
        ]

    else:
        msg = (
            "The version of the executables will not be checked, because the "
            "`version_tag` is not passed to `get_flopy_bin_directories()`."
        )
        meta_list_validversion = meta_list
    logger.debug(msg)

    path_list = [
        Path(meta["bindir"])
        for meta in meta_list_validversion
        if Path(meta["bindir"]).exists()
    ]
    return path_list


def download_mfbinaries(bindir=None, version_tag="latest", repo="executables"):
    """Download and unpack platform-specific modflow binaries.

    Source: USGS

    Parameters
    ----------
    binpath : str, optional
        path to directory to download binaries to, if it doesnt exist it
        is created. Default is None which sets dir to nlmod/bin.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default),
        "modflow6", or "modflow6-nightly-build".
    version_tag : str, default "latest"
        GitHub release ID.
    """
    if bindir is None:
        # Path objects are immutable so a copy is implied
        bindir = nlmod_bindir

    if not os.path.isdir(bindir):
        os.makedirs(bindir)

    get_modflow(bindir=str(bindir), release_id=version_tag, repo=repo)

    # Ensure metadata is saved.
    # https://github.com/modflowpy/flopy/blob/
    # 0748dcb9e4641b5ad9616af115dd3be906f98f50/flopy/utils/get_modflow.py#L623
    flopy_metadata_fp = flopy_appdata_path / "get_modflow.json"

    if not flopy_metadata_fp.exists():
        if "pytest" not in str(bindir) and "pytest" not in sys.modules:
            logger.warning(
                f"flopy metadata file not found at {flopy_metadata_fp}. "
                "After downloading and installing the executables. "
                "Creating a new metadata file."
            )

        release_metadata = get_release(tag=version_tag, repo=repo, quiet=True)
        install_metadata = {
            "release_id": release_metadata["tag_name"],
            "repo": repo,
            "bindir": str(bindir),
        }

        with open(flopy_metadata_fp, "w", encoding="UTF-8") as f:
            json.dump([install_metadata], f, indent=4)

    # download the provisional version of modpath from Github
    download_modpath_provisional_exe(bindir=bindir, timeout=120)


def get_ds_empty(ds, keep_coords=None):
    """Get a copy of a dataset with only coordinate information.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with coordinates
    keep_coords : tuple or None, optional
        the coordinates in ds the you want keep in your empty ds. If None all
        coordinates are kept from original ds. The default is None.

    Returns
    -------
    empty_ds : xr.Dataset
        dataset with only coordinate information
    """
    if keep_coords is None:
        keep_coords = list(ds.coords)

    empty_ds = xr.Dataset()
    for coord in list(ds.coords):
        if coord in keep_coords:
            empty_ds = empty_ds.assign_coords(coords={coord: ds[coord]})

    return empty_ds


def get_da_from_da_ds(da_ds, dims=("y", "x"), data=None):
    """Get a dataarray from ds with certain dimensions.

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
    """Find the most recent file in a folder.

    File must startwith name and end width extension. If you want to look for the most
    recent folder use extension = ''.

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
    """Check overlap between two model extents.

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


def extent_to_polygon(extent):
    """Generate a shapely Polygon from an extent ([xmin, xmax, ymin, ymax])

    Parameters
    ----------
    extent : tuple, list or array
        extent (xmin, xmax, ymin, ymax).

    Returns
    -------
    shapely.geometry.Polygon
        polygon of the extent.

    """
    nw = (extent[0], extent[2])
    no = (extent[1], extent[2])
    zo = (extent[1], extent[3])
    zw = (extent[0], extent[3])
    return Polygon([nw, no, zo, zw])


def extent_to_gdf(extent, crs="EPSG:28992"):
    """Create a geodataframe with a single polygon with the extent given.

    Parameters
    ----------
    extent : tuple, list or array
        extent.
    crs : str, optional
        coördinate reference system of the extent, default is EPSG:28992
        (RD new)

    Returns
    -------
    gdf_extent : geopandas.GeoDataFrame
        geodataframe with extent.
    """
    geom_extent = extent_to_polygon(extent)
    gdf_extent = gpd.GeoDataFrame(geometry=[geom_extent], crs=crs)

    return gdf_extent


def polygon_from_extent(extent):
    """Create a shapely polygon from a given extent.

    Parameters
    ----------
    extent : tuple, list or array
        extent (xmin, xmax, ymin, ymax).

    Returns
    -------
    polygon_ext : shapely.geometry.polygon.Polygon
        polygon of the extent.
    """
    logger.warning(
        "nlmod.util.polygon_from_extent is deprecated. "
        "Use nlmod.util.extent_to_polygon instead"
    )
    bbox = (extent[0], extent[2], extent[1], extent[3])
    polygon_ext = box(*tuple(bbox))

    return polygon_ext


def gdf_from_extent(extent, crs="EPSG:28992"):
    """Create a geodataframe with a single polygon with the extent given.

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
    logger.warning(
        "nlmod.util.gdf_from_extent is deprecated. "
        "Use nlmod.util.extent_to_gdf instead"
    )
    geom_extent = polygon_from_extent(extent)
    gdf_extent = gpd.GeoDataFrame(geometry=[geom_extent], crs=crs)

    return gdf_extent


def gdf_within_extent(gdf, extent):
    """Select only parts of the geodataframe within the extent.

    Only accepts Polygon and Linestring geometry types.

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
    gdf_extent = extent_to_gdf(extent, crs=gdf.crs)

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
    """Get the filename of a google drive file.

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
        stacklevel=1,
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
    """Download a file from google drive using it's id.

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
        stacklevel=1,
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


def download_modpath_provisional_exe(bindir=None, timeout=120):
    """Download the provisional version of modpath to the folder with binaries."""
    if bindir is None:
        bindir = os.path.join(os.path.dirname(__file__), "bin")
    if not os.path.isdir(bindir):
        os.makedirs(bindir)
    if sys.platform.startswith("win"):
        fname = "mp7_win64_20231016_86b38df.exe"
    elif sys.platform.startswith("darwin"):
        fname = "mp7_mac_20231016_86b38df"
    elif sys.platform.startswith("linux"):
        fname = "mp7_linux_20231016_86b38df"
    else:
        raise (OSError(f"Unknown platform: {sys.platform}"))
    url = "https://github.com/MODFLOW-USGS/modpath-v7/raw/develop/msvs/bin_PROVISIONAL"
    url = f"{url}/{fname}"
    r = requests.get(url, allow_redirects=True, timeout=timeout)
    ext = os.path.splitext(fname)[-1]
    fname = os.path.join(bindir, f"mp7_2_002_provisional{ext}")
    with open(fname, "wb") as file:
        file.write(r.content)


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
    """Get a logger with colored output.

    Parameters
    ----------
    level : str, optional
        The logging level to set for the logger. Default is "INFO".

    Returns
    -------
    logger : logging.Logger
        The configured logger object.
    """
    if level == "DEBUG":
        FORMAT = "{color}{levelname}:{name}.{funcName}:{lineno}:{message}{reset}"
    else:
        FORMAT = "{color}{levelname}:{name}.{funcName}:{message}{reset}"
    formatter = ColoredFormatter(
        FORMAT,
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


def _get_value_from_ds_attr(
    ds, varname, attr=None, value=None, default=None, warn=True
):
    """Internal function to get value from dataset attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing model data
    varname : str
        name of the variable in flopy package
    attr : str, optional
        name of the attribute in dataset (is sometimes different to varname)
    value : Any, optional
        variable value, by default None
    default : Any, optional
        When default is not None, value is None, and attr is not present in ds,
        this default is returned. The default is None.
    warn : bool, optional
        log warning if value not found

    Returns
    -------
    value : Any
        returns variable value, if value was None, attempts to obtain
        variable from dataset attributes.
    """
    if attr is None:
        attr = varname

    if value is not None and (attr in ds.attrs):
        logger.info(
            f"Using user-provided '{varname}' and not stored attribute 'ds.{attr}'"
        )
    elif value is None and (attr in ds.attrs):
        logger.debug(f"Using stored data attribute '{attr}' for '{varname}'")
        value = ds.attrs[attr]
    elif value is None:
        if default is not None:
            logger.debug(f"Using default value of {default} for '{varname}'")
            value = default
        elif warn:
            msg = (
                f"No value found for '{varname}', returning None. "
                f"To fix this error pass '{varname}' to function or set 'ds.{attr}'."
            )
            logger.warning(msg)
        # raise ValueError(msg)
    return value


def _get_value_from_ds_datavar(
    ds,
    varname,
    datavar=None,
    default=None,
    warn=True,
    return_da=False,
    checkcoords=True,
):
    """Internal function to get value from dataset data variables.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing model data
    varname : str
        name of the variable in flopy package (used for logging)
    datavar : Any, optional
        if str, treated as the name of the data variable (which can be
        different to varname) in dataset, if not provided is assumed to be
        the same as varname. If not passed as string, it is treated as data
    default : Any, optional
        When default is not None, datavar is a string, and not present in ds, this
        default is returned. The default is None.
    warn : bool, optional
        log warning if value not found
    return_da : bool, optional
        if True, a DataArray can be returned. If False, a DataArray is always
        converted to a numpy array before being returned. The default is False.
    checkcoords : bool, optional
        if True and datavar is a DataArray, the DataArray coords are checked against
        the Dataset coordinates. Raises an AssertionError if they do not match.

    Returns
    -------
    value : Any
        returns variable value, if value is None or str, attempts to obtain
        variable from dataset data variables.

    Note
    ----
    For optional data, use warn=False, e.g.::

        _get_value_from_ds_datavar(ds, "ss", datavar=None, warn=False)
    """
    # parsing datavar to check things:
    # - varname is the name of the variable in the original function/flopy package
    # - datavar is converted to str or None, used to check for presence in dataset
    # - value is used to store value
    if isinstance(datavar, xr.DataArray):
        value = datavar
        datavar = datavar.name
        if checkcoords:
            check_da_dims_coords(value, ds)
    elif isinstance(datavar, str):
        value = datavar
    else:
        value = datavar
        datavar = None

    # inform user that user-provided variable is used over stored copy
    if (value is not None and not isinstance(value, str)) and (datavar in ds):
        logger.info(
            f"Using user-provided '{varname}' and not"
            f" stored data variable 'ds.{datavar}'"
        )
    # get datavar from dataset if value is None or value is string
    elif ((value is None) or isinstance(value, str)) and (datavar in ds):
        logger.debug(f"Using stored data variable '{datavar}' for '{varname}'")
        value = ds[datavar]
    # warn if value is None
    elif isinstance(value, str):
        if default is not None:
            logger.debug(f"Using default value of {default} for '{varname}'")
            value = default
        else:
            value = None
            if warn:
                msg = (
                    f"No value found for '{varname}', returning None. "
                    f"To silence this warning pass '{varname}' data directly "
                    f"to function or check whether 'ds.{datavar}' was set correctly."
                )
                logger.warning(msg)
    if not return_da:
        if isinstance(value, xr.DataArray):
            value = value.values

    return value


def gdf_intersection_join(
    gdf_from,
    gdf_to,
    columns=None,
    desc="",
    silent=False,
    min_total_overlap=0.5,
    geom_type="Polygon",
    add_index_from_column=None,
):
    """Add information from 'gdf_from' to 'gdf_to', based on the spatial intersection.

    Parameters
    ----------
    gdf_from : gpd.GeoDataFrame
        The GeoDataFrame to add information to.
    gdf_to : gpd.GeoDataFrame
        The GeoDataFrame containing the information to be added.
    columns : list, optional
        A list of the columns to add from gdf_from to gdf_to. When columns is None,
        columns is set to all columns that are present in gdf_from but not in gdf_to.
        The default is None.
    desc : string, optional
        The description of the progressbar. The default is "".
    silent : bool, optional
        If true, do not show the prgressbar. The default is False.
    min_total_overlap : float, optional
        The minimum required total overlap between geometries. If the total overlap of a
        feature in gdf_to with the features of gdf_from, no information is added to the
        feature of gdf_to. The default is 0.5.
    geom_type : string, optional
        The type of Geometries to evaluate. Can be "Polygon" or "LineString". When
        geom_type is "Polygon" the overlap of features are ranked by the area of
        features. When geom_type is "LineString", the overlap is ranked by the length of
        features. The default is "Polygon".
    add_index_from_column : string, optional
        The name of the column to add the index of gdf_from to gdf_to. The default is
        None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    gdf_to : gpd.GeoDataFrame
        gdf_to with extra columns from gdf_from.

    """
    gdf_to = gdf_to.copy()
    if columns is None:
        columns = gdf_from.columns[~gdf_from.columns.isin(gdf_to.columns)]
    s = STRtree(gdf_from.geometry)
    for index in tqdm(gdf_to.index, desc=desc, disable=silent):
        geom_to = gdf_to.geometry[index]
        inds = s.query(geom_to)
        if len(inds) == 0:
            continue
        overlap = gdf_from.geometry.iloc[inds].intersection(geom_to)
        if geom_type is None:
            geom_type = overlap.geom_type.iloc[0]
        if geom_type in ["Polygon", "MultiPolygon"]:
            measure_org = geom_to.area
            measure = overlap.area
        elif geom_type in ["LineString", "MultiLineString"]:
            measure_org = geom_to.length
            measure = overlap.length
        else:
            msg = f"Unsupported geometry type: {geom_type}"
            raise TypeError(msg)

        if np.any(measure.sum() > min_total_overlap * measure_org):
            # take the largest
            ind = measure.idxmax()
            gdf_to.loc[index, columns] = gdf_from.loc[ind, columns]
            if add_index_from_column:
                gdf_to.loc[index, add_index_from_column] = ind
    return gdf_to


def zonal_statistics(
    gdf,
    da,
    columns=None,
    buffer=0.0,
    engine="geocube",
    all_touched=True,
    statistics="mean",
    add_to_gdf=True,
):
    """Calculate raster statistics in the features of a GeoDataFrame

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame with features for which to calculate the statistics.
    da : xr.DataArray
        A DataArray of the raster.
    columns : str or list of str, optional
        The columns in gdf to add the statistics to. When columns is None, use the name
        of the statistics. When columns is a string, and more than one statistic is
        calculated, the value of columns is used as the prefix before each statistic, to
        form the column-names. The default is None.
    buffer : float, optional
        The buffer, in m, that is added to each of the features of gdf, before
        calculating the statistics. The default is 0.0.
    engine : str, optional
        The engine to use for the calculation of the statistics. The possible values are
        'geocube' and 'rasterstats'. The two engines should approximately result in the
        same values. If features overlap (which will happen more frequently when
        buffer > 0), the result will differ. If engine=='geocube', each raster-cell is
        only designated to one feature. If engine='rasterstats' a raster-cell can be
        designated to multiple features, if features overlap. The default is "geocube".
    all_touched : bool, optional
        If True, include every raster cell touched by a geometry. Otherwise only include
        those having a center point within the polygon. The default is True.
    statistics : str or list of str, optional
        The name or a list of names of the statistics to be calculated. The default is
        "mean".
    add_to_gdf : bool, optional
        Add the result to the orignal GeoDataFrame if True. Otherwise return a
        GeoDataFrame with only the statistics. The default is True.


    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the the statistics in some of its columns.
    """
    if isinstance(statistics, str):
        statistics = [statistics]
    if columns is None:
        columns = statistics
    elif isinstance(columns, str):
        if len(statistics) == 1:
            columns = [columns]
        else:
            columns = [f"{columns}_{stat}" for stat in statistics]
    if add_to_gdf:
        for column in columns:
            if column in gdf.columns:
                logger.warning(
                    f"Column {column} allready exists. It is overwritten by new data."
                )
    # for geocube we need a unique integer index
    geometry = gpd.GeoDataFrame(geometry=gdf.geometry.values, index=range(len(gdf)))
    if buffer != 0.0:
        geometry.geometry = geometry.buffer(buffer)
    if engine == "geocube":
        from geocube.api.core import make_geocube
        from geocube.rasterize import rasterize_image

        gc = make_geocube(
            vector_data=geometry.reset_index(),
            measurements=["index"],
            like=da,  # ensure the data are on the same grid
            rasterize_function=partial(rasterize_image, all_touched=all_touched),
        )
        if gc.index.isnull().all():
            raise (ValueError("There is no overlap between gdf and da"))
        gc["values"] = da
        gc = gc.set_coords("index")
        groups = gc.groupby("index")
        for stat, column in zip(statistics, columns):
            if stat == "min":
                values = groups.min()
            elif stat == "max":
                values = groups.max()
            elif stat == "count":
                values = groups.count()
            elif stat == "mean":
                values = groups.mean()
            elif stat == "median":
                values = groups.median()
            elif stat == "std":
                values = groups.std()
            elif stat == "sum":
                values = groups.sum()
            else:
                raise (NotImplementedError(f"Statistic {stat} not implemented"))
            values = values["values"].to_pandas()
            values.index = values.index.astype(int)
            geometry[column] = values

    elif engine == "rasterstats":
        from rasterstats import zonal_stats

        if isinstance(da, xr.DataArray):
            stats = zonal_stats(
                geometry,
                da.data,
                stats=statistics,
                all_touched=all_touched,
                affine=da.rio.transform(),
                nodata=da.rio.nodata,
            )
        else:
            # we assume da is a filename
            stats = zonal_stats(
                geometry,
                da,
                stats=stat,
                all_touched=all_touched,
            )
        for stat, column in zip(statistics, columns):
            geometry[column] = [x[stat] for x in stats]

    else:
        raise (ValueError(f"Unknown engine: {engine}"))

    if add_to_gdf:
        gdf[columns] = geometry[columns].values
        return gdf
    else:
        geometry.index = gdf.index
        return geometry
