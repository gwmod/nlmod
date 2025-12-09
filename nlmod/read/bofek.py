import logging
import shutil
import warnings
import zipfile
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import requests

from .. import cache, util, read
from ..util import tqdm

logger = logging.getLogger(__name__)


def get_gdf_bofek(*args, **kwargs):
    """Get geodataframe of bofek 2020 wihtin the extent of the model.

    It does so by downloading a zip file (> 100 MB) and extracting the relevant
    geodatabase. Therefore the function can be slow, ~35 seconds depending on your
    internet connection.

    .. deprecated:: 0.10.0
        `get_gdf_bofek` will be removed in nlmod 1.0.0, it is replaced by
        `download_bofek_gdf` because of new naming convention
        https://github.com/gwmod/nlmod/issues/47

    Parameters
    ----------
    extent : list, tuple or np.array
        extent xmin, xmax, ymin, ymax.
    dirname : str
        Directory name for the bofek2020 files. This is a temporary directory used to
        store and unpack zip files. The directory will be created if it does not exist.
    timeout : int, optional
        timeout time of request in seconds. Default is 3600.

    Returns
    -------
    gdf_bofek : GeoDataframe
        Bofek2020 geodataframe with a column 'BOFEK2020' containing the bofek cluster
        codes

    Notes
    -----
    An attempt was made to read the geodatabase in memory from the zip file wihtout
    writing data to disk, but this was not successful. Mainly because of the difficulty
    to read the geodatabase in memory.
    """

    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use nlmod.read.bofek.download_bofek_gdf() in the future.",
        DeprecationWarning,
    )

    return download_bofek_gdf(*args, **kwargs)


@cache.cache_pickle
def download_bofek_gdf(extent, dirname, timeout=3600):
    """Get geodataframe of bofek 2020 wihtin the extent of the model.

    It does so by downloading a zip file (> 100 MB) and extracting the relevant
    geodatabase. Therefore the function can be slow, ~35 seconds depending on your
    internet connection.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent xmin, xmax, ymin, ymax.
    dirname : str
        Directory name for the bofek2020 files. This is a temporary directory used to
        store and unpack zip files. The directory will be created if it does not exist.
    timeout : int, optional
        timeout time of request in seconds. Default is 3600.

    Returns
    -------
    gdf_bofek : GeoDataframe
        Bofek2020 geodataframe with a column 'BOFEK2020' containing the bofek cluster
        codes

    Notes
    -----
    An attempt was made to read the geodatabase in memory from the zip file wihtout
    writing data to disk, but this was not successful. Mainly because of the difficulty
    to read the geodatabase in memory.
    """
    import py7zr

    # set paths
    dirname = Path(dirname)
    fname_bofek_gdb = dirname / "GIS" / "BOFEK2020_bestanden" / "BOFEK2020.gdb"

    # create directories if they do not exist
    dirname.mkdir(exist_ok=True, parents=True)

    # url
    bofek_zip_url = "https://www.wur.nl/nl/show/bofek-2020-gis-1.htm"

    # download zip
    logger.info("Downloading BOFEK2020 GIS data (~35 seconds)")
    r = requests.get(bofek_zip_url, timeout=timeout, stream=True)

    # show download progress
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    file_unzipped = BytesIO()
    with tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading BOFEK"
    ) as progress_bar:
        for data in r.iter_content(block_size):
            progress_bar.update(len(data))
            file_unzipped.write(data)

    # extract geodatabase from 7z
    with zipfile.ZipFile(file_unzipped, mode="r") as zf:
        with py7zr.SevenZipFile(BytesIO(zf.read(zf.filelist[0])), mode="r") as z:
            z.extract(
                targets=["GIS/BOFEK2020_bestanden/BOFEK2020.gdb"],
                path=dirname,
                recursive=True,
            )

    # read geodatabase
    logger.debug("convert geodatabase to geojson")
    gdf_bofek = gpd.read_file(fname_bofek_gdb)

    # slice to extent
    gdf_bofek = util.gdf_within_extent(gdf_bofek, extent)

    # clean up
    logger.debug("Remove geodatabase")
    shutil.rmtree(fname_bofek_gdb)

    return gdf_bofek
