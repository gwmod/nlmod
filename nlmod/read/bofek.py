import logging
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import requests

from nlmod import NLMOD_DATADIR, cache, util

logger = logging.getLogger(__name__)


@cache.cache_pickle
def get_gdf_bofek(extent, dirname=None, timeout=3600):
    """Get geodataframe of bofek 2020 wihtin the extent of the model.

    It does so by downloading a zip file (> 100 MB) and extracting the relevant
    geodatabase. Therefore the function can be slow, ~35 seconds depending on your
    internet connection.

    Parameters
    ----------
    extent : list, tuple or np.array
        extent xmin, xmax, ymin, ymax.
    dirname : str, optional
        directory name for the bofek2020 files. If None the NLMOD_DATADIR is used to
        store the data. The default is None.
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
    if dirname is None:
        dirname = Path(NLMOD_DATADIR)
    else:
        dirname = Path(dirname)
    fname_bofek = dirname / "GIS" / "BOFEK2020_bestanden" / "BOFEK2020.gdb"
    fname_bofek_geojson = dirname / "bofek" / "BOFEK2020.geojson"
    dirname_bofek = dirname / "bofek"
    dirname_bofek.mkdir(exist_ok=True, parents=True)

    # url
    bofek_zip_url = "https://www.wur.nl/nl/show/bofek-2020-gis-1.htm"

    if not fname_bofek_geojson.exists():
        # download zip
        logger.info("Downloading BOFEK2020 GIS data (~35 seconds)")
        r = requests.get(bofek_zip_url, timeout=timeout, stream=True)

        # download file and unzip in memory
        with zipfile.ZipFile(BytesIO(r.content)) as zf:
            unzipped = zf.read(zf.namelist()[0])

        # unpack unzipped file further, write gdb files to disk (not possible in memory)
        logger.debug("Extracting zipped BOFEK2020 GIS data")
        with py7zr.SevenZipFile(BytesIO(unzipped), mode="r") as z:
            z.extract(
                path=dirname,
                targets=["GIS/BOFEK2020_bestanden/BOFEK2020.gdb"],
                recursive=True,
            )

        # read geodatabase
        logger.debug("convert geodatabase to geojson")
        gdf_bofek = gpd.read_file(fname_bofek)

        # save to geojson
        gdf_bofek.to_file(fname_bofek_geojson, driver="GeoJSON")

        # clean up
        logger.debug("Remove geodatabase")
        shutil.rmtree(fname_bofek)

    # read geojson
    msg = f"read bofek2020 geojson from {fname_bofek_geojson}"
    logger.debug(msg)
    gdf_bofek = gpd.read_file(fname_bofek_geojson)

    if extent is not None:
        # slice to extent
        gdf_bofek = util.gdf_within_extent(gdf_bofek, extent)

    return gdf_bofek
