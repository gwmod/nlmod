import logging
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from tqdm.auto import tqdm

from nlmod import NLMOD_DATADIR, cache, dims, util

logger = logging.getLogger(__name__)


@cache.cache_pickle
def get_gdf_bofek(ds=None, extent=None, timeout=3600):
    """Get geodataframe of bofek 2020 wihtin the extent of the model.

    It does so by downloading a zip file (> 100 MB) and extracting the relevant
    geodatabase. Therefore the function can be slow, ~35 seconds depending on your
    internet connection.

    Parameters
    ----------
    ds : xr.DataSet, optional
        dataset containing relevant model information. The default is None.
    extent : list, tuple or np.array, optional
        extent xmin, xmax, ymin, ymax. Only used if ds is None. The default is None.
    timeout : int, optional
        timeout time of request in seconds. Default is 3600.

    Returns
    -------
    gdf_bofek : GeoDataframe
        Bofek2020 geodataframe with a column 'BOFEK2020' containing the bofek cluster
        codes
    """
    import py7zr

    if extent is None and ds is not None:
        extent = dims.get_extent(ds)

    # set paths
    tmpdir = Path(NLMOD_DATADIR)

    fname_7z = tmpdir / "BOFEK2020_GIS.7z"
    fname_bofek = tmpdir / "GIS" / "BOFEK2020_bestanden" / "BOFEK2020.gdb"
    fname_bofek_geojson = tmpdir / "bofek" / "BOFEK2020.geojson"
    bofek_zip_url = "https://www.wur.nl/nl/show/bofek-2020-gis-1.htm"

    if not fname_bofek_geojson.exists():
        # download zip
        logger.info("Downloading BOFEK2020 GIS data (~35 seconds)")
        r = requests.get(bofek_zip_url, timeout=timeout, stream=True)

        # show download progress
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading BOFEK"
        ) as progress_bar:
            with open(tmpdir / "bofek.zip", "wb") as file:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        # unpack zips
        logger.debug("Extracting zipped BOFEK2020 GIS data")
        with zipfile.ZipFile(tmpdir / "bofek.zip") as z:
            # extract 7z
            z.extractall(tmpdir)

        with py7zr.SevenZipFile(fname_7z, mode="r") as z:
            z.extract(
                targets=["GIS/BOFEK2020_bestanden/BOFEK2020.gdb"],
                path=tmpdir,
                recursive=True,
            )

        # clean up
        logger.debug("Remove zip files")
        Path(tmpdir / "bofek.zip").unlink()
        Path(fname_7z).unlink()

        # read geodatabase
        logger.debug("convert geodatabase to geojson")
        gdf_bofek = gpd.read_file(fname_bofek)

        # save to geojson
        gdf_bofek.to_file(fname_bofek_geojson, driver="GeoJSON")

        # remove geodatabase
        shutil.rmtree(fname_bofek)

    # read geojson
    logger.debug(f"read bofek2020 geojson from {fname_bofek_geojson}")
    gdf_bofek = gpd.read_file(fname_bofek_geojson)

    if extent is not None:
        # slice to extent
        gdf_bofek = util.gdf_within_extent(gdf_bofek, extent)

    return gdf_bofek
