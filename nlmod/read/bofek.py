import requests
import zipfile
import io
import geopandas as gpd
from pathlib import Path
from .. import NLMOD_DATADIR, cache, dims, util


@cache.cache_pickle
def get_gdf_bofek(ds=None, extent=None, timeout=120):
    """get geodataframe of bofek 2020 wihtin the extent of the model. It does so by
    downloading a zip file (> 100 MB) and extracting the relevant geodatabase. Therefore
    the function can be slow, ~35 seconds depending on your internet connection.

    Parameters
    ----------
    ds : xr.DataSet, optional
        dataset containing relevant model information. The default is None.
    extent : list, tuple or np.array, optional
        extent xmin, xmax, ymin, ymax. Only used if ds is None. The default is None.
    timeout : int, optional
        timeout time of request in seconds. Default is 120.

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
    bofek_zip_url = "https://www.wur.nl/nl/show/bofek-2020-gis-1.htm"

    if not fname_bofek.exists():
        # download zip
        r = requests.get(bofek_zip_url, timeout=timeout)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # extract 7z
            z.extractall(tmpdir)
        with py7zr.SevenZipFile(fname_7z, mode="r") as z:
            z.extract(
                targets=["GIS/BOFEK2020_bestanden/BOFEK2020.gdb"],
                path=tmpdir,
                recursive=True,
            )

        # clean up
        Path(fname_7z).unlink()

    # read geodatabase
    gdf_bofek = gpd.read_file(fname_bofek)

    if extent is not None:
        # slice to extent
        gdf_bofek = util.gdf_within_extent(gdf_bofek, extent)

    return gdf_bofek
