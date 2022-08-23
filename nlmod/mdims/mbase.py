import datetime as dt
import os
import sys

from .. import util


def set_ds_attrs(ds, model_name, model_ws, mfversion="mf6", exe_name=None):
    """ set the attribute of a model dataset.

    Parameters
    ----------
    ds : xarray dataset
        An existing model dataset
    model_name : str
        name of the model.
    model_ws : str or None
        workspace of the model. This is where modeldata is saved to.
    mfversion : str, optional
        modflow version. The default is "mf6".
    exe_name: str, optional
        path to modflow executable, default is None, which assumes binaries
        are available in nlmod/bin directory. Binaries can be downloaded
        using `nlmod.util.download_mfbinaries()`.

    Returns
    -------
    ds : xarray dataset
        model dataset.
    """

    ds.attrs["model_name"] = model_name
    ds.attrs["mfversion"] = mfversion
    fmt = "%Y%m%d_%H:%M:%S"
    ds.attrs["model_dataset_created_on"] = dt.datetime.now().strftime(fmt)

    if exe_name is None:
        exe_name = os.path.join(
            os.path.dirname(__file__), "..", "bin", mfversion
        )

    # if working on Windows add .exe extension
    if sys.platform.startswith("win"):
        exe_name += ".exe"

    ds.attrs["exe_name"] = exe_name

    # add some directories
    if model_ws is not None:
        figdir, cachedir = util.get_model_dirs(model_ws)
        ds.attrs["model_ws"] = model_ws
        ds.attrs["figdir"] = figdir
        ds.attrs["cachedir"] = cachedir

    return ds
