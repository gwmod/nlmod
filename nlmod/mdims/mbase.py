import os
import sys

import xarray as xr
import datetime as dt

from .. import util


def get_empty_model_ds(model_name, model_ws, mfversion="mf6",
                       exe_name=None):
    """ get an empty model dataset

    Parameters
    ----------
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
    model_ds : xarray dataset
        model dataset.
    """

    model_ds = xr.Dataset()

    model_ds.attrs['model_name'] = model_name
    model_ds.attrs['mfversion'] = mfversion
    model_ds.attrs['model_dataset_created_on'] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")

    if exe_name is None:
        exe_name = os.path.join(os.path.dirname(__file__),
                                '..', '..', 'bin', model_ds.mfversion)

    # if working on Windows add .exe extension
    if sys.platform.startswith('win'):
        exe_name += ".exe"

    model_ds.attrs["exe_name"] = exe_name

    # add some directories
    if model_ws is not None:
        figdir, cachedir = util.get_model_dirs(model_ws)
        model_ds.attrs['model_ws'] = model_ws
        model_ds.attrs['figdir'] = figdir
        model_ds.attrs['cachedir'] = cachedir

    return model_ds
