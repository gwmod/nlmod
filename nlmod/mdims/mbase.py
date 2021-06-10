import sys
import xarray as xr


def get_empty_model_ds(model_name, model_ws, mfversion="mf6", exe_name="mf6"):

    model_ds = xr.Dataset()

    model_ds.attrs['model_name'] = model_name
    model_ds.attrs['model_ws'] = model_ws
    model_ds.attrs['mfversion'] = mfversion

    # if working on Windows add .exe extension
    if sys.platform.startswith('win'):
        exe_name += ".exe"

    model_ds.attrs["exe_name"] = exe_name

    return model_ds
