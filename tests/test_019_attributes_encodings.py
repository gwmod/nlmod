import os
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from nlmod.dims.attributes_encodings import get_encodings


def test_encodings_float_as_int16():
    """Test if the encodings are correct."""
    # Test is the encodings work for floats where degradation to int16 is allowed
    heads_data = np.arange(1.0, 6.0)
    heads_data[1] = np.nan

    por_data = np.linspace(0.0, 1.0, 5)
    por_data[1] = np.nan

    ds = xr.Dataset(
        data_vars=dict(
            heads=xr.DataArray(data=heads_data),
            porosity=xr.DataArray(data=por_data),
        )
    )
    encodings = get_encodings(
        ds, set_encoding_inplace=False, allowed_to_read_data_vars_for_minmax=True
    )

    assert encodings["heads"]["dtype"] == "int16", "dtype should be int16"
    assert encodings["porosity"]["dtype"] == "int16", "dtype should be int16"

    # test writing to temporary netcdf file
    with TemporaryDirectory() as tmpdir:
        fp_test = os.path.join(tmpdir, "test2.nc")
        ds.to_netcdf(fp_test, encoding=encodings)

        with xr.open_dataset(fp_test, mask_and_scale=True) as ds2:
            ds2.load()

    dval_max = float(ds["heads"].max() - ds["heads"].min()) / (32766 + 32767)

    assert np.allclose(
        ds["heads"].values, ds2["heads"].values, atol=dval_max, rtol=0.0, equal_nan=True
    )
    assert np.all(np.isnan(ds["heads"]) == np.isnan(ds2["heads"]))

    # Test is the encodings work for floats where degradation to int16 is not allowed
    data = np.arange(1.0, 1e6)
    data[1] = np.nan

    ds = xr.Dataset(data_vars=dict(heads=xr.DataArray(data=data)))
    encodings = get_encodings(
        ds, set_encoding_inplace=False, allowed_to_read_data_vars_for_minmax=True
    )
    assert encodings["heads"]["dtype"] == "float32", "dtype should be float32"
    pass


def test_encondings_inplace():
    """Test if the encodings inplace are correct."""
    # Test is the encodings work for floats where degradation to int16 is allowed
    data = np.arange(1.0, 5.0)
    data[1] = np.nan

    ds = xr.Dataset(data_vars=dict(heads=xr.DataArray(data=data)))
    ds_inplace = ds.copy(deep=True)

    encodings = get_encodings(
        ds, set_encoding_inplace=False, allowed_to_read_data_vars_for_minmax=True
    )
    get_encodings(
        ds_inplace,
        set_encoding_inplace=False,
        allowed_to_read_data_vars_for_minmax=True,
    )

    # test writing to temporary netcdf file
    with TemporaryDirectory() as tmpdir:
        fp_test = os.path.join(tmpdir, "test2.nc")
        ds.to_netcdf(fp_test, encoding=encodings)

        with xr.open_dataset(fp_test, mask_and_scale=True) as ds2:
            ds2.load()

        fp_test_inplace = os.path.join(tmpdir, "test_inplace.nc")
        ds_inplace.to_netcdf(fp_test_inplace)

        with xr.open_dataset(fp_test_inplace, mask_and_scale=True) as ds_inplace2:
            ds_inplace2.load()

    assert np.allclose(ds2["heads"].values, ds_inplace2["heads"].values, equal_nan=True)
    pass


test_encodings_float_as_int16()
