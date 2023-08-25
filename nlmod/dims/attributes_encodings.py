import numpy as np

dim_attrs = {
    "botm": dict(
        name="Bottom elevation",
        description="Bottom elevation for each model cell",
        units="mNAP",
        valid_min=-2000.0,
        valid_max=500.0,
    ),
    "top": dict(
        name="Top elevation",
        description="Top elevation for each model cell",
        units="mNAP",
        valid_min=-2000.0,
        valid_max=500.0,
    ),
    "kh": dict(
        name="Horizontal hydraulic conductivity",
        description="Horizontal hydraulic conductivity for each model cell",
        units="m/day",
        valid_min=0.0,
        valid_max=1000.0,
    ),
    "kv": dict(
        name="Vertical hydraulic conductivity",
        description="Vertical hydraulic conductivity for each model cell",
        units="m/day",
        valid_min=0.0,
        valid_max=1000.0,
    ),
    "ss": dict(
        name="Specific storage",
        description="Specific storage for each model cell",
        units="1/m",
        valid_min=0.0,
        valid_max=1.0,
    ),
    "sy": dict(
        name="Specific yield",
        description="Specific yield for each model cell",
        units="-",
        valid_min=0.0,
        valid_max=1.0,
    ),
    "porosity": dict(
        name="Porosity",
        description="Porosity for each model cell",
        units="-",
        valid_min=0.0,
        valid_max=1.0,
    ),
    "recharge": dict(
        name="Recharge",
        description="Recharge for each model cell",
        units="m/day",
        valid_min=0.0,
        valid_max=0.5,
    ),
    "heads": dict(
        name="Heads",
        description="Point water heads for each model cell",
        units="mNAP",
        valid_min=-100.0,
        valid_max=500.0,
    ),
    "starting_head": dict(
        name="Starting head",
        description="Starting head for each model cell",
        units="mNAP",
        valid_min=-100.0,
        valid_max=500.0,
    ),
    "freshwater_head": dict(
        name="Freshwater head",
        description="Freshwater head for each model cell",
        units="mNAP",
        valid_min=-100.0,
        valid_max=500.0,
    ),
    "pointwater_head": dict(
        name="Pointwater head",
        description="Pointwater head for each model cell",
        units="mNAP",
        valid_min=-100.0,
        valid_max=500.0,
    ),
    "density": dict(
        name="Density",
        description="Density for each model cell",
        units="kg/m3",
        valid_min=950.0,
        valid_max=1200.0,
    ),
    "area": dict(
        name="Cell area",
        description="Cell area for each model cell",
        units="m2",
        valid_min=0.0,
        valid_max=1e8,
    ),
}


encoding_requirements = {
    "heads": dict(dval_max=0.005),
    "botm": dict(dval_max=0.005),
    "top": dict(dval_max=0.005),
    "kh": dict(dval_max=1e-6),
    "kv": dict(dval_max=1e-6),
    "ss": dict(dval_max=0.005),
    "sy": dict(dval_max=0.005),
    "porosity": dict(dval_max=0.005),
    "recharge": dict(dval_max=0.0005),
    "starting_head": dict(dval_max=0.005),
    "freshwater_head": dict(dval_max=0.005),
    "pointwater_head": dict(dval_max=0.005),
    "density": dict(dval_max=0.005),
    "area": dict(dval_max=0.05),
}


def get_encodings(
    ds, set_encoding_inplace=True, allowed_to_read_data_vars_for_minmax=True
):
    """Get the encoding for the data_vars. Based on the minimum values and maximum
    values set in `dim_attrs` and the maximum allowed difference from
    `encoding_requirements`.

    If a loss of data resolution is allowed floats can also be stored at int16, halfing
    the space required for storage. The maximum acceptabel loss in resolution
    (`dval_max`) is compared with the expected loss in resolution
    (`is_int16_allowed()`).

    If `set_encoding_inplace` is False, a dictionary with encodings is returned that
    can be passed as argument to `ds.to_netcdf()`. If True, the encodings are set
    inplace; they are stored in the `ds["var"].encoding` for each var seperate.

    If encoding is specified as argument in `ds.to_netcdf()` the encoding stored in the
    `ds["var"].encoding` for each var is ignored.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the data_vars
    set_encoding_inplace : bool
        Set the encoding inplace, by default True
    allowed_to_data_vars : bool
        If True, only data_vars that are allowed to be read are used to calculate the
        minimum and maximum values to estimate the effect of precision loss.
        If False, min max from dim_attrs are used. By default True.

    Returns
    -------
    encodings : dict
        Dictionary containing the encodings for each data_var

    TODO: add support for strings
    """
    encodings = {}
    for varname, da in ds.data_vars.items():
        encodings[varname] = dict(
            fletcher32=True,  # Store checksums to detect corruption
        )
        encoding = encodings[varname]

        assert "_FillValue" not in da.attrs, "Custom fillvalues are not supported."

        isfloat = np.issubdtype(da.dtype, np.floating)
        isint = np.issubdtype(da.dtype, np.integer)

        # set the dtype, scale_factor and add_offset
        if isfloat and varname in encoding_requirements and varname in dim_attrs:
            dval_max = encoding_requirements[varname]["dval_max"]

            if allowed_to_read_data_vars_for_minmax:
                vmin = float(da.min())
                vmax = float(da.max())
            else:
                vmin = dim_attrs[varname]["valid_min"]
                vmax = dim_attrs[varname]["valid_max"]

            float_as_int16 = is_int16_allowed(vmin, vmax, dval_max)

            if float_as_int16:
                scale_factor, add_offset = compute_scale_and_offset(vmin, vmax)
                encoding["dtype"] = "int16"
                encoding["scale_factor"] = scale_factor
                encoding["add_offset"] = add_offset
                encoding["_FillValue"] = -32767  # default for NC_SHORT
                # result = (np.array([vmin, vmax]) - add_offset) / scale_factor
            else:
                encoding["dtype"] = "float32"

        elif isint and allowed_to_read_data_vars_for_minmax:
            vmin = int(da.min())
            vmax = int(da.max())

            if vmin >= -32766 and vmax <= 32767:
                encoding["dtype"] = "int16"
            elif vmin >= -2147483646 and vmax <= 2147483647:
                encoding["dtype"] = "int32"
            else:
                encoding["dtype"] = "int64"
        else:
            pass

        # set the compression
        if isfloat or isint:
            # Strings dont support compression. Only floats and ints for now.
            encoding["zlib"] = True
            encoding["complevel"] = 5

    if set_encoding_inplace:
        da.encoding = encoding
    else:
        return encodings


def compute_scale_and_offset(minValue, maxValue):
    """
    Computes the scale_factor and offset for the dataset using a minValue and maxValue,
    and int16. Useful for maximizing the compression of a dataset.

    Parameters
    ----------
    minValue : float
        Minimum value of the dataset
    maxValue : float
        Maximum value of the dataset

    Returns
    -------
    scale_factor : float
        Scale factor for the dataset
    add_offset : float
        Add offset for the dataset
    """
    # stretch/compress data to the available packed range
    # from -32766 to 32767, because -32767 is the default fillvalue.
    width = 32766 + 32767
    scale_factor = (maxValue - minValue) / width
    add_offset = 1.0 + scale_factor * (width - 1) / 2
    return scale_factor, add_offset


def is_int16_allowed(vmin, vmax, dval_max):
    """compute the loss of resolution by storing a float as int16 (`dval`). And
    compare it with the maximum allowed loss of resolution (`dval_max`).

    Parameters
    ----------
    vmin : float
        Minimum value of the dataset
    vmax : float
        Maximum value of the dataset
    dval_max : float
        Maximum allowed loss of resolution

    Returns
    -------
    bool
        True if the loss of resolution is allowed, False otherwise"""
    nsteps = 32766 + 32767
    dval = (vmax - vmin) / nsteps
    return dval <= dval_max
