import numpy as np

dim_attrs = {
    "botm": dict(
        name="Bottom elevation",
        description="Bottom elevation for each model cell",
        units="mNAP",
        valid_min=-2000.,
        valid_max=500.
    ),
    "top": dict(
        name="Top elevation",
        description="Top elevation for each model cell",
        units="mNAP",
        valid_min=-2000.,
        valid_max=500.
    ),
    "kh": dict(
        name="Horizontal hydraulic conductivity",
        description="Horizontal hydraulic conductivity for each model cell",
        units="m/day",
        valid_min=0.,
        valid_max=1000.),
    "kv": dict(
        name="Vertical hydraulic conductivity",
        description="Vertical hydraulic conductivity for each model cell",
        units="m/day",
        valid_min=0.,
        valid_max=1000.),
    "ss": dict(
        name="Specific storage",
        description="Specific storage for each model cell",
        units="1/m",
        valid_min=0.,
        valid_max=1.),
    "sy": dict(
        name="Specific yield",
        description="Specific yield for each model cell",
        units="1",
        valid_min=0.,
        valid_max=1.),
    "porosity": dict(
        name="Porosity",
        description="Porosity for each model cell",
        units="1",
        valid_min=0.,
        valid_max=1.),
    "recharge": dict(
        name="Recharge",
        description="Recharge for each model cell",
        units="m/day",
        valid_min=0.,
        valid_max=0.5),
    "head": dict(
        name="Head",
        description="Head for each model cell",
        units="mNAP",
        valid_min=-100.,
        valid_max=500.),
    "starting_head": dict(
        name="Starting head",
        description="Starting head for each model cell",
        units="mNAP",
        valid_min=-100.,
        valid_max=500.),
    "freshwater_head": dict(
        name="Freshwater head",
        description="Freshwater head for each model cell",
        units="mNAP",
        valid_min=-100.,
        valid_max=500.),
    "pointwater_head": dict(
        name="Pointwater head",
        description="Pointwater head for each model cell",
        units="mNAP",
        valid_min=-100.,
        valid_max=500.),
    "density": dict(
        name="Density",
        description="Density for each model cell",
        units="kg/m3",
        valid_min=950.,
        valid_max=1200.),
    "area": dict(
        name="Cell area",
        description="Cell area for each model cell",
        units="m2",
        valid_min=0.,
        valid_max=1e8),
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


def get_encodings(ds, set_encoding_inplace=True, allowed_to_read_data_vars_for_minmax=True):
    """Get the encoding for the data_vars. Based on the minimum values and maximum values
    set in `dim_attrs` and the maximum allowed difference from `encoding_requirements`.

    If a loss of data resolution is allowed floats can also be stored at int16, halfing
    the space required for storage. The maximum acceptabel loss in resolution (`dval_max`)
    is compared with the expected loss in resolution (`is_int16_allowed()`).

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
    """
    encodings = {}
    for varname, da in ds.data_vars.items():
        encodings[varname] = dict(
            fletcher32=True,  # Store checksums to detect corruption
        )
        encoding = encodings[varname]

        isfloat = np.issubdtype(da.dtype, np.floating)
        isint = np.issubdtype(da.dtype, np.integer)

        # set the dtype, scale_factor and add_offset
        if isfloat and varname in encoding_requirements and varname in dim_attrs:
            valid_min = dim_attrs[varname]["valid_min"]
            valid_max = dim_attrs[varname]["valid_max"]
            dval_max = encoding_requirements[varname]["dval_max"]
            float_as_int16 = is_int16_allowed(valid_min, valid_max, dval_max)

            if float_as_int16:
                # Fillvalue currently clashes with scaling. See:
                # https://stackoverflow.com/questions/75755441/why-does-saving-to-
                # netcdf-without-encoding-change-some-values-to-nan
                if allowed_to_read_data_vars_for_minmax:
                    vmin = float(da.min())
                    vmax = float(da.max())
                else:
                    vmin = valid_min
                    vmax = valid_max

                scale_factor, add_offset = compute_scale_and_offset(vmin, vmax, 16)
                encoding["dtype"] = "int16"
                encoding["scale_factor"] = scale_factor
                encoding["add_offset"] = add_offset

            else:
                encoding["dtype"] = "float32"

        elif isint and allowed_to_read_data_vars_for_minmax:
            vmin = int(da.min())
            vmax = int(da.max())
            
            if vmin >= -32768 and vmax <= 32767:
                encoding["dtype"] = "int16"
            elif vmin >= -2147483648 and vmax <= 2147483647:
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
            
    return encodings


def compute_scale_and_offset(minValue, maxValue, n):
    """
    Computes the scale_factor and offset for the dataset using a minValue and maxValue,
    and int n. Useful for maximizing the compression of a dataset.

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
    scale_factor = (maxValue - minValue) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = minValue + 2 ** (n - 1) * scale_factor
    return scale_factor, add_offset


def is_int16_allowed(vmin, vmax, dval_max):
    nsteps = 2 * 32768
    dval = (vmax - vmin) / nsteps
    print(f"dval: {dval}, dval_max: {dval_max}")
    return (vmax - vmin) / nsteps < dval_max
