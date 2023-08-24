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


def get_encodings(ds):
    """Get the encoding for the data_vars. Based on the minimum values and maximum values
    set in `dim_attrs` and the maximum allowed difference from `encoding_requirements`.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the data_vars

    Returns
    -------
    encodings : dict
        Dictionary containing the encodings for each data_var
    """
    encodings = {}
    for varname, da in ds.data_vars.items():
        encodings[varname] = dict()
        encoding = encodings[varname]

        isfloat = np.issubdtype(da.dtype, np.floating)

        if isfloat and varname in encoding_requirements and varname in dim_attrs:
            vmin = dim_attrs[varname]["valid_min"]
            vmax = dim_attrs[varname]["valid_max"]
            dval_max = encoding_requirements[varname]["dval_max"]
            float_as_int16 = is_int16_allowed(vmin, vmax, dval_max)

            if float_as_int16:
                scale_factor, add_offset = compute_scale_and_offset(vmin, vmax, 16)
                encoding["dtype"] = "int16"
                encoding["scale_factor"] = scale_factor
                encoding["add_offset"] = add_offset

            else:
                encoding["dtype"] = "float32"

        if isfloat or np.issubdtype(da.dtype, np.integer):
            # Strings dont support compression
            encoding["zlib"] = True
            encoding["complevel"] = 5
            
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
