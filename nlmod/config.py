from contextlib import contextmanager

NLMOD_CACHE_OPTIONS = {
    # compare hash for stored netcdf, default is True:
    "nc_hash": True,
    # compare hash for dataset coordinates, default is True:
    "dataset_coords_hash": True,
    # compare hash for dataset data variables, default is True:
    "dataset_data_vars_hash": True,
    # perform explicit comparison of dataset coordinates, default is False:
    "explicit_dataset_coordinate_comparison": False,
}

_DEFAULT_CACHE_OPTIONS = {
    "nc_hash": True,
    "dataset_coords_hash": True,
    "dataset_data_vars_hash": True,
    "explicit_dataset_coordinate_comparison": False,
}


@contextmanager
def cache_options(**kwargs):
    """Context manager for nlmod cache options."""
    set_options(**kwargs)
    try:
        yield get_options()
    finally:
        reset_options(list(kwargs.keys()))


def set_options(**kwargs):
    """
    Set options for the nlmod package.

    Parameters
    ----------
    **kwargs : dict
        Options to set.

    """
    for key, value in kwargs.items():
        if key in NLMOD_CACHE_OPTIONS:
            NLMOD_CACHE_OPTIONS[key] = value
        else:
            raise ValueError(
                f"Unknown option: {key}. Options are: "
                f"{list(NLMOD_CACHE_OPTIONS.keys())}"
            )


def get_options(key=None):
    """
    Get options for the nlmod package.

    Parameters
    ----------
    key : str, optional
        Option to get.

    Returns
    -------
    dict or value
        The options or the value of the requested option.

    """
    if key is None:
        return NLMOD_CACHE_OPTIONS
    else:
        return {key: NLMOD_CACHE_OPTIONS[key]}


def reset_options(options=None):
    """Reset options to default."""
    if options is None:
        set_options(**_DEFAULT_CACHE_OPTIONS)
    else:
        for opt in options:
            set_options(**{opt: _DEFAULT_CACHE_OPTIONS[opt]})
