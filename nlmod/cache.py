import functools
import hashlib
import importlib
import inspect
import logging
import numbers
import os
import pickle

import dask
import flopy
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from xarray.testing import assert_identical

from .config import NLMOD_CACHE_OPTIONS

logger = logging.getLogger(__name__)


def clear_cache(cachedir, prompt=True):
    """Clears the cache in a given cache directory by removing all .pklz and
    corresponding .nc files.

    Parameters
    ----------
    cachedir : str
        path to cache directory.
    prompt : bool, optional
        Ask for confirmation before removing the cache. The default is True.

    Returns
    -------
    None.
    """
    if prompt:
        ans = input(f"this will remove all cached files in {cachedir} are you sure [Y/N]")
        if ans.lower() != "y":
            return

    for fname in os.listdir(cachedir):
        # assuming all pklz files belong to a cached netcdf file
        if fname.endswith(".pklz"):
            fname_nc = fname.replace(".pklz", ".nc")

            # remove pklz file
            os.remove(os.path.join(cachedir, fname))
            msg = f"removed {fname}"
            logger.info(msg)

            # remove netcdf file
            fpath_nc = os.path.join(cachedir, fname_nc)
            if os.path.exists(fname_nc):
                # make sure cached netcdf is closed
                cached_ds = xr.open_dataset(fpath_nc, decode_coords='all')
                cached_ds.close()
                os.remove(fpath_nc)
                msg = f"removed {fname_nc}"
                logger.info(msg)


def cache_netcdf(
    coords_2d=False,
    coords_3d=False,
    coords_time=False,
    attrs_ds=False,
    datavars=None,
    coords=None,
    attrs=None,
    nc_hash=True,
):
    """Decorator to read/write the result of a function from/to a file to speed up
    function calls with the same arguments. Should only be applied to functions that:
        - return an Xarray Dataset
        - have no more than one xarray dataset as function argument
        - have functions arguments of types that can be checked using the
        _is_valid_cache functions

    1. The directory and filename of the cache should be defined by the person
    calling a function with this decorator. If not defined no cache is
    created nor used.
    2. Create a new cached file if it is impossible to check if the function
    arguments used to create the cached file are the same as the current
    function arguments. This can happen if one of the function arguments has a
    type that cannot be checked using the _is_valid_cache function.
    3. Function arguments are pickled together with the cache to check later
    if the cache is valid.
    4. If one of the function arguments is an xarray Dataset it is not pickled.
    Therefore we cannot check if this function argument is identical for the
    cached data and the new function call. We do check if the xarray Dataset
    coördinates correspond to the coördinates of the cached netcdf file.
    5. This function uses `functools.wraps` and some home made
    magic in _update_docstring_and_signature to add arguments of the decorator
    to the decorated function. This assumes that the decorated function has a
    docstring with a "Returns" heading. If this is not the case an error is
    raised when trying to decorate the function.

    If all kwargs are left to their defaults, the function caches the full dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions and coordinates.
    coords_2d : bool, optional
        Shorthand for adding 2D coordinates. The default is False.
    coords_3d : bool, optional
        Shorthand for adding 3D coordinates. The default is False.
    coords_time : bool, optional
        Shorthand for adding time coordinates. The default is False.
    attrs_ds : bool, optional
        Shorthand for adding model dataset attributes. The default is False.
    datavars : list, optional
        List of data variables to check for. The default is an empty list.
    coords : list, optional
        List of coordinates to check for. The default is an empty list.
    attrs : list, optional
        List of attributes to check for. The default is an empty list.
    nc_hash: bool, optional
        check if the pickled function arguments belong to the cached netcdf file.
        Default is True.
    """

    def decorator(func):
        # add cachedir and cachename to docstring
        _update_docstring_and_signature(func)

        @functools.wraps(func)
        def wrapper(*args, cachedir=None, cachename=None, **kwargs):
            # 1 check if cachedir and name are provided
            if cachedir is None or cachename is None:
                return func(*args, **kwargs)

            if not cachename.endswith(".nc"):
                cachename += ".nc"

            fname_cache = os.path.join(cachedir, cachename)  # netcdf file
            fname_pickle_cache = fname_cache.replace(".nc", ".pklz")

            # adjust args and kwargs with minimal dataset
            args_adj = []
            kwargs_adj = {}

            datasets = []
            func_args_dic = {}

            for i, arg in enumerate(args):
                if isinstance(arg, xr.Dataset):
                    arg_adj = ds_contains(
                        arg,
                        coords_2d=coords_2d,
                        coords_3d=coords_3d,
                        coords_time=coords_time,
                        attrs_ds=attrs_ds,
                        datavars=datavars,
                        coords=coords,
                        attrs=attrs,
                    )
                    args_adj.append(arg_adj)
                    datasets.append(arg_adj)
                else:
                    args_adj.append(arg)
                    func_args_dic[f"arg{i}"] = arg

            for key, arg in kwargs.items():
                if isinstance(arg, xr.Dataset):
                    arg_adj = ds_contains(
                        arg,
                        coords_2d=coords_2d,
                        coords_3d=coords_3d,
                        coords_time=coords_time,
                        attrs_ds=attrs_ds,
                        datavars=datavars,
                        coords=coords,
                        attrs=attrs,
                    )
                    kwargs_adj[key] = arg_adj
                    datasets.append(arg_adj)
                else:
                    kwargs_adj[key] = arg
                    func_args_dic[key] = arg

            if len(datasets) == 0:
                dataset = None
            elif len(datasets) == 1:
                dataset = datasets[0]
            else:
                raise NotImplementedError(
                    "Function was called with multiple xarray dataset arguments. "
                    "Currently unsupported."
                )

            # only use cache if the cache file and the pickled function arguments exist
            if os.path.exists(fname_cache) and os.path.exists(fname_pickle_cache):
                # check if you can read the pickle, there are several reasons why a
                # pickle can not be read.
                try:
                    with open(fname_pickle_cache, "rb") as f:
                        func_args_dic_cache = pickle.load(f)
                    pickle_check = True

                except (pickle.UnpicklingError, ModuleNotFoundError):
                    logger.info("could not read pickle, not using cache")
                    pickle_check = False
                    argument_check = False

                # check if the module where the function is defined was changed
                # after the cache was created
                time_mod_func = _get_modification_time(func)
                time_mod_cache = os.path.getmtime(fname_cache)
                modification_check = time_mod_cache > time_mod_func

                if not modification_check:
                    logger.info(
                        f"module of function {func.__name__} recently modified, "
                        "not using cache"
                    )

                with xr.open_dataset(fname_cache, decode_coords='all') as cached_ds:
                    cached_ds.load()

                if pickle_check:
                    # Ensure that the pickle pairs with the netcdf, see #66.
                    if NLMOD_CACHE_OPTIONS["nc_hash"] and nc_hash:
                        with open(fname_cache, "rb") as myfile:
                            cache_bytes = myfile.read()
                        func_args_dic["_nc_hash"] = hashlib.sha256(
                            cache_bytes
                        ).hexdigest()

                    if dataset is not None:
                        if NLMOD_CACHE_OPTIONS["dataset_coords_hash"]:
                            # Check the coords of the dataset argument
                            func_args_dic["_dataset_coords_hash"] = dask.base.tokenize(
                                dict(dataset.coords)
                            )
                        else:
                            func_args_dic_cache.pop("_dataset_coords_hash", None)
                            logger.warning(
                                "cache -> dataset coordinates not checked, "
                                "disabled in global config. See "
                                "`nlmod.config.NLMOD_CACHE_OPTIONS`."
                            )
                            if not NLMOD_CACHE_OPTIONS[
                                "explicit_dataset_coordinate_comparison"
                            ]:
                                logger.warning(
                                    "It is recommended to turn on "
                                    "`explicit_dataset_coordinate_comparison` "
                                    "in global config when hash check is turned off!"
                                )

                        if NLMOD_CACHE_OPTIONS["dataset_data_vars_hash"]:
                            # Check the data_vars of the dataset argument
                            func_args_dic["_dataset_data_vars_hash"] = (
                                dask.base.tokenize(dict(dataset.data_vars))
                            )
                        else:
                            func_args_dic_cache.pop("_dataset_data_vars_hash", None)
                            logger.warning(
                                "cache -> dataset data vars not checked, "
                                "disabled in global config. See "
                                "`nlmod.config.NLMOD_CACHE_OPTIONS`."
                            )

                    # check if cache was created with same function arguments as
                    # function call
                    argument_check = _same_function_arguments(
                        func_args_dic, func_args_dic_cache
                    )

                    # explicit check on input dataset coordinates and cached dataset
                    if NLMOD_CACHE_OPTIONS[
                        "explicit_dataset_coordinate_comparison"
                    ] and isinstance(dataset, (xr.DataArray, xr.Dataset)):
                        b = _explicit_dataset_coordinate_comparison(dataset, cached_ds)
                        # update argument check
                        argument_check = argument_check and b

                cached_ds = _check_for_data_array(cached_ds)
                if modification_check and argument_check and pickle_check:
                    msg = f"using cached data -> {cachename}"
                    logger.info(msg)
                    return cached_ds

            # create cache
            result = func(*args_adj, **kwargs_adj)
            msg = f"caching data -> {cachename}"
            logger.info(msg)

            if isinstance(result, xr.DataArray):
                # set the DataArray as a variable in a new Dataset
                result = xr.Dataset({"__xarray_dataarray_variable__": result})

            if isinstance(result, xr.Dataset):
                # close cached netcdf (otherwise it is impossible to overwrite)
                if os.path.exists(fname_cache):
                    with xr.open_dataset(fname_cache, decode_coords='all') as cached_ds:
                        cached_ds.load()

                # write netcdf cache
                # check if dataset is chunked for writing with dask.delayed
                first_data_var = next(iter(result.data_vars.keys()))
                if result[first_data_var].chunks:
                    delayed = result.to_netcdf(fname_cache, compute=False)
                    with ProgressBar():
                        delayed.compute()
                    # close and reopen dataset to ensure data is read from
                    # disk, and not from opendap
                    result.close()
                    result = xr.open_dataset(fname_cache, decode_coords='all', chunks="auto")
                else:
                    result.to_netcdf(fname_cache)

                # add netcdf hash to function arguments dic, see #66
                if NLMOD_CACHE_OPTIONS["nc_hash"] and nc_hash:
                    with open(fname_cache, "rb") as myfile:
                        cache_bytes = myfile.read()
                    func_args_dic["_nc_hash"] = hashlib.sha256(cache_bytes).hexdigest()

                # Add dataset argument hash to function arguments dic
                if dataset is not None:
                    if NLMOD_CACHE_OPTIONS["dataset_coords_hash"]:
                        func_args_dic["_dataset_coords_hash"] = dask.base.tokenize(
                            dict(dataset.coords)
                        )
                    else:
                        logger.warning(
                            "cache -> not writing dataset coordinates hash to "
                            "pickle file, disabled in global config. See "
                            "`nlmod.config.NLMOD_CACHE_OPTIONS`."
                        )
                    if NLMOD_CACHE_OPTIONS["dataset_data_vars_hash"]:
                        func_args_dic["_dataset_data_vars_hash"] = dask.base.tokenize(
                            dict(dataset.data_vars)
                        )
                    else:
                        logger.warning(
                            "cache -> not writing dataset data vars hash to "
                            "pickle file, disabled in global config. See "
                            "`nlmod.config.NLMOD_CACHE_OPTIONS`."
                        )

                # pickle function arguments
                with open(fname_pickle_cache, "wb") as fpklz:
                    pickle.dump(func_args_dic, fpklz)
            else:
                msg = f"expected xarray Dataset, got {type(result)} instead"
                raise TypeError(msg)
            return _check_for_data_array(result)

        return wrapper

    return decorator


def cache_pickle(func):
    """Decorator to read/write the result of a function from/to a file to speed
    up function calls with the same arguments. Should only be applied to
    functions that:

        - return a picklable object
        - have functions arguments of types that can be checked using the
        _is_valid_cache functions

    1. The directory and filename of the cache should be defined by the person
    calling a function with this decorator. If not defined no cache is
    created nor used.
    2. Create a new cached file if it is impossible to check if the function
    arguments used to create the cached file are the same as the current
    function arguments. This can happen if one of the function arguments has a
    type that cannot be checked using the _is_valid_cache function.
    3. Function arguments are pickled together with the cache to check later
    if the cache is valid.
    4. This function uses `functools.wraps` and some home made
    magic in _update_docstring_and_signature to add arguments of the decorator
    to the decorated function. This assumes that the decorated function has a
    docstring with a "Returns" heading. If this is not the case an error is
    raised when trying to decorate the function.
    """
    # add cachedir and cachename to docstring
    _update_docstring_and_signature(func)

    @functools.wraps(func)
    def decorator(*args, cachedir=None, cachename=None, **kwargs):
        # 1 check if cachedir and name are provided
        if cachedir is None or cachename is None:
            return func(*args, **kwargs)

        if not cachename.endswith(".pklz"):
            cachename += ".pklz"

        fname_cache = os.path.join(cachedir, cachename)  # pklz file
        fname_pickle_cache = fname_cache.replace(".pklz", "__cache__.pklz")

        # create dictionary with function arguments
        func_args_dic = {f"arg{i}": args[i] for i in range(len(args))}
        func_args_dic.update(kwargs)

        # only use cache if the cache file and the pickled function arguments exist
        if os.path.exists(fname_cache) and os.path.exists(fname_pickle_cache):
            # check if you can read the function argument pickle, there are
            # several reasons why a pickle can not be read.
            try:
                with open(fname_pickle_cache, "rb") as f:
                    func_args_dic_cache = pickle.load(f)
                pickle_check = True
            except (pickle.UnpicklingError, ModuleNotFoundError):
                logger.info("could not read pickle, not using cache")
                pickle_check = False
                argument_check = False

            # check if the module where the function is defined was changed
            # after the cache was created
            time_mod_func = _get_modification_time(func)
            time_mod_cache = os.path.getmtime(fname_cache)
            modification_check = time_mod_cache > time_mod_func

            if not modification_check:
                msg = (
                    f"module of function {func.__name__} recently modified, "
                    "not using cache"
                )
                logger.info(msg)

            # check if you can read the cached pickle, there are
            # several reasons why a pickle can not be read.
            try:
                with open(fname_cache, "rb") as f:
                    cached_pklz = pickle.load(f)
            except (pickle.UnpicklingError, ModuleNotFoundError):
                logger.info("could not read pickle, not using cache")
                pickle_check = False

            if pickle_check:
                # add dataframe hash to function arguments dic
                func_args_dic["_pklz_hash"] = joblib.hash(cached_pklz)

                # check if cache was created with same function arguments as
                # function call
                argument_check = _same_function_arguments(
                    func_args_dic, func_args_dic_cache
                )

            if modification_check and argument_check and pickle_check:
                msg = f"using cached data -> {cachename}"
                logger.info(msg)
                return cached_pklz

        # create cache
        result = func(*args, **kwargs)
        msg = f"caching data -> {cachename}"
        logger.info(msg)

        if isinstance(result, pd.DataFrame):
            # write pklz cache
            result.to_pickle(fname_cache)

            # add dataframe hash to function arguments dic
            with open(fname_cache, "rb") as f:
                temp = pickle.load(f)
            func_args_dic["_pklz_hash"] = joblib.hash(temp)

            # pickle function arguments
            with open(fname_pickle_cache, "wb") as fpklz:
                pickle.dump(func_args_dic, fpklz)
        else:
            msg = f"expected DataFrame, got {type(result)} instead"
            raise TypeError(msg)
        return result

    return decorator


def _same_function_arguments(func_args_dic, func_args_dic_cache):
    """Checks if two dictionaries with function arguments are identical.

    The following items are checked:
        1. if they have the same keys
        2. if the items have the same type
        3. if the items have the same values (only implemented for the types: int,
           float, bool, str, bytes, list, tuple, dict, np.ndarray, xr.DataArray,
           flopy.mf6.ModflowGwf).

    Parameters
    ----------
    func_args_dic : dictionary
        dictionary with all the args and kwargs of a function call.
    func_args_dic_cache : dictionary
        dictionary with all the args and kwargs of a previous function call of
        which the results are cached.

    Returns
    -------
    bool
        if True the dictionaries are identical which means that the cached
        data was created using the same function arguments as the requested
        data.

    Notes
    -----
    Keys that end with '_hash' are assumed to be hashes and not function arguments. They
    are checked equally.

    """
    for key, item in func_args_dic.items():
        # check if cache and function call have same argument names
        if key not in func_args_dic_cache:
            msg = (
                f"cache was created using different function argument '{key}' "
                "not in cached arguments, do not use cached data"
            )
            logger.info(msg)
            return False

        # check if cache and function call have same argument types
        if not isinstance(item, type(func_args_dic_cache[key])):
            msg = (
                f"cache was created using different function argument types for {key}: "
                f"current '{type(item)}' cache: '{type(func_args_dic_cache[key])}', "
                "do not use cached data"
            )
            logger.info(msg)
            return False

        # check if cache and function call have same argument values
        if item is None:
            # Value of None type is always None so the check happens in previous if statement
            pass
        elif isinstance(item, (numbers.Number, bool, str, bytes, list, tuple)):
            if item != func_args_dic_cache[key]:
                if key.endswith("_hash") and isinstance(item, str):
                    logger.info(
                        f"cached hashes do not match: {key}, do not use cached data"
                    )
                else:
                    logger.info(
                        f"cache was created using different function argument: {key}, "
                        "do not use cached data"
                    )
                logger.debug(f"{key}: {item} != {func_args_dic_cache[key]}")
                return False
        elif isinstance(item, np.ndarray):
            if not np.allclose(item, func_args_dic_cache[key]):
                logger.info(
                    f"cache was created using different numpy array for: {key}, "
                    "do not use cached data"
                )
                logger.debug(
                    f"array '{key}' max difference with stored copy is "
                    f"{np.max(np.abs(item - func_args_dic_cache[key]))}"
                )
                return False
        elif isinstance(item, (pd.DataFrame, pd.Series, xr.DataArray)):
            if not item.equals(func_args_dic_cache[key]):
                logger.info(
                    "cache was created using different DataFrame/Series/DataArray for: "
                    f"{key}, do not use cached data"
                )
                return False
        elif isinstance(item, dict):
            # recursive checking
            if not _same_function_arguments(item, func_args_dic_cache[key]):
                logger.info(
                    f"cache was created using a different dictionary for: {key}, "
                    "do not use cached data"
                )
                return False
        elif isinstance(item, (flopy.mf6.ModflowGwf, flopy.modflow.mf.Modflow)):
            if str(item) != str(func_args_dic_cache[key]):
                logger.info(
                    "cache was created using different groundwater flow model for: "
                    f"{key}, do not use cached data"
                )
                return False

        elif isinstance(item, flopy.utils.gridintersect.GridIntersect):
            i2 = func_args_dic_cache[key]
            is_method_equal = item.method == i2.method

            # check if mfgrid is equal except for cache_dict and polygons
            excl = ("_cache_dict", "_polygons")
            mfgrid1 = {k: v for k, v in item.mfgrid.__dict__.items() if k not in excl}
            mfgrid2 = {k: v for k, v in i2.mfgrid.__dict__.items() if k not in excl}

            is_same_length_props = all(
                np.all(np.size(v) == np.size(mfgrid2[k])) for k, v in mfgrid1.items()
            )

            if (
                not is_method_equal
                or mfgrid1.keys() != mfgrid2.keys()
                or not is_same_length_props
            ):
                logger.info(
                    f"cache was created using different gridintersect object: {key}, "
                    "do not use cached data"
                )
                return False

            is_other_props_equal = all(
                np.all(v == mfgrid2[k]) for k, v in mfgrid1.items()
            )

            if not is_other_props_equal:
                logger.info(
                    f"cache was created using different gridintersect object: {key}, "
                    "do not use cached data"
                )
                return False

        else:
            logger.info(
                f"cannot check if cache argument {key} is valid, assuming invalid cache"
                f", function argument of type {type(item)}"
            )
            return False

    return True


def _get_modification_time(func):
    """Return the modification time of the module where func is defined.

    Parameters
    ----------
    func : function
        function.

    Returns
    -------
    float
        modification time of module.
    """
    mod = func.__module__
    active_mod = importlib.import_module(mod.split(".")[0])
    if "." in mod:
        for submod in mod.split(".")[1:]:
            active_mod = getattr(active_mod, submod)

    return os.path.getmtime(active_mod.__file__)


def _update_docstring_and_signature(func):
    """Add function arguments 'cachedir' and 'cachename' to the docstring and signature
    of a function.

    The function arguments are added before the "Returns" header in the
    docstring. If the function has no Returns header in the docstring, the function
    arguments are not added to the docstring.

    Parameters
    ----------
    func : function
        function that is decorated.

    Returns
    -------
    None
    """
    # add cachedir and cachename to signature
    sig = inspect.signature(func)
    cur_param = tuple(sig.parameters.values())
    if cur_param[-1].name == "kwargs":
        add_kwargs = cur_param[-1]
        cur_param = cur_param[:-1]
    else:
        add_kwargs = None
    new_param = (
        *cur_param,
        inspect.Parameter(
            "cachedir", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        ),
        inspect.Parameter(
            "cachename", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        ),
    )
    if add_kwargs is not None:
        new_param = (*new_param, add_kwargs)
    sig = sig.replace(parameters=new_param)
    func.__signature__ = sig

    # add cachedir and cachename to docstring
    original_doc = func.__doc__
    if original_doc is None:
        msg = f'Function "{func.__name__}" has no docstring'
        logger.warning(msg)
        return
    if "Returns" not in original_doc:
        msg = f'Function "{func.__name__}" has no "Returns" header in docstring'
        logger.warning(msg)
        return
    before, after = original_doc.split("Returns")
    mod_before = (
        before.strip() + "\n    cachedir : str or None, optional\n"
        "        directory to save cache. If None no cache is used."
        " Default is None.\n    cachename : str or None, optional\n"
        "        filename of netcdf cache. If None no cache is used."
        " Default is None.\n\n    Returns"
    )
    new_doc = f"{mod_before}{after}"
    func.__doc__ = new_doc
    return


def _check_for_data_array(ds):
    """Check if the saved NetCDF-file represents a DataArray or a Dataset, and return
    this data-variable.

    The file contains a DataArray when a variable called "__xarray_dataarray_variable__"
    is present in the Dataset. If so, return a DataArray, otherwise return the Dataset.

    By saving the DataArray, the coordinate "spatial_ref" was saved as a separate
    variable. Therefore, add this variable as a coordinate to the DataArray again.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions and coordinates.

    Returns
    -------
    ds : xr.Dataset or xr.DataArray
        A Dataset or DataArray containing the cached data.
    """
    if "__xarray_dataarray_variable__" in ds:
        spatial_ref = ds.spatial_ref if "spatial_ref" in ds else None
        # the method returns a DataArray, so we return only this DataArray
        ds = ds["__xarray_dataarray_variable__"]
        if spatial_ref is not None:
            ds = ds.assign_coords({"spatial_ref": spatial_ref})
    return ds


def ds_contains(
    ds,
    coords_2d=False,
    coords_3d=False,
    coords_time=False,
    attrs_ds=False,
    datavars=None,
    coords=None,
    attrs=None,
):
    """Returns a Dataset containing only the required data.

    If all kwargs are left to their defaults, the function returns the full dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions and coordinates.
    coords_2d : bool, optional
        Shorthand for adding 2D coordinates. The default is False.
    coords_3d : bool, optional
        Shorthand for adding 3D coordinates. The default is False.
    coords_time : bool, optional
        Shorthand for adding time coordinates. The default is False.
    attrs_ds : bool, optional
        Shorthand for adding model dataset attributes. The default is False.
    datavars : list, optional
        List of data variables to check for. The default is an empty list.
    coords : list, optional
        List of coordinates to check for. The default is an empty list.
    attrs : list, optional
        List of attributes to check for. The default is an empty list.

    Returns
    -------
    ds : xr.Dataset
        A Dataset containing only the required data.
    """
    # Return the full dataset if not configured
    if ds is None:
        msg = "No dataset provided"
        raise ValueError(msg)
    isdefault_args = not any(
        [coords_2d, coords_3d, coords_time, attrs_ds, datavars, coords, attrs]
    )
    if isdefault_args:
        return ds

    isvertex = ds.attrs["gridtype"] == "vertex"

    # Initialize lists
    if datavars is None:
        datavars = []
    if coords is None:
        coords = []
    if attrs is None:
        attrs = []

    # Add coords, datavars and attrs via shorthands
    if coords_2d or coords_3d:
        coords.append("x")
        coords.append("y")
        attrs.append("extent")
        attrs.append("gridtype")

        if isvertex:
            datavars.append("xv")
            datavars.append("yv")
            datavars.append("icvert")

        if "angrot" in ds.attrs:
            # set by `nlmod.base.to_model_ds()` and `nlmod.dims.resample._set_angrot_attributes()`
            attrs_angrot_required = ["angrot", "xorigin", "yorigin"]
            attrs.extend(attrs_angrot_required)

    if coords_3d:
        coords.append("layer")
        datavars.append("top")
        datavars.append("botm")

    if coords_time:
        coords.append("time")
        datavars.append("steady")
        datavars.append("nstp")
        datavars.append("tsmult")

    if attrs_ds:
        # set by `nlmod.base.to_model_ds()` and `nlmod.base.set_ds_attrs()`,
        # excluding "created_on"
        attrs_ds_required = [
            "model_name",
            "mfversion",
            "exe_name",
            "model_ws",
            "figdir",
            "cachedir",
            "transport",
        ]
        attrs.extend(attrs_ds_required)

    # User-friendly error messages if missing from ds
    if "northsea" in datavars and "northsea" not in ds.data_vars:
        msg = "Northsea not in dataset. Run nlmod.read.rws.add_northsea() first."
        raise ValueError(msg)

    if coords_time:
        if "time" not in ds.coords:
            msg = "time not in dataset. Run nlmod.time.set_ds_time() first."
            raise ValueError(msg)

        # Check if time-coord is complete
        time_attrs_required = ["start", "time_units"]

        for t_attr in time_attrs_required:
            if t_attr not in ds["time"].attrs:
                msg = (
                    f"{t_attr} not in dataset['time'].attrs. "
                    + "Run nlmod.time.set_ds_time() to set time."
                )
                raise ValueError(msg)

    if attrs_ds:
        for attr in attrs_ds_required:
            if attr not in ds.attrs:
                msg = f"{attr} not in dataset.attrs. Run nlmod.set_ds_attrs() first."
                raise ValueError(msg)

    # User-unfriendly error messages
    for datavar in datavars:
        if datavar not in ds.data_vars:
            msg = f"{datavar} not in dataset.data_vars"
            raise ValueError(msg)

    for coord in coords:
        if coord not in ds.coords:
            msg = f"{coord} not in dataset.coords"
            raise ValueError(msg)

    for attr in attrs:
        if attr not in ds.attrs:
            msg = f"{attr} not in dataset.attrs"
            raise ValueError(msg)

    # Return only the required data
    return xr.Dataset(
        data_vars={k: ds.data_vars[k] for k in datavars},
        coords={k: ds.coords[k] for k in coords},
        attrs={k: ds.attrs[k] for k in attrs},
    )


def _explicit_dataset_coordinate_comparison(ds_in, ds_cache):
    """Perform explicit dataset coordinate comparison.

    Uses `xarray.testing.assert_identical()`.

    Parameters
    ----------
    ds_in : xr.Dataset
        Input dataset.
    ds_cache : xr.Dataset
        Cached dataset.

    Returns
    -------
    bool
        True if coordinates are identical, else False.

    Raises
    ------
    AssertionError
        If the coordinates are not equal.
    """
    logger.debug("cache -> performing explicit dataset coordinate comparison")
    for coord in ds_cache.coords:
        logger.debug(f"cache -> comparing coordinate {coord}")
        try:
            assert_identical(ds_in[coord], ds_cache[coord])
        except AssertionError as e:
            logger.debug(f"cache -> coordinate {coord} not equal")
            logger.debug(e)
            return False
    logger.debug("cache -> all coordinates equal")
    return True
