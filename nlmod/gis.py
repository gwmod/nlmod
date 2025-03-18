import logging
import os

import geopandas as gpd
import numpy as np

from nlmod.dims.grid import get_affine_mod_to_world, polygons_from_model_ds
from nlmod.dims.layers import calculate_thickness
from nlmod.epsg28992 import EPSG_28992

logger = logging.getLogger(__name__)


def vertex_da_to_gdf(
    model_ds, data_variables, polygons=None, dealing_with_time="mean", crs=EPSG_28992
):
    """Convert one or more DataArrays from a vertex model dataset to a Geodataframe.

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    data_variables : list, tuple or set
        data_variables in model_ds that will be stored as attributes in the
        shapefile.
    polygons : list of shapely Polygons, optional
        geometries used for the GeoDataframe, if None the polygons are created
        from the data variable 'vertices' in model_ds. The default is None.
    dealing_with_time : str, optional
        when there is time variant data in the model dataset this function
        becomes very slow. For now only the time averaged data will be
        saved in the geodataframe. Later this can be extended with multiple
        possibilities. The default is 'mean'.
    crs : str, optional
        coordinate reference system for the geodataframe. The default
        is EPSG:28992 (RD).

    Raises
    ------
    ValueError
        for DataArrays with unexpected dimension names.
    NotImplementedError
        for DataArrays with more than 2 dimensions or dealing_with_time!='mean'

    Returns
    -------
    gdf : geopandas.GeoDataframe
        geodataframe of one or more DataArrays.
    """
    assert (
        model_ds.gridtype == "vertex"
    ), f"expected model dataset with gridtype vertex, got {model_ds.gridtype}"

    if isinstance(data_variables, str):
        data_variables = [data_variables]

    # create dictionary with column names and values of the geodataframe
    dv_dic = {}
    for da_name in data_variables:
        da = model_ds[da_name]
        no_dims = len(da.dims)
        if no_dims == 1:
            dv_dic[da_name] = da.values
        elif no_dims == 2:
            if da.dims == ("layer", "icell2d"):
                for i, da_lay in enumerate(da):
                    dv_dic[f"{da_name}_lay{i}"] = da_lay
            elif "time" in da.dims:
                da_mean = da.mean(dim="time")
                if dealing_with_time == "mean":
                    dv_dic[f"{da_name}_mean"] = da_mean.values
                else:
                    raise NotImplementedError(
                        "Can only use the mean of a DataArray with dimension time, "
                        "use dealing_with_time='mean'"
                    )
            else:
                logger.warning(
                    "expected dimensions ('layer', 'icell2d') for data variable "
                    f"{da_name}, got {da.dims}"
                )
        else:
            logger.warning(
                f"expected one or two dimensions for data variable "
                f"{da_name}, got {no_dims} dimensions"
            )

    # create geometries
    if polygons is None:
        polygons = polygons_from_model_ds(model_ds)

    # construct geodataframe
    gdf = gpd.GeoDataFrame(dv_dic, geometry=polygons, crs=crs)

    return gdf


def struc_da_to_gdf(
    model_ds, data_variables, polygons=None, dealing_with_time="mean", crs=EPSG_28992
):
    """Convert one or more DataArrays from a structured model dataset to a Geodataframe.

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    data_variables : list, tuple or set
        data_variables in model_ds that will be stored as attributes in the
        shapefile.
    polygons : list of shapely Polygons, optional
        geometries used for the GeoDataframe, if None the polygons are created
        from the data variable 'vertices' in model_ds. The default is None.
    crs : str, optional
        coordinate reference system for the geodataframe. The default
        is EPSG:28992 (RD).

    Raises
    ------
    ValueError
        for DataArrays with unexpected dimension names.
    NotImplementedError
        for DataArrays with more than 2 dimensions or dealing_with_time!='mean'

    Returns
    -------
    gdf : geopandas.GeoDataframe
        geodataframe of one or more DataArrays.
    """
    assert (
        model_ds.gridtype == "structured"
    ), f"expected model dataset with gridtype vertex, got {model_ds.gridtype}"

    if isinstance(data_variables, str):
        data_variables = [data_variables]

    # create dictionary with column names and values of the geodataframe
    dv_dic = {}
    for da_name in data_variables:
        da = model_ds[da_name]
        no_dims = len(da.dims)
        if no_dims == 2:
            dv_dic[da_name] = da.values.flatten("F")
        elif no_dims == 3:
            if da.dims == ("layer", "y", "x"):
                for i, da_lay in enumerate(da):
                    dv_dic[f"{da_name}_lay{i}"] = da_lay.values.flatten("F")
            elif "time" in da.dims:
                da_mean = da.mean(dim="time")
                if dealing_with_time == "mean":
                    dv_dic[f"{da_name}_mean"] = da_mean.values.flatten("F")
                else:
                    raise NotImplementedError(
                        "Can only use the mean of a DataArray with dimension time, use dealing_with_time='mean'"
                    )
            else:
                raise ValueError(
                    f"expected dimensions ('layer', 'y', 'x'), got {da.dims}"
                )
        else:
            raise NotImplementedError(
                f"expected two or three dimensions got {no_dims} for data variable {da_name}"
            )

    # create geometries
    if polygons is None:
        polygons = polygons_from_model_ds(model_ds)

    # construct geodataframe
    gdf = gpd.GeoDataFrame(dv_dic, geometry=polygons, crs=crs)

    return gdf


def dataarray_to_shapefile(model_ds, data_variables, fname, polygons=None):
    """Save one or more DataArrays from a model dataset as a shapefile.

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    data_variables : list, tuple or set
        data_variables in model_ds that will be stored as attributes in the
        shapefile.
    fname : str
        filename of the shapefile.
    polygons : list of shapely Polygons, optional
        geometries used for the GeoDataframe, if None the polygons are created
        from the data variable 'vertices' in model_ds. The default is None.

    Returns
    -------
    None.
    """
    if model_ds.gridtype == "vertex":
        gdf = vertex_da_to_gdf(model_ds, data_variables, polygons=polygons)
    else:
        gdf = struc_da_to_gdf(model_ds, data_variables, polygons=polygons)
    gdf.to_file(fname)


def ds_to_vector_file(
    model_ds,
    gisdir=None,
    driver="GPKG",
    combine_dic=None,
    exclude=("x", "y", "time_steps", "area", "vertices", "rch_name", "icvert"),
    crs=EPSG_28992,
):
    """Save all data variables in a model dataset to multiple shapefiles.

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    gisdir : str, optional
        gis directory to save shapefiles, if None a subdirectory 'gis' of the
        model_ws (which is an attribute of model_ds) is used. The default is
        None.
    driver : str, optional
        determines if the data variables are saved as seperate "ESRI Shapefile"
        or a single "GPKG" (geopackage). The default is geopackage.
    combine_dic : dictionary of str:set(), optional
        The items in this dictionary are data variables in model_ds that
        should be combined in one shapefile. The key defines the name of the
        shapefile. If None the default combine_dic is used. The default is
        None.
    exclude : tuple of str, optional
        data variables that are not exported to shapefiles. The default is
        ('x', 'y', 'time_steps', 'area', 'vertices').
    crs : str, optional
        coordinate reference system for the vector file. The default
        is EPSG:28992 (RD).

    Returns
    -------
    fnames : str or list of str
        filename(s) of exported geopackage or shapefiles.
    """
    # get default combination dictionary
    if combine_dic is None:
        combine_dic = {
            "topbot": {"top", "botm"},
            "sea": {"northsea", "bathymetry"},
        }

    # create gis directory and filenames
    if gisdir is None:
        gisdir = os.path.join(model_ds.model_ws, "gis")
    if not os.path.exists(gisdir):
        os.mkdir(gisdir)

    if driver == "GPKG":
        fname_gpkg = os.path.join(gisdir, f"{model_ds.model_name}.gpkg")
    elif driver == "ESRI Shapefile":
        fnames = []
    else:
        raise ValueError(f"invalid driver -> {driver}")

    # get all data variables in the model dataset
    da_names = set(model_ds)

    # exclude some data variables
    for da_name in da_names:
        # add data variables with only time variant data to exclude list
        if ("time",) == model_ds[da_name].dims:
            exclude += (da_name,)
        # add data variables with vertex data to exclude list
        elif "iv" in model_ds[da_name].dims:
            exclude += (da_name,)
        # add data variables with vertex data to exclude list
        elif "nvert" in model_ds[da_name].dims:
            exclude += (da_name,)

    # exclude some names from export
    da_names -= set(exclude)

    # create list of polygons
    polygons = polygons_from_model_ds(model_ds)

    # combine some data variables in one shapefile
    for key, item in combine_dic.items():
        if set(item).issubset(da_names):
            if model_ds.gridtype == "structured":
                gdf = struc_da_to_gdf(model_ds, item, polygons=polygons, crs=crs)
            elif model_ds.gridtype == "vertex":
                gdf = vertex_da_to_gdf(model_ds, item, polygons=polygons, crs=crs)
            if driver == "GPKG":
                gdf.to_file(fname_gpkg, layer=key, driver=driver)
            else:
                fname = os.path.join(gisdir, f"{key}.shp")
                gdf.to_file(fname, driver=driver)
                fnames.append(fname)
            da_names -= item
        else:
            logger.info(
                f"could not add {item} into to geopackage because 1 or more of the data variables do not exist"
            )

    # create unique shapefiles for the other data variables
    for da_name in da_names:
        if model_ds.gridtype == "structured":
            gdf = struc_da_to_gdf(model_ds, (da_name,), polygons=polygons, crs=crs)
        elif model_ds.gridtype == "vertex":
            gdf = vertex_da_to_gdf(model_ds, (da_name,), polygons=polygons, crs=crs)
        if driver == "GPKG":
            gdf.to_file(fname_gpkg, layer=da_name, driver=driver)
        else:
            fname = os.path.join(gisdir, f"{da_name}.shp")
            gdf.to_file(fname, driver=driver)
            fnames.append(fname)

    # return filename(s)
    if driver == "GPKG":
        return fname_gpkg
    else:
        return fnames


def ds_to_ugrid_nc_file(
    model_ds,
    fname=None,
    variables=None,
    dummy_var="mesh_topology",
    xv="xv",
    yv="yv",
    face_node_connectivity="icvert",
    split_layer_dimension=True,
    split_time_dimension=False,
    for_imod_qgis_plugin=False,
):
    """Save a model dataset to a UGRID NetCDF file, so it can be opened as a Mesh Layer
    in qgis.

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray Dataset with model data
    fname : str, optional
        filename of the UGRID NetCDF-file, preferably with the extension .nc. When fname
        is None,only a ugird-ready Dataset is created, without saving this Dataset to
        file. The defaults is None.
    variables : str or list of str, optional
        THe variables to be saved in the NetCDF file. The default is None,
        which means all variables will be saved in the file.
    dummy_var : str, optional
        The name of the new dummy-variable that contains the gridinformation in
        its attributes. The default is 'mesh_topology'.
    xv : str, optional
        The name of the variable that contains the x-coordinate of the vertices
        that together form the edges of the faces. The default is 'xv'.
    yv : str, optional
        The name of the variable that contains the y-coordinate of the vertices
        that together form the edges of the faces. The default is 'yv'.
    face_node_connectivity : str, optional
        The name of the variable that contains the indexes of the vertices for
        each face. The default is 'icvert'.
    split_layer_dimension : bool, optional
        Splits the layer dimension into seperate variables when True. The defaults is
        True.
    split_time_dimension : bool, optional
        Splits the time dimension into seperate variables when True. The defaults is
        False.
    for_imod_qgis_plugin : bool, optional
        When True, set some properties of the netcdf file to improve compatibility with
        the iMOD-QGIS plugin. Layers are renamed to 'layer_i' until 'layer_n', a
        variable 'top' is added for each layer, and the variable 'botm' is renamed to
        'bottom'. The default is False.

    Returns
    -------
    ds : xr.DataSet
        The dataset that was saved to a NetCDF-file. Can be used for debugging.
    """
    assert model_ds.gridtype == "vertex", "Only vertex grids are supported for now"

    # copy the dataset, so we do not alter the original one
    ds = model_ds.copy()

    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # rotate the model coordinates to real coordinates
        affine = get_affine_mod_to_world(ds)
        ds[xv], ds[yv] = affine * (ds[xv], ds[yv])

    # add a dummy variable with the required grid-information
    ds[dummy_var] = 0
    ds[dummy_var].attrs["node_coordinates"] = f"{xv} {yv}"
    ds[dummy_var].attrs["cf_role"] = "mesh_topology"
    ds[dummy_var].attrs["topology_dimension"] = 2
    ds[dummy_var].attrs["face_node_connectivity"] = face_node_connectivity

    # drop the first vertex of each face, as the faces should be open in ugrid
    nvert_per_cell_dim = ds[face_node_connectivity].dims[1]
    ds = ds[{nvert_per_cell_dim: ds[nvert_per_cell_dim][1:]}]
    # make sure vertices (nodes) in faces are sprecified in counterclokcwise-
    # direction. Flopy specifies them in clockwise direction, so we need to
    # reverse the direction.
    data = np.flip(ds[face_node_connectivity].data, 1)
    nodata = ds[face_node_connectivity].attrs.get("nodata")
    if nodata is not None:
        # move the nodata values from the first columns to the last
        data_new = np.full(data.shape, nodata)
        for i in range(data.shape[0]):
            mask = data[i, :] != nodata
            data_new[i, : mask.sum()] = data[i, mask]
        data = data_new
    ds[face_node_connectivity].data = data
    ds[face_node_connectivity].attrs["cf_role"] = "face_node_connectivity"
    ds[face_node_connectivity].attrs["start_index"] = 0

    if for_imod_qgis_plugin and "botm" in ds:
        ds["top"] = ds["botm"] + calculate_thickness(ds)
        ds = ds.rename({"botm": "bottom"})

    # set for each of the variables that they describe the faces
    if variables is None:
        variables = list(ds.keys())
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        if var in [dummy_var, face_node_connectivity]:
            continue
        ds[var].attrs["location"] = "face"

    # Make sure time is encoded as a float for MDAL.
    # Copied from imod-python.
    for var in ds.coords:
        if np.issubdtype(ds[var].dtype, np.datetime64):
            ds[var].encoding["dtype"] = np.float64

    # Convert boolean layers to integer and int64 to int32
    for var in variables:
        if np.issubdtype(ds[var].dtype, bool):
            ds[var].encoding["dtype"] = np.int32
        elif np.issubdtype(ds[var].dtype, str) or np.issubdtype(ds[var].dtype, object):
            # convert the string to an index of unique strings
            index = np.unique(ds[var], return_inverse=True)[1]
            ds[var] = ds[var].dims, index
        if np.issubdtype(ds[var].dtype, np.int64):
            ds[var].encoding["dtype"] = np.int32

    # Breaks down variables with a layer dimension into separate variables.
    if split_layer_dimension:
        if for_imod_qgis_plugin:
            ds, variables = _break_down_dimension(
                ds, variables, "layer", add_dim_name=True, add_one_based_index=True
            )
        else:
            ds, variables = _break_down_dimension(ds, variables, "layer")
    if split_time_dimension:
        # Breaks down variables with a time dimension into separate variables.
        ds, variables = _break_down_dimension(ds, variables, "time")

    # only keep the selected variables
    ds = ds[variables + [dummy_var, xv, yv, face_node_connectivity]]
    if fname is not None:
        # and save to file
        ds.to_netcdf(fname)
    return ds


def _break_down_dimension(
    ds, variables, dim, add_dim_name=False, add_one_based_index=False
):
    """Internal method to split a dimension of a variable into multiple variables.

    Copied and altered from imod-python.
    """
    keep_vars = []
    for var in variables:
        if dim in ds[var].dims:
            stacked = ds[var]
            for i, value in enumerate(stacked[dim].values):
                name = var
                if add_dim_name:
                    name = f"{name}_{dim}"
                if add_one_based_index:
                    name = f"{name}_{i + 1}"
                else:
                    name = f"{name}_{value}"

                ds[name] = stacked.sel({dim: value}, drop=True)
                if "long_name" in ds[name].attrs:
                    long_name = ds[name].attrs["long_name"]
                    ds[name].attrs["long_name"] = f"{long_name} {value}"
                if "standard_name" in ds[name].attrs:
                    standard_name = ds[name].attrs["standard_name"]
                    ds[name].attrs["standard_name"] = f"{standard_name}_{value}"
                keep_vars.append(name)
        else:
            keep_vars.append(var)
    if dim in ds.coords:
        ds = ds.drop_vars(dim)

    return ds, keep_vars
