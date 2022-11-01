import logging
import os

import geopandas as gpd
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from ..mdims import resample

logger = logging.getLogger(__name__)


def _polygons_from_model_ds(model_ds):
    """create polygons of each cell in a model dataset.

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data

    Raises
    ------
    ValueError
        for wrong gridtype or inconsistent grid definition.

    Returns
    -------
    polygons : list of shapely Polygons
        list with polygon of each raster cell.
    """

    if model_ds.gridtype == "structured":
        # check if coördinates are consistent with delr/delc values
        delr_x = np.unique(model_ds.x.values[1:] - model_ds.x.values[:-1])
        delc_y = np.unique(model_ds.y.values[:-1] - model_ds.y.values[1:])
        if not ((delr_x == model_ds.delr) and (delc_y == model_ds.delc)):
            raise ValueError(
                "delr and delc attributes of model_ds inconsistent "
                "with x and y coordinates"
            )

        xmins = model_ds.x - (model_ds.delr * 0.5)
        xmaxs = model_ds.x + (model_ds.delr * 0.5)
        ymins = model_ds.y - (model_ds.delc * 0.5)
        ymaxs = model_ds.y + (model_ds.delc * 0.5)
        polygons = [
            Polygon(
                [
                    (xmins[i], ymins[j]),
                    (xmins[i], ymaxs[j]),
                    (xmaxs[i], ymaxs[j]),
                    (xmaxs[i], ymins[j]),
                ]
            )
            for i in range(len(xmins))
            for j in range(len(ymins))
        ]

    elif model_ds.gridtype == "vertex":
        polygons = [
            Polygon(vertices) for vertices in model_ds["vertices"].values
        ]
    else:
        raise ValueError(
            f"gridtype must be 'structured' or 'vertex', not {model_ds.gridtype}"
        )
    if "angrot" in model_ds.attrs and model_ds.attrs["angrot"] != 0.0:
        # rotate the model coordinates to real coordinates
        affine = resample.get_affine_mod_to_world(model_ds).to_shapely()
        polygons = [affine_transform(polygon, affine) for polygon in polygons]

    return polygons


def vertex_dataarray_to_gdf(
    model_ds, data_variables, polygons=None, dealing_with_time="mean"
):
    """Convert one or more DataArrays from a vertex model dataset to a
    Geodataframe.

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
                        "Can only use the mean of a DataArray with dimension time, use dealing_with_time='mean'"
                    )
            else:
                raise ValueError(
                    f"expected dimensions ('layer', 'icell2d'), got {da.dims}"
                )
        else:
            raise NotImplementedError(
                f"expected one or two dimensions got {no_dims} for data variable {da_name}"
            )

    # create geometries
    if polygons is None:
        polygons = _polygons_from_model_ds(model_ds)

    # construct geodataframe
    gdf = gpd.GeoDataFrame(dv_dic, geometry=polygons)

    return gdf


def struc_dataarray_to_gdf(
    model_ds, data_variables, polygons=None, dealing_with_time="mean"
):
    """Convert one or more DataArrays from a structured model dataset to a
    Geodataframe.

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
        polygons = _polygons_from_model_ds(model_ds)

    # construct geodataframe
    gdf = gpd.GeoDataFrame(dv_dic, geometry=polygons)

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
        gdf = vertex_dataarray_to_gdf(
            model_ds, data_variables, polygons=polygons
        )
    else:
        gdf = struc_dataarray_to_gdf(
            model_ds, data_variables, polygons=polygons
        )
    gdf.to_file(fname)


def model_dataset_to_vector_file(
    model_ds,
    gisdir=None,
    driver="GPKG",
    combine_dic=None,
    exclude=("x", "y", "time_steps", "area", "vertices", "rch_name"),
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

    Returns
    -------
    fnames : str or list of str
        filename(s) of exported geopackage or shapefiles.
    """

    # get default combination dictionary
    if combine_dic is None:
        combine_dic = {
            "idomain": {"first_active_layer", "idomain"},
            "topbot": {"top", "botm", "thickness"},
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
    polygons = _polygons_from_model_ds(model_ds)

    # combine some data variables in one shapefile
    for key, item in combine_dic.items():
        if set(item).issubset(da_names):
            if model_ds.gridtype == "structured":
                gdf = struc_dataarray_to_gdf(model_ds, item, polygons=polygons)
            elif model_ds.gridtype == "vertex":
                gdf = vertex_dataarray_to_gdf(
                    model_ds, item, polygons=polygons
                )
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
            gdf = struc_dataarray_to_gdf(
                model_ds, (da_name,), polygons=polygons
            )
        elif model_ds.gridtype == "vertex":
            gdf = vertex_dataarray_to_gdf(
                model_ds, (da_name,), polygons=polygons
            )
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


def model_dataset_to_ugrid_nc_file(
    model_ds,
    fname,
    variables=None,
    dummy_var="mesh_topology",
    xv="xv",
    yv="yv",
    face_node_connectivity="icvert",
):
    """Save a model dataset to a UGRID NetCDF file, so it can be opened as a
    Mesh Layer in qgis.

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    fname : str
        filename of the UGRID NetCDF-file, preferably with the extension .nc.
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

    Returns
    -------
    ds : xr.DataSet
        The dataset that was saved to a NetCDF-file. Can be used for debugging.
    """
    assert model_ds.gridtype == "vertex", "Only vertex grids are supported"

    # copy the dataset, so we do not alter the original one
    ds = model_ds.copy()

    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # rotate the model coordinates to real coordinates
        affine = resample.get_affine_mod_to_world(ds)
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
    nodata = ds[face_node_connectivity].attrs.get("_FillValue")
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
            ds[var].encoding["dtype"] = np.int
        if np.issubdtype(ds[var].dtype, str):
            # convert the string to an index of unique strings
            index = np.unique(model_ds[var], return_inverse=True)[1]
            ds[var] = ds[var].dims, index
        if np.issubdtype(ds[var].dtype, np.int64):
            ds[var].encoding["dtype"] = np.int32

    # Breaks down variables with a layer dimension into separate variables.
    # Copied from imod-python.
    for var in variables:
        if "layer" in ds[var].dims:
            stacked = ds[var]
            ds = ds.drop_vars(var)
            for layer in stacked["layer"].values:
                name = f"{var}_layer_{layer}"
                ds[name] = stacked.sel(layer=layer, drop=True)
                variables.append(name)
            variables.remove(var)
    if "layer" in ds.coords:
        ds = ds.drop_vars("layer")

    # only keep the selected variables
    ds = ds[variables + [dummy_var, xv, yv, face_node_connectivity]]
    # and save to file
    ds.to_netcdf(fname)
    return ds
