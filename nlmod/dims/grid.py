"""Module containing model grid functions.

-   project data on different grid types
-   obtain various types of reclists from a grid that
    can be used as input for a MODFLOW package
-   fill, interpolate and resample grid data
"""

import logging
import os
import warnings

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from affine import Affine
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.gridgen import Gridgen
from flopy.utils.gridintersect import GridIntersect
from packaging import version
from scipy.interpolate import griddata
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from .. import cache, util
from .layers import (
    fill_nan_top_botm_kh_kv,
    get_first_active_layer,
    get_idomain,
    remove_inactive_layers,
)
from .rdp import rdp
from .shared import (
    GridTypeDims,
    get_area,
    get_delc,
    get_delr,
    is_structured,
    is_vertex,
    is_layered,
)  # noqa: F401

logger = logging.getLogger(__name__)


def snap_extent(extent, delr, delc):
    """Snap the extent in such a way that an integer number of columns and rows fit in
    the extent. The new extent is always equal to, or bigger than the original extent.

    Parameters
    ----------
    extent : list, tuple or np.array
        original extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy

    Returns
    -------
    extent : list, tuple or np.array
        adjusted extent
    """
    extent = list(extent).copy()

    logger.debug(f"redefining extent: {extent}")

    if delr <= 0 or delc <= 0:
        raise ValueError("delr and delc should be positive values")

    # if xmin can be divided by delr do nothing, otherwise rescale xmin
    if not extent[0] % delr == 0:
        extent[0] -= extent[0] % delr

    # get number of columns and xmax
    ncol = int(np.ceil((extent[1] - extent[0]) / delr))
    extent[1] = extent[0] + (ncol * delr)  # round xmax up to close grid

    # if ymin can be divided by delc do nothing, otherwise rescale ymin
    if not extent[2] % delc == 0:
        extent[2] -= extent[2] % delc

    # get number of rows and ymax
    nrow = int(np.ceil((extent[3] - extent[2]) / delc))
    extent[3] = extent[2] + (nrow * delc)  # round ymax up to close grid

    logger.debug(f"new extent is {extent} and has {nrow} rows and {ncol} columns")

    return extent


def xy_to_icell2d(xy, ds):
    """Get the icell2d value of a point defined by its x and y coordinates.

    Parameters
    ----------
    xy : list, tuple
        coordinates of ta point.
    ds : xr.Dataset
        model dataset.

    Returns
    -------
    icell2d : int
        number of the icell2d value of a cell containing the xy point.
    """
    logger.warning(
        "nlmod.grid.xy_to_icell2d is deprecated. "
        "Use nlmod.grid.get_icell2d_from_xy instead"
    )
    msg = "xy_to_icell2d can only be applied to a vertex grid"
    assert ds.gridtype == "vertex", msg
    icell2d = (np.abs(ds.x.data - xy[0]) + np.abs(ds.y.data - xy[1])).argmin().item()

    return icell2d


def get_icell2d_from_xy(x, y, ds, gi=None, rotated=True):
    """Get the icell2d value of a point defined by its x and y coordinates.

    Parameters
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    ds : xr.Dataset
        The model dataset.
    gi : flopy.utils.GridIntersect, optional
        Can be supplied to speed up the calculation, as the generation of the
        GridIntersect-instance can take some time. The default is None.
    rotated : bool, optional
        If the model grid has a rotation, and rotated is False, x and y are in model
        coordinates. Otherwise x and y are in real world coordinates. The default is
        True.

    Raises
    ------
    ValueError
        Raises a ValueError if the point is outside of the model grid.

    Returns
    -------
    icell2d : int
        The icell2d-number of the model cell containing the point (zero-based)

    """
    msg = "get_icell2d_from_xy can only be applied to a vertex grid"
    assert ds.gridtype == "vertex", msg
    if gi is None:
        gi = flopy.utils.GridIntersect(
            modelgrid_from_ds(ds, rotated=rotated), method="vertex"
        )
    cellids = gi.intersects(Point(x, y))["cellids"]
    if len(cellids) < 1:
        raise (ValueError(f"Point ({x}, {y}) is outside of the model grid"))
    icell2d = cellids[0]
    return icell2d


def xy_to_row_col(xy, ds):
    """Get the row and column values of a point defined by its x and y coordinates.

    Parameters
    ----------
    xy : list, tuple
        coordinates of ta point.
    ds : xr.Dataset
        model dataset.

    Returns
    -------
    row : int
        number of the row value of a cell containing the xy point.
    col : int
        number of the column value of a cell containing the xy point.
    """
    logger.warning(
        "nlmod.grid.xy_to_row_col is deprecated. "
        "Use nlmod.grid.get_row_col_from_xy instead"
    )
    msg = "xy_to_row_col can only be applied to a structured grid"
    assert ds.gridtype == "structured", msg
    row = np.abs(ds.y.data - xy[1]).argmin()
    col = np.abs(ds.x.data - xy[0]).argmin()
    return row, col


def get_row_col_from_xy(x, y, ds, rotated=True, gi=None):
    """Get the row and column of a point defined by a x and y coordinate.

    Parameters
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    ds : xr.Dataset
        The model dataset.
    rotated : bool, optional
        If the model grid has a rotation, and rotated is False, x and y are in model
        coordinates. Otherwise x and y are in real world coordinates. The default is
        True.
    gi : flopy.utils.GridIntersect, optional
        Can be supplied to use a GridIntersect-class instead of our own calculation

    Raises
    ------
    ValueError
        Raises a ValueError if the point is outside of the model grid.

    Returns
    -------
    row : int
        The row-number of the model cell containing the point (zero-based)
    col : int
        The column-number of the model cell containing the point (zero-based)
    """
    msg = "get_row_col_from_xy can only be applied to a structured grid"
    assert ds.gridtype == "structured", msg
    if gi is not None:
        cellids = gi.intersects(Point(x, y))["cellids"]
        if len(cellids) < 1:
            raise (ValueError(f"Point ({x}, {y}) is outside of the model grid"))
        row, col = cellids[0]
        return row, col
    if rotated and ("angrot" in ds.attrs) and (ds.attrs["angrot"] != 0.0):
        # calculate the x and y in model coordinates
        affine = get_affine_world_to_mod(ds)
        x, y = affine * (x, y)
    if x < ds.extent[0] or x > ds.extent[1] or y < ds.extent[2] or y > ds.extent[3]:
        raise (ValueError(f"Point ({x}, {y}) is outside of the model grid"))

    y_bot = ds.y - get_delc(ds) / 2
    row = np.where(y >= y_bot)[0][0]

    x_left = ds.x - get_delr(ds) / 2
    col = np.where(x >= x_left)[0][-1]
    return row, col


def xyz_to_cid(xyz, ds=None, modelgrid=None):
    """Get the icell2d value of a point defined by its x and y coordinates.

    Parameters
    ----------
    xyz : list, tuple
        coordinates of a point.
    ds : xr.Dataset
        model dataset.
    modelgrid : StructuredGrid, VertexGrid, optional
        A flopy grid-object


    Returns
    -------
    cid : tuple
        (layer, cid) for vertex grid, (layer, row, column) for structured grid.
    """
    if modelgrid is None:
        modelgrid = modelgrid_from_ds(ds)

    cid = modelgrid.intersect(x=xyz[0], y=xyz[1], z=xyz[2])

    return cid


def modelgrid_from_ds(ds, rotated=True, nlay=None, top=None, botm=None, **kwargs):
    """Get flopy modelgrid from ds.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset.

    Returns
    -------
    modelgrid : StructuredGrid, VertexGrid
        grid information.
    """
    if rotated and ("angrot" in ds.attrs) and (ds.attrs["angrot"] != 0.0):
        xoff = ds.attrs["xorigin"]
        yoff = ds.attrs["yorigin"]
        angrot = ds.attrs["angrot"]
    else:
        if ds.gridtype == "structured":
            xoff = ds.extent[0]
            yoff = ds.extent[2]
        else:
            xoff = 0.0
            yoff = 0.0
        angrot = 0.0
    if top is None and "top" in ds:
        top = ds["top"].data
    if botm is None and "botm" in ds:
        botm = ds["botm"].data
    if nlay is None:
        if "layer" in ds:
            nlay = len(ds.layer)
        elif botm is not None:
            nlay = len(botm)

    if nlay is not None and botm is not None and nlay < len(botm):
        botm = botm[:nlay]

    kwargs = dict(
        xoff=xoff, yoff=yoff, angrot=angrot, nlay=nlay, top=top, botm=botm, **kwargs
    )
    if ds.gridtype == "structured":
        if not isinstance(ds.extent, (tuple, list, np.ndarray)):
            raise TypeError(
                f"extent should be a list, tuple or numpy array, not {type(ds.extent)}"
            )
        modelgrid = StructuredGrid(
            delc=get_delc(ds),
            delr=get_delr(ds),
            **kwargs,
        )
    elif ds.gridtype == "vertex":
        vertices = get_vertices_from_ds(ds)
        cell2d = get_cell2d_from_ds(ds)
        modelgrid = VertexGrid(
            vertices=vertices,
            cell2d=cell2d,
            **kwargs,
        )
    return modelgrid


def modelgrid_to_vertex_ds(mg, ds, nodata=-1):
    """Add information about the calculation-grid to a model dataset."""
    warnings.warn(
        "'modelgrid_to_vertex_ds' is deprecated and will be removed in a"
        "future version, please use 'modelgrid_to_ds' instead",
        DeprecationWarning,
    )

    # add modelgrid to ds
    ds["xv"] = ("iv", mg.verts[:, 0])
    ds["yv"] = ("iv", mg.verts[:, 1])

    cell2d = mg.cell2d
    ncvert_max = np.max([x[3] for x in cell2d])
    icvert = np.full((mg.ncpl, ncvert_max), nodata)
    for i in range(mg.ncpl):
        icvert[i, : cell2d[i][3]] = cell2d[i][4:]
    ds["icvert"] = ("icell2d", "icv"), icvert
    ds["icvert"].attrs["nodata"] = nodata
    return ds


def modelgrid_to_ds(mg=None, grbfile=None):
    """Create Dataset from flopy modelgrid object.

    Parameters
    ----------
    mg : flopy.discretization.Grid
        flopy modelgrid object
    grbfile : str
        path to a binary grid file

    Returns
    -------
    ds : xr.Dataset
        Dataset containing grid information
    """
    if mg is None and grbfile is not None:
        mg = flopy.mf6.utils.MfGrdFile(grbfile).modelgrid
    elif mg is None and grbfile is None:
        raise ValueError("Either 'mg' or 'grbfile' should be specified!")
    if mg.grid_type == "structured":
        x, y = mg.xyedges
        from .base import _get_structured_grid_ds

        ds = _get_structured_grid_ds(
            xedges=x,
            yedges=y,
            nlay=mg.nlay,
            botm=mg.botm,
            top=mg.top,
            xorigin=mg.xoffset,
            yorigin=mg.yoffset,
            angrot=mg.angrot,
            attrs=None,
            crs=None,
        )
    elif mg.grid_type == "vertex":
        from .base import _get_vertex_grid_ds

        ds = _get_vertex_grid_ds(
            x=mg.xcellcenters,
            y=mg.ycellcenters,
            xv=mg.verts[:, 0],
            yv=mg.verts[:, 1],
            cell2d=mg.cell2d,
            extent=mg.extent,
            nlay=mg.nlay,
            angrot=mg.angrot,
            xorigin=np.concatenate(mg.xvertices).min(),
            yorigin=np.concatenate(mg.yvertices).min(),
            botm=mg.botm,
            top=mg.top,
            attrs=None,
            crs=None,
        )
    else:
        raise NotImplementedError(f"Grid type '{mg.grid_type}' not supported!")

    return ds


def get_dims_coords_from_modelgrid(mg):
    """Get dimensions and coordinates from modelgrid.

    Used to build new xarray DataArrays with appropriate dimensions and coordinates.

    Parameters
    ----------
    mg : flopy.discretization.Grid
        flopy modelgrid object

    Returns
    -------
    dims : tuple of str
        tuple containing dimensions
    coords : dict
        dictionary containing spatial coordinates derived from modelgrid

    Raises
    ------
    ValueError
        for unsupported grid types
    """
    if mg.grid_type == "structured":
        layers = np.arange(mg.nlay)
        x, y = mg.xycenters  # local coordinates
        if mg.angrot == 0.0:
            x += mg.xoffset  # convert to global coordinates
            y += mg.yoffset  # convert to global coordinates
        coords = {"layer": layers, "y": y, "x": x}
        dims = ("layer", "y", "x")
    elif mg.grid_type == "vertex":
        layers = np.arange(mg.nlay)
        y = mg.ycellcenters
        x = mg.xcellcenters
        coords = {"layer": layers, "y": ("icell2d", y), "x": ("icell2d", x)}
        dims = ("layer", "icell2d")
    else:
        raise ValueError(f"grid type '{mg.grid_type}' not supported.")
    return dims, coords


def gridprops_to_vertex_ds(gridprops, ds, nodata=-1):
    """Gridprops is a dictionary containing keyword arguments needed to generate a flopy
    modelgrid instance.
    """
    _, xv, yv = zip(*gridprops["vertices"])
    ds["xv"] = ("iv", np.array(xv))
    ds["yv"] = ("iv", np.array(yv))

    cell2d = gridprops["cell2d"]
    ncvert_max = np.max([x[3] for x in cell2d])
    icvert = np.full((gridprops["ncpl"], ncvert_max), nodata)
    for i in range(gridprops["ncpl"]):
        icvert[i, : cell2d[i][3]] = cell2d[i][4:]
    ds["icvert"] = ("icell2d", "icv"), icvert
    ds["icvert"].attrs["nodata"] = nodata
    return ds


def get_vertices_from_ds(ds):
    """Get the vertices-list from a model dataset.

    Flopy needs needs this list to build a disv-package
    """
    vertices = list(zip(ds["iv"].data, ds["xv"].data, ds["yv"].data))
    return vertices


def get_cell2d_from_ds(ds):
    """Get the cell2d-list from a model dataset.

    Flopy needs this list to build a disv-package
    """
    icell2d = ds["icell2d"].data
    x = ds["x"].data
    y = ds["y"].data
    icvert = ds["icvert"].data
    if "nodata" in ds["icvert"].attrs:
        nodata = ds["icvert"].attrs["nodata"]
    else:
        nodata = -1
        icvert = icvert.copy()
        icvert[np.isnan(icvert)] = nodata
        icvert = icvert.astype(int)
    cell2d = []
    for i, cid in enumerate(icell2d):
        mask = icvert[i] != nodata
        cell2d.append((cid, x[i], y[i], mask.sum(), *icvert[i, mask]))
    return cell2d


def refine(
    ds,
    model_ws=None,
    refinement_features=None,
    exe_name=None,
    remove_nan_layers=True,
    model_coordinates=False,
    version_tag=None,
):
    """Refine the grid (discretization by vertices, disv), using Gridgen.

    Parameters
    ----------
    ds : xr.Dataset
        A structured model Dataset.
    model_ws : str, optional
        The working directory for GridGen. Get from ds when model_ws is None.
        The default is None.
    refinement_features : list of tuples of length 2 or 3, optional
        List of tuples containing refinement features. Each tuple must be of
        the form (GeoDataFrame, level) or (geometry, shape_type, level). When
        refinement_features is None, no refinement is added, but the structured model
        Dataset is transformed to a Vertex Dataset. The default is None.
    exe_name : str, optional
        Filepath to the gridgen executable. The file path within nlmod is chose
        if exe_name is None. The default is None.
    remove_nan_layers : bool, optional
        if True layers that are inactive everywhere are removed from the model.
        If False nan layers are kept which might be usefull if you want
        to keep some layers that exist in other models. The default is True.
    model_coordinates : bool, optional
        When model_coordinates is True, the features supplied in refinement_features are
        already in model-coordinates. Only used when a grid is rotated. The default is
        False.
    version_tag : str, default None
        GitHub release ID: for example "18.0" or "latest". If version_tag is provided,
        the most recent installation location of MODFLOW is found in flopy metadata
        that respects `version_tag`. If not found, the executables are downloaded.
        Not compatible with exe_name.

    Returns
    -------
    xr.Dataset
        A Vertex model Dataset
    """
    assert ds.gridtype == "structured", "Can only refine a structured grid"
    logger.info("create vertex grid using gridgen")

    if exe_name is None:
        exe_name = util.get_exe_path(exe_name="gridgen", version_tag=version_tag)
    else:
        exe_name = util.get_exe_path(exe_name=exe_name, version_tag=version_tag)

    if model_ws is None:
        model_ws = os.path.join(ds.model_ws, "gridgen")
        os.makedirs(model_ws, exist_ok=True)

    if version.parse(flopy.__version__) < version.parse("3.3.6"):
        sim = flopy.mf6.MFSimulation()
        gwf = flopy.mf6.MFModel(sim)
        dis = flopy.mf6.ModflowGwfdis(
            gwf,
            nrow=len(ds.y),
            ncol=len(ds.x),
            delr=get_delr(ds),
            delc=get_delc(ds),
            xorigin=ds.extent[0],
            yorigin=ds.extent[2],
        )
        g = Gridgen(dis, model_ws=model_ws, exe_name=exe_name)
    else:
        # create a modelgrid with only one layer, to speed up Gridgen
        modelgrid = modelgrid_from_ds(ds, rotated=False, nlay=1)
        g = Gridgen(modelgrid, model_ws=model_ws, exe_name=exe_name)

    ds_has_rotation = "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0
    if model_coordinates:
        if not ds_has_rotation:
            msg = "The supplied shapes need to be in realworld coordinates"
            raise ValueError(msg)

    if refinement_features is not None:
        for refinement_feature in refinement_features:
            if len(refinement_feature) == 3:
                # the feature is a file or a list of geometries
                fname, geom_type, level = refinement_feature
                if not model_coordinates and ds_has_rotation:
                    msg = "Converting files to model coordinates not supported"
                    raise NotImplementedError(msg)
                g.add_refinement_features(fname, geom_type, level, layers=[0])
            elif len(refinement_feature) == 2:
                # the feature is a geodataframe
                gdf, level = refinement_feature
                if not model_coordinates and ds_has_rotation:
                    affine = get_affine_world_to_mod(ds)
                    gdf = affine_transform_gdf(gdf, affine)
                geom_types = gdf.geom_type.str.replace("Multi", "")
                geom_types = geom_types.str.replace("String", "")
                geom_types = geom_types.str.lower()
                for geom_type in geom_types.unique():
                    if flopy.__version__ == "3.3.5" and geom_type == "line":
                        # a bug in flopy that is fixed in the dev branch
                        msg = (
                            "geom_type line is buggy in flopy 3.3.5. "
                            "See https://github.com/modflowpy/flopy/issues/1405"
                        )
                        raise ValueError(msg)
                    mask = geom_types == geom_type
                    # features = [gdf[mask].unary_union]
                    features = list(gdf[mask].geometry.explode(index_parts=True))
                    g.add_refinement_features(features, geom_type, level, layers=[0])
    g.build()
    gridprops = g.get_gridprops_disv()
    gridprops["area"] = g.get_area()
    ds = ds_to_gridprops(ds, gridprops=gridprops)
    if remove_nan_layers:
        ds = remove_inactive_layers(ds)
    return ds


def ds_to_gridprops(ds_in, gridprops, method="nearest", icvert_nodata=-1):
    """Resample a xarray dataset of a structured grid to a new dataset with a vertex
    grid.

    Returns a dataset with resampled variables and the untouched variables.

    Parameters
    ----------
    ds_in : xr.Dataset
        dataset with dimensions (layer, y, x). y and x are from the original
        structured grid
    gridprops : dictionary
        dictionary with grid properties output from gridgen.  Used as the
        definition of the vertex grid.
    method : str, optional
        type of interpolation used to resample. The default is 'nearest'.
    icvert_nodata : int, optional
        integer to represent nodata-values in cell2d array. Defaults to -1.

    Returns
    -------
    ds_out : xr.Dataset
        dataset with resampled variables and the untouched variables.
    """
    logger.info("resample model Dataset to vertex modelgrid")

    assert isinstance(ds_in, xr.core.dataset.Dataset)

    xyi, _ = get_xyi_icell2d(gridprops)
    x = xr.DataArray(xyi[:, 0], dims=("icell2d",))
    y = xr.DataArray(xyi[:, 1], dims=("icell2d",))

    if method in ["nearest", "linear"]:
        # resample the entire dataset in one line. Leaves not_interp_vars untouched
        ds_out = ds_in.interp(x=x, y=y, method=method, kwargs={"fill_value": None})

    else:
        # apply method to numeric data variables
        interp_vars = []
        not_interp_vars = []
        for key, var in ds_in.items():
            if "x" in var.dims or "y" in var.dims:
                if np.issubdtype(var.dtype, np.number):
                    interp_vars.append(key)
                else:
                    logger.info(
                        f"Data variable {key} has spatial coordinates but it cannot be refined "
                        "because of its non-numeric dtype. It is not available in the output Dataset."
                    )
            else:
                not_interp_vars.append(key)

        ds_out = ds_in[not_interp_vars]
        ds_out.coords.update({"layer": ds_in.layer, "x": x, "y": y})

        # add other variables
        from .resample import structured_da_to_ds

        for not_interp_var in not_interp_vars:
            ds_out[not_interp_var] = structured_da_to_ds(
                da=ds_in[not_interp_var], ds=ds_out, method=method, nodata=np.nan
            )
    has_rotation = "angrot" in ds_out.attrs and ds_out.attrs["angrot"] != 0.0
    if has_rotation:
        affine = get_affine_mod_to_world(ds_out)
        ds_out["xc"], ds_out["yc"] = affine * (ds_out.x, ds_out.y)

    if "area" in gridprops:
        if "area" in ds_out:
            ds_out = ds_out.drop_vars("area")

        # only keep the first layer of area
        area = gridprops["area"][: len(ds_out["icell2d"])]
        ds_out["area"] = ("icell2d", area)

    # add information about the vertices
    ds_out = gridprops_to_vertex_ds(gridprops, ds_out, nodata=icvert_nodata)

    # then finally change the gridtype in the attributes
    ds_out.attrs["gridtype"] = "vertex"
    return ds_out


def get_xyi_icell2d(gridprops=None, ds=None):
    """Get x and y coordinates of the cell mids from the cellids in the grid properties.

    Parameters
    ----------
    gridprops : dictionary, optional
        dictionary with grid properties output from gridgen. If gridprops is
        None xyi and icell2d will be obtained from ds.
    ds : xr.Dataset
        dataset with model data. Should have dimension (layer, icell2d).

    Returns
    -------
    xyi : numpy.ndarray
        array with x and y coördinates of cell centers, shape(len(icell2d), 2).
    icell2d : numpy.ndarray
        array with cellids, shape(len(icell2d))
    """
    if gridprops is not None:
        xc_gwf = [cell2d[1] for cell2d in gridprops["cell2d"]]
        yc_gwf = [cell2d[2] for cell2d in gridprops["cell2d"]]
        xyi = np.vstack((xc_gwf, yc_gwf)).T
        icell2d = np.array([c[0] for c in gridprops["cell2d"]])
    elif ds is not None:
        xyi = np.array(list(zip(ds.x.values, ds.y.values)))
        icell2d = ds.icell2d.values
    else:
        raise ValueError("either gridprops or ds should be specified")

    return xyi, icell2d


def update_ds_from_layer_ds(ds, layer_ds, method="nearest", **kwargs):
    """Add variables from a layer Dataset to a model Dataset. Keep de grid- information
    from the model Dataset (x and y or icell2d), but update the layer dimension when
    neccesary.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with model data. Can have dimension (layer, y, x) or
        (layer, icell2d).
    layer_ds : xr.Dataset
        dataset with layer data.
    method : str
        The method used for resampling layer_ds to the grid of ds.
    **kwargs : keyword arguments
        keyword arguments are passed to the fill_nan_top_botm_kh_kv-method.

    Returns
    -------
    ds : xr.Dataset
        Dataset with variables from layer_ds.
    """
    if not layer_ds.layer.equals(ds.layer):
        # update layers in ds
        drop_vars = []
        for var in ds.data_vars:
            if "layer" in ds[var].dims:
                if var not in layer_ds.data_vars:
                    logger.info(
                        f"Variable {var} is dropped, as it has dimension layer, "
                        "but is not defined in layer_ds"
                    )
                drop_vars.append(var)
        if len(drop_vars) > 0:
            ds = ds.drop_vars(drop_vars)
        ds = ds.assign_coords({"layer": layer_ds.layer})
    has_rotation = "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0
    if method in ["nearest", "linear"]:
        if has_rotation:
            x = ds.xc
            y = ds.yc
        else:
            x = ds.x
            y = ds.y
        layer_ds = layer_ds.interp(x=x, y=y, method=method, kwargs={"fill_value": None})
        for var in layer_ds.data_vars:
            ds[var] = layer_ds[var]
    else:
        from .resample import structured_da_to_ds

        for var in layer_ds.data_vars:
            ds[var] = structured_da_to_ds(layer_ds[var], ds, method=method)
    from .base import extrapolate_ds

    ds = extrapolate_ds(ds)
    ds = fill_nan_top_botm_kh_kv(ds, **kwargs)
    return ds


def col_to_list(col_in, ds, cellids):
    """Convert array data in ds to a list of values for specific cells.

    This function is typically used to create a rec_array with stress period
    data for the modflow packages. Can be used for structured and
    vertex grids.

    Parameters
    ----------
    col_in : xr.DataArray, str, int or float
        if col_in is a str type it is the name of the variable in ds (if it exists).
        if col_in is an int or a float it is a value that will be used for all
        cells in cellids.
    ds : xr.Dataset
        dataset with model data. Can have dimension (layer, y, x) or
        (layer, icell2d).
    cellids : tuple of numpy arrays
        tuple with indices of the cells that will be used to create the list
        with values. There are 3 options:
            1.   cellids contains (layers, rows, columns)
            2.   cellids contains (rows, columns) or (layers, icell2ds)
            3.   cellids contains (icell2ds)

    Raises
    ------
    ValueError
        raised if the cellids are in the wrong format.

    Returns
    -------
    col_lst : list
        raster values from ds presented in a list per cell.
    """
    if isinstance(col_in, str) and col_in in ds:
        col_in = ds[col_in]
    if isinstance(col_in, xr.DataArray):
        if len(cellids) == 3:
            # 3d grid
            col_lst = [
                col_in.data[lay, row, col]
                for lay, row, col in zip(cellids[0], cellids[1], cellids[2])
            ]
        elif len(cellids) == 2:
            # 2d grid or vertex 3d grid
            col_lst = [
                col_in.data[row, col] for row, col in zip(cellids[0], cellids[1])
            ]
        elif len(cellids) == 1:
            # 2d vertex grid
            col_lst = col_in.data[cellids[0]]
        else:
            raise ValueError(f"could not create a column list for col_in={col_in}")
    else:
        col_lst = [col_in] * len(cellids[0])

    return col_lst


def lrc_to_reclist(
    layers, rows, columns, cellids, ds, col1=None, col2=None, col3=None, aux=None
):
    """Create a reclist for stress period data from a set of cellids.

    Used for structured grids.


    Parameters
    ----------
    layers : list or numpy.ndarray
        list with the layer for each cell in the reclist.
    rows : list or numpy.ndarray
        list with the rows for each cell in the reclist.
    columns : list or numpy.ndarray
        list with the columns for each cell in the reclist.
    cellids : tuple of numpy arrays
        tuple with indices of the cells that will be used to create the list
        with values.
    ds : xr.Dataset
        dataset with model data. Can have dimension (layer, y, x) or
        (layer, icell2d).
    col1 : str, int or float, optional
        1st column of the reclist, if None the reclist will be a list with
        ((layer,row,column)) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]

    col2 : str, int or float, optional
        2nd column of the reclist, if None the reclist will be a list with
        ((layer,row,column), col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the reclist, if None the reclist will be a list with
        ((layer,row,column), col1, col2) for each row.

        col3 should be the following value for each package (can also be the
            name of a timeseries):

    aux : str or list of str
        list of auxiliary variables to include in reclist

    Raises
    ------
    ValueError
        Question: will this error ever occur?.

    Returns
    -------
    reclist : list of tuples
        every row consist of ((layer,row,column), col1, col2, col3).
    """
    cols = []

    if col1 is not None:
        cols.append(col_to_list(col1, ds, cellids))
    if col2 is not None and len(cols) == 1:
        cols.append(col_to_list(col2, ds, cellids))
    elif col2 is not None and len(cols) != 1:
        raise ValueError("col2 is set, but col1 is not!")
    if col3 is not None and len(cols) == 2:
        cols.append(col_to_list(col3, ds, cellids))
    elif col3 is not None and len(cols) != 2:
        raise ValueError("col3 is set, but col1 and/or col2 are not!")

    if aux is not None:
        if isinstance(aux, (str, int, float)):
            aux = [aux]

        for i_aux in aux:
            if isinstance(i_aux, str):
                if "layer" in ds[i_aux].dims and len(cellids) != 3:
                    cols.append(col_to_list(i_aux, ds, (np.array(layers),) + cellids))
                else:
                    cols.append(col_to_list(i_aux, ds, cellids))
            else:
                cols.append(col_to_list(i_aux, ds, cellids))

    reclist = list(zip(zip(layers, rows, columns), *cols))
    return reclist


def lcid_to_reclist(
    layers,
    cellids,
    ds,
    col1=None,
    col2=None,
    col3=None,
    aux=None,
):
    """Create a reclist for stress period data from a set of cellids.

    Used for vertex grids.


    Parameters
    ----------
    layers : list or numpy.ndarray
        list with the layer for each cell in the reclist.
    cellids : tuple of numpy arrays
        tuple with indices of the cells that will be used to create the list
        with values for a column. There are 2 options:
            1. cellids contains (layers, cids)
            2. cellids contains (cids)
    ds : xr.Dataset
        dataset with model data. Should have dimensions (layer, icell2d).
    col1 : str, int or float, optional
        1st column of the reclist, if None the reclist will be a list with
        ((layer,icell2d)) for each row. col1 should be the following value for
        each package (can also be the name of a timeseries):
        -   rch: recharge [L/T]
        -   ghb: head [L]
        -   drn: drain level [L]
        -   chd: head [L]
        -   riv: stage [L]

    col2 : str, int or float, optional
        2nd column of the reclist, if None the reclist will be a list with
        ((layer,icell2d), col1) for each row. col2 should be the following
        value for each package (can also be the name of a timeseries):
        -   ghb: conductance [L^2/T]
        -   drn: conductance [L^2/T]
        -   riv: conductacnt [L^2/T]

    col3 : str, int or float, optional
        3th column of the reclist, if None the reclist will be a list with
        ((layer,icell2d), col1, col2) for each row. col3 should be the following
        value for each package (can also be the name of a timeseries):
        -   riv: bottom [L]

    aux : str or list of str
        list of auxiliary variables to include in reclist

    Raises
    ------
    ValueError
        Question: will this error ever occur?.

    Returns
    -------
    reclist : list of tuples
        every row consist of ((layer, icell2d), col1, col2, col3)
        grids.
    """
    cols = []

    if col1 is not None:
        cols.append(col_to_list(col1, ds, cellids))
    if col2 is not None and len(cols) == 1:
        cols.append(col_to_list(col2, ds, cellids))
    elif col2 is not None and len(cols) != 1:
        raise ValueError("col2 is set, but col1 is not!")
    if col3 is not None and len(cols) == 2:
        cols.append(col_to_list(col3, ds, cellids))
    elif col3 is not None and len(cols) != 2:
        raise ValueError("col3 is set, but col1 and/or col2 are not!")

    if aux is not None:
        if isinstance(aux, (str, int, float)):
            aux = [aux]

        for i_aux in aux:
            if isinstance(i_aux, str):
                if "layer" in ds[i_aux].dims and len(cellids) != 2:
                    cols.append(col_to_list(i_aux, ds, (np.array(layers),) + cellids))
                else:
                    cols.append(col_to_list(i_aux, ds, cellids))
            else:
                cols.append(col_to_list(i_aux, ds, cellids))

    reclist = list(zip(zip(layers, cellids[-1]), *cols))
    return reclist


def cols_to_reclist(ds, cellids, *args, cellid_column=0):
    """Create a reclist for stress period data from a set of cellids.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with model data. Should have dimensions (layer, icell2d).
    cellids : tuple of length 2 or 3
        tuple with indices of the cells that will be used to create the list. For a
        structured grid, cellids represents (layer, row, column). For a vertex grid
        cellid reprsents (layer, icell2d).
    args : xr.DataArray, str, int or float
        the args parameter represents the data to be used as the columns in the reclist.
        See col_to_list of the allowed values.
    cellid_column : int, optional
        Adds the cellid ((layer, row, col) or (layer, icell2d)) to the reclist in this
        column number. Do not add cellid when cellid_column is None. The default is 0.
    """
    cols = [col_to_list(col, ds, cellids) for col in args]
    if cellid_column is not None:
        cols.insert(cellid_column, list(zip(*cellids)))
    return list(zip(*cols))


def da_to_reclist(
    ds,
    mask,
    col1=None,
    col2=None,
    col3=None,
    layer=0,
    aux=None,
    first_active_layer=False,
    only_active_cells=True,
):
    """Create a reclist for stress period data from a model dataset.

    Used for vertex grids.


    Parameters
    ----------
    ds : xr.Dataset
        dataset with model data and dimensions (layer, icell2d)
    mask : xr.DataArray for booleans
        True for the cells that will be used in the rec list.
    col1 : str, int or float, optional
        1st column of the reclist, if None the reclist will be a list with
        (cellid,) for each row.

        col1 should be the following value for each package (can also be the
            name of a timeseries):
            rch: recharge [L/T]
            ghb: head [L]
            drn: drain level [L]
            chd: head [L]

    col2 : str, int or float, optional
        2nd column of the reclist, if None the reclist will be a list with
        (cellid, col1) for each row.

        col2 should be the following value for each package (can also be the
            name of a timeseries):
            ghb: conductance [L^2/T]
            drn: conductance [L^2/T]

    col3 : str, int or float, optional
        3th column of the reclist, if None the reclist will be a list with
        (cellid, col1, col2) for each row.

        col3 should be the following value for each package (can also be the
            name of a timeseries):
            riv: bottom [L]
    aux : str or list of str, optional
        list of auxiliary variables to include in reclist
    layer : int, optional
        layer used in the reclist. Not used if layer is in the dimensions of
        mask or if first_active_layer is True. The default is 0
    first_active_layer : bool, optional
        If True an extra mask is applied to use the first active layer of each
        cell in the grid. Not used if layer is in the dimensions of mask. The
        default is False.
    only_active_cells : bool, optional
        If True an extra mask is used to only include cells with an idomain
        of 1. The default is True.

    Returns
    -------
    reclist : list of tuples
        every row consists of ((layer, icell2d), col1, col2, col3).
    """
    if "layer" in mask.dims:
        if only_active_cells:
            idomain = get_idomain(ds)
            cellids = np.where((mask) & (idomain == 1))
            ignore_cells = int(np.sum((mask) & (idomain != 1)))
            if ignore_cells > 0:
                logger.info(
                    f"ignore {ignore_cells} out of {np.sum(mask.values)} cells "
                    "because idomain is inactive"
                )
        else:
            cellids = np.where(mask)

        if "icell2d" in mask.dims:
            layers = cellids[0]
            return lcid_to_reclist(layers, cellids, ds, col1, col2, col3, aux=aux)
        else:
            layers = cellids[0]
            rows = cellids[1]
            columns = cellids[2]
            return lrc_to_reclist(
                layers, rows, columns, cellids, ds, col1, col2, col3, aux=aux
            )
    else:
        if first_active_layer:
            fal = get_first_active_layer(ds)
            cellids = np.where((mask.squeeze()) & (fal != fal.attrs["nodata"]))
            layers = col_to_list(fal, ds, cellids)
        elif only_active_cells:
            idomain = get_idomain(ds)
            cellids = np.where((mask) & (idomain[layer] == 1))
            ignore_cells = int(np.sum((mask) & (idomain[layer] != 1)))
            if ignore_cells > 0:
                logger.info(
                    f"ignore {ignore_cells} out of {np.sum(mask.values)} cells because idomain is inactive"
                )
            layers = col_to_list(layer, ds, cellids)
        else:
            cellids = np.where(mask)
            layers = col_to_list(layer, ds, cellids)

        if "icell2d" in mask.dims:
            return lcid_to_reclist(layers, cellids, ds, col1, col2, col3, aux=aux)
        else:
            rows = cellids[-2]
            columns = cellids[-1]

            return lrc_to_reclist(
                layers, rows, columns, cellids, ds, col1, col2, col3, aux=aux
            )


def polygon_to_area(modelgrid, polygon, da, gridtype="structured"):
    """Create a grid with the surface area in each cell based on a polygon value.

    Parameters
    ----------
    modelgrid : flopy.discretization.structuredgrid.StructuredGrid
        grid.
    polygon : shapely.geometry.polygon.Polygon
        polygon feature.
    da : xr.DataArray
        data array that is filled with polygon data

    Returns
    -------
    area_array : xr.DataArray
        area of polygon within each modelgrid cell
    """
    if polygon.geom_type == "Polygon":
        pass
    elif polygon.geom_type == "MultiPolygon":
        warnings.warn(
            "function not tested for MultiPolygon type, can have unexpected results"
        )
    else:
        raise TypeError(
            f'input geometry should by of type "Polygon" not {polygon.geom_type}'
        )

    ix = GridIntersect(modelgrid, method="vertex")
    opp_cells = ix.intersect(polygon)

    if gridtype == "structured":
        area_array = util.get_da_from_da_ds(da, dims=("y", "x"), data=0)
        for cellid, area in zip(opp_cells["cellids"], opp_cells["areas"]):
            area_array[cellid[0], cellid[1]] = area
    elif gridtype == "vertex":
        area_array = util.get_da_from_da_ds(da, dims=("icell2d",), data=0)
        cids = opp_cells.cellids
        area = opp_cells.areas
        area_array[cids.astype(int)] = area

    return area_array


def gdf_to_data_array_struc(
    gdf, gwf, field="VALUE", agg_method=None, interp_method=None
):
    """Project vector data on a structured grid. Aggregate data if multiple geometries
    are in a single cell.

    Parameters
    ----------
    gdf : geopandas.GeoDataframe
        vector data can only contain a single geometry type.
    gwf : flopy groundwater flow model
        model with a structured grid.
    field : str, optional
        column name in the geodataframe. The default is 'VALUE'.
    interp_method : str or None, optional
        method to obtain values in cells without geometry by interpolating
        between cells with values. Options are 'nearest' and 'linear'.
    agg_method : str, optional
        aggregation method to handle multiple geometries in one cell, options
        are:
        - max, min, mean,
        - length_weighted (lines), max_length (lines),
        - area_weighted (polygon), max_area (polygon).
        The default is 'max'.

    Returns
    -------
    da : xr DataArray
        The DataArray with the projected vector data.
    """
    warnings.warn(
        "The method gdf_to_data_array_struc is deprecated. Please use gdf_to_da instead.",
        DeprecationWarning,
    )

    x = gwf.modelgrid.get_xcellcenters_for_layer(0)[0]
    y = gwf.modelgrid.get_ycellcenters_for_layer(0)[:, 0]
    da = xr.DataArray(np.nan, dims=("y", "x"), coords={"y": y, "x": x})

    # interpolate data
    if interp_method is not None:
        arr = interpolate_gdf_to_array(gdf, gwf, field=field, method=interp_method)
        da.values = arr

        return da

    gdf_cellid = gdf_to_grid(gdf, gwf)

    if gdf_cellid.cellid.duplicated().any():
        # aggregate data
        if agg_method is None:
            raise ValueError(
                "multiple geometries in one cell please define aggregation method"
            )
        gdf_agg = aggregate_vector_per_cell(gdf_cellid, {field: agg_method}, gwf)
    else:
        # aggregation not neccesary
        gdf_agg = gdf_cellid[[field]]
        gdf_agg.set_index(
            pd.MultiIndex.from_tuples(gdf_cellid.cellid.values), inplace=True
        )

    for ind, row in gdf_agg.iterrows():
        da.values[ind[0], ind[1]] = row[field]

    return da


def gdf_to_da(
    gdf, ds, column, agg_method=None, fill_value=np.nan, min_total_overlap=0.0, ix=None
):
    """Project vector data on a grid. Aggregate data if multiple geometries are in a
    single cell. Supports structured and vertex grids. This method replaces
    gdf_to_data_array_struc.

    Parameters
    ----------
    gdf : geopandas.GeoDataframe
        vector data can only contain a single geometry type.
    ds : xr.Dataset
        model Datset
    column : str
        column name in the geodataframe.
    agg_method : str, optional
        aggregation method to handle multiple geometries in one cell, options
        are:
        - max, min, mean,
        - length_weighted (lines), max_length (lines),
        - area_weighted (polygon), max_area (polygon).
        The default is 'max'.
    fill_value : float or int, optional
        The value to fill in da outside gdf. The default is np.NaN
    min_total_overlap: float, optional
        Only assign cells with a gdf-area larger than min_total_overlap * cell-area. The
        default is 0.0
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    da : xr DataArray
        The DataArray with the projected vector data.
    """
    da = util.get_da_from_da_ds(ds, dims=ds.top.dims, data=fill_value)

    gdf_cellid = gdf_to_grid(gdf, ds, ix=ix)

    if gdf_cellid.size == 0:
        return da

    if min_total_overlap > 0:
        gdf_cellid["area"] = gdf_cellid.area
        area_sum = gdf_cellid[["cellid", "area"]].groupby("cellid").sum()
        min_area = min_total_overlap * ds["area"].data[area_sum.index]
        cellids = area_sum.index[area_sum["area"] > min_area]
        gdf_cellid = gdf_cellid[gdf_cellid["cellid"].isin(cellids)]

    if gdf_cellid.cellid.duplicated().any():
        # aggregate data
        if agg_method is None:
            raise ValueError(
                "multiple geometries in one cell please define aggregation method"
            )
        if agg_method in ["nearest"]:
            modelgrid = modelgrid_from_ds(ds)
            gdf_agg = aggregate_vector_per_cell(
                gdf_cellid, {column: agg_method}, modelgrid
            )
        else:
            gdf_agg = aggregate_vector_per_cell(gdf_cellid, {column: agg_method})
    else:
        # aggregation not neccesary
        gdf_agg = gdf_cellid[[column]]
        if isinstance(gdf_cellid.cellid.iloc[0], tuple):
            gdf_agg.set_index(
                pd.MultiIndex.from_tuples(gdf_cellid.cellid.values), inplace=True
            )
        else:
            gdf_agg.set_index(gdf_cellid.cellid.values, inplace=True)

    if ds.gridtype == "structured":
        ixs, iys = zip(*gdf_agg.index.values)
        da.values[ixs, iys] = gdf_agg[column]
    elif ds.gridtype == "vertex":
        da[gdf_agg.index] = gdf_agg[column]

    return da


def interpolate_gdf_to_array(gdf, gwf, field="values", method="nearest"):
    """Interpolate data from a point gdf.

    Parameters
    ----------
    gdf : geopandas.GeoDataframe
        vector data can only contain a single geometry type.
    gwf : flopy groundwater flow model
        model with a structured grid.
    field : str, optional
        column name in the geodataframe. The default is 'values'.
    method : str or None, optional
        method to obtain values in cells without geometry by interpolating
        between cells with values. Options are 'nearest' and 'linear'.

    Returns
    -------
    arr : np.array
        numpy array with interpolated data.
    """
    # check geometry
    geom_types = gdf.geometry.type.unique()
    if geom_types[0] != "Point":
        raise NotImplementedError("can only use interpolation with point geometries")

    # check field
    if field not in gdf.columns:
        raise ValueError(f"Missing column in DataFrame: {field}")

    points = np.array([[g.x, g.y] for g in gdf.geometry])
    values = gdf[field].values
    xi = np.vstack(
        (
            gwf.modelgrid.xcellcenters.flatten(),
            gwf.modelgrid.ycellcenters.flatten(),
        )
    ).T
    vals = griddata(points, values, xi, method=method)
    arr = np.reshape(vals, (gwf.modelgrid.nrow, gwf.modelgrid.ncol))

    return arr


def _agg_max_area(gdf, col):
    return gdf.loc[gdf.area.idxmax(), col]


def _agg_area_weighted(gdf, col):
    aw = (gdf.area * gdf[col]).sum(skipna=True) / gdf.area.sum(skipna=True)
    return aw


def _agg_max_length(gdf, col):
    return gdf.loc[gdf.length.idxmax(), col]


def _agg_length_weighted(gdf, col):
    nanmask = gdf[col].isna()
    aw = (gdf.length * gdf[col]).sum(skipna=True) / gdf.loc[~nanmask].length.sum()
    return aw


def _agg_nearest(gdf, col, modelgrid):
    if modelgrid.grid_type == "structured":
        cid = gdf["cellid"].values[0]
        cellcenter = Point(
            modelgrid.xcellcenters[0, cid[1]], modelgrid.ycellcenters[cid[0], 0]
        )
        val = gdf.iloc[gdf.distance(cellcenter).argmin()].loc[col]
    elif modelgrid.grid_type == "vertex":
        cid = gdf["cellid"].values[0]
        cellcenter = Point(modelgrid.xcellcenters[cid], modelgrid.ycellcenters[cid])
        val = gdf.iloc[gdf.distance(cellcenter).argmin()].loc[col]

    return val


def _get_aggregates_values(group, fields_methods, modelgrid=None):
    agg_dic = {}
    for field, method in fields_methods.items():
        # aggregation is only necesary if group shape is greater than 1
        if (group.shape[0] == 1) or (method == "first"):
            agg_dic[field] = group[field].values[0]
        elif method == "max":
            agg_dic[field] = group[field].max()
        elif method == "min":
            agg_dic[field] = group[field].min()
        elif method == "mean":
            agg_dic[field] = group[field].mean()
        elif method == "sum":
            agg_dic[field] = group[field].sum()
        elif method == "nearest":
            agg_dic[field] = _agg_nearest(group, field, modelgrid)
        elif method == "length_weighted":  # only for lines
            agg_dic[field] = _agg_length_weighted(group, field)
        elif method == "max_length":  # only for lines
            agg_dic[field] = _agg_max_length(group, field)
        elif method == "area_weighted":  # only for polygons
            agg_dic[field] = _agg_area_weighted(group, field)
        elif method == "max_area":  # only for polygons
            agg_dic[field] = _agg_max_area(group, field)
        elif method == "center_grid":  # only for polygons
            raise NotImplementedError
        else:
            raise ValueError(f"Method '{method}' not recognized!")

    return agg_dic


def aggregate_vector_per_cell(gdf, fields_methods, modelgrid=None):
    """Aggregate vector features per cell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing points, lines or polygons per grid cell.
    fields_methods: dict
        fields (keys) in the Geodataframe with their aggregation method (items)
        aggregation methods can be:
        max, min, mean, sum, length_weighted (lines), max_length (lines),
        area_weighted (polygon), max_area (polygon), and functions supported by
        pandas.*GroupBy.agg().
    modelgrid : flopy Groundwater flow modelgrid
        only necesary if one of the field methods is 'nearest'

    Returns
    -------
    celldata : pd.DataFrame
        DataFrame with aggregated surface water parameters per grid cell
    """
    # check geometry types
    geom_types = gdf.geometry.type.unique()
    if len(geom_types) > 1:
        if (
            len(geom_types) == 2
            and ("Polygon" in geom_types)
            and ("MultiPolygon" in geom_types)
        ):
            pass
        else:
            raise TypeError("cannot aggregate geometries of different types")
    if bool({"length_weighted", "max_length"} & set(fields_methods.values())):
        assert (
            geom_types[0] == "LineString"
        ), "can only use length methods with line geometries"
    if bool({"area_weighted", "max_area"} & set(fields_methods.values())):
        if ("Polygon" in geom_types) or ("MultiPolygon" in geom_types):
            pass
        else:
            raise TypeError("can only use area methods with polygon geometries")

    # check fields
    missing_cols = set(fields_methods.keys()).difference(gdf.columns)
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # aggregate data
    gr = gdf.groupby(by="cellid")
    celldata = pd.DataFrame(index=gr.groups.keys())

    for field, method in fields_methods.items():
        if method == "area_weighted":
            gdf_copy = gdf.copy()
            gdf_copy["area_times_field"] = gdf_copy["area"] * gdf_copy[field]

            # skipna is not implemented by groupby therefore we use min_count=1
            celldata[field] = gdf_copy.groupby(by="cellid")["area_times_field"].sum(
                min_count=1
            ) / gdf_copy.groupby(by="cellid")["area"].sum(min_count=1)

        elif method in (
            "nearest",
            "length_weighted",
            "max_length",
            "max_area",
            "center_grid",
        ):
            for cid, group in tqdm(gr, desc="Aggregate vector data"):
                agg_dic = _get_aggregates_values(group, {field: method}, modelgrid)

                for key, item in agg_dic.items():
                    celldata.loc[cid, key] = item

        else:
            celldata[field] = gr[field].agg(method)

    return celldata


def gdf_to_bool_da(
    gdf,
    ds,
    ix=None,
    buffer=0.0,
    contains_centroid=False,
    min_area_fraction=None,
    **kwargs,
):
    """Return True if grid cell is partly in polygons, False otherwise.

    This function returns True for grid cells that are partly in the polygons. If
    contains_centroid is True, only cells are returned where the centroid is in the
    polygon. If min_area_fraction is set, only cells are returned where the area of the
    intersection is larger than min_area_fraction * cell_area.

    ix can be provided to speed up the process. If not provided it is computed from ds.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or shapely.geometry
        shapes that will be rasterised.
    ds : xr.Dataset
        Dataset with model data.
    ix : flopy.utils.GridIntersect, optional
        If not provided it is computed from ds. Speeds up the the function
    buffer : float, optional
        The distance to buffer around the geometries in meters. A positive distance
        produces a dilation, a negative distance an erosion. The default is 0.
    contains_centroid :  bool, optional
        if True, only store intersection result if cell centroid is
        contained within intersection shape, only used if shape type is
        "polygon"
    min_area_fraction : float, optional
        float defining minimum intersection area threshold, if intersection
        area is smaller than min_frac_area * cell_area, do not store
        intersection result.
    **kwargs : keyword arguments
        keyword arguments are passed to the intersect-method.

    Returns
    -------
    da : xr.DataArray
        True if polygon is in cell, False otherwise.
    """
    if ds.gridtype == "structured":
        da = util.get_da_from_da_ds(ds, dims=("y", "x"), data=False)
    elif ds.gridtype == "vertex":
        da = util.get_da_from_da_ds(ds, dims=("icell2d",), data=False)
    else:
        msg = "gdf_to_bool_da() only support structured or vertex gridtypes"
        raise ValueError(msg)

    if isinstance(gdf, gpd.GeoDataFrame):
        if len(gdf) == 0:
            return da
        elif len(gdf) == 1:
            multipolygon = gdf.geometry.values[0]
        else:
            multipolygon = unary_union(gdf.geometry)
    elif isinstance(gdf, shapely.geometry.base.BaseGeometry):
        multipolygon = gdf
    else:
        msg = "gdf_to_bool_da() only support GeoDataFrame or shapely"
        raise TypeError(msg)

    if buffer != 0.0:
        multipolygon = multipolygon.buffer(buffer)

    # Rotate the polygon instead of the modelgrid
    if ix is None:
        modelgrid = modelgrid_from_ds(ds, rotated=False)
        ix = GridIntersect(modelgrid, method="vertex")

    grid_rotation = ds.attrs.get("angrot", 0.0)
    ix_rotation = ix.mfgrid.angrot

    if grid_rotation != 0.0 and ix_rotation == 0.0:
        affine = get_affine_world_to_mod(ds).to_shapely()
        multipolygon = affine_transform(multipolygon, affine)

    if kwargs or contains_centroid or min_area_fraction is not None:
        r = ix.intersect(
            multipolygon,
            contains_centroid=contains_centroid,
            min_area_fraction=min_area_fraction,
            **kwargs,
        )
    else:
        r = ix.intersects(multipolygon)

    if r.size > 0 and ds.gridtype == "structured":
        ixs, iys = zip(*r["cellids"], strict=True)
        da.values[ixs, iys] = True
    elif r.size > 0 and ds.gridtype == "vertex":
        da[r["cellids"].astype(int)] = True

    return da


def gdf_to_bool_ds(
    gdf,
    ds,
    da_name,
    keep_coords=None,
    ix=None,
    buffer=0.0,
    contains_centroid=False,
    min_area_fraction=None,
    **kwargs,
):
    """Return True if grid cell is partly in polygons, False otherwise.

    This function returns True for grid cells that are partly in the polygons. If
    contains_centroid is True, only cells are returned where the centroid is in the
    polygon. If min_area_fraction is set, only cells are returned where the area of the
    intersection is larger than min_area_fraction * cell_area.

    ix can be provided to speed up the process. If not provided it is computed from ds.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or shapely.geometry
        shapes that will be rasterised.
    ds : xr.Dataset
        Dataset with model data.
    da_name : str
        The name of the variable with boolean data in the ds_out
    keep_coords : tuple or None, optional
        the coordinates in ds the you want keep in your empty ds. If None all
        coordinates are kept from original ds. The default is None.
    ix : flopy.utils.GridIntersect, optional
        If not provided it is computed from ds. Speeds up the the function
    buffer : float, optional
        The distance to buffer around the geometries in meters. A positive distance
        produces a dilation, a negative distance an erosion. The default is 0.
    contains_centroid :  bool, optional
        if True, only store intersection result if cell centroid is
        contained within intersection shape, only used if shape type is
        "polygon"
    min_area_fraction : float, optional
        float defining minimum intersection area threshold, if intersection
        area is smaller than min_frac_area * cell_area, do not store
        intersection result.
    **kwargs : keyword arguments
        keyword arguments are passed to the intersect_*-methods.

    Returns
    -------
    ds_out : xr.Dataset
        True if polygon is in cell, False otherwise.
    """
    ds_out = util.get_ds_empty(ds, keep_coords=keep_coords)
    ds_out[da_name] = gdf_to_bool_da(
        gdf,
        ds,
        ix=ix,
        buffer=buffer,
        contains_centroid=contains_centroid,
        min_area_fraction=min_area_fraction,
        **kwargs,
    )

    return ds_out


def gdf_to_count_da(gdf, ds, ix=None, buffer=0.0, **kwargs):
    """Counts in how many polygons a coordinate of ds appears.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or shapely.geometry
        shapes that will be rasterised.
    ds : xr.Dataset
        Dataset with model data.
    ix : flopy.utils.GridIntersect, optional
        If not provided it is computed from ds.
    buffer : float, optional
        buffer around the geometries. The default is 0.
    **kwargs : keyword arguments
        keyword arguments are passed to the intersect_*-methods.

    Returns
    -------
    da : xr.DataArray
        1 if polygon is in cell, 0 otherwise. Grid dimensions according to ds.
    """
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # transform gdf into model coordinates
        affine = get_affine_world_to_mod(ds)
        gdf = affine_transform_gdf(gdf, affine)

    # build list of gridcells
    if ix is None:
        modelgrid = modelgrid_from_ds(ds, rotated=False)
        ix = GridIntersect(modelgrid, method="vertex")

    if ds.gridtype == "structured":
        da = util.get_da_from_da_ds(ds, dims=("y", "x"), data=0)
    elif ds.gridtype == "vertex":
        da = util.get_da_from_da_ds(ds, dims=("icell2d",), data=0)
    else:
        raise ValueError("function only support structured or vertex gridtypes")

    if isinstance(gdf, gpd.GeoDataFrame):
        geoms = gdf.geometry.values
    elif isinstance(gdf, shapely.geometry.base.BaseGeometry):
        geoms = [gdf]

    for geom in geoms:
        if buffer > 0.0:
            cids = ix.intersects(geom.buffer(buffer), **kwargs)["cellids"]
        else:
            cids = ix.intersects(geom, **kwargs)["cellids"]

        if len(cids) == 0:
            continue

        if ds.gridtype == "structured":
            ixs, iys = zip(*cids)
            da.values[ixs, iys] += 1
        elif ds.gridtype == "vertex":
            da[cids.astype(int)] += 1

    return da


def gdf_to_count_ds(gdf, ds, da_name, keep_coords=None, ix=None, buffer=0.0, **kwargs):
    """Counts in how many polygons a coordinate of ds appears.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        polygon shapes with surface water.
    ds : xr.Dataset
        Dataset with model data.
    da_name : str
        The name of the variable with boolean data in the ds_out
    keep_coords : tuple or None, optional
        the coordinates in ds the you want keep in your empty ds. If None all
        coordinates are kept from original ds. The default is None.
    ix : flopy.utils.GridIntersect, optional
        If not provided it is computed from ds.
    buffer : float, optional
        buffer around the geometries. The default is 0.
    **kwargs : keyword arguments
        keyword arguments are passed to the intersect_*-methods.

    Returns
    -------
    ds_out : xr.Dataset
        Dataset with a single DataArray, this DataArray is 1 if polygon is in
        cell, 0 otherwise. Grid dimensions according to ds and mfgrid.
    """
    ds_out = util.get_ds_empty(ds, keep_coords=keep_coords)
    ds_out[da_name] = gdf_to_count_da(gdf, ds, ix=ix, buffer=buffer, **kwargs)

    return ds_out


@cache.cache_pickle
def gdf_to_grid(
    gdf,
    ml=None,
    method="vertex",
    ix=None,
    desc="Intersecting with grid",
    silent=False,
    **kwargs,
):
    """Intersect a geodataframe with the grid of a MODFLOW model.

    Note: This method is a wrapper around the GridIntersect method in flopy.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame that needs to be cut by the grid. The GeoDataFrame can consist of
        multiple types (Point, LineString, Polygon and the Multi-variants).
    ml : flopy.modflow.Modflow or flopy.mf6.ModflowGwf or xr.Dataset, optional
        The flopy model or xarray dataset that defines the grid. When a Dataset is
        supplied, and the grid is rotated, the geodataframe is transformed in model
        coordinates. The default is None.
    method : string, optional
        Method passed to the GridIntersect-class. The default is 'vertex'.
    ix : flopy.utils.GridIntersect, optional
        GridIntersect, if not provided the modelgrid in ml is used.
    desc : string, optional
        The description of the progressbar. The default is 'Intersecting with grid'.
    silent : bool, optional
        Do not show a progressbar when silent is True. The default is False.
    **kwargs : keyword arguments
        keyword arguments are passed to the intersect_*-methods.

    Returns
    -------
    gdfg : geopandas.GeoDataFrame
        The GeoDataFrame with the geometries per grid-cell.
    """
    if ml is None and ix is None:
        raise (ValueError("Either specify ml or ix"))

    if gdf.index.has_duplicates or gdf.columns.has_duplicates:
        raise ValueError("gdf should not have duplicate columns or index.")

    if ml is not None:
        if isinstance(ml, xr.Dataset):
            ds = ml
            modelgrid = modelgrid_from_ds(ds, rotated=False)
            if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
                # transform gdf into model coordinates
                affine = get_affine_world_to_mod(ds)
                gdf = affine_transform_gdf(gdf, affine)
        else:
            modelgrid = ml.modelgrid
            if modelgrid.angrot != 0:
                raise NotImplementedError(
                    "please use a model dataset instead of a model"
                )

    if ix is None:
        ix = flopy.utils.GridIntersect(modelgrid, method=method)
    shps = []
    geometry = gdf.geometry.name
    for _, shp in tqdm(gdf.iterrows(), total=gdf.shape[0], desc=desc, disable=silent):
        r = ix.intersect(shp[geometry], **kwargs)
        for i in range(r.shape[0]):
            shpn = shp.copy()
            shpn["cellid"] = r["cellids"][i]
            shpn[geometry] = r["ixshapes"][i]
            if shp[geometry].geom_type == ["LineString", "MultiLineString"]:
                shpn["length"] = r["lengths"][i]
            elif shp[geometry].geom_type in ["Polygon", "MultiPolygon"]:
                shpn["area"] = r["areas"][i]
            shps.append(shpn)

    if len(shps) == 0:
        # Unable to determine the column names, because no geometries intersect with the grid
        logger.info("No geometries intersect with the grid")
        columns = gdf.columns.to_list() + ["area", "length", "cellid"]
    else:
        columns = None  # adopt from shps

    gdfg = gpd.GeoDataFrame(shps, columns=columns, geometry=geometry, crs=gdf.crs)
    gdfg.index.name = gdf.index.name
    return gdfg


def get_thickness_from_topbot(top, bot):
    """Get thickness from data arrays with top and bots.

    Parameters
    ----------
    top : xr.DataArray
        raster with top of each cell. dimensions should be (y,x) or (icell2d).
    bot : xr.DataArray
        raster with bottom of each cell. dimensions should be (layer, y,x) or
        (layer, icell2d).

    Returns
    -------
    thickness : xr.DataArray
        raster with thickness of each cell. dimensions should be (layer, y,x)
        or (layer, icell2d).
    """
    warnings.warn(
        "The method get_thickness_from_topbot is deprecated. Please use nlmod.layers.calculate_thickness instead.",
        DeprecationWarning,
    )

    if np.ndim(top) > 2:
        raise NotImplementedError("function works only for 2d top")

    # get thickness
    if bot.ndim == 3:
        thickness = util.get_da_from_da_ds(bot, dims=("layer", "y", "x"))
    elif bot.ndim == 2:
        thickness = util.get_da_from_da_ds(bot, dims=("layer", "icell2d"))
    else:
        raise ValueError("function only support structured or vertex gridtypes")

    for lay, botlay in enumerate(bot):
        if lay == 0:
            thickness[lay] = top - botlay
        else:
            thickness[lay] = bot[lay - 1] - botlay

    return thickness


def get_vertices_arr(ds, modelgrid=None, vert_per_cid=4, epsilon=0, rotated=False):
    """Get vertices of a vertex modelgrid from a ds or the modelgrid. Only return the 4
    corners of each cell and not the corners of adjacent cells thus limiting the
    vertices per cell to 4 points.

    This method uses the xvertices and yvertices attributes of the modelgrid.
    When no modelgrid is supplied, a modelgrid-object is created from ds.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset, attribute grid_type should be 'vertex'
    modelgrid : flopy.discretization.vertexgrid.VertexGrid
        vertex grid with attributes xvertices and yvertices.
    vert_per_cid : int or None:
        number of vertices per cell:
        - 4 return the 4 vertices of each cell
        - 5 return the 4 vertices of each cell + one duplicate vertex
        (sometimes useful if you want to create polygons)
        - anything else, the maximum number of vertices. For locally refined
        cells this includes all the vertices adjacent to the cell.

        if vert_per_cid is 4 or 5 vertices are removed using the
        Ramer-Douglas-Peucker Algorithm -> https://github.com/fhirschmann/rdp.
    epsilon : int or float, optional
        epsilon in the rdp algorithm. I (Onno) think this is: the maximum
        distance between a line and a point for which the point is considered
        to be on the line. The default is 0.

    Returns
    -------
    vertices_arr : numpy array
         Vertex coördinates per cell with dimensions(cid, no_vert, 2).
    """
    warnings.warn(
        "this function is deprecated and will eventually be removed, "
        "please use 'modelgrid_from_ds' and 'modelgrid.map_polygons' in the future.",
        DeprecationWarning,
    )
    if modelgrid is None:
        modelgrid = modelgrid_from_ds(ds, rotated=rotated)
    xvert = modelgrid.xvertices
    yvert = modelgrid.yvertices
    if vert_per_cid == 4:
        coord_list = []
        for xv, yv in zip(xvert, yvert):
            coords = rdp(list(zip(xv, yv)), epsilon=epsilon)[:-1]
            if len(coords) > 4:
                raise RuntimeError(
                    "unexpected number of coördinates, you probably want to change epsilon"
                )
            coord_list.append(coords)
        vertices_arr = np.array(coord_list)
    elif vert_per_cid == 5:
        coord_list = []
        for xv, yv in zip(xvert, yvert):
            coords = rdp(list(zip(xv, yv)), epsilon=epsilon)
            if len(coords) > 5:
                raise RuntimeError(
                    "unexpected number of coördinates, you probably want to change epsilon"
                )
            coord_list.append(coords)
        vertices_arr = np.array(coord_list)
    else:
        raise NotImplementedError()

    return vertices_arr


def get_vertices(ds, vert_per_cid=4, epsilon=0, rotated=False):
    """Get vertices of a vertex modelgrid from a ds or the modelgrid. Only return the 4
    corners of each cell and not the corners of adjacent cells thus limiting the
    vertices per cell to 4 points.

    This method uses the xvertices and yvertices attributes of the modelgrid.
    When no modelgrid is supplied, a modelgrid-object is created from ds.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset, attribute grid_type should be 'vertex'
    modelgrid : flopy.discretization.vertexgrid.VertexGrid
        vertex grid with attributes xvertices and yvertices.
    vert_per_cid : int or None:
        number of vertices per cell:
        - 4 return the 4 vertices of each cell
        - 5 return the 4 vertices of each cell + one duplicate vertex
        (sometimes useful if you want to create polygons)
        - anything else, the maximum number of vertices. For locally refined
        cells this includes all the vertices adjacent to the cell.

        if vert_per_cid is 4 or 5 vertices are removed using the
        Ramer-Douglas-Peucker Algorithm -> https://github.com/fhirschmann/rdp.
    epsilon : int or float, optional
        epsilon in the rdp algorithm. I (Onno) think this is: the maximum
        distance between a line and a point for which the point is considered
        to be on the line. The default is 0.

    Returns
    -------
    vertices_da : xr.DataArray
         Vertex coördinates per cell with dimensions(cid, no_vert, 2).
    """
    warnings.warn(
        "get_vertices is deprecated and will eventually be removed, "
        "please use 'modelgrid_from_ds' and 'modelgrid.map_polygons'.",
        DeprecationWarning,
    )

    vertices_arr = get_vertices_arr(
        ds,
        vert_per_cid=vert_per_cid,
        epsilon=epsilon,
        rotated=rotated,
    )

    vertices_da = xr.DataArray(
        vertices_arr,
        dims=("icell2d", "vert_per_cid", "xy"),
        coords={"xy": ["x", "y"]},
    )

    return vertices_da


@cache.cache_netcdf(coords_2d=True)
def mask_model_edge(ds, idomain=None):
    """Get data array which is 1 for every active cell (defined by idomain) at the
    boundaries of the model (xmin, xmax, ymin, ymax). Other cells are 0.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with model data.
    idomain : xr.DataArray, optional
        idomain used to get active cells and shape of DataArray. Calculate from ds when
        None. The default is None.

    Returns
    -------
    ds_out : xr.Dataset
        dataset with edge mask array
    """
    ds = ds.copy()  # avoid side effects

    # add constant head cells at model boundaries
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        raise NotImplementedError("model edge not yet calculated for rotated grids")

    # get mask with grid edges
    xmin = ds["x"] == ds["x"].min()
    xmax = ds["x"] == ds["x"].max()
    ymin = ds["y"] == ds["y"].min()
    ymax = ds["y"] == ds["y"].max()

    ds_out = util.get_ds_empty(ds)

    if ds.gridtype == "structured":
        mask2d = ymin | ymax | xmin | xmax

        # assign 1 to cells that are on the edge and have an active idomain
        if idomain is None:
            idomain = get_idomain(ds)
        ds_out["edge_mask"] = xr.zeros_like(idomain)
        for lay in ds.layer:
            ds_out["edge_mask"].loc[lay] = np.where(
                mask2d & (idomain.loc[lay] == 1), 1, 0
            )

    elif ds.gridtype == "vertex":
        polygons_grid = polygons_from_ds(ds)
        gdf_grid = gpd.GeoDataFrame(geometry=polygons_grid)
        extent_edge = get_extent_polygon(ds).exterior
        cids_edge = gdf_grid.loc[gdf_grid.touches(extent_edge)].index
        ds_out["edge_mask"] = util.get_da_from_da_ds(
            ds, dims=("layer", "icell2d"), data=0
        )

        for lay in ds.layer:
            ds_out["edge_mask"].loc[lay, cids_edge] = 1

    return ds_out


def polygons_from_ds(ds):
    """Create polygons of each cell in a model dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with model data.

    Raises
    ------
    ValueError
        for wrong gridtype or inconsistent grid definition.

    Returns
    -------
    polygons : list of shapely Polygons
        list with polygon of each raster cell.
    """
    if ds.gridtype == "structured":
        delr = get_delr(ds)
        delc = get_delc(ds)

        xmins = ds.x - (delr * 0.5)
        xmaxs = ds.x + (delr * 0.5)
        ymins = ds.y - (delc * 0.5)
        ymaxs = ds.y + (delc * 0.5)
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

    elif ds.gridtype == "vertex":
        modelgrid = modelgrid_from_ds(ds)
        polygons = [Polygon(v.vertices) for v in modelgrid.map_polygons]
    else:
        raise ValueError(
            f"gridtype must be 'structured' or 'vertex', not {ds.gridtype}"
        )
    if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        # rotate the model coordinates to real coordinates
        affine = get_affine_mod_to_world(ds).to_shapely()
        polygons = [affine_transform(polygon, affine) for polygon in polygons]

    return polygons


def _get_attrs(ds):
    if isinstance(ds, dict):
        return ds
    else:
        return ds.attrs


def get_extent_polygon(ds, rotated=True):
    """Get the model extent, as a shapely Polygon."""
    attrs = _get_attrs(ds)
    polygon = util.extent_to_polygon(attrs["extent"])
    if rotated and "angrot" in ds.attrs and attrs["angrot"] != 0.0:
        affine = get_affine_mod_to_world(ds)
        polygon = affine_transform(polygon, affine.to_shapely())
    return polygon


def get_extent_gdf(ds, rotated=True, crs="EPSG:28992"):
    polygon = get_extent_polygon(ds, rotated=rotated)
    return gpd.GeoDataFrame(geometry=[polygon], crs=crs)


def affine_transform_gdf(gdf, affine):
    """Apply an affine transformation to a geopandas GeoDataFrame."""
    if isinstance(affine, Affine):
        affine = affine.to_shapely()
    gdfm = gdf.copy()
    gdfm.geometry = gdf.affine_transform(affine)
    return gdfm


def get_extent(ds, rotated=True, xmargin=0.0, ymargin=0.0):
    """Get the model extent, corrected for angrot if necessary.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset.
    rotated : bool, optional
        if True, the extent is corrected for angrot. The default is True.
    xmargin : float, optional
        margin to add to the x-extent. The default is 0.0.
    ymargin : float, optional
        margin to add to the y-extent. The default is 0.0.

    Returns
    -------
    extent : list
        [xmin, xmax, ymin, ymax]
    """
    attrs = _get_attrs(ds)
    extent = attrs["extent"]
    extent = [
        extent[0] - xmargin,
        extent[1] + xmargin,
        extent[2] - ymargin,
        extent[3] + ymargin,
    ]
    if rotated and "angrot" in attrs and attrs["angrot"] != 0.0:
        affine = get_affine_mod_to_world(ds)
        xc = np.array([extent[0], extent[1], extent[1], extent[0]])
        yc = np.array([extent[2], extent[2], extent[3], extent[3]])
        xc, yc = affine * (xc, yc)
        extent = [xc.min(), xc.max(), yc.min(), yc.max()]
    return extent


def get_affine_mod_to_world(ds):
    """Get the affine-transformation from model to real-world coordinates."""
    attrs = _get_attrs(ds)
    xorigin = attrs["xorigin"]
    yorigin = attrs["yorigin"]
    angrot = attrs["angrot"]
    return Affine.translation(xorigin, yorigin) * Affine.rotation(angrot)


def get_affine_world_to_mod(ds):
    """Get the affine-transformation from real-world to model coordinates."""
    attrs = _get_attrs(ds)
    xorigin = attrs["xorigin"]
    yorigin = attrs["yorigin"]
    angrot = attrs["angrot"]
    return Affine.rotation(-angrot) * Affine.translation(-xorigin, -yorigin)


def get_affine(ds, sx=None, sy=None):
    """Get the affine-transformation, from pixel to real-world coordinates."""
    attrs = _get_attrs(ds)
    if sx is None:
        sx = get_delr(ds)
        assert len(np.unique(sx)) == 1, "Affine-transformation needs a constant delr"
        sx = sx[0]
    if sy is None:
        sy = get_delc(ds)
        assert len(np.unique(sy)) == 1, "Affine-transformation needs a constant delc"
        sy = -sy[0]

    if "angrot" in attrs:
        xorigin = attrs["xorigin"]
        yorigin = attrs["yorigin"]
        angrot = -attrs["angrot"]
        # xorigin and yorigin represent the lower left corner, while for the transform we
        # need the upper left
        dy = attrs["extent"][3] - attrs["extent"][2]
        xoff = xorigin + dy * np.sin(angrot * np.pi / 180)
        yoff = yorigin + dy * np.cos(angrot * np.pi / 180)
        return (
            Affine.translation(xoff, yoff)
            * Affine.scale(sx, sy)
            * Affine.rotation(angrot)
        )
    else:
        xoff = attrs["extent"][0]
        yoff = attrs["extent"][3]
        return Affine.translation(xoff, yoff) * Affine.scale(sx, sy)
