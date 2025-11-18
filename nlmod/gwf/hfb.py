import logging
import warnings

import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from geopandas import GeoSeries
from shapely.geometry import Point, Polygon

from ..dims.grid import (
    gdf_to_da,
    gdf_to_grid,
    get_node_structured,
    is_structured,
    is_vertex,
    modelgrid_from_ds,
    node_to_lrc,
)
from ..dims.layers import calculate_thickness, get_idomain

logger = logging.getLogger(__name__)


def get_hfb_spd(ds, linestrings, hydchr, depth=None, elevation=None):
    """Generate a stress period data for horizontal flow barrier between two cell nodes,
    with several limitations. The stress period data can be used directly in the HFB
    package of flopy. The hfb is placed at the cell interface; it follows the sides of
    the cells.

    The estimation of the cross-sectional area at the interface is pretty crude, as the
    thickness at the cell interface is just the average of the thicknesses of the two
    cells.

    Parameters
    ----------
    ds : xr.Dataset
        model dataset
    linestrings : geopandas.geodataframe
        geodataframe with line elements.
    hydchr : float
        Conductance of the horizontal flow barrier, e.g. 1 / 100 means
        a resistance of 100 days for a unit gradient.
    depth : float
        Depth with respect to groundlevel. For example for cases where the depth of the
        barrier is only limited by the construction method. Use depth or elevation
        argument.
    elevation : float
        The elevation of the bottom of barrier. Top of the barrier is at groundlevel.

    Returns
    -------
    spd : List of Tuple
        Stress period data used to configure the hfb package of Flopy.
    """
    assert (
        sum([depth is None, elevation is None]) == 1
    ), "Use either depth or elevation argument"

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Please pass a model dataset!")

    thick = calculate_thickness(ds)
    idomain = get_idomain(ds)
    tops = np.concatenate((ds["top"].values[np.newaxis], ds["botm"].values))
    cells = line_to_hfb(linestrings, ds)

    # drop cells on the edge of the model
    cells = [x for x in cells if len(x) > 1]

    spd = []
    for icell2d1, icell2d2 in cells:
        # TODO: Improve assumption of the thickness between the cells.
        if isinstance(icell2d1, (int, np.integer)):
            thicki = (thick[:, icell2d1] + thick[:, icell2d2]) / 2
            topi = (tops[:, icell2d1] + tops[:, icell2d2]) / 2
        else:
            thicki = (thick[:, *icell2d1] + thick[:, *icell2d2]) / 2
            topi = (tops[*icell2d1] + tops[*icell2d2]) / 2

        for ilay in range(ds.sizes["layer"]):
            cellid1 = (ilay,) + (
                icell2d1 if isinstance(icell2d1, tuple) else (icell2d1,)
            )
            cellid2 = (ilay,) + (
                icell2d2 if isinstance(icell2d2, tuple) else (icell2d2,)
            )

            if idomain.values[cellid1] <= 0:
                continue

            if idomain.values[cellid2] <= 0:
                continue

            if depth is not None:
                if sum(thicki[: ilay + 1]) <= depth:
                    # hfb spans the entire cell
                    spd.append([cellid1, cellid2, hydchr])

                elif sum(thicki[:ilay]) <= depth:
                    # hfb spans the cell partially
                    hydchr_frac = (depth - sum(thicki[:ilay])) / thicki[ilay]
                    assert 0 <= hydchr_frac <= 1, "Something is wrong"

                    spd.append([cellid1, cellid2, hydchr * hydchr_frac])
                    break  # go to next cell

            else:
                if topi[ilay + 1] >= elevation:
                    # hfb spans the entire cell
                    spd.append([cellid1, cellid2, hydchr])

                else:
                    # hfb spans the cell partially
                    hydchr_frac = (topi[ilay] - elevation) / thicki[ilay]
                    assert 0 <= hydchr_frac <= 1, "Something is wrong"

                    spd.append([cellid1, cellid2, hydchr * hydchr_frac])
                    break  # go to next cell

    return spd


def line2hfb(gdf, ds=None, gwf=None, prevent_rings=True, plot=False):
    warnings.warn(
        "The function 'line2hfb' is deprecated and will be removed in a future version. "
        "Please use 'line_to_hfb' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return line_to_hfb(
        gdf=gdf,
        ds=ds,
        gwf=gwf,
        prevent_rings=prevent_rings,
        plot=plot,
    )


def line_to_hfb(gdf, ds=None, gwf=None, prevent_rings=True, plot=False):
    """Snap line to grid and return list of cellids that share faces.

    Used for determining where to place horizontal flow barriers.

    Parameters
    ----------
    gdf : gpd.GeoDataframe
        geodataframe with line elements.
    ds : xarray.Dataset, optional
        Dataset with the grid information. The default is None.
        Must pass one of ds or gwf.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        grondwater flow model or modelgrid object. The default is None.
        Must pass one of ds or gwf.
    prevent_rings : bool, optional
        Prevent cells with segments on each side when True. Remove the segments whose
        centroid is farthest from the line. The default is True.
    plot : bool, optional
        If True create a simple plot of the grid cells and shapefile. For a
        more complex plot you can use plot_hfb. The default is False.

    Returns
    -------
    cellids : 2d list of ints
        a list with pairs of cells that have a hfb between them.
    """
    # for the idea, see:
    # https://gis.stackexchange.com/questions/188755/how-to-snap-a-road-network-to-a-hexagonal-grid-in-qgis

    if gwf is not None:
        if isinstance(gwf, flopy.discretization.grid.Grid):
            mgrid = gwf
        elif isinstance(gwf, flopy.mf6.ModflowGwf):
            mgrid = gwf.modelgrid
        else:
            raise TypeError(
                "Please pass either a flopy.discretization.grid.Grid or "
                "flopy.mf6.ModflowGwf object as gwf."
            )
    elif ds is not None:
        mgrid = modelgrid_from_ds(ds)
    else:
        raise ValueError(
            "Please pass either a dataset or a flopy.mf6.ModflowGwf object."
        )

    gdfg = gdf_to_grid(gdf, gwf if gwf is not None else ds, desc="Intersecting HFB(s)")

    # add support for structured grid
    if mgrid.grid_type == "structured":
        vertices = pd.DataFrame(
            index=np.arange(mgrid.nvert),
            data=mgrid.verts,
            columns=["xv", "yv"],
        )
        vertices.index.name = "iv"
        cell2d = pd.DataFrame(index=np.arange(mgrid.ncpl))
        cell2d["ncvert"] = 5
        cell2d["xc"] = mgrid.xcellcenters.flatten()
        cell2d["yc"] = mgrid.ycellcenters.flatten()
        cell2d.index.name = "icell2d"
        icvert = np.array(
            [
                mgrid._build_structured_iverts(*mgrid.get_lrc(icpl)[0][1:])
                for icpl in range(mgrid.ncpl)
            ]
        )
        # add first vertex to the end of the list
        icvert = np.hstack([icvert, icvert[:, :1]])
        gdfg["cellid_structured"] = gdfg["cellid"]
        gdfg["cellid"] = gdfg["cellid"].map(
            lambda cid: get_node_structured(0, *cid, shape=mgrid.shape)
        )
    elif mgrid.grid_type == "vertex":
        cell2d = pd.DataFrame(mgrid.cell2d)
        cell2d.columns = ["icell2d", "xc", "yc", "ncvert"] + [
            f"icvert_{i}" for i in range(cell2d.columns.size - 4)
        ]
        cell2d.set_index("icell2d", inplace=True)
        vertices = pd.DataFrame(
            index=np.arange(mgrid.nvert),
            data=mgrid.verts,
            columns=["xv", "yv"],
        )
        vertices.index.name = "iv"
        icvert = cell2d.loc[:, cell2d.columns.str.startswith("icvert")].values
    else:
        raise ValueError(
            f"gridtype {mgrid.grid_type} not supported. Only 'structured' "
            "and 'vertex' are supported."
        )

    # for every cell determine which cell-edge could form the line
    # by testing for an intersection with a triangle to the cell-center
    hfb_seg = []
    for index in gdfg.index.unique():
        # Get the nearest hexagon sides where routes cross
        for icell2d in gdfg.loc[index, "cellid"]:
            for i in range(cell2d.at[icell2d, "ncvert"] - 1):
                iv1 = icvert[icell2d, i]
                iv2 = icvert[icell2d, i + 1]
                # make sure vert1 is lower than vert2
                if iv1 > iv2:
                    iv1, iv2 = iv2, iv1
                coords = [
                    (cell2d.at[icell2d, "xc"], cell2d.at[icell2d, "yc"]),
                    (vertices.at[iv1, "xv"], vertices.at[iv1, "yv"]),
                    (vertices.at[iv2, "xv"], vertices.at[iv2, "yv"]),
                ]
                triangle = Polygon(coords)
                if triangle.intersects(gdf.loc[index, "geometry"]):
                    hfb_seg.append((icell2d, iv1, iv2))
    hfb_seg = np.array(hfb_seg)

    if prevent_rings:
        # find out if there are cells with segments on each side
        # remove the segments whose centroid is farthest from the line
        for icell2d in np.unique(hfb_seg[:, 0]):
            mask = hfb_seg[:, 0] == icell2d
            if mask.sum() >= cell2d.at[icell2d, "ncvert"] - 1:
                segs = hfb_seg[mask]
                dist = []
                for seg in segs:
                    p = Point(
                        vertices.loc[seg[1:3], "xv"].mean(),
                        vertices.loc[seg[1:3], "yv"].mean(),
                    )
                    dist.append(gdf.distance(p).min())
                iv1, iv2 = segs[np.argmax(dist), [1, 2]]
                mask = (hfb_seg[:, 1] == iv1) & (hfb_seg[:, 2] == iv2)
                hfb_seg = hfb_seg[~mask]

    # get unique segments
    hfb_seg_unique = np.unique(hfb_seg[:, 1:], axis=0)
    # NOTE: see next note describing the idea behind this code. Leaving here for future
    # review.
    # hfb_seg_unique, uidx = np.unique(hfb_seg[:, 1:], return_index=True, axis=0)
    # ucids = hfb_seg[uidx, 0]
    # try:
    #     endpoint_cellid = mgrid.intersect(get_coordinates(gdf.geometry.union_all()[-1]))
    # except Exception:  # e.g. point is on our outside model boundary
    #     endpoint_cellid = ucids[-1]  # pick last cellid (this might not be the endpoint)
    # n_segments_endpoint_cell = 0

    # Get rid of disconnected (or 'open') segments
    # Let's remove disconnected/open segments
    iv = np.unique(hfb_seg_unique)
    segments_per_iv = pd.Series([np.sum(hfb_seg_unique == x) for x in iv], index=iv)
    mask = np.full(hfb_seg_unique.shape[0], True)
    for i, segment in enumerate(hfb_seg_unique):
        # one vertex is not connected and the other one at least to two other segments
        if (segments_per_iv[segment[0]] == 1 and segments_per_iv[segment[1]] >= 3) or (
            segments_per_iv[segment[1]] == 1 and segments_per_iv[segment[0]] >= 3
        ):
            mask[i] = False
        # NOTE: the code below can be used to avoid dropping all segments in the
        # final cell, leaving it here for review in the future.
        #     if ensure_endpoint_segment:
        #         # keep at least one segment in final cell
        #         if ucids[i] == endpoint_cellid and n_segments_endpoint_cell == 0:
        #             continue
        #         else:
        #             mask[i] = False
        #     else:
        #         mask[i] = False
        # elif ensure_endpoint_segment and ucids[i] == endpoint_cellid:
        #     n_segments_endpoint_cell += 1
    hfb_seg_unique = hfb_seg_unique[mask]

    if plot:
        # test by plotting
        ax = gdfg.plot()
        for _, seg in enumerate(hfb_seg_unique):
            x = [vertices.at[seg[0], "xv"], vertices.at[seg[1], "xv"]]
            y = [vertices.at[seg[0], "yv"], vertices.at[seg[1], "yv"]]
            ax.plot(x, y, color="k")

    # find out between which cellid's these segments are
    segments = []
    for icell2d in cell2d.index:
        for i in range(cell2d.at[icell2d, "ncvert"] - 1):
            iv1 = icvert[icell2d, i]
            iv2 = icvert[icell2d, i + 1]
            # make sure vert1 is lower than vert2
            if iv1 > iv2:
                iv1, iv2 = iv2, iv1
            segments.append((icell2d, (iv1, iv2)))
    segments = pd.DataFrame(segments, columns=["icell2d", "verts"])
    segments = segments.set_index(["verts"])

    cellids = []
    for seg in hfb_seg_unique:
        if mgrid.grid_type == "structured":
            iseg = [
                node_to_lrc(cid, mgrid.shape)[1:]
                for cid in segments.loc[[tuple(seg)]].values[:, 0]
            ]
            cellids.append(iseg)
        else:
            cellids.append(list(segments.loc[[tuple(seg)]].values[:, 0]))

    return cellids


def line_to_hfb_buffer(gdf, ds, buffer_distance=None, gi=None):
    """Snap line to grid and return list of cellids that share faces.

    Uses a two single-sided buffers to compute where the interface to compute the
    locations of the horizontal flow barriers. This method is more robust than the
    other methods, and is also faster.

    Parameters
    ----------
    gdf : gpd.GeoDataframe
        geodataframe with line elements.
    ds : xarray.Dataset
        model dataset
    buffer_distance : float, optional
        Buffer distance. The default is None, which will be set to 1.5 times the
        maximum cell dimension in the model.
    gi : int, optional
        GridIntersect instance, if not passed, the function will create one.
        The default is None.

    Returns
    -------
    hfb_seg : list of tuples
        List of tuples with cellids that share a face.
    """
    if buffer_distance is None:
        # buffer distance ~ 1.5 * max cell size
        buffer_distance = 1.5 * np.max(
            [np.max(np.abs(np.diff(ds["x"]))), np.max(np.abs(np.diff(ds["y"])))]
        )

    zone_left = gdf.buffer(-buffer_distance, single_sided=True)
    if isinstance(zone_left, GeoSeries):
        zone_left = zone_left.to_frame()
    zone_left["zone_id"] = -1
    zone_right = gdf.buffer(buffer_distance, single_sided=True)
    if isinstance(zone_right, GeoSeries):
        zone_right = zone_right.to_frame()
    zone_right["zone_id"] = 1
    if (zone_left.intersection(zone_right).area > 0).any():
        logger.warning(
            "The left and right buffer zones intersect. Please check your input."
            "Perhaps there are multiple linestrings, or the linestring geometry is "
            "causing potential issues. You can also try modifying the buffer_distance."
        )
    zone_gdf = pd.concat([zone_left, zone_right], axis=0).reset_index(drop=True)

    da = gdf_to_da(zone_gdf, ds, column="zone_id", agg_method="max_area", ix=gi)
    cellids = np.nonzero(da.values == 1)
    mgrid = modelgrid_from_ds(ds)
    hfb_seg = []
    if is_structured(ds):
        for cid in zip(*cellids):
            neighbors = mgrid.neighbors(0, *cid)
            for n in neighbors:
                if da.values[n[1:]] == -1:
                    hfb_seg.append((cid, n[1:]))
    elif is_vertex(ds):
        for cid in cellids[0]:
            neighbors = mgrid.neighbors(cid)
            for n in neighbors:
                if da.values[n] == -1:
                    hfb_seg.append((cid, n))
    else:
        raise ValueError(
            f"gridtype {mgrid.grid_type} not supported. Only 'structured' "
            "and 'vertex' are supported."
        )
    return hfb_seg


def polygon_to_hfb(
    gdf, ds, hydchr, column=None, gwf=None, lay=0, add_data=False, include_nan=True
):
    """Snap polygon exterior to grid to form a horizontal flow barrier.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        geodataframe with polygon elements.
    ds : xarray.Dataset
        model dataset
    hydchr : float
        Conductance of the horizontal flow barrier, e.g. 1 / 100 means
        a resistance of 100 days for a unit gradient.
    column : str, optional
        Column name to use for the data. The default is None.
    gwf : flopy.mf6.ModflowGwf, optional
        Groundwater flow model. The default is None. If passed,
        function returns a HFB package.
    lay : int, optional
        Layer number. The default is 0.
    add_data : bool, optional
        If True, add the data to the stress period data. The default is False.
    include_nan : bool, optional
        If true, add hfb's to and from nan-values

    Returns
    -------
    spd : list of lists
        List of lists with the cell ids and the conductance, if gwf is None.
    hfb : flopy.mf6.ModflowGwfhfb, optional
        If gwf is passed, returns a HFB package.
    """
    if isinstance(gdf, xr.DataArray):
        da = gdf
    elif isinstance(gdf, str):
        da = ds[gdf]
    else:
        if column is None:
            column = gdf.index.name
            if column is None:
                column = "index"
            gdf = gdf.reset_index()
        da = gdf_to_da(gdf, ds, column, agg_method="max_area", fill_value=-1)
    data = da.data

    spd = []
    if ds.gridtype == "structured":
        for row in range(len(ds.y) - 1):
            for col in range(len(ds.x) - 1):
                if data[row, col] != data[row + 1, col]:
                    spd.append([(lay, row, col), (lay, row + 1, col), hydchr])
                    if add_data:
                        spd[-1].extend([data[row, col], data[row + 1, col]])
                if data[row, col] != data[row, col + 1]:
                    spd.append([(lay, row, col), (lay, row, col + 1), hydchr])
                    if add_data:
                        spd[-1].extend([data[row, col], data[row, col + 1]])
    else:
        # find connections
        icvert = ds["icvert"].data
        nodata = ds["icvert"].attrs["nodata"]

        edges = []
        for icell2d in range(icvert.shape[0]):
            if not include_nan and np.isnan(data[icell2d]):
                continue
            for j in range(icvert.shape[1] - 1):
                if icvert[icell2d, j + 1] == nodata:
                    break
                edge = [icell2d, data[icell2d]]
                if icvert[icell2d, j + 1] > icvert[icell2d, j]:
                    edge.extend([icvert[icell2d, j], icvert[icell2d, j + 1]])
                else:
                    edge.extend([icvert[icell2d, j + 1], icvert[icell2d, j]])
                edges.append(edge)
        edges = np.array(edges)
        edges_un, inverse = np.unique(edges[:, 2:], axis=0, return_inverse=True)
        icell2ds = []
        for i in range(len(edges_un)):
            mask = inverse == i
            if len(np.unique(edges[mask, 1])) > 1:
                icell2ds.append(edges[mask, 0])
        # icell2ds = np.array(icell2ds)
        for icell2d1, icell2d2 in icell2ds:
            spd.append([(lay, int(icell2d1)), (lay, int(icell2d2)), hydchr])
            if add_data:
                spd[-1].extend([data[icell2d1], data[icell2d2]])
    if gwf is None:
        return spd
    else:
        return flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: spd})


def plot_hfb(cellids, gwf, ax=None, color="red", **kwargs):
    """Plots a horizontal flow barrier.

    Parameters
    ----------
    cellids : list of lists of integers or flopy.mf6.ModflowGwfhfb
        list with the ids of adjacent cells that should get a horizontal
        flow barrier, hfb is the output of line_to_hfb.
    gwf : flopy groundwater flow model, xarray Dataset or flopy modelgrid
        The object to determine the properties of the model grid. It can be a flopy
        modelgrid-object, a flopy groundwater flow model, or an nlmod xarray dataset.
    ax : matplotlib axes
        The Axes to plot the hfb-data in. When ax is None, a new figure is created.
        The default is None.
    color : str
        The color of the hfb-lines in the plot. The default is 'red'.


    Returns
    -------
    ax : matplotlib.Axes
        The ax that conatins the hfb-lines.
    """
    if ax is None:
        _, ax = plt.subplots()

    if isinstance(gwf, xr.Dataset):
        modelgrid = modelgrid_from_ds(gwf)
    elif isinstance(gwf, flopy.discretization.grid.Grid):
        modelgrid = gwf
    else:
        modelgrid = gwf.modelgrid

    if modelgrid.grid_type == "structured":
        if isinstance(cellids, flopy.mf6.ModflowGwfhfb):
            spd = cellids.stress_period_data.data[0]
            cellids = [[row[0][1:], row[1][1:]] for row in spd]
        for line in cellids:
            pc1 = Polygon(modelgrid.get_cell_vertices(*line[0]))
            pc2 = Polygon(modelgrid.get_cell_vertices(*line[1]))
            x, y = pc1.intersection(pc2).xy
            ax.plot(x, y, color=color, **kwargs)

    elif modelgrid.grid_type == "vertex":
        if isinstance(cellids, flopy.mf6.ModflowGwfhfb):
            spd = cellids.stress_period_data.data[0]
            cellids = [[line[0][1], line[1][1]] for line in spd]
        for line in cellids:
            pc1 = Polygon(modelgrid.get_cell_vertices(line[0]))
            pc2 = Polygon(modelgrid.get_cell_vertices(line[1]))
            x, y = pc1.intersection(pc2).xy
            ax.plot(x, y, color=color, **kwargs)
    else:
        raise ValueError(f"not supported gridtype -> {modelgrid.grid_type}")

    return ax
