import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, Polygon

from ..dims.grid import gdf_to_da, gdf_to_grid


def get_hfb_spd(gwf, linestrings, hydchr=1 / 100, depth=None, elevation=None):
    """Generate a stress period data for horizontal flow barrier between two cell nodes,
    with several limitations. The stress period data can be used directly in the HFB
    package of flopy. The hfb is placed at the cell interface; it follows the sides of
    the cells.

    The estimation of the cross-sectional area at the interface is pretty crude, as the
    thickness at the cell interface is just the average of the thicknesses of the two
    cells.

    Parameters
    ----------
    gwf : Groundwater flow
        Groundwaterflow model from flopy.
    linestrings : geopandas.geodataframe
        DESCRIPTION
    hydchr : float
        Conductance of the horizontal flow barrier
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

    tops = np.concatenate((gwf.disv.top.array[None], gwf.disv.botm.array))
    thick = tops[:-1] - tops[1:]

    cells = line2hfb(linestrings, gwf)

    # drop cells on the edge of the model
    cells = [x for x in cells if len(x) > 1]

    spd = []

    # hydchr = 1 / 100  # resistance of 100 days
    for icell2d1, icell2d2 in cells:
        # TODO: Improve assumption of the thickness between the cells.
        thicki = (thick[:, icell2d1] + thick[:, icell2d2]) / 2
        topi = (tops[:, icell2d1] + tops[:, icell2d2]) / 2

        for ilay in range(gwf.disv.nlay.array):
            cellid1 = (ilay, icell2d1)
            cellid2 = (ilay, icell2d2)

            if gwf.disv.idomain.array[cellid1] <= 0:
                continue

            if gwf.disv.idomain.array[cellid2] <= 0:
                continue

            if depth is not None:
                if sum(thicki[: ilay + 1]) <= depth:
                    # hfb pierces the entire cell
                    spd.append([cellid1, cellid2, hydchr])

                elif sum(thicki[:ilay]) <= depth:
                    # hfb pierces the cell partially
                    hydchr_frac = (depth - sum(thicki[:ilay])) / thicki[ilay]
                    assert 0 <= hydchr_frac <= 1, "Something is wrong"

                    spd.append([cellid1, cellid2, hydchr * hydchr_frac])
                    break  # go to next cell

                else:
                    pass

            else:
                if topi[ilay + 1] >= elevation:
                    # hfb pierces the entire cell
                    spd.append([cellid1, cellid2, hydchr])

                else:
                    # hfb pierces the cell partially
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


    Parameters
    ----------
    gdf : gpd.GeoDataframe
        geodataframe with line elements.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        grondwater flow model.
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

    gdfg = gdf_to_grid(gdf, gwf)

    cell2d = pd.DataFrame(gwf.disv.cell2d.array).set_index("icell2d")
    vertices = pd.DataFrame(gwf.disv.vertices.array).set_index("iv")

    # for every cell determine which cell-edge could form the line
    # by testing for an intersection with a triangle to the cell-center
    icvert = cell2d.loc[:, cell2d.columns.str.startswith("icvert")].values

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
    hfb_seg = np.unique(hfb_seg[:, 1:], axis=0)

    # Get rid of disconnected (or 'open') segments
    # Let's remove disconnected/open segments
    iv = np.unique(hfb_seg)
    segments_per_iv = pd.Series([np.sum(hfb_seg == x) for x in iv], index=iv)
    mask = np.full(hfb_seg.shape[0], True)
    for i, segment in enumerate(hfb_seg):
        # one vertex is not connected and the other one at least to two other segments
        if (segments_per_iv[segment[0]] == 1 and segments_per_iv[segment[1]] >= 3) or (
            segments_per_iv[segment[1]] == 1 and segments_per_iv[segment[0]] >= 3
        ):
            mask[i] = False
    hfb_seg = hfb_seg[mask]

    if plot:
        # test by plotting
        ax = gdfg.plot()
        for i, seg in enumerate(hfb_seg):
            x = [vertices.at[seg[0], "xv"], vertices.at[seg[1], "xv"]]
            y = [vertices.at[seg[0], "yv"], vertices.at[seg[1], "yv"]]
            ax.plot(x, y)

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
    for seg in hfb_seg:
        cellids.append(list(segments.loc[[tuple(seg)]].values[:, 0]))
    return cellids


def polygon_to_hfb(
    gdf, ds, column=None, gwf=None, lay=0, hydchr=1 / 100, add_data=False
):
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
            spd.append([(lay, icell2d1), (lay, icell2d2), hydchr])
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
        flow barrier, hfb is the output of line2hfb.
    gwf : flopy groundwater flow model
        DESCRIPTION.
    ax : matplotlib axes


    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    """
    if ax is None:
        _, ax = plt.subplots()

    if gwf.modelgrid.grid_type == "structured":
        if isinstance(cellids, flopy.mf6.ModflowGwfhfb):
            spd = cellids.stress_period_data.data[0]
            cellids = [[row[0][1:], row[1][1:]] for row in spd]
        for line in cellids:
            pc1 = Polygon(gwf.modelgrid.get_cell_vertices(*line[0]))
            pc2 = Polygon(gwf.modelgrid.get_cell_vertices(*line[1]))
            x, y = pc1.intersection(pc2).xy
            ax.plot(x, y, color=color, **kwargs)

    elif gwf.modelgrid.grid_type == "vertex":
        if isinstance(cellids, flopy.mf6.ModflowGwfhfb):
            spd = cellids.stress_period_data.data[0]
            cellids = [[line[0][1], line[1][1]] for line in spd]
        for line in cellids:
            pc1 = Polygon(gwf.modelgrid.get_cell_vertices(line[0]))
            pc2 = Polygon(gwf.modelgrid.get_cell_vertices(line[1]))
            x, y = pc1.intersection(pc2).xy
            ax.plot(x, y, color=color, **kwargs)
    else:
        raise ValueError(f"not supported gridtype -> {gwf.modelgrid.grid_type}")

    return ax
