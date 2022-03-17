import flopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

from .. import mdims


def line2hfb(gdf, gwf, prevent_rings=True, plot=False):
    """ Obtain the cells with a horizontal flow barrier between them from a 
    geodataframe with line elements.


    Parameters
    ----------
    gdf : gpd.GeoDataframe
        geodataframe with line elements.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        grondwater flow model.
    prevent_rings : bool, optional
        DESCRIPTION. The default is True.
    plot : bool, optional
        If True create a simple plot of the grid cells and shapefile. For a
        more complex plot you can use plot_hfb. The default is False.

    Returns
    -------
    cellids : 2d list of ints
        a list with pairs of cells that have a hfb between them.

    """
    # for the idea, sea:
    # https://gis.stackexchange.com/questions/188755/how-to-snap-a-road-network-to-a-hexagonal-grid-in-qgis

    gdfg = mdims.gdf2grid(gdf, gwf)

    cell2d = pd.DataFrame(gwf.disv.cell2d.array).set_index('icell2d')
    vertices = pd.DataFrame(gwf.disv.vertices.array).set_index('iv')

    # for every cell determine which cell-edge could form the line
    # by testing for an intersection with a triangle to the cell-center
    icvert = cell2d.loc[:, cell2d.columns.str.startswith('icvert')].values

    hfb_seg = []
    for index in gdfg.index.unique():
        # Get the nearest hexagon sides where routes cross
        for icell2d in gdfg.loc[index, 'cellid']:
            for i in range(cell2d.at[icell2d, 'ncvert'] - 1):
                iv1 = icvert[icell2d, i]
                iv2 = icvert[icell2d, i + 1]
                # make sure vert1 is lower than vert2
                if iv1 > iv2:
                    iv1, iv2 = iv2, iv1
                coords = [(cell2d.at[icell2d, 'xc'], cell2d.at[icell2d, 'yc']),
                          (vertices.at[iv1, 'xv'], vertices.at[iv1, 'yv']),
                          (vertices.at[iv2, 'xv'], vertices.at[iv2, 'yv'])]
                triangle = Polygon(coords)
                if triangle.intersects(gdf.loc[index, 'geometry']):
                    hfb_seg.append((icell2d, iv1, iv2))
    hfb_seg = np.array(hfb_seg)

    if prevent_rings:
        # find out if there are cells with segments on each side
        # remove the segments whose centroid is farthest from the line
        for icell2d in np.unique(hfb_seg[:, 0]):
            mask = hfb_seg[:, 0] == icell2d
            if mask.sum() >= cell2d.at[icell2d, 'ncvert'] - 1:
                segs = hfb_seg[mask]
                dist = []
                for seg in segs:
                    p = Point(vertices.loc[seg[1:3], 'xv'].mean(),
                              vertices.loc[seg[1:3], 'yv'].mean())
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
        if ((segments_per_iv[segment[0]] == 1 and
             segments_per_iv[segment[1]] >= 3) or
            (segments_per_iv[segment[1]] == 1 and
             segments_per_iv[segment[0]] >= 3)):
            mask[i] = False
    hfb_seg = hfb_seg[mask]

    if plot:
        # test by plotting
        ax = gdfg.plot()
        for i, seg in enumerate(hfb_seg):
            x = [vertices.at[seg[0], 'xv'], vertices.at[seg[1], 'xv']]
            y = [vertices.at[seg[0], 'yv'], vertices.at[seg[1], 'yv']]
            ax.plot(x, y)

    # find out between which cellid's these segments are
    segments = []
    for icell2d in cell2d.index:
        for i in range(cell2d.at[icell2d, 'ncvert'] - 1):
            iv1 = icvert[icell2d, i]
            iv2 = icvert[icell2d, i + 1]
            # make sure vert1 is lower than vert2
            if iv1 > iv2:
                iv1, iv2 = iv2, iv1
            segments.append((icell2d, (iv1, iv2)))
    segments = pd.DataFrame(segments, columns=['icell2d', 'verts'])
    segments = segments.set_index(['verts'])

    cellids = []
    for seg in hfb_seg:
        cellids.append(list(segments.loc[[tuple(seg)]].values[:, 0]))
    return cellids


def plot_hfb(cellids, gwf, ax=None):
    """ plots a horizontal flow barrier


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
        fig, ax = plt.subplots()

    if isinstance(cellids, flopy.mf6.ModflowGwfhfb):
        spd = cellids.stress_period_data.data[0]
        cellids = [[line[0][1], line[1][1]] for line in spd]

    for line in cellids:
        pc1 = Polygon(gwf.modelgrid.get_cell_vertices(line[0]))
        pc2 = Polygon(gwf.modelgrid.get_cell_vertices(line[1]))
        x, y = pc1.intersection(pc2).xy
        ax.plot(x, y, color='red')

    return ax
