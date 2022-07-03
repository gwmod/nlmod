# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 21:32:49 2021.

@author: oebbe
"""

import os
import warnings

import flopy
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ..read import rws
from ..mdims import get_vertices


def plot_surface_water(model_ds, ax=None):
    surf_water = rws.get_gdf_surface_water(model_ds)

    if ax is None:
        _, ax = plt.subplots()
    surf_water.plot(ax=ax)

    return ax


def plot_modelgrid(model_ds, gwf, ax=None, add_surface_water=True):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    gwf.modelgrid.plot(ax=ax)
    ax.axis('scaled')
    if add_surface_water:
        plot_surface_water(model_ds, ax=ax)
        ax.set_title('modelgrid with surface water')
    else:
        ax.set_title('modelgrid')
    ax.set_ylabel('y [m RD]')
    ax.set_xlabel('x [m RD]')

    return ax


def facet_plot(gwf, arr, lbl="", plot_dim="layer", layer=None, period=None,
               cmap="viridis", scale_cbar=True, vmin=None, vmax=None,
               norm=None, xlim=None, ylim=None, grid=False, figdir=None,
               figsize=(10, 8), plot_bc=None, plot_grid=False):

    if arr.ndim == 4 and plot_dim == "layer":
        nplots = arr.shape[1]
    elif arr.ndim == 4 and plot_dim == "time":
        nplots = arr.shape[0]
    elif arr.ndim == 3:
        nplots = arr.shape[0]
    else:
        raise ValueError("Array must have at least 3 dimensions.")

    plots_per_row = int(np.ceil(np.sqrt(nplots)))
    plots_per_col = nplots // plots_per_row + 1

    fig, axes = plt.subplots(
        plots_per_col, plots_per_row, figsize=figsize,
        sharex=True, sharey=True, constrained_layout=True)

    if scale_cbar:
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)

    for i in range(nplots):
        iax = axes.flat[i]
        iax.set_aspect("equal")
        if plot_dim == "layer":
            ilay = i
            iper = period
            if arr.ndim == 4:
                if iper is None:
                    raise ValueError("Pass 'period' to select "
                                     "timestep to plot.")
                a = arr[iper]
        elif plot_dim == "time":
            ilay = layer
            iper = i
            if arr.ndim == 4:
                if ilay is None:
                    raise ValueError("Pass 'layer' to select "
                                     "layer to plot.")
                a = arr[iper]
        else:
            raise ValueError("'plot_dim' must be one of ['layer', 'time']")

        mp = flopy.plot.PlotMapView(model=gwf, layer=ilay, ax=iax)
        qm = mp.plot_array(a, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)

        mp.plot_ibound(color_vpt="darkgray")

        if plot_grid:
            mp.plot_grid(ls=0.25, color="k")

        if plot_bc is not None:
            for bc, bc_kwargs in plot_bc.items():
                mp.plot_bc(bc, **bc_kwargs)

        iax.grid(grid)
        iax.set_xticklabels([])
        iax.set_yticklabels([])

        if plot_dim == "layer":
            iax.set_title(f"Layer {ilay}", fontsize=6)
        elif plot_dim == "time":
            iax.set_title(f"Timestep {iper}", fontsize=6)

        if xlim is not None:
            iax.set_xlim(xlim)
        if ylim is not None:
            iax.set_ylim(ylim)

    for iax in axes.ravel()[nplots:]:
        iax.set_visible(False)

    cb = fig.colorbar(qm, ax=axes, shrink=1.0)
    cb.set_label(lbl)

    if figdir:
        fig.savefig(os.path.join(figdir, f"{lbl}_per_{plot_dim}.png"),
                    dpi=150, bbox_inches="tight")

    return fig, axes


def facet_plot_ds(gwf, model_ds, figdir, plot_var='bot', plot_time=None,
                  plot_bc=('CHD',), color='k', grid=False,
                  xlim=None, ylim=None):
    """make a 2d plot of every modellayer, store them in a grid.

    Parameters
    ----------
    gwf : Groundwater flow
        Groundwaterflow model.
    model_ds : xr.DataSet
        model data.
    figdir : str
        file path figures.
    plot_var : str, optional
        variable in model_ds. The default is 'bot'.
    plot_time : int, optional
        time step if plot_var is time variant. The default is None.
    plot_bc : list of str, optional
        name of packages of which boundary conditions are plot. The default
        is ['CHD'].
    color : str, optional
        color. The default is 'k'.
    grid : bool, optional
        if True a grid is plotted. The default is False.
    xlim : tuple, optional
        xlimits. The default is None.
    ylim : tuple, optional
        ylimits. The default is None.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axes : TYPE
        DESCRIPTION.
    """
    for key in plot_bc:
        if key not in gwf.get_package_list():
            raise ValueError(
                f'cannot plot boundary condition {key} because it is not in the package list')

    nlay = len(model_ds.layer)

    plots_per_row = int(np.ceil(np.sqrt(nlay)))
    plots_per_col = nlay // plots_per_row + 1

    fig, axes = plt.subplots(
        plots_per_col, plots_per_row, figsize=(11, 10),
        sharex=True, sharey=True, dpi=150
    )
    if plot_time is None:
        plot_arr = model_ds[plot_var]
    else:
        plot_arr = model_ds[plot_var][plot_time]

    vmin = plot_arr.min()
    vmax = plot_arr.max()
    for ilay in range(nlay):
        iax = axes.ravel()[ilay]
        mp = flopy.plot.PlotMapView(model=gwf, layer=ilay, ax=iax)
        # mp.plot_grid()
        qm = mp.plot_array(plot_arr[ilay].values, cmap="viridis",
                           vmin=vmin, vmax=vmax)
        # qm = mp.plot_array(hf[-1], cmap="viridis", vmin=-0.1, vmax=0.1)
        # mp.plot_ibound()
        # plt.colorbar(qm)
        for bc_var in plot_bc:
            mp.plot_bc(bc_var, kper=0, color=color)

        iax.set_aspect("equal", adjustable="box")
        iax.set_title(f"Layer {ilay}")

        iax.grid(grid)
        if xlim is not None:
            iax.set_xlim(xlim)
        if ylim is not None:
            iax.set_ylim(ylim)

    for iax in axes.ravel()[nlay:]:
        iax.set_visible(False)

    cb = fig.colorbar(qm, ax=axes, shrink=1.0)
    cb.set_label(f'{plot_var}', rotation=270)
    fig.suptitle(
        f"{plot_var} Time = {(model_ds.nper*model_ds.perlen)/365} year")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{plot_var}_per_layer.png"), dpi=150,
                bbox_inches="tight")

    return fig, axes


def plot_array(gwf, array, figsize=(8, 8), colorbar=True, ax=None, **kwargs):

    warnings.warn("The 'plot_array' functions is deprecated please use"
                  "'plot_vertex_array' instead", DeprecationWarning)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    yticklabels = ax.yaxis.get_ticklabels()
    plt.setp(yticklabels, rotation=90, verticalalignment='center')
    ax.axis('scaled')
    pmv = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid, ax=ax)
    pcm = pmv.plot_array(array, **kwargs)
    if colorbar:
        fig = ax.get_figure()
        fig.colorbar(pcm, ax=ax, orientation='vertical')
        # plt.colorbar(pcm)
    if hasattr(array, 'name'):
        ax.set_title(array.name)
    # set rotation of y ticks to zero
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    return ax


def plot_vertex_array(da, vertices, ax=None, gridkwargs=None, **kwargs):
    """plot dataarray with gridtype vertex.

    Parameters
    ----------
    da : xarray.Datarray
        plot data with dimension(icell2d).
    vertices : xarray.Datarray or numpy.ndarray
        Vertex coördinates per cell with dimensions(icell2d, 4, 2)
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    gridkwargs : dict or None, optional
        layout parameters to plot the cells. For example
        {'edgecolor':'k'} to create black cell lines. The default is None.
    **kwargs :
        passed to quadmesh.set().

    Returns
    -------
    ax : TYPE
        DESCRIPTION.
    """

    if isinstance(vertices, xr.Dataset):
        vertices = get_vertices(vertices)
    if isinstance(vertices, xr.DataArray):
        vertices = vertices.values

    if ax is None:
        _, ax = plt.subplots()

    patches = [Polygon(vert) for vert in vertices]
    if gridkwargs is None:
        pc = PatchCollection(patches)
    else:
        pc = PatchCollection(patches, **gridkwargs)
    pc.set_array(da)

    # set max and min
    if "vmin" in kwargs:
        vmin = kwargs.pop("vmin")
    else:
        vmin = None

    if "vmax" in kwargs:
        vmax = kwargs.pop("vmax")
    else:
        vmax = None

    # limit the color range
    pc.set_clim(vmin=vmin, vmax=vmax)
    pc.set(**kwargs)

    ax.add_collection(pc)
    ax.set_xlim(vertices[:, :, 0].min(), vertices[:, :, 0].max())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(vertices[:, :, 1].min(), vertices[:, :, 1].max())
    ax.set_aspect('equal')

    ax.get_figure().colorbar(pc, ax=ax, orientation='vertical')
    if hasattr(da, 'name'):
        ax.set_title(da.name)

    return ax


def da(da, ds=None, ax=None, **kwargs):
    """
    PLot an xarray DataArray

    Parameters
    ----------
    da : xarray.DataArray
        DESCRIPTION.
    ds : xarray.DataSet, optional
        Needed when the calculation grid is . The default is None.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    matplotlib QuadMesh or PatchCollection
        The object containing the cells.

    """
    if ax is None:
        ax = plt.gca()
    if 'icell2d' in da.dims:
        if ds is None:
            raise(Exception('Supply model dataset (ds) for grid information'))
        vertices = get_vertices(ds)
        patches = [Polygon(vert) for vert in vertices]
        pc = PatchCollection(patches, **kwargs)
        pc.set_array(da)
        ax.add_collection(pc)
        return pc
    else:
        return ax.pcolormesh(da.x, da.y, da, **kwargs)


def get_map(extent, figsize=10., nrows=1, ncols=1, base=1000., fmt='{:.0f}',
            sharex=False, sharey=True):
    """
    Generate a motplotlib Figure with a map with the axis set to extent

    Parameters
    ----------
    extent : list of 4 floats
        The model extent .
    figsize : float or list of 2 floats, optional
        The size of the figure, in inches. The default is 10, which means the
        figsize is determined automatically.
    nrows : int, optional
        THe number of rows. The default is 1.
    ncols : int, optional
        THe number of columns. The default is 1.
    base : float, optional
        The interval for ticklabels on the x- and y-axis. The default is 1000.
        m.
    fmt : string, optional
        The format of the ticks on the x- and y-axis. The default is '{:.0f}'.
    sharex : bool, optional
        Only display the ticks on the lowest x-axes, when nrows > 1. The
        default is False.
    sharey : bool, optional
        Only display the ticks on the left y-axes, when ncols > 1. The default
        is True.

    Returns
    -------
    f : matplotlib.Figure
        The resulting figure.
    axes : matplotlib.Axes or numpy array of matplotlib.Axes
        the ax or axes (when ncols/nrows > 1).

    """
    if isinstance(figsize, float) or isinstance(figsize, int):
        xh = 0.2
        if base is None:
            xh = 0.0
        figsize = get_figsize(extent, nrows=nrows, ncols=ncols, figw=figsize,
                              xh=xh)
    f, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                           sharex=sharex, sharey=sharey)

    def set_ax_in_map(ax, extent, base=1000., fmt='{:.0f}'):
        ax.axis('scaled')
        ax.axis(extent)
        rotate_yticklabels(ax)
        if base is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            rd_ticks(ax, base=base, fmt=fmt)
    if nrows == 1 and ncols == 1:
        set_ax_in_map(axes, extent, base=base, fmt=fmt)
    else:
        for ax in axes.ravel():
            set_ax_in_map(ax, extent, base=base, fmt=fmt)
    f.tight_layout(pad=0.0)

    return f, axes


def get_figsize(extent, figw=10., nrows=1, ncols=1, xh=0.2):
    """Get a figure size in inches, calculated from a model extent"""
    w = extent[1] - extent[0]
    h = extent[3] - extent[2]
    axh = (figw/ncols) * (h / w) + xh
    figh = nrows * axh
    figsize = (figw, figh)
    return figsize


def rotate_yticklabels(ax):
    """Rotate the labels on the y-axis 90 degrees to save space"""
    yticklabels = ax.yaxis.get_ticklabels()
    plt.setp(yticklabels, rotation=90, verticalalignment='center')


def rd_ticks(ax, base=1000., fmt_base=1000., fmt='{:.0f}'):
    """Add ticks every 1000 (base) m, and divide ticklabels by 1000 (fmt_base)
    """
    def fmt_rd_ticks(x, y):
        return fmt.format(x / fmt_base)
    if base is not None:
        ax.xaxis.set_major_locator(MultipleLocator(base))
        ax.yaxis.set_major_locator(MultipleLocator(base))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_rd_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_rd_ticks))


def colorbar_inside(mappable=None, ax=None, norm=None, cmap=None,
                    bounds=None, **kw):
    """Place a colorbar inside an axes"""
    if ax is None:
        ax = plt.gca()
    if bounds is None:
        bounds = [0.95, 0.05, 0.02, 0.9]
    cax = ax.inset_axes(bounds, facecolor='none')
    if mappable is None and norm is not None and cmap is not None:
        # make an empty dataset...
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable._A = []
    cb = plt.colorbar(mappable, cax=cax, ax=ax, **kw)
    if bounds[0] > 0.5:
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position("left")
    return cb
