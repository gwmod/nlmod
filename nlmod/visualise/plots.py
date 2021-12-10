# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 21:32:49 2021.

@author: oebbe
"""

import os
import warnings

import flopy
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


from ..read import rws


def plot_surface_water(model_ds, ax=None):
    surf_water = rws.get_gdf_surface_water(model_ds)

    if ax is None:
        fig, ax = plt.subplots()
    surf_water.plot(ax=ax)

    return ax


def plot_modelgrid(model_ds, gwf, ax=None, add_surface_water=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

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
               figsize=(10, 8), plot_bc={}, plot_grid=False):

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
                  plot_bc=['CHD'], plot_bc_kwargs=[{'color': 'k'}], grid=False,
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
    plot_bc_kwargs : list of dictionaries, optional
        kwargs per boundary conditions. The default is [{'color':'k'}].
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
        if not key in gwf.get_package_list():
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
        for ibc, bc_var in enumerate(plot_bc):
            mp.plot_bc(bc_var, kper=0, **plot_bc_kwargs[ibc])

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
                  "'plot_vertex_array' instead", DeprecationWarning )
    
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)

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
    """ plot dataarray with gridtype vertex

    Parameters
    ----------
    da : xarray.Datarray
        plot data with dimension(cid).
    vertices : xarray.Datarray or numpy.ndarray
        Vertex coördinates per cell with dimensions(cid, 4, 2)
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
    
    if isinstance(vertices, xr.DataArray):
        vertices = vertices.values
    
    if ax is None:
        fig, ax = plt.subplots()
        
    patches = [Polygon(vert) for vert in vertices]
    if gridkwargs is None:
        quadmesh = PatchCollection(patches)
    else:
        quadmesh = PatchCollection(patches, **gridkwargs)
    quadmesh.set_array(da)
    
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
    quadmesh.set_clim(vmin=vmin, vmax=vmax)
    quadmesh.set(**kwargs)

    ax.add_collection(quadmesh)
    ax.set_xlim(vertices[:,:,0].min(), vertices[:,:,0].max())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(vertices[:,:,1].min(), vertices[:,:,1].max())
    ax.set_aspect('equal')
    
    ax.get_figure().colorbar(quadmesh, ax=ax, orientation='vertical')
    if hasattr(da, 'name'):
        ax.set_title(da.name)
    
    return ax