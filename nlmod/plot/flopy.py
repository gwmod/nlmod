from functools import partial

import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .plotutil import add_background_map, get_figsize, get_map, title_inside


def _get_figure(ax=None, gwf=None, figsize=None):
    # figure
    if ax is not None:
        f = ax.figure
    else:
        if figsize is None:
            figsize = get_figsize(gwf.modelgrid.extent)
            # try to ensure pixel size is divisible by 2
            figsize = (figsize[0], np.round(figsize[1] / 0.02, 0) * 0.02)

        base = 10 ** int(np.log10(gwf.modelgrid.extent[1] - gwf.modelgrid.extent[0]))
        f, ax = get_map(gwf.modelgrid.extent, base=base, figsize=figsize)
        ax.set_aspect("equal", adjustable="box")
    return f, ax


def map_array(
    arr,
    gwf,
    ilay=0,
    iper=0,
    extent=None,
    ax=None,
    title="",
    xlabel="X [km RD]",
    ylabel="Y [km RD]",
    norm=None,
    vmin=None,
    vmax=None,
    levels=None,
    cmap="viridis",
    alpha=1.0,
    colorbar=True,
    colorbar_label="",
    plot_grid=False,
    add_to_plot=None,
    background=False,
    figsize=None,
    animate=False,
):
    """Plot an array using flopy PlotMapView.

    Parameters
    ----------
    arr : np.array, xarray.DataArray
        array to plot
    gwf : flopy.mf6.ModflowGwf or flopy.mf6.ModflowGwt
        flopy groundwater flow or transport model
    ilay : int, optional
        layer to plot, by default 0
    iper : int, optional
        timestep to plot, by default 0
    extent : list or tuple, optional
        plot extent: (xmin, xmax, ymin, ymax), by default None which defaults
        model extent
    ax : matplotlib Axes, optional
        axis handle to plot on, by default None
    title : str, optional
        title of the plot, by default "" (blank)
    xlabel : str, optional
        x-axis label, by default "X [km RD]"
    ylabel : str, optional
        y-axis label, by default "Y [km RD]"
    norm : matplotlib.colors.Norm
        colorbar norm
    vmin : float, optional
        minimum value for colorbar
    vmax : float, optional
        maximum value for colorbar
    levels : np.array, optional
        colorbar levels, used for setting colorbar ticks
    cmap : str or colormap, optional
        colormap, default is "viridis"
    alpha : float, optional
        transparency, by default 1.0
    plot_grid : bool, optional
        plot model grid, by default False
    add_to_plot : tuple of func, optional
        tuple or list of plotting functions that take ax as the
        only argument, by default None. Use to add features to plot, e.g.
        plotting shapefiles, or other data.
    background : bool, optional
        add background map, by default False
    figsize : tuple, optional
        figure size, by default None
    animate : bool, optional
        if True return figure, axis and quadmesh handles, by default
        False (returns only axes handle)

    Returns
    -------
    ax : matplotlib Axes
        axes handle
    f, ax, qm :
        only if animate is True, return figure, axes and quadmesh handles.
    """
    # get data
    if isinstance(arr, xr.DataArray):
        arr = arr.values

    # get correct timestep and layer if need be
    if len(arr.shape) == 4:
        arr = arr[iper]
    if len(arr.shape) == 3:
        arr = arr[ilay]

    # get figure
    f, ax = _get_figure(ax=ax, gwf=gwf, figsize=figsize)

    # get normalization if vmin/vmax are passed
    if vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # get plot obj
    pmv = flopy.plot.PlotMapView(gwf, layer=ilay, ax=ax, extent=extent)

    # plot data
    qm = pmv.plot_array(arr, cmap=cmap, norm=norm, alpha=alpha)

    # bgmap
    if background:
        add_background_map(ax, map_provider="nlmaps.water", alpha=0.5)

    # add other info to plot
    if add_to_plot is not None:
        for fplot in add_to_plot:
            fplot(ax)

    if plot_grid:
        pmv.plot_grid(lw=0.25, alpha=0.5)

    # axes properties
    axprops = {"xlabel": xlabel, "ylabel": ylabel, "title": title}
    ax.set(**axprops)

    # colorbar
    divider = make_axes_locatable(ax)
    if colorbar:
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = f.colorbar(qm, cax=cax)
        if levels is not None:
            cbar.set_ticks(levels)
        cbar.set_label(colorbar_label)

    if animate:
        return f, ax, qm
    else:
        return ax


def contour_array(
    arr,
    gwf,
    ilay=0,
    iper=0,
    extent=None,
    ax=None,
    title="",
    xlabel="X [km RD]",
    ylabel="Y [km RD]",
    levels=10,
    alpha=1.0,
    labels=True,
    label_kwargs=None,
    plot_grid=False,
    add_to_plot=None,
    background=False,
    figsize=None,
    animate=False,
    **kwargs,
):
    """Contour an array using flopy PlotMapView.

    Parameters
    ----------
    arr : np.array, xarray.DataArray
        array to contour
    gwf : flopy.mf6.ModflowGwf or flopy.mf6.ModflowGwt
        flopy groundwater flow or transport model
    ilay : int, optional
        layer to contour, by default 0
    iper : int, optional
        timestep to contour, by default 0
    extent : list or tuple, optional
        plot extent: (xmin, xmax, ymin, ymax), by default None which defaults
        model extent
    ax : matplotlib Axes, optional
        axis handle to plot on, by default None
    title : str, optional
        title of the plot, by default "" (blank)
    xlabel : str, optional
        x-axis label, by default "X [km RD]"
    ylabel : str, optional
        y-axis label, by default "Y [km RD]"
    levels : int, list, np.array, optional
        contour levels, when passed as int draw that many contours, when
        list or array draw contours at provided levels, by default 10
    alpha : float, optional
        transparency of contour lines, by default 1.0
    labels : bool, optional
        add contour labels showing contour levels, by default True
    label_kwargs : dict, optional
        keyword arguments passed onto ax.clabel(), by default None
    plot_grid : bool, optional
        plot model grid, by default False
    add_to_plot : tuple of func, optional
        tuple or list of plotting functions that take ax as the
        only argument, by default None. Use to add features to plot, e.g.
        plotting shapefiles, or other data.
    background : bool, optional
        add background map, by default False
    figsize : tuple, optional
        figure size, by default None
    animate : bool, optional
        if True return figure, axis and contour handles, by default
        False (returns only axes handle)

    Returns
    -------
    ax : matplotlib Axes
        axes handle
    f, ax, cs :
        only if animate is True, return figure, axes and contour handles.
    """
    # get data
    if isinstance(arr, xr.DataArray):
        arr = arr.values

    # get correct timestep and layer if need be
    if len(arr.shape) == 4:
        arr = arr[iper]
    if len(arr.shape) == 3:
        arr = arr[ilay]

    # get figure
    f, ax = _get_figure(ax=ax, gwf=gwf, figsize=figsize)

    # get plot obj
    pmv = flopy.plot.PlotMapView(gwf, layer=ilay, ax=ax, extent=extent)

    # plot data
    cs = pmv.contour_array(arr, levels=levels, alpha=alpha, **kwargs)
    if labels:
        if label_kwargs is None:
            label_kwargs = {}
        ax.clabel(cs, **label_kwargs)

    # bgmap
    if background:
        add_background_map(ax, map_provider="nlmaps.water", alpha=0.5)

    # add other info to plot
    if add_to_plot is not None:
        for fplot in add_to_plot:
            fplot(ax)

    if plot_grid:
        pmv.plot_grid(lw=0.25, alpha=0.5)

    # axes properties
    axprops = {"xlabel": xlabel, "ylabel": ylabel, "title": title}
    ax.set(**axprops)

    if animate:
        return f, ax, cs
    else:
        return ax


def animate_map(
    arr,
    times,
    gwf,
    ilay=0,
    extent=None,
    ax=None,
    title="",
    xlabel="X [km RD]",
    ylabel="Y [km RD]",
    datefmt="%Y-%m",
    norm=None,
    vmin=None,
    vmax=None,
    levels=None,
    cmap="viridis",
    alpha=1.0,
    colorbar=True,
    colorbar_label="",
    plot_grid=True,
    add_to_plot=None,
    background=False,
    figsize=(9.24, 10.042),
    save=False,
    fname=None,
):
    """Animate a map over time.

    Parameters
    ----------
    arr : np.ndarray or xr.DataArray
        Array to animate with shape (time, layer, y, x).
    times : list
        List of time values for each frame.
    gwf : flopy ModflowGwf
        Groundwater flow model object.
    ilay : int, optional
        Layer index to plot. The default is 0.
    extent : tuple, optional
        Extent for the plot. The default is None.
    ax : matplotlib Axes, optional
        Axes to plot on. The default is None.
    title : str, optional
        Plot title. The default is "".
    xlabel : str, optional
        X-axis label. The default is "X [km RD]".
    ylabel : str, optional
        Y-axis label. The default is "Y [km RD]".
    datefmt : str, optional
        Date format string. The default is "%Y-%m".
    norm : matplotlib.colors.Normalize, optional
        Color normalization. The default is None.
    vmin : float, optional
        Minimum value for color scale. The default is None.
    vmax : float, optional
        Maximum value for color scale. The default is None.
    levels : list, optional
        Contour levels. The default is None.
    cmap : str, optional
        Colormap name. The default is "viridis".
    alpha : float, optional
        Transparency. The default is 1.0.
    colorbar : bool, optional
        Show colorbar. The default is True.
    colorbar_label : str, optional
        Colorbar label. The default is "".
    plot_grid : bool, optional
        Plot grid. The default is True.
    add_to_plot : callable, optional
        Function to add additional elements to the plot. The default is None.
    background : bool, optional
        Use background. The default is False.
    figsize : tuple, optional
        Figure size. The default is (9.24, 10.042).
    save : bool, optional
        Save animation. The default is False.
    fname : str, optional
        Filename for saved animation. The default is None.

    Returns
    -------
    FuncAnimation
        matplotlib animation object.
    """
    # get data
    if isinstance(arr, xr.DataArray):
        arr = arr.values

    # get correct layer if need be
    if isinstance(arr, list):
        arr = np.stack(arr)
    if len(arr.shape) == 4 and arr.shape[1] > 1:
        arr = arr[:, ilay]
    elif len(arr.shape) < 3:
        raise ValueError("Array has too few dimensions!")

    # plot base image
    f, ax, qm = map_array(
        arr,
        gwf,
        ilay=ilay,
        iper=0,
        extent=extent,
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        alpha=alpha,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        plot_grid=plot_grid,
        add_to_plot=add_to_plot,
        background=background,
        figsize=figsize,
        animate=True,
    )
    # add updating title
    t = pd.Timestamp(times[0])
    title = title_inside(
        f"Layer {ilay}, t = {t.strftime(datefmt)}",
        ax,
        x=0.025,
        bbox={"facecolor": "w"},
        horizontalalignment="left",
    )

    # write update func
    def update(iper, qm, title):
        # select timestep
        ai = arr[iper]

        # update quadmesh
        qm.set_array(ai.ravel())

        # update title
        t = pd.Timestamp(times[iper])
        title.set_text(f"Layer {ilay}, t = {t.strftime(datefmt)}")

        return qm, title

    # create animation
    anim = FuncAnimation(
        f,
        partial(update, qm=qm, title=title),
        frames=len(times),
        blit=False,
        interval=100,
    )

    # save animation as mp4
    if save:
        writer = FFMpegWriter(
            fps=10,
            bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p"],
            codec="libx264",
        )
        anim.save(fname, writer=writer)

    return f, anim


def facet_plot(
    gwf,
    arr,
    lbl="",
    plot_dim="layer",
    layer=None,
    period=None,
    cmap="viridis",
    scale_cbar=True,
    vmin=None,
    vmax=None,
    norm=None,
    xlim=None,
    ylim=None,
    grid=False,
    figsize=(10, 8),
    plot_bc=None,
    plot_grid=False,
):
    """Create a facet plot for model results.

    Parameters
    ----------
    gwf : flopy ModflowGwf
        Groundwater flow model object.
    arr : np.ndarray
        Array to plot with shape (time, layer, y, x) or (layer, y, x).
    lbl : str, optional
        Label for colorbar. The default is "".
    plot_dim : str, optional
        Dimension to plot ("layer" or "time"). The default is "layer".
    layer : int, optional
        Layer index to plot. The default is None (all layers).
    period : int, optional
        Stress period to plot. The default is None (all periods).
    cmap : str, optional
        Colormap name. The default is "viridis".
    scale_cbar : bool, optional
        Scale colorbar. The default is True.
    vmin : float, optional
        Minimum value for color scale. The default is None.
    vmax : float, optional
        Maximum value for color scale. The default is None.
    norm : matplotlib.colors.Normalize, optional
        Color normalization. The default is None.
    xlim : tuple, optional
        X-axis limits. The default is None.
    ylim : tuple, optional
        Y-axis limits. The default is None.
    grid : bool, optional
        Show grid. The default is False.
    figsize : tuple, optional
        Figure size. The default is (10, 8).
    plot_bc : callable, optional
        Function to plot boundary conditions. The default is None.
    plot_grid : bool, optional
        Plot grid. The default is False.

    Returns
    -------
    fig : matplotlib Figure
        Figure object.
    axes : list
        List of Axes objects.
    """
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
        plots_per_col,
        plots_per_row,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

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
                    raise ValueError("Pass 'period' to select timestep to plot.")
                a = arr[iper]
            else:
                a = arr
        elif plot_dim == "time":
            ilay = layer
            iper = i
            if arr.ndim == 4:
                if ilay is None:
                    raise ValueError("Pass 'layer' to select layer to plot.")
                a = arr[iper]
            else:
                a = arr
        else:
            raise ValueError("'plot_dim' must be one of ['layer', 'time']")

        mp = flopy.plot.PlotMapView(model=gwf, layer=ilay, ax=iax)
        qm = mp.plot_array(a, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)

        mp.plot_ibound(color_vpt="darkgray")

        if plot_grid:
            mp.plot_grid(lw=0.25, color="k")

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

    return fig, axes
