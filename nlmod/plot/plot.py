import os
from functools import partial

import flopy as fp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..dcs import DatasetCrossSection
from ..dims.grid import modelgrid_from_ds
from ..dims.resample import get_affine_mod_to_world, get_extent
from ..read import geotop, rws
from .plotutil import (
    add_background_map,
    get_figsize,
    get_map,
    get_patches,
    rd_ticks,
    title_inside,
)


def surface_water(model_ds, ax=None, **kwargs):
    surf_water = rws.get_gdf_surface_water(model_ds)

    if ax is None:
        _, ax = plt.subplots()
    surf_water.plot(ax=ax, **kwargs)

    return ax


def modelgrid(ds, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
        ax.axis("scaled")
    modelgrid = modelgrid_from_ds(ds)
    modelgrid.plot(ax=ax, **kwargs)
    return ax


def facet_plot(
    gwf,
    ds,
    figdir,
    plot_var="bot",
    plot_time=None,
    plot_bc=("CHD",),
    color="k",
    grid=False,
    xlim=None,
    ylim=None,
):
    """make a 2d plot of every modellayer, store them in a grid.

    Parameters
    ----------
    gwf : Groundwater flow
        Groundwaterflow model.
    ds : xr.DataSet
        model dataset.
    figdir : str
        file path figures.
    plot_var : str, optional
        variable in ds. The default is 'bot'.
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
                f"cannot plot boundary condition {key} "
                "because it is not in the package list"
            )

    nlay = len(ds.layer)

    plots_per_row = int(np.ceil(np.sqrt(nlay)))
    plots_per_col = nlay // plots_per_row + 1

    fig, axes = plt.subplots(
        plots_per_col,
        plots_per_row,
        figsize=(11, 10),
        sharex=True,
        sharey=True,
        dpi=150,
    )
    if plot_time is None:
        plot_arr = ds[plot_var]
    else:
        plot_arr = ds[plot_var][plot_time]

    vmin = plot_arr.min()
    vmax = plot_arr.max()
    for ilay in range(nlay):
        iax = axes.ravel()[ilay]
        mp = fp.plot.PlotMapView(model=gwf, layer=ilay, ax=iax)
        # mp.plot_grid()
        qm = mp.plot_array(plot_arr[ilay].values, cmap="viridis", vmin=vmin, vmax=vmax)
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
    cb.set_label(f"{plot_var}", rotation=270)
    fig.suptitle(f"{plot_var} Time = {(ds.nper*ds.perlen)/365} year")
    fig.tight_layout()
    fig.savefig(
        os.path.join(figdir, f"{plot_var}_per_layer.png"),
        dpi=150,
        bbox_inches="tight",
    )

    return fig, axes


def data_array(da, ds=None, ax=None, rotated=False, edgecolor=None, **kwargs):
    """Plot an xarray DataArray, using information from the model Dataset ds.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray (structured or vertex) you like to plot.
    ds : xarray.DataSet, optional
        Needed when the gridtype is vertex or rotated is True. The default is None.
    ax : matplotlib.Axes, optional
        The axes used for plotting. Set to current axes when None. The default is None.
    rotated : bool, optional
        Plot the data-array in rotated coordinates
    **kwargs : cit
        Kwargs are passed to PatchCollection (vertex) or pcolormesh (structured).

    Returns
    -------
    matplotlib QuadMesh or PatchCollection
        The object containing the cells.
    """
    if ax is None:
        ax = plt.gca()
    if "icell2d" in da.dims:
        if ds is None:
            raise (Exception("Supply model dataset (ds) for grid information"))
        if isinstance(ds, list):
            patches = ds
        else:
            patches = get_patches(ds, rotated=rotated)
        if edgecolor is None:
            edgecolor = "face"
        pc = PatchCollection(patches, edgecolor=edgecolor, **kwargs)
        pc.set_array(da)
        ax.add_collection(pc)
        if ax.get_autoscale_on():
            extent = get_extent(ds, rotated=rotated)
            ax.axis(extent)
        return pc
    else:
        x = da.x
        y = da.y
        if rotated:
            if ds is None:
                raise (Exception("Supply model dataset (ds) for grid information"))
            if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
                affine = get_affine_mod_to_world(ds)
                x, y = affine * np.meshgrid(x, y)
        return ax.pcolormesh(x, y, da, shading="nearest", edgecolor=edgecolor, **kwargs)


def geotop_lithok_in_cross_section(
    line, gt=None, ax=None, legend=True, legend_loc=None, lithok_props=None, **kwargs
):
    """PLot the lithoclass-data of GeoTOP in a cross-section.

    Parameters
    ----------
    line : sahpely.LineString
        The line along which the GeoTOP data is plotted
    gt : xr.Dataset, optional
        The voxel-dataset from GeoTOP. It is downloaded with the method
        nlmod.read.geaotop.get_geotop_raw_within_extent if None. The default is None.
    ax : matplotlib.Axes, optional
        The axes in whcih the cross-section is plotted. Will default to the current axes
        if None. The default is None.
    legend : bool, optional
        When True, add a legend to the plot with the lithology-classes. The default is
        True.
    legend_loc : None or str, optional
        The location of the legend. See matplotlib documentation. The default is None.
    lithok_props : pd.DataFrame, optional
        A DataFrame containing the properties of the lithoclasses.
        Will call nlmod.read.geotop.get_lithok_props() when None. The default is None.

    **kwargs : dict
        kwargs are passed onto DatasetCrossSection.

    Returns
    -------
    cs : DatasetCrossSection
        The instance of DatasetCrossSection that is used to plot the cross-section.
    """
    if ax is None:
        ax = plt.gca()

    if gt is None:
        # download geotop
        x = [coord[0] for coord in line.coords]
        y = [coord[1] for coord in line.coords]
        extent = [min(x), max(x), min(y), max(y)]
        gt = geotop.get_geotop_raw_within_extent(extent)

    if "top" not in gt or "botm" not in gt:
        gt = geotop.add_top_and_botm(gt)

    if lithok_props is None:
        lithok_props = geotop.get_lithok_props()

    cs = DatasetCrossSection(gt, line, layer="z", ax=ax, **kwargs)
    lithoks = gt["lithok"].data
    lithok_un = np.unique(lithoks[~np.isnan(lithoks)])
    array = np.full(lithoks.shape, np.NaN)

    colors = []
    for i, lithok in enumerate(lithok_un):
        lithok = int(lithok)
        array[lithoks == lithok] = i
        colors.append(lithok_props.at[lithok, "color"])
    cmap = ListedColormap(colors)
    norm = Normalize(-0.5, np.nanmax(array) + 0.5)
    cs.plot_array(array, norm=norm, cmap=cmap)
    if legend:
        # make a legend with dummy handles
        handles = []
        for i, lithok in enumerate(lithok_un):
            label = lithok_props.at[int(lithok), "name"]
            handles.append(Patch(facecolor=colors[i], label=label))
        ax.legend(handles=handles, loc=legend_loc)

    return cs


def map_array(
    da,
    ds,
    ilay=0,
    iper=0,
    ax=None,
    title="",
    xlabel="X [km RD]",
    ylabel="Y [km RD]",
    norm=None,
    vmin=None,
    vmax=None,
    levels=None,
    cmap="viridis",
    colorbar=True,
    colorbar_label="",
    plot_grid=True,
    add_to_plot=None,
    backgroundmap=False,
    figsize=None,
    save=False,
    fname=None,
):
    # get data
    if isinstance(da, str):
        da = ds[da]

    # select layer
    try:
        nlay = da["layer"].shape[0]
    except IndexError:
        nlay = 1
    if nlay > 1:
        layer = da["layer"].isel(layer=ilay).item()
        da = da.isel(layer=ilay)
    else:
        ilay = 0
        layer = ds["layer"].item()

    # select time
    if "time" in da:
        try:
            nper = da["time"].shape[0]
        except IndexError:
            nper = 1
        if nper > 1:
            t = da["time"].isel(time=iper).item()
            da = da.isel(time=iper)
        else:
            iper = 0
            t = ds["time"].item()
    else:
        t = None

    # figure
    if ax is not None:
        f = ax.figure
    else:
        if figsize is None:
            figsize = get_figsize(ds.extent)
        f, ax = plt.subplots(1, 1, figsize=figsize)
        rd_ticks(ax, base=1e4, fmt="{:.0f}")
        plt.yticks(rotation=90, va="center")
        ax.set_aspect("equal", adjustable="box")

        # get normalization if vmin/vmax are passed
    if vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    qm = data_array(da, cmap=cmap, norm=norm)

    # bgmap
    if backgroundmap:
        add_background_map(ax, map_provider="nlmaps.water", alpha=0.5)

    # add other info to plot
    if add_to_plot is not None:
        for fplot in add_to_plot:
            fplot(ax)

    if plot_grid:
        modelgrid(ds, ax=ax, lw=0.25, alpha=0.5)

    # axes properties
    title += f" (layer={layer})"
    if t is not None:
        title += f" (t={t})"
    axprops = {"xlabel": xlabel, "ylabel": ylabel, "title": title}
    ax.set(**axprops)

    f.tight_layout()

    # colorbar
    divider = make_axes_locatable(ax)
    if colorbar:
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = f.colorbar(qm, cax=cax)
        cbar.set_ticks(levels)
        cbar.set_label(colorbar_label)

    if save:
        f.savefig(fname, bbox_inches="tight", dpi=150)

    return ax


def animate_map(
    da,
    ds=None,
    ilay=0,
    xlabel="X",
    ylabel="Y",
    title="",
    datefmt="%Y-%m",
    cmap="viridis",
    vmin=None,
    vmax=None,
    norm=None,
    levels=None,
    colorbar=True,
    colorbar_label="",
    plot_grid=True,
    backgroundmap=False,
    figsize=None,
    ax=None,
    add_to_plot=None,
    save=True,
    fname=None,
):
    """Animates a map visualization using a DataArray.

    Parameters
    ----------
    da : DataArray, str
        The DataArray containing the data to be animated. If passed as a string,
        and the model dataset `ds` is also provided, the DataArray will be
        obtained from the model dataset: `da = ds[da]`.
    ds : Dataset, optional
        The model Dataset containing grid information, etc.
    ilay : int, optional
        The index of the layer to be visualized.
    xlabel : str, optional
        The label for the x-axis. Default is "X".
    ylabel : str, optional
        The label for the y-axis. Default is "Y".
    title : str, optional
        The title of the plot. Default is an empty string.
    datefmt : str, optional
        The date format string for the title. Default is "%Y-%m".
    cmap : str, optional
        The colormap to be used for the visualization. Default is "viridis".
    vmin : float, optional
        The minimum value for the colormap normalization. Default is None.
    vmax : float, optional
        The maximum value for the colormap normalization. Default is None.
    norm : Normalize, optional
        The normalization object for the colormap. Default is None.
    levels : array-like, optional
        levels for colorbar
    colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    colorbar_label : str, optional
        The label for the colorbar. Default is an empty string.
    plot_grid : bool, optional
        Whether to plot the model grid. Default is True.
    backgroundmap : bool, optional
        Whether to add a background map. Default is False.
    figsize : tuple, optional
        figure size in inches, default is None.
    ax : Axes, optional
        The matplotlib Axes object to be used for the plot.
        If None, a new figure and Axes will be created.
    add_to_plot : list, optional
        A list of functions that accept `ax` as an argument that add
        additional elements to the plot. Default is None.
    save : bool, optional
        Whether to save the animation as an mp4 file. Default is True.
    fname : str, optional
        The filename to save the animation. Required if save is True.

    Raises
    ------
    ValueError :
        If the DataArray does not have a time dimension.
    ValueError :
        If plotting modelgrid is requested but no model Dataset is provided.

    Returns
    -------
    f : Figure
        matplotlib figure handle
    anim : FuncAnimation
        The matplotlib FuncAnimation object representing the animation.
    """
    # if da is a string and ds is provided select data array from model dataset
    if isinstance(da, str) and ds is not None:
        da = ds[da]

    # check da
    if "time" not in ds.dims:
        raise ValueError("DataArray needs to have time dimension!")

    # select layer
    try:
        nlay = da["layer"].shape[0]
    except IndexError:
        nlay = 1
    if nlay > 1:
        layer = da["layer"].isel(layer=ilay).item()
        da = da.isel(layer=ilay)
    else:
        ilay = 0
        layer = ds["layer"].item()

    # figure
    if ax is not None:
        f = ax.figure
    else:
        if ds is None:
            extent = [
                da.x.values.min(),
                da.x.values.max(),
                da.y.values.min(),
                da.y.values.max(),
            ]
        else:
            extent = ds.extent
        base = 10 ** int(np.log10(extent[1] - extent[0]))
        if figsize is None:
            figsize = get_figsize(ds.extent)
        f, ax = get_map(extent, base=base, figsize=figsize, tight_layout=False)
        ax.set_aspect("equal", adjustable="box")

    # get normalization if vmin/vmax are passed
    if vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # plot initial data
    pc = data_array(da.isel(time=0), ds=ds, norm=norm, cmap=cmap)

    # plot modelgrid
    if plot_grid:
        if ds is None:
            raise ValueError("Plotting modelgrid requires model Dataset!")
        modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")

    if add_to_plot is not None:
        for fplot in add_to_plot:
            fplot(ax)

    # axes properties
    axprops = {"xlabel": xlabel, "ylabel": ylabel, "title": title}
    ax.set(**axprops)

    # add updating title
    t = pd.Timestamp(da.time.values[0])
    title = title_inside(
        f"Layer {layer}, t = {t.strftime(datefmt)}",
        ax,
        x=0.025,
        bbox={"facecolor": "w"},
        horizontalalignment="left",
    )
    # tight layout
    f.tight_layout()

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = f.colorbar(pc, cax=cax)
        if levels is not None:
            cbar.set_ticks(levels)
        cbar.set_label(colorbar_label)

    # bgmap
    if backgroundmap:
        add_background_map(ax, map_provider="nlmaps.water", alpha=0.5)

    # write update func
    def update(iper, pc, title):
        # select timestep
        da_i = da.isel(time=iper)

        # update pcolormesh
        pc.set_array(da_i.values.ravel())

        # update title
        t = pd.Timestamp(da.time.values[iper])
        title.set_text(f"Layer {ilay}, t = {t.strftime(datefmt)}")

        return pc, title

    # create animation
    anim = FuncAnimation(
        f,
        partial(update, pc=pc, title=title),
        frames=da["time"].shape[0],
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