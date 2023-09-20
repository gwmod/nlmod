import warnings
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

from ..dims.grid import modelgrid_from_ds
from ..dims.resample import get_affine_mod_to_world, get_extent
from ..read import geotop, rws
from .dcs import DatasetCrossSection
from .plotutil import (
    add_background_map,
    get_figsize,
    get_map,
    get_patches,
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
    plot_var,
    plot_time=None,
    plot_bc=None,
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
    plot_var : str
        variable in ds
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

    warnings.warn(
        "this function is out of date and will probably be removed in a future version",
        DeprecationWarning,
    )

    if plot_bc is not None:
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
        iax = axes.flat[ilay]
        mp = fp.plot.PlotMapView(model=gwf, layer=ilay, ax=iax)
        # mp.plot_grid()
        qm = mp.plot_array(plot_arr[ilay].values, cmap="viridis", vmin=vmin, vmax=vmax)
        # qm = mp.plot_array(hf[-1], cmap="viridis", vmin=-0.1, vmax=0.1)
        # mp.plot_ibound()
        # plt.colorbar(qm)
        if plot_bc is not None:
            for bc_var in plot_bc:
                mp.plot_bc(bc_var, color=color, kper=0)

        iax.set_aspect("equal", adjustable="box")
        iax.set_title(f"Layer {ilay}")

        iax.grid(grid)
        if xlim is not None:
            iax.set_xlim(xlim)
        if ylim is not None:
            iax.set_ylim(ylim)

    for iax in axes.flat[nlay:]:
        iax.set_visible(False)

    cb = fig.colorbar(qm, ax=axes, shrink=1.0)
    cb.set_label(f"{plot_var}", rotation=270)
    fig.suptitle(f"{plot_var} Time = {(ds.nper*ds.perlen)/365} year")
    fig.tight_layout()

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
    if "layer" in da.dims:
        msg = (
            "The suppplied DataArray in nlmod.plot.data_darray contains multiple "
            "layers. Please select a layer first."
        )
        raise (Exception(msg))
    if "icell2d" in da.dims:
        if ds is None:
            raise (ValueError("Supply model dataset (ds) for grid information"))
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
    line : shapely.LineString
        The line along which the GeoTOP data is plotted
    gt : xr.Dataset, optional
        The voxel-dataset from GeoTOP. It is downloaded with the method
        `nlmod.read.geotop.get_geotop()` if None. The default is None.
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
        gt = geotop.get_geotop(extent)

    if "top" not in gt or "botm" not in gt:
        gt = geotop.add_top_and_botm(gt)

    if lithok_props is None:
        lithok_props = geotop.get_lithok_props()

    cs = DatasetCrossSection(gt, line, layer="z", ax=ax, **kwargs)
    lithoks = gt["lithok"].values
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


def _get_figure(ax=None, da=None, ds=None, figsize=None, rotated=True):
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
            extent = get_extent(ds, rotated=rotated)

        if figsize is None:
            figsize = get_figsize(extent)
            # try to ensure pixel size is divisible by 2
            figsize = (figsize[0], np.round(figsize[1] / 0.02, 0) * 0.02)

        base = 10 ** int(np.log10(extent[1] - extent[0])) / 2
        if base < 1000:
            fmt = "{:.1f}"
        else:
            fmt = "{:.0f}"
        f, ax = get_map(extent, base=base, figsize=figsize, tight_layout=False, fmt=fmt)
        ax.set_aspect("equal", adjustable="box")
    return f, ax


def map_array(
    da,
    ds,
    ilay=0,
    iper=0,
    extent=None,
    ax=None,
    title="",
    xlabel="X [km RD]",
    ylabel="Y [km RD]",
    date_fmt="%Y-%m-%d",
    norm=None,
    vmin=None,
    vmax=None,
    levels=None,
    cmap="viridis",
    alpha=1.0,
    colorbar=True,
    colorbar_label="",
    plot_grid=True,
    rotated=True,
    add_to_plot=None,
    background=False,
    figsize=None,
    animate=False,
):
    # get data
    if isinstance(da, str):
        da = ds[da]

    # select layer
    try:
        nlay = da["layer"].shape[0]
    except IndexError:
        nlay = 0  # only one layer
    except KeyError:
        nlay = -1  # no dim layer
    if nlay >= 1:
        layer = da["layer"].isel(layer=ilay).item()
        da = da.isel(layer=ilay)
    elif nlay < 0:
        ilay = None
    else:
        ilay = 0
        layer = da["layer"].item()

    # select time
    if "time" in da.dims:
        try:
            nper = da["time"].shape[0]
        except IndexError:
            nper = 0  # only one timestep
        except KeyError:
            nper = -1  # no dim time
        if nper >= 1:
            t = pd.Timestamp(da["time"].isel(time=iper).item())
            da = da.isel(time=iper)
        elif nper < 0:
            iper = None
        else:
            iper = 0
            t = pd.Timestamp(ds["time"].item())
    else:
        t = None

    f, ax = _get_figure(ax=ax, da=da, ds=ds, figsize=figsize, rotated=rotated)

    # get normalization if vmin/vmax are passed
    if vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # plot data
    pc = data_array(
        da, ds=ds, cmap=cmap, alpha=alpha, norm=norm, ax=ax, rotated=rotated
    )

    # set extent
    if extent is not None:
        ax.axis(extent)

    # bgmap
    if background:
        add_background_map(ax, map_provider="nlmaps.water", alpha=0.5)

    # add other info to plot
    if add_to_plot is not None:
        for fplot in add_to_plot:
            fplot(ax=ax)

    # plot modelgrid
    if plot_grid:
        if ds is None:
            raise ValueError("Plotting modelgrid requires model Dataset!")
        modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")

    # axes properties
    if ilay is not None:
        title += f" (layer={layer})"
    if t is not None:
        title += f" (t={t.strftime(date_fmt)})"
    axprops = {"xlabel": xlabel, "ylabel": ylabel, "title": title}
    ax.set(**axprops)

    f.tight_layout()

    # colorbar
    divider = make_axes_locatable(ax)
    if colorbar:
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = f.colorbar(pc, cax=cax)
        if levels is not None:
            cbar.set_ticks(levels)
        cbar.set_label(colorbar_label)

    if animate:
        return f, ax, pc
    else:
        return ax


def animate_map(
    da,
    ds=None,
    ilay=0,
    xlabel="X",
    ylabel="Y",
    title="",
    date_fmt="%Y-%m-%d",
    cmap="viridis",
    alpha=1.0,
    vmin=None,
    vmax=None,
    norm=None,
    levels=None,
    colorbar=True,
    colorbar_label="",
    plot_grid=True,
    rotated=True,
    background=False,
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
    date_fmt : str, optional
        The date format string for the title. Default is "%Y-%m-%d".
    cmap : str, optional
        The colormap to be used for the visualization. Default is "viridis".
    alpha : float, optional
        transparency, default is 1.0 (not transparent)
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
    rotated : bool, optional
        Whether to plot rotated model, if applicable. Default is True.
    background : bool, optional
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
        da = ds[da].isel(layer=ilay)
    else:
        if "layer" in da.dims:
            da = da.isel(layer=ilay)

    # check da
    if "time" not in da.dims:
        raise ValueError("DataArray needs to have time dimension!")

    # plot base image
    f, ax, pc = map_array(
        da=da,
        ds=ds,
        ilay=ilay,
        iper=0,
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        date_fmt=date_fmt,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        alpha=alpha,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        plot_grid=plot_grid,
        rotated=rotated,
        add_to_plot=add_to_plot,
        background=background,
        figsize=figsize,
        animate=True,
    )
    # remove timestamp from title
    axtitle = ax.get_title()
    ax.set_title(axtitle.replace("(t=", "(tstart="))

    # add updating title
    t = pd.Timestamp(da.time.values[0])
    title = title_inside(
        f"t = {t.strftime(date_fmt)}",
        ax,
        x=0.025,
        bbox={"facecolor": "w"},
        horizontalalignment="left",
    )

    # write update func
    def update(iper, pc, title):
        # select timestep
        da_i = da.isel(time=iper)

        # update pcolormesh
        pc.set_array(da_i.values.ravel())

        # update title
        t = pd.Timestamp(da.time.values[iper])
        title.set_text(f"Layer {ilay}, t = {t.strftime(date_fmt)}")

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
        if fname is None:
            raise ValueError("please specify a fname or use save=False")
        writer = FFMpegWriter(
            fps=10,
            bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p"],
            codec="libx264",
        )
        anim.save(fname, writer=writer)

    return f, anim
