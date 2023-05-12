import os
import warnings
from functools import partial

import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .dcs import DatasetCrossSection
from .dims.grid import get_vertices, modelgrid_from_ds
from .dims.resample import get_affine_mod_to_world, get_extent
from .read import geotop, rws


def surface_water(model_ds, ax=None, **kwargs):
    surf_water = rws.get_gdf_surface_water(model_ds)

    if ax is None:
        _, ax = plt.subplots()
    surf_water.plot(ax=ax, **kwargs)

    return ax


def modelgrid(ds, ax=None, add_surface_water=False, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    modelgrid = modelgrid_from_ds(ds)
    modelgrid.plot(ax=ax, **kwargs)
    ax.axis("scaled")
    if add_surface_water:
        surface_water(ds, ax=ax)
        ax.set_title("modelgrid with surface water")
    else:
        ax.set_title("modelgrid")
    ax.set_ylabel("y [m RD]")
    ax.set_xlabel("x [m RD]")

    return ax


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
    figdir=None,
    figsize=(10, 8),
    plot_bc=None,
    plot_grid=False,
):
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
        elif plot_dim == "time":
            ilay = layer
            iper = i
            if arr.ndim == 4:
                if ilay is None:
                    raise ValueError("Pass 'layer' to select layer to plot.")
                a = arr[iper]
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

    if figdir:
        fig.savefig(
            os.path.join(figdir, f"{lbl}_per_{plot_dim}.png"),
            dpi=150,
            bbox_inches="tight",
        )

    return fig, axes


def facet_plot_ds(
    gwf,
    model_ds,
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
                f"cannot plot boundary condition {key} "
                "because it is not in the package list"
            )

    nlay = len(model_ds.layer)

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
        plot_arr = model_ds[plot_var]
    else:
        plot_arr = model_ds[plot_var][plot_time]

    vmin = plot_arr.min()
    vmax = plot_arr.max()
    for ilay in range(nlay):
        iax = axes.ravel()[ilay]
        mp = flopy.plot.PlotMapView(model=gwf, layer=ilay, ax=iax)
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
    fig.suptitle(f"{plot_var} Time = {(model_ds.nper*model_ds.perlen)/365} year")
    fig.tight_layout()
    fig.savefig(
        os.path.join(figdir, f"{plot_var}_per_layer.png"),
        dpi=150,
        bbox_inches="tight",
    )

    return fig, axes


def plot_array(gwf, array, figsize=(8, 8), colorbar=True, ax=None, **kwargs):
    warnings.warn(
        "The 'plot.plot_array' function is deprecated please use"
        "'plot.data_array' instead",
        DeprecationWarning,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    yticklabels = ax.yaxis.get_ticklabels()
    plt.setp(yticklabels, rotation=90, verticalalignment="center")
    ax.axis("scaled")
    pmv = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid, ax=ax)
    pcm = pmv.plot_array(array, **kwargs)
    if colorbar:
        fig = ax.get_figure()
        fig.colorbar(pcm, ax=ax, orientation="vertical")
        # plt.colorbar(pcm)
    if hasattr(array, "name"):
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
    DeprecationWarning("plot.plot_vertex_array is deprecated. Use plot.data_array.")

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
        vmin = da.min()

    if "vmax" in kwargs:
        vmax = kwargs.pop("vmax")
    else:
        vmax = da.max()

    # limit the color range
    pc.set_clim(vmin=vmin, vmax=vmax)
    pc.set(**kwargs)

    ax.add_collection(pc)
    ax.set_xlim(vertices[:, :, 0].min(), vertices[:, :, 0].max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(vertices[:, :, 1].min(), vertices[:, :, 1].max())
    ax.set_aspect("equal")

    ax.get_figure().colorbar(pc, ax=ax, orientation="vertical")
    if hasattr(da, "name"):
        ax.set_title(da.name)

    return ax


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


def get_patches(ds, rotated=False):
    """Get the matplotlib patches for a vertex grid."""
    assert "icell2d" in ds.dims
    xy = np.column_stack((ds["xv"].data, ds["yv"].data))
    if rotated and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        affine = get_affine_mod_to_world(ds)
        xy[:, 0], xy[:, 1] = affine * (xy[:, 0], xy[:, 1])
    icvert = ds["icvert"].data
    if "_FillValue" in ds["icvert"].attrs:
        nodata = ds["icvert"].attrs["_FillValue"]
    else:
        nodata = -1
        icvert = icvert.copy()
        icvert[np.isnan(icvert)] = nodata
        icvert = icvert.astype(int)
    patches = []
    for icell2d in ds.icell2d.data:
        patches.append(Polygon(xy[icvert[icell2d, icvert[icell2d] != nodata]]))
    return patches


def get_map(
    extent,
    figsize=10.0,
    nrows=1,
    ncols=1,
    base=1000.0,
    fmt="{:.0f}",
    sharex=False,
    sharey=True,
    crs=28992,
    background=False,
    alpha=0.5,
    tight_layout=True,
):
    """Generate a motplotlib Figure with a map with the axis set to extent.

    Parameters
    ----------
    extent : list of 4 floats
        The model extent .
    figsize : float or list of 2 floats, optional
        The size of the figure, in inches. The default is 10, which means the
        figsize is determined automatically.
    nrows : int, optional
        The number of rows. The default is 1.
    ncols : int, optional
        The number of columns. The default is 1.
    base : float, optional
        The interval for ticklabels on the x- and y-axis. The default is 1000.
        m.
    fmt : string, optional
        The format of the ticks on the x- and y-axis. The default is "{:.0f}".
    sharex : bool, optional
        Only display the ticks on the lowest x-axes, when nrows > 1. The
        default is False.
    sharey : bool, optional
        Only display the ticks on the left y-axes, when ncols > 1. The default
        is True.
    background : bool or str, optional
        Draw a background using contextily when True or when background is a string.
        When background is a string it repesents the map-provider. Use
        nlmod.plot._list_contextily_providers().keys() to show possible map-providers.
        THe defaults is False.
    alpha: float, optional
        The alpha value of the background. The default is 0.5.
    tight_layout : bool, optional
        set tight_layout, default is True. Set to False for e.g. saving animations.

    Returns
    -------
    f : matplotlib.Figure
        The resulting figure.
    axes : matplotlib.Axes or numpy array of matplotlib.Axes
        the ax or axes (when ncols/nrows > 1).
    """
    if isinstance(figsize, (float, int)):
        xh = 0.0
        if base is None:
            xh = 0.0
        figsize = get_figsize(extent, nrows=nrows, ncols=ncols, figw=figsize, xh=xh)
    f, axes = plt.subplots(
        figsize=figsize, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey
    )
    if isinstance(background, bool) and background is True:
        background = "nlmaps.standaard"

    def set_ax_in_map(ax):
        ax.axis("scaled")
        ax.axis(extent)
        rotate_yticklabels(ax)
        if base is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            rd_ticks(ax, base=base, fmt=fmt)
        if background:
            add_background_map(ax, crs=crs, map_provider=background, alpha=alpha)

    if nrows == 1 and ncols == 1:
        set_ax_in_map(axes)
    else:
        for ax in axes.ravel():
            set_ax_in_map(ax)
    if tight_layout:
        f.tight_layout(pad=0.0)
    return f, axes


def _list_contextily_providers():
    """List contextily providers.

    Taken from contextily notebooks.

    Returns
    -------
    providers : dict
        dictionary containing all providers. See keys for names
        that can be passed as map_provider arguments.
    """
    import contextily as ctx

    providers = {}

    def get_providers(provider):
        if "url" in provider:
            providers[provider["name"]] = provider
        else:
            for prov in provider.values():
                get_providers(prov)

    get_providers(ctx.providers)
    return providers


def add_background_map(ax, crs=28992, map_provider="nlmaps.standaard", **kwargs):
    """Add background map to axes using contextily.

    Parameters
    ----------
    ax: matplotlib.Axes
        axes to add background map to
    map_provider: str, optional
        name of map provider, see `contextily.providers` for options.
        Default is 'nlmaps.standaard'
    proj: pyproj.Proj or str, optional
        projection for background map, default is 'epsg:28992'
        (RD Amersfoort, a projection for the Netherlands)
    """
    import contextily as ctx

    if isinstance(crs, (str, int)):
        import pyproj

        proj = pyproj.Proj(crs)

    providers = _list_contextily_providers()
    ctx.add_basemap(ax, source=providers[map_provider], crs=proj.srs, **kwargs)


def get_figsize(extent, figw=10.0, nrows=1, ncols=1, xh=0.2):
    """Get a figure size in inches, calculated from a model extent."""
    w = extent[1] - extent[0]
    h = extent[3] - extent[2]
    axh = (figw / ncols) * (h / w) + xh
    figh = nrows * axh
    figsize = (figw, figh)
    return figsize


def rotate_yticklabels(ax):
    """Rotate the labels on the y-axis 90 degrees to save space."""
    yticklabels = ax.yaxis.get_ticklabels()
    plt.setp(yticklabels, rotation=90, verticalalignment="center")


def rd_ticks(ax, base=1000.0, fmt_base=1000.0, fmt="{:.0f}"):
    """Add ticks every 1000 (base) m, and divide ticklabels by 1000
    (fmt_base)"""

    def fmt_rd_ticks(x, _):
        return fmt.format(x / fmt_base)

    if base is not None:
        ax.xaxis.set_major_locator(MultipleLocator(base))
        ax.yaxis.set_major_locator(MultipleLocator(base))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_rd_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_rd_ticks))


def colorbar_inside(
    mappable=None, ax=None, norm=None, cmap=None, bounds=None, bbox_labels=True, **kw
):
    """Place a colorbar inside an axes."""
    if ax is None:
        ax = plt.gca()
    if bounds is None:
        bounds = [0.95, 0.05, 0.02, 0.9]
    cax = ax.inset_axes(bounds, facecolor="none")
    if mappable is None and norm is not None and cmap is not None:
        # make an empty dataset...
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable._A = []
    cb = plt.colorbar(mappable, cax=cax, ax=ax, **kw)
    if bounds[0] > 0.5:
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position("left")
    if isinstance(bbox_labels, bool) and bbox_labels is True:
        bbox_labels = dict(facecolor="w", alpha=0.5)
    if isinstance(bbox_labels, dict):
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_bbox(bbox_labels)
        cb.ax.yaxis.get_label().set_bbox(bbox_labels)
    return cb


def title_inside(
    title,
    ax=None,
    x=0.5,
    y=0.98,
    horizontalalignment="center",
    verticalalignment="top",
    bbox=True,
    **kwargs,
):
    """Place a title inside a matplotlib axes, at the top."""
    if ax is None:
        ax = plt.gca()
    if isinstance(bbox, bool):
        if bbox:
            bbox = dict(facecolor="w", alpha=0.5)
        else:
            bbox = None
    return ax.text(
        x,
        y,
        title,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        transform=ax.transAxes,
        bbox=bbox,
        **kwargs,
    )


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
    figsize=(10, 10),
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
        figure size in inches, default is (10, 10).
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
        layer = da["layer"].isel(layer=ilay).values[()]
        da = da.isel(layer=ilay)
    else:
        ilay = 0
        layer = ds["layer"].values[()]

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
