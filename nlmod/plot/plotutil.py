from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely import LineString

from ..dims.grid import get_affine_mod_to_world
from ..epsg28992 import EPSG_28992


def get_patches(ds, rotated=False):
    """Get the matplotlib patches for a vertex grid."""
    assert "icell2d" in ds.dims
    xy = np.column_stack((ds["xv"].data, ds["yv"].data))
    if rotated and "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
        affine = get_affine_mod_to_world(ds)
        xy[:, 0], xy[:, 1] = affine * (xy[:, 0], xy[:, 1])
    icvert = ds["icvert"].data
    if "nodata" in ds["icvert"].attrs:
        nodata = ds["icvert"].attrs["nodata"]
    else:
        nodata = -1
        icvert = icvert.copy()
        icvert[np.isnan(icvert)] = nodata
        icvert = icvert.astype(int)
    patches = []
    for icell2d in ds.icell2d.data:
        patches.append(Polygon(xy[icvert[icell2d, icvert[icell2d] != nodata]]))
    return patches


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


def add_background_map(ax, crs=EPSG_28992, map_provider="nlmaps.standaard", **kwargs):
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

    if isinstance(crs, str):
        import pyproj

        proj = pyproj.CRS.from_string(crs)
    elif isinstance(crs, int):
        import pyproj

        proj = pyproj.CRS.from_epsg(crs)

    providers = _list_contextily_providers()
    ctx.add_basemap(ax, source=providers[map_provider], crs=proj.srs, **kwargs)


def get_map(
    extent,
    figsize=10.0,
    nrows=1,
    ncols=1,
    base=1000.0,
    fmt_base=1000.0,
    fmt="{:.0f}",
    sharex=False,
    sharey=True,
    crs=EPSG_28992,
    background=False,
    alpha=0.5,
    tight_layout=False,
    layout="constrained",
    xh=0.0,
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
        The interval for ticklabels on the x- and y-axis. The default is 1000 m.
    fmt_base : float, optional
        divide ticklabels by this number, by default 1000, so units become km.
    fmt : string, optional
        The format of the ticks on the x- and y-axis. The default is "{:.0f}".
    sharex : bool, optional
        Only display the ticks on the lowest x-axes, when nrows > 1. The
        default is False.
    sharey : bool, optional
        Only display the ticks on the left y-axes, when ncols > 1. The default
        is True.
    background : bool or str, optional
        Draw a background map using contextily when True or when background is a string.
        When background is a string it repesents the map-provider. Use
        nlmod.plot._list_contextily_providers().keys() to show possible map-providers.
        The defaults is False.
    alpha: float, optional
        The alpha value of the background. The default is 0.5.
    tight_layout : bool, optional
        Set tight_layout of the figure. The default value used to be True, but it has
        been replaced by layout="constrained". Set to False for e.g. saving animations.
        The default is False.
    layout : str, optional
        Used to set the layout of the figure to constrained. For more information see
        https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html .
        The default is "constrained".
    xh : float
        The extra height of the x-axis in inches, compared to the y-axis. The default is
        0.0.

    Returns
    -------
    f : matplotlib.Figure
        The resulting figure.
    axes : matplotlib.Axes or numpy array of matplotlib.Axes
        the ax or axes (when ncols/nrows > 1).
    """
    if isinstance(figsize, (float, int)):
        figsize = get_figsize(extent, nrows=nrows, ncols=ncols, figw=figsize, xh=xh)
    f, axes = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        layout=layout,
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
            rd_ticks(ax, base=base, fmt=fmt, fmt_base=fmt_base)
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
    """Add ticks every 1000 (base) m, and divide ticklabels by 1000 (fmt_base)."""

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
        bbox_labels = {"facecolor": "w", "alpha": 0.5}
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
            bbox = {"facecolor": "w", "alpha": 0.5}
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


def _get_axes_aspect_ratio(ax):
    pos = ax.get_position()
    width = pos.width * ax.figure.get_figwidth()
    height = pos.height * ax.figure.get_figheight()
    return width / height


def get_inset_map_bounds(
    ax: plt.Axes,
    extent: Union[tuple[float], list[float]],
    height: Optional[float] = None,
    width: Optional[float] = None,
    margin: Optional[float] = 0.025,
    right: Optional[bool] = True,
    bottom: Optional[bool] = True,
):
    """Get the bounds of the inset_map from a width or height, and a margin.

    These bounds can be used for the parameter `axes_bounds` in the `inset_map` method.
    The horizontal and vertical margin (in pixels) around this map are equal (unless the
    figure is reshaped).

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the inset map to.
    extent : list of 4 floats
        The extent of the inset map.
    height : float, optional
        The height of the inset axes, in axes coordinates. Either height or width needs
        to be specified. The default is None.
    width : float, optional
        The width of the inset axes, in axes coordinates. Either height or width needs
        to be specified. The default is None.
    margin : float, optional
        The margin around, in axes coordinates. When height is specified, margin is
        relative to the height of ax. When width is specified, margin is relative to the
        width of ax. The default is 0.025.
    right : bool, optional
        If True, the inset axes is placed at the right corner. The default is True.
    bottom : bool, optional
        If True, the inset axes is placed at the bottom corner. The default is True.

    Returns
    -------
    bounds: list of 4 floats
        The bounds (left, right, width, height) of the inset axes.

    """
    msg = "Please specify either height or width"
    assert (height is None) + (width is None) == 1, msg
    ar = _get_axes_aspect_ratio(ax)
    dxdy = (extent[1] - extent[0]) / (extent[3] - extent[2])
    if height is None:
        # the bounds are determined by width
        height = width * ar / dxdy
        bounds = [margin, margin * ar, width, height]
    else:
        # the bounds are determined by height
        width = height * dxdy / ar
        bounds = [margin / ar, margin, width, height]
    if right:
        # put the axes on the right side
        bounds[0] = 1 - bounds[0] - width
    if not bottom:
        # put the axes on the top side
        bounds[1] = 1 - bounds[1] - height
    return bounds


def inset_map(
    ax: plt.Axes,
    extent: Union[tuple[float], list[float]],
    axes_bounds: Union[tuple[float], list[float]] = (0.63, 0.025, 0.35, 0.35),
    anchor: str = "SE",
    provider: Optional[str] = "nlmaps.water",
    add_to_plot: Optional[list[Callable]] = None,
):
    """Add an inset map to an axes.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the inset map to.
    extent : list of 4 floats
        The extent of the inset map.
    axes_bounds : list or tuple of 4 floats, optional
        The bounds (left, right, width, height) of the inset axes, default
        is (0.63, 0.025, 0.35, 0.35). This is rescaled according to the extent of
        the inset map.
    anchor : str, optional
        The anchor point of the inset map, default is 'SE'.
    provider : str, optional
        Add a backgroundmap if map provider is passed, default is 'nlmaps.water'. To
        turn off the backgroundmap set provider to None.
    add_to_plot : list of functions, optional
        List of functions to plot on the inset map, default is None. The functions
        must accept an ax argument. Hint: use `functools.partial` to set plot style,
        and pass the partial function to add_to_plot.

    Returns
    -------
    mapax : matplotlib.Axes
        The inset map axes.
    """
    mapax = ax.inset_axes(axes_bounds)
    mapax.axis(extent)
    mapax.set_aspect("equal", adjustable="box", anchor=anchor)
    mapax.set_xticks([])
    mapax.set_yticks([])
    mapax.set_xlabel("")
    mapax.set_ylabel("")

    if provider:
        add_background_map(mapax, map_provider=provider, attribution=False)

    if add_to_plot:
        for fplot in add_to_plot:
            fplot(ax=mapax)

    return mapax


def add_xsec_line_and_labels(
    line: Union[list, LineString],
    ax: plt.Axes,
    mapax: plt.Axes,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    label: str = "A",
    **kwargs,
):
    """Add a cross-section line to an overview map and label the start and end points.

    Parameters
    ----------
    line : list or shapely LineString
        The line to plot.
    ax : matplotlib.Axes
        The axes to plot the labels on.
    mapax : matplotlib.Axes
        The axes of the overview map to plot the line on.
    x_offset : float, optional
        The x offset of the labels, default is 0.0.
    y_offset : float, optional
        The y offset of the labels, default is 0.0.
    kwargs : dict
        Keyword arguments to pass to the line plot function.

    Raises
    ------
    ValueError
        If the line is not a list or a shapely LineString.
    """
    if isinstance(line, list):
        x, y = np.array(line).T
    elif isinstance(line, LineString):
        x, y = line.xy
    else:
        raise ValueError("line should be a list or a shapely LineString")
    mapax.plot(x, y, **kwargs)
    stroke = [patheffects.withStroke(linewidth=2, foreground="w")]
    mapax.text(
        x[0] - x_offset,
        y[0] - y_offset,
        f"{label}",
        fontweight="bold",
        path_effects=stroke,
        fontsize=7,
    )
    mapax.text(
        x[-1] + x_offset,
        y[-1] + y_offset,
        f"{label}'",
        fontweight="bold",
        path_effects=stroke,
        fontsize=7,
    )
    ax.text(
        0.01,
        0.99,
        f"{label}",
        transform=ax.transAxes,
        path_effects=stroke,
        fontsize=14,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax.text(
        0.99,
        0.99,
        f"{label}'",
        transform=ax.transAxes,
        path_effects=stroke,
        fontsize=14,
        ha="right",
        va="top",
        fontweight="bold",
    )
