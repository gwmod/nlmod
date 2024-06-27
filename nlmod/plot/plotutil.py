import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ..dims.resample import get_affine_mod_to_world
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

    if isinstance(crs, (str, int)):
        import pyproj

        proj = pyproj.Proj(crs)

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
    """Add ticks every 1000 (base) m, and divide ticklabels by 1000 (fmt_base)"""

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
