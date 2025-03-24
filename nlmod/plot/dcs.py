import logging
from functools import partial

import flopy
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
from shapely.affinity import affine_transform
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from ..dims.grid import get_affine_world_to_mod, modelgrid_from_ds
from .plotutil import get_map

logger = logging.getLogger(__name__)


class DatasetCrossSection:
    # assumes:
    # x and y are 1d-vectors
    # x is increasing, y is decreasing
    # the layers are ordered from the top down

    def __init__(
        self,
        ds,
        line,
        ax=None,
        zmin=None,
        zmax=None,
        set_extent=True,
        top="top",
        bot="botm",
        x="x",
        y="y",
        layer="layer",
        icell2d="icell2d",
    ):
        if ax is None:
            ax = plt.gca()
        self.ax = ax

        self.ds = ds
        if isinstance(line, list):
            line = LineString(line)
        self.line = line
        self.x = x
        self.y = y
        if isinstance(layer, str):
            layer = ds[layer].data
        self.layer = layer
        self.icell2d = icell2d

        if "angrot" in ds.attrs and ds.attrs["angrot"] != 0.0:
            # transform the line to model coordinates
            line = affine_transform(line, get_affine_world_to_mod(ds).to_shapely())
            self.rotated = True
        else:
            self.rotated = False

        # first determine where the cross-section crosses grid-lines
        if self.icell2d in ds.dims:
            # determine the cells that are crossed
            modelgrid = modelgrid_from_ds(ds, rotated=False)
            gi = flopy.utils.GridIntersect(modelgrid, method="vertex")
            r = gi.intersect(line)
            s_cell = []
            for i, ic2d in enumerate(r["cellids"]):
                intersection = r["ixshapes"][i]
                if intersection.length == 0:
                    continue
                if isinstance(intersection, MultiLineString):
                    for ix in intersection.geoms:
                        s_cell.append([line.project(Point(ix.coords[0])), 1, ic2d])
                        s_cell.append([line.project(Point(ix.coords[-1])), 0, ic2d])
                    continue
                assert isinstance(intersection, LineString)
                s_cell.append([line.project(Point(intersection.coords[0])), 1, ic2d])
                s_cell.append([line.project(Point(intersection.coords[-1])), 0, ic2d])
            s_cell = np.array(s_cell)
            ind = np.lexsort((s_cell[:, 1], s_cell[:, 0]))
            s_cell = s_cell[ind, :]
            self.icell2ds = s_cell[::2, -1].astype(int)
            self.s = s_cell[:, 0].reshape((len(self.icell2ds), 2))
        else:
            self.xedge, self.yedge = self.get_grid_edges()
            xys = self.line_intersect_grid(line)
            self.s = np.column_stack((xys[:-1, -1], xys[1:, -1]))
            # get the row and column of the centers
            sm = self.s[:, 0] + (self.s[:, 1] - self.s[:, 0]) / 2
            self.cols = []
            self.rows = []
            for s in sm:
                x, y = line.interpolate(s).coords[0]
                if self.xedge[1] - self.xedge[0] > 0:
                    self.cols.append(np.where(x >= self.xedge[:-1])[0][-1])
                else:
                    self.cols.append(np.where(x <= self.xedge[:-1])[0][-1])
                if self.yedge[1] - self.yedge[0] > 0:
                    self.rows.append(np.where(y >= self.yedge[:-1])[0][-1])
                else:
                    self.rows.append(np.where(y <= self.yedge[:-1])[0][-1])
        self.zmin = zmin
        self.zmax = zmax
        self.top, self.bot = self.get_top_and_bot(top, bot)
        if self.zmin is None:
            self.zmin = np.nanmin(self.bot)
        if self.zmax is None:
            self.zmax = np.nanmax(self.top)
        if set_extent:
            extent = [0, self.line.length, self.zmin, self.zmax]
            self.ax.axis(extent)

    def get_grid_edges(self):
        """Get the x and y values of all gridcell edges

        Returns
        -------
        tuple
            x and y edges of the dataset.
        """
        x = self.ds[self.x].values
        x = np.hstack((x[:-1] - np.diff(x) / 2, x[-2:] + np.diff(x[-3:]) / 2))
        y = self.ds[self.y].values
        y = np.hstack((y[:-1] - np.diff(y) / 2, y[-2:] + np.diff(y[-3:]) / 2))
        return x, y

    def coordinates_in_dataset(self, xy):
        return (
            xy[0] > self.xedge.min()
            and xy[1] > self.yedge.min()
            and xy[0] < self.xedge.max()
            and xy[1] < self.yedge.max()
        )

    @staticmethod
    def add_intersections(gr_line, cs_line, points):
        intersection = cs_line.intersection(gr_line)
        if intersection.geom_type == "Point":
            points.append(intersection)
        elif intersection.geom_type == "MultiPoint":
            for point in intersection.geoms:
                points.append(point)

    def line_intersect_grid(self, cs_line):
        points = []
        # add the starting point
        if self.coordinates_in_dataset(cs_line.coords[0]):
            points.append(Point(cs_line.coords[0]))
        # add all intersections with yedge
        for y in self.yedge:
            gr_line = LineString(zip(self.xedge[[0, -1]], [y, y]))
            self.add_intersections(gr_line, cs_line, points)
        # add all intersections with xedge
        for x in self.xedge:
            gr_line = LineString(zip([x, x], self.yedge[[0, -1]]))
            self.add_intersections(gr_line, cs_line, points)
        # add the ending point
        if self.coordinates_in_dataset(cs_line.coords[-1]):
            points.append(Point(cs_line.coords[-1]))
        # generate ax x, y, d -array
        xys = []
        for point in points:
            xys.append([point.x, point.y, cs_line.project(point)])
        xys = np.array(xys)
        if xys.size == 0:
            raise ValueError("The line does not instersect with the dataset")
        # sort the points along the line
        xys = xys[xys[:, -1].argsort()]
        return xys

    def plot_layers(
        self,
        colors=None,
        min_label_area=np.inf,
        fontsize=None,
        only_labels=False,
        **kwargs,
    ):
        if colors is None:
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i) for i in range(len(self.layer))]
        if isinstance(colors, pd.DataFrame):
            colors = colors["color"]
        if isinstance(colors, (dict, pd.Series)):
            # split on _ in case layers were split and appended with a numbered suffix
            colors = [colors[layer.split("_")[0]] for layer in self.layer]

        if colors == "none":
            colors = ["none"] * len(self.layer)

        polygons = []
        for i, _ in enumerate(self.layer):
            if np.all(np.isnan(self.bot[i]) | (self.bot[i] == self.zmax)):
                continue
            if np.all(np.isnan(self.top[i]) | (self.top[i] == self.zmin)):
                continue
            z_not_nan = np.where(~np.isnan(self.top[i]) & ~np.isnan(self.bot[i]))[0]
            vans = [z_not_nan[0]]
            tots = []
            for x in np.where(np.diff(z_not_nan) > 1)[0]:
                tots.append(z_not_nan[x] + 1)
                vans.append(z_not_nan[x + 1])
            tots.append(z_not_nan[-1] + 1)
            for van, tot in zip(vans, tots):
                t = self.top[i, van:tot]
                b = self.bot[i, van:tot]
                n = tot - van

                x = self.s[van:tot].ravel()
                x = np.concatenate((x, x[::-1]))
                y = np.concatenate(
                    (
                        t[sorted(list(range(n)) * 2)],
                        b[sorted(list(range(n)) * 2, reverse=True)],
                    )
                )
                xy = list(zip(x, y))
                # xy = np.vstack((x, y)).T
                color = colors[i]
                pol = matplotlib.patches.Polygon(xy, facecolor=color, **kwargs)
                if not only_labels:
                    self.ax.add_patch(pol)
                    polygons.append(pol)

                if not np.isinf(min_label_area):
                    pols = Polygon(xy)
                    if not pols.is_valid:
                        pols = pols.buffer(0)
                    if isinstance(pols, Polygon):
                        pols = [pols]
                    else:
                        pols = pols.geoms
                    for pol in pols:
                        if pol.area > min_label_area:
                            xt = pol.centroid.x
                            xp = x[: int(len(x) / 2)]
                            yp1 = np.interp(xt, xp, y[: int(len(x) / 2)])
                            yp = list(reversed(y[int(len(x) / 2) :]))
                            yp2 = np.interp(xt, xp, yp)
                            yt = np.mean([yp1, yp2])
                            ht = self.ax.text(
                                xt,
                                yt,
                                self.layer[i],
                                ha="center",
                                va="center",
                                fontsize=fontsize,
                            )
                            if only_labels:
                                polygons.append(ht)
        return polygons

    def label_layers(self, min_label_area=None):
        if min_label_area is None:
            # plot labels of layers with an average thickness of 1 meter
            # in entire cross-section
            min_label_area = self.line.length * 1
        return self.plot_layers(min_label_area=min_label_area, only_labels=True)

    def plot_grid(
        self,
        edgecolor="k",
        facecolor="none",
        horizontal=True,
        vertical=True,
        ilayers=None,
        **kwargs,
    ):
        lines = []
        if ilayers is None:
            ilayers = range(self.top.shape[0])
        if horizontal and not vertical:
            for i in ilayers:
                for j in range(self.bot.shape[1]):
                    if not np.isnan(self.top[i, j]):
                        lines.append(
                            [
                                (self.s[j, 0], self.top[i, j]),
                                (self.s[j, 1], self.top[i, j]),
                            ]
                        )
                        # add vertical connection when necessary
                        if (
                            j < self.top.shape[1] - 1
                            and not np.isnan(self.top[i, j + 1])
                            and self.top[i, j] != self.top[i, j + 1]
                        ):
                            lines.append(
                                [
                                    (self.s[j + 1, 0], self.top[i, j]),
                                    (self.s[j + 1, 0], self.top[i, j + 1]),
                                ]
                            )
                    if not np.isnan(self.bot[i, j]):
                        lines.append(
                            [
                                (self.s[j, 0], self.bot[i, j]),
                                (self.s[j, 1], self.bot[i, j]),
                            ]
                        )
                        # add vertical connection when necessary
                        if (
                            j < self.top.shape[1] - 1
                            and not np.isnan(self.bot[i, j + 1])
                            and self.bot[i, j] != self.bot[i, j + 1]
                        ):
                            lines.append(
                                [
                                    (self.s[j + 1, 0], self.bot[i, j]),
                                    (self.s[j + 1, 0], self.bot[i, j + 1]),
                                ]
                            )
            line_collection = LineCollection(lines, edgecolor=edgecolor, **kwargs)
            self.ax.add_collection(line_collection)
            return line_collection
        if vertical and not horizontal:
            raise NotImplementedError("Why would you want this!?")
        patches = []
        for i in range(self.top.shape[0]):
            for j in range(self.bot.shape[1]):
                if not (np.isnan(self.top[i, j]) or np.isnan(self.bot[i, j])):
                    if self.bot[i, j] == self.zmax or self.top[i, j] == self.zmin:
                        continue
                    width = self.s[j, 1] - self.s[j, 0]
                    height = self.top[i, j] - self.bot[i, j]
                    rect = Rectangle((self.s[j, 0], self.bot[i, j]), width, height)
                    patches.append(rect)
        patch_collection = PatchCollection(
            patches, edgecolor=edgecolor, facecolor=facecolor, **kwargs
        )
        self.ax.add_collection(patch_collection)
        return patch_collection

    def plot_map_cs(
        self,
        ax=None,
        figsize=5,
        background=True,
        lw=5,
        ls="--",
        label="cross section",
        **kwargs,
    ):
        """Creates a different figure with the map of the cross section.

        Parameters
        ----------
        ax : None or matplotlib.Axes, optional
            if None a new axis object is created using nlmod.plot.get_map()
        figsize : int, optional
            size of the figure, only used if ax is None, by default 5
        background : bool, optional
            add a backgroun map, only used if ax is None, by default True
        lw : int, optional
            linewidth of the cross section, by default 10
        ls : str, optional
            linestyle of the cross section, by default "--"
        label : str, optional
            label of the cross section, by default "cross section"
        **kwargs are passed to the nlmod.plot.get_map() function. Only if ax is None

        Returns
        -------
        matplotlib Axes
            axes
        """
        if ax is None:
            _, ax = get_map(
                self.ds.extent, background=background, figsize=figsize, **kwargs
            )
        gpd.GeoDataFrame(geometry=[self.line]).plot(ax=ax, ls=ls, lw=lw, label=label)
        ax.legend()

        return ax

    def iterate_active_cells(self, zcs):
        """Iterate over the cell indices of the cells in an array that are visible in the cross section and active in the model.
        
        Parameters
        ----------
        zcs : np.ndarray
            2d array with dimensions (layer, distance along cross section).

        Yields
        ------
        tuple
            i, j indices of the cells that are active and visible in cross section. 
        
        """
        for i in range(zcs.shape[0]):
            for j in range(zcs.shape[1]):
                if not (
                    np.isnan(self.top[i, j])
                    or np.isnan(self.bot[i, j])
                    or np.isnan(zcs[i, j])
                    or (self.bot[i, j] >= self.zmax)
                    or (self.top[i, j] <= self.zmin)
                ):
                    yield i, j

    def _get_rect(self, i, j, hcs=None):
        """Get a rectangle patch for a cell in the cross section.

        Parameters
        ----------
        i : int
            layer index.
        j : int
            cell index along cross section.
        hcs : np.ndarray, optional
            head array with dimensions (layer, distance along cross section), by
            default None.

        Returns
        -------
        Rectangle
            rectangle patch.
        """
        width = self.s[j, 1] - self.s[j, 0]
        top = self.top[i, j]
        if hcs is not None:
            top = max(min(top, hcs[i, j]), self.bot[i, j])
        height = top - self.bot[i, j]
        xy = (self.s[j, 0], self.bot[i, j])
        rect = Rectangle(xy, width, height)
        return rect


    def array_on_cs(self, z):
        """Select cells in an array that are in the cross section.

        Parameters
        ----------
        z : np.ndarray or xr.DataArray
            array with dimensions (layer, y, x) or (layer, cellid).

        Returns
        -------
        np.ndarray
            array with dimensions (layer, distance along cross section).
        """

        if isinstance(z, xr.DataArray):
            z = z.data

        if self.icell2d in self.ds.dims:
            assert len(z.shape) == 2
            assert z.shape[0] == len(self.layer)
            assert z.shape[1] == len(self.ds[self.icell2d])

            return z[:, self.icell2ds]
        else:
            assert len(z.shape) == 3
            assert z.shape[0] == len(self.layer)
            assert z.shape[1] == len(self.ds[self.y])
            assert z.shape[2] == len(self.ds[self.x])

            return z[:, self.rows, self.cols]

    def plot_array(self, z, head=None, **kwargs):

        zcs = self.array_on_cs(z)

        if head is not None:
            assert head.shape == z.shape
            hcs = self.array_on_cs(head)
        else:
            hcs = None

        patches = [self._get_rect(i,j, hcs=hcs) for i, j in self.iterate_active_cells(zcs)]
        array = [zcs[i, j] for i, j in self.iterate_active_cells(zcs)]

        patch_collection = PatchCollection(patches, **kwargs)
        patch_collection.set_array(np.array(array))
        self.ax.add_collection(patch_collection)
        return patch_collection


    def plot_surface(self, z, **kwargs):
        if isinstance(z, xr.DataArray):
            z = z.data
        # check if z has the same dimensions as ds
        if self.icell2d in self.ds.dims:
            assert len(z.shape) == 1
            assert z.shape[0] == len(self.ds[self.icell2d])

            zcs = z[self.icell2ds]
        else:
            assert len(z.shape) == 2
            assert z.shape[0] == len(self.ds[self.y])
            assert z.shape[1] == len(self.ds[self.x])

            zcs = z[self.rows, self.cols]
        x = self.s.ravel()
        y = zcs[sorted(list(range(len(zcs))) * 2)]
        return self.ax.plot(x, y, **kwargs)

    def get_top_and_bot(self, top, bot):
        # then determine the top and botm of each cell
        if isinstance(top, str):
            top = self.ds[top].data
        if isinstance(bot, str):
            bot = self.ds[bot].data
        # # hack for single layer datasets
        # if len(bot.shape) == 2:
        #     bot = np.vstack([bot[np.newaxis], bot[np.newaxis]])
        if len(top.shape) == len(bot.shape) - 1:
            # the top is defines as the top of the model (like modflow)
            top = np.vstack([top[np.newaxis], bot[:-1]])
        if self.icell2d in self.ds.dims:
            top = top[:, self.icell2ds]
            bot = bot[:, self.icell2ds]
        else:
            top = top[:, self.rows, self.cols]
            bot = bot[:, self.rows, self.cols]
        if self.zmin is not None:
            top[top < self.zmin] = self.zmin
            bot[bot < self.zmin] = self.zmin
        if self.zmax is not None:
            top[top > self.zmax] = self.zmax
            bot[bot > self.zmax] = self.zmax
        return top, bot

    def animate(
        self,
        da,
        cmap="Spectral_r",
        norm=None,
        head=None,
        plot_title="",
        date_fmt="%Y-%m-%d",
        cbar_label=None,
        fname=None,
    ):
        """Animate a cross section.

        Parameters
        ----------
        da : DataArray
            should have dimensions structured: time, y, x or vertex: time, icell2d
        cmap : str, optional
            passed to plot_array function, by default "Spectral_r"
        norm : , optional
            norm for the colorbar of the datarray, by default None
        head : DataArray, optional
            If not given the top cell is completely filled, by default None
        plot_title : str or None, optional
            if not None a title is added which is updated with every timestep (using
            date_fmt for the date format), by default ""
        date_fmt : str, optional
            date format for plot title, by default "%Y-%m-%d"
        cbar_label : str, optional
            label for the colorbar, by default None
        fname : str or Path, optional
            filename if not None this is where the aniation is saved as mp4, by
            default None

        Returns
        -------
        matplotlib.animation.FuncAnimation
            animation object
        """
        f = self.ax.get_figure()

        if norm is None:
            vmin = np.nanmin(da)
            vmax = np.nanmax(da)
            norm = matplotlib.colors.Normalize(vmin, vmax)

        # plot first timeframe
        iper = 0
        if head is not None:
            plot_head = head
            self.pc = self.plot_array(da[iper].squeeze(), cmap=cmap, norm=norm, head=head.values[iper].squeeze())
        else:
            self.pc = self.plot_array(da[iper].squeeze(), cmap=cmap, norm=norm)
            plot_head = None

        cbar = f.colorbar(self.pc, ax=self.ax, shrink=1.0)

        if cbar_label is not None:
            cbar.set_label(cbar_label)
        elif "units" in da.attrs:
            cbar.set_label(da.units)

        if da.time.dtype.kind == "M":
            t = pd.Timestamp(da.time.values[iper]).strftime(date_fmt)
        elif da.time.dtype.kind == "O":
            t = da.time.values[iper].strftime(date_fmt)
        else:
            t = f"{da.time.values[iper]} {da.time.time_units}"

        if plot_title is None:
            self.title = None
        else:
            self.title = self.ax.set_title(f"{plot_title}, t = {t}")

        # update func
        def update(iper):
            zcs = self.array_on_cs(da[iper].squeeze())
            if plot_head is not None:
                # create new patches
                hcs = self.array_on_cs(plot_head[iper].squeeze())
                patches = [self._get_rect(i,j, hcs=hcs) for i, j in self.iterate_active_cells(zcs)]
                array = [zcs[i, j] for i, j in self.iterate_active_cells(zcs)]
                self.pc.remove() # remove previous patches
                self.pc = PatchCollection(patches, cmap=cmap, norm=norm)
                self.pc.set_array(np.array(array))
                self.ax.add_collection(self.pc)
            else:
                # only set new values of existing patches
                array = [zcs[i, j] for i, j in self.iterate_active_cells(zcs)]
                self.pc.set_array(np.array(array))


            # update title
            if da.time.dtype.kind == "M":
                t = pd.Timestamp(da.time.values[iper]).strftime(date_fmt)
            elif da.time.dtype.kind == "O":
                t = da.time.values[iper].strftime(date_fmt)
            else:
                t = f"{da.time.values[iper]} {da.time.time_units}"

            if self.title is not None:
                self.title.set_text(f"{plot_title}, t = {t}")

        # create animation
        anim = FuncAnimation(
            f,
            update,
            frames=da["time"].shape[0],
            blit=False,
            interval=100,
        )

        # save animation
        if fname is None:
            return anim
        else:
            # save animation as mp4
            writer = FFMpegWriter(
                fps=10,
                bitrate=-1,
                extra_args=["-pix_fmt", "yuv420p"],
                codec="libx264",
            )
            anim.save(fname, writer=writer)
            return anim
