import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PatchCollection
from shapely.geometry import Point, LineString, Polygon
from shapely.algorithms.polylabel import polylabel


class DatasetCrossSection:
    # assumes:
    # x and y are 1d-vectors
    # x is increasing, y is decreasing
    # the layers are ordered from the top down

    def __init__(self, ds, line, ax=None, zmin=None, zmax=None,
                 set_extent=True, top='t', bot='b', x='x', y='y',
                 layer='layer'):
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
        # first determine where the cross-section crosses grid-lines
        self.xedge, self.yedge = self.get_grid_edges()
        self.xys = self.line_intersect_grid(line)
        # get the row and column of the centers
        sm = self.xys[:-1, -1] + np.diff(self.xys[:, -1]) / 2
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
        x = self.ds[self.x].values
        x = np.hstack((x[:-1] - np.diff(x) / 2, x[-2:] + np.diff(x[-3:]) / 2))
        y = self.ds[self.y].values
        y = np.hstack((y[:-1] - np.diff(y) / 2, y[-2:] + np.diff(y[-3:]) / 2))
        return x, y

    def coordinates_in_dataset(self, xy):
        return (xy[0] > self.xedge.min() and xy[1] > self.yedge.min() and
                xy[0] < self.xedge.max() and xy[1] < self.yedge.max())

    @staticmethod
    def add_intersections(gr_line, cs_line, points):
        intersection = cs_line.intersection(gr_line)
        if intersection.type == 'Point':
            points.append(intersection)
        elif intersection.type == 'MultiPoint':
            for point in intersection:
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
            raise(Exception('The line does not instersect with the dataset'))
        # sort the points along the line
        xys = xys[xys[:, -1].argsort()]
        return xys

    def plot_layers(self, colors=None, zmin=None, zmax=None, min_label_area=np.inf,
                    **kwargs):
        if colors is None:
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i) for i in range(len(self.layer))]

        polygons = []
        for i in range(len(self.layer)):
            if np.all(np.isnan(self.bot[i]) | (self.bot[i] == self.zmax)):
                continue
            if np.all(np.isnan(self.top[i]) | (self.top[i] == self.zmin)):
                continue
            z_not_nan = np.where(~np.isnan(self.top[i]) &
                                 ~np.isnan(self.bot[i]))[0]
            vans = [z_not_nan[0]]
            tots = []
            for x in np.where(np.diff(z_not_nan) > 1)[0]:
                tots.append(z_not_nan[x] + 1)
                vans.append(z_not_nan[x + 1])
            tots.append(z_not_nan[-1] + 1)
            for van, tot in zip(vans, tots):
                s = self.xys[van:tot + 1, -1]
                t = self.top[i, van:tot]
                b = self.bot[i, van:tot]
                n = tot - van

                x = s[sorted([0] + list(range(1, n)) * 2 + [n])]
                x = np.concatenate((x, x[::-1]))
                y = np.concatenate((t[sorted(list(range(n)) * 2)],
                                    b[sorted(list(range(n)) * 2, reverse=True)]))
                xy = list(zip(x, y))
                # xy = np.vstack((x, y)).T
                color = colors[i]
                pol = matplotlib.patches.Polygon(xy, facecolor=color, **kwargs)
                self.ax.add_patch(pol)
                polygons.append(pol)

                if not np.isinf(min_label_area):
                    pols = Polygon(xy)
                    if not pols.is_valid:
                        pols = pols.buffer(0)
                    if isinstance(pols, Polygon):
                        pols = [pols]
                    for pol in pols:
                        if pol.area > min_label_area:
                            p = pol.centroid
                            if not pol.contains(p):
                                p = polylabel(pol, 100.)
                            self.ax.text(p.x, p.y, self.layer[i],
                                         ha='center', va='center')
        return polygons

    def plot_grid(self, edgecolor='k', facecolor='none', horizontal=True,
                  vertical=True, **kwargs):
        lines = []
        if horizontal and not vertical:
            for i in range(self.top.shape[0]):
                for j in range(self.bot.shape[1]):
                    if not np.isnan(self.top[i, j]):
                        lines.append([(self.xys[j, -1], self.top[i, j]),
                                      (self.xys[j + 1, -1], self.top[i, j])])
                        # add vertical connection when necessary
                        if (j < self.top.shape[1] - 1 and
                            not np.isnan(self.top[i, j + 1]) and
                                self.top[i, j] != self.top[i, j + 1]):
                            lines.append([(self.xys[j + 1, -1], self.top[i, j]),
                                          (self.xys[j + 1, -1], self.top[i, j + 1])])
                    if not np.isnan(self.bot[i, j]):
                        lines.append([(self.xys[j, -1], self.bot[i, j]),
                                      (self.xys[j + 1, -1], self.bot[i, j])])
                        # add vertical connection when necessary
                        if (j < self.top.shape[1] - 1 and
                            not np.isnan(self.bot[i, j + 1]) and
                                self.bot[i, j] != self.bot[i, j + 1]):
                            lines.append([(self.xys[j + 1, -1], self.bot[i, j]),
                                          (self.xys[j + 1, -1], self.bot[i, j + 1])])
            line_collection = LineCollection(lines, edgecolor=edgecolor,
                                             **kwargs)
            self.ax.add_collection(line_collection)
            return line_collection
        if vertical and not horizontal:
            raise(Exception('Not implemented yet. Why would you want this!?'))
        patches = []
        for i in range(self.top.shape[0]):
            for j in range(self.bot.shape[1]):
                if not (np.isnan(self.top[i, j]) or
                        np.isnan(self.bot[i, j])):
                    if (self.bot[i, j] == self.zmax or
                            self.top[i, j] == self.zmin):
                        continue
                    width = self.xys[j + 1, -1] - self.xys[j, -1]
                    height = self.top[i, j] - self.bot[i, j]
                    rect = Rectangle((self.xys[j, -1], self.bot[i, j]), width,
                                     height)
                    patches.append(rect)
        patch_collection = PatchCollection(patches, edgecolor=edgecolor,
                                           facecolor=facecolor, **kwargs)
        self.ax.add_collection(patch_collection)
        return patch_collection

    def plot_array(self, z, **kwargs):
        if isinstance(z, xr.DataArray):
            z = z.data
        assert len(z.shape) == 3
        assert z.shape[0] == len(self.layer)
        assert z.shape[1] == len(self.ds[self.y])
        assert z.shape[2] == len(self.ds[self.x])

        zcs = z[:, self.rows, self.cols]
        patches = []
        array = []
        for i in range(zcs.shape[0]):
            for j in range(zcs.shape[1]):
                if not (np.isnan(self.top[i, j]) or
                        np.isnan(self.bot[i, j]) or
                        np.isnan(zcs[i, j])):
                    if (self.bot[i, j] == self.zmax or
                            self.top[i, j] == self.zmin):
                        continue
                    width = self.xys[j + 1, -1] - self.xys[j, -1]
                    height = self.top[i, j] - self.bot[i, j]
                    rect = Rectangle((self.xys[j, -1], self.bot[i, j]), width,
                                     height)
                    patches.append(rect)
                    array.append(zcs[i, j])
        patch_collection = PatchCollection(patches, **kwargs)
        patch_collection.set_array(np.array(array))
        self.ax.add_collection(patch_collection)
        return patch_collection

    def plot_surface(self, z, **kwargs):
        if isinstance(z, xr.DataArray):
            z = z.data
        # check if z has the same dimensions as ds
        assert len(z.shape) == 2
        assert z.shape[0] == len(self.ds[self.y])
        assert z.shape[1] == len(self.ds[self.x])

        zcs = z[self.rows, self.cols]
        n = len(zcs)
        x = self.xys[:, -1][sorted([0] + list(range(1, n)) * 2 + [n])]
        y = zcs[sorted(list(range(n)) * 2)]
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
        if len(top.shape) == 2:
            # the top is defines as the top of the model (like modflow)
            top = np.vstack([top[np.newaxis], bot[:-1]])
        top = top[:, self.rows, self.cols]
        bot = bot[:, self.rows, self.cols]
        if self.zmin:
            top[top < self.zmin] = self.zmin
            bot[bot < self.zmin] = self.zmin
        if self.zmax:
            top[top > self.zmax] = self.zmax
            bot[bot > self.zmax] = self.zmax
        return top, bot