# %%
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.geometry import Polygon

import nlmod


# %%
def string_to_gdf(text, font, size=500, x_offset=0, **kwargs):
    tp = TextPath((x_offset, 0), text, size=size, prop=font)
    polygons = []

    # Matplotlib returns list of polygons (outer + inner contours)
    for poly in tp.to_polygons():
        if len(poly) >= 3:
            polygons.append(Polygon(poly))

    # if a polygon has holes, we need to reconstruct it
    final_polygons = []
    used = [False] * len(polygons)
    for i, outer in enumerate(polygons):
        if used[i]:
            continue
        holes = []
        for j, inner in enumerate(polygons):
            if i != j and not used[j]:
                if outer.contains(inner):
                    holes.append(inner.exterior.coords)
                    used[j] = True
        final_polygons.append(Polygon(outer.exterior.coords, holes))
        used[i] = True

    gdf = gpd.GeoDataFrame(geometry=final_polygons, **kwargs)
    return gdf


text = "NLMOD"
# font_path = "C:/Windows/Fonts/consola.ttf"  # Adjust for your system
font_path = "C:/Windows/Fonts/cour.ttf"  # Courier font
font = FontProperties(fname=font_path)
text_size = 100  # Larger size -> smoother curves

gdf = string_to_gdf(text, font, size=text_size)

if False:
    gdf.plot()
    plt.axis("equal")
    plt.show()

dx = 10
extent = gdf.total_bounds[[0, 2, 1, 3]]

# add a buffer of 10 % of width
buffer = 0.1 * (extent[1] - extent[0])
extent = np.array(
    [
        extent[0] - buffer,
        extent[1] + buffer,
        extent[2] - buffer,
        extent[3] + buffer,
    ]
)
if False:
    # make sure vertical extent is at least half of the horizontal extent
    height = extent[3] - extent[2]
    width = extent[1] - extent[0]
    center_y = 0.5 * (extent[2] + extent[3])
    if height < 0.5 * width:
        height = 0.5 * width
    extent = np.array(
        [
            extent[0],
            extent[1],
            center_y - 0.5 * height,
            center_y + 0.5 * height,
        ]
    )

extent = (extent / dx).round() * dx
ds = nlmod.get_ds(extent, dx)
ds = nlmod.grid.refine(ds, "logo", [(gdf, 2)])

nlmod.plot.modelgrid(ds)

# %% plot the logo
# add 1 percent margin
margin = 0.01 * (extent[1] - extent[0])
extent = (
    extent[0] - margin,
    extent[1] + margin,
    extent[2] - margin,
    extent[3] + margin,
)
figwidth = 5
figheight = figwidth * (extent[3] - extent[2]) / (extent[1] - extent[0])
f = plt.figure(figsize=(figwidth, figheight))
ax = f.add_axes([0, 0, 1, 1])
ax.axis("equal")
ax.axis(extent)
color = "k"
nlmod.plot.modelgrid(
    ds, color=color, ax=ax, linewidth=0.5, clip_on=False, antialiased=False
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
ax.axis("off")

# %% save logo
fname = "logo_text"
dpi = 300
if figwidth != 5:
    fname = f"{fname}_{figwidth}"
    dpi = None
f.savefig(os.path.join("_static", f"{fname}.png"), dpi=dpi)
f.savefig(os.path.join("_static", f"{fname}.svg"))

# %%
