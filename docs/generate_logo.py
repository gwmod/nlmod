import os
import matplotlib.pyplot as plt
import nlmod

# %%
filled = False
n = 2
dx = 10_000
# dx = 20000
figwidth = 5

# %%
nederland = nlmod.read.administrative.download_netherlands_gdf()
# add a small buffer to remove holes that are introduced by dissolve later on
nederland.geometry = nederland.buffer(0.1)
nederland = nederland.dissolve()
nederland["geometry"] = nederland.simplify(1000)

# %%
refine = nederland.copy()
if not filled:
    refine["geometry"] = refine.boundary

if dx == 10000:
    extent = [0, 290000, 295000, 625000]
elif dx == 20000:
    extent = [-5000, 295000, 295000, 635000]
else:
    raise (Exception(f"Unsupported dx: {dx}"))

# f, ax = nlmod.plot.get_map(extent, figsize=figsize, base=50000)
# nederland.plot(ax=ax, color="k")

# %% generate a model dataset
ds = nlmod.get_ds(extent, dx)
ds = nlmod.grid.refine(ds, "logo", [(refine, n)])

# %% plot the logo
figheight = figwidth * (extent[3] - extent[2]) / (extent[1] - extent[0])
f = plt.figure(figsize=(figwidth, figheight))
ax = f.add_axes([0, 0, 1, 1])
ax.axis("equal")
ax.axis(extent)

# f, ax = nlmod.plot.get_map(extent, figsize=(figwidth, figsize), base=50000)
color = "darkslategray"
if False:
    # filled rectangles
    nederland.plot(facecolor=color, ax=ax, edgecolor=color)
    nlmod.plot.modelgrid(ds, color="w", ax=ax, linewidth=1.0)
else:
    if figwidth < 5:
        linewidth = 0.5
    else:
        linewidth = 1.5
    # only plot the rectangles that are partly in the netherlands
    ds["nederland"] = nlmod.grid.gdf_to_bool_da(nederland, ds)
    gdf = nlmod.gis.vertex_da_to_gdf(ds, "nederland")
    gdf[gdf["nederland"] == 1].plot(
        edgecolor=color, facecolor="none", ax=ax, linewidth=linewidth
    )

# nlmod.plot.data_array(in_nl, ds=ds, ax=ax, cmap="RdBu")
# nlmod.plot.data_array(ds["area"] * np.NaN, ds, edgecolor="k", clip_on=False)  # light plot
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
ax.axis("off")
ax.text(
    100_000,
    330_000,
    "nlmod",
    fontname="Consolas",
    fontsize=50,
    ha="center",
    va="center",
    color=color,
    # fontweight="bold",
)

# %%
fname = f"logo_{dx}_{n}"
dpi = 150
if filled:
    fname = f"{fname}_filled"
if figwidth != 5:
    fname = f"{fname}_{figwidth}"
    dpi = None
f.savefig(os.path.join("_static", f"{fname}.png"), dpi=dpi)
f.savefig(os.path.join("_static", f"{fname}.svg"))

# %%
