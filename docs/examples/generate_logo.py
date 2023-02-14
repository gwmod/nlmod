# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:40:21 2022

@author: ruben
"""
import os
import numpy as np
import nlmod
import flopy
import art_tools
import xarray as xr

nederland = art_tools.shapes.nederland()
filled = False
n = 2
dx = 10_000
# dx = 20000
figsize = 5

nederland["geometry"] = nederland.simplify(1000)

if not filled:
    nederland["geometry"] = nederland.boundary

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

ds = nlmod.grid.refine(ds, "logo", [(nederland, n)])


nl2 = art_tools.shapes.nederland()
ix = flopy.utils.GridIntersect(nlmod.grid.modelgrid_from_ds(ds), method="vertex")
r = ix.intersect(nl2.geometry.iloc[0])
in_nl = xr.ones_like(ds["top"])
in_nl.data[r["cellids"].astype(int)] = np.nan


# %% plot this
f, ax = nlmod.plot.get_map(extent, figsize=5, base=50000)
nederland.plot(facecolor="k", ax=ax, edgecolor="k")
nlmod.plot.modelgrid(ds, color="w", ax=ax, linewidth=1.0)
# nlmod.plot.data_array(in_nl, ds=ds, ax=ax, cmap="RdBu")
# nlmod.plot.data_array(ds["area"] * np.NaN, ds, edgecolor="k", clip_on=False)  # light plot
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
ax.axis("off")

# ax.text(50000, 550000, "nlmod", fontsize=30, ha="center", va="center")

fname = f"logo_{dx}_{n}"
if filled:
    fname = f"{fname}_filled"
if figsize != 5:
    fname = f"{fname}_{figsize}"
f.savefig(os.path.join("..", "_static", f"{fname}.png"), bbox_inches="tight", dpi=150)
f.savefig(os.path.join("..", "_static", f"{fname}.svg"), bbox_inches="tight")

# %%
