# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:40:21 2022

@author: ruben
"""
import os
import numpy as np
import nlmod
import art_tools

nederland = art_tools.shapes.nederland()
filled = False
n = 2
dx = 10000
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

# %% plot this
f, ax = nlmod.plot.get_map(extent, figsize=figsize, base=50000)
nlmod.plot.data_array(ds["area"] * np.NaN, ds, edgecolor="k", clip_on=False)
# nlmod.plot.modelgrid(ds, ax=ax)
# modelgrid = nlmod.grid.modelgrid_from_ds(ds)
# modelgrid.plot(ax=ax)
# ax.axis(extent)
ax.axis("off")

# ax.text(50000, 550000, "nlmod", fontsize=30, ha="center", va="center")

fname = f"logo_{dx}_{n}"
if filled:
    fname = f"{fname}_filled"
if figsize != 5:
    fname = f"{fname}_{figsize}"
f.savefig(os.path.join("..", "_static", f"{fname}.png"), bbox_inches="tight")
f.savefig(os.path.join("..", "_static", f"{fname}.svg"), bbox_inches="tight")
