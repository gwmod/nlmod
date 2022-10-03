# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:07:55 2022

@author: Ruben
"""

import numpy as np
import pandas as pd
import xarray as xr
import logging
import nlmod


# toon informatie bij het aanroepen van functies
logging.basicConfig(level=logging.INFO)


# %%
extent = [0.0, 1000.0, 0.0, 500.0]
ds = nlmod.get_default_ds(
    extent, xorigin=103000, yorigin=400000, model_name="test_rch", model_ws="models"
)
ds = nlmod.refine(ds)
ds = nlmod.set_ds_time(ds, time=pd.date_range("2020", "2022"))

knmi = nlmod.read.knmi.get_recharge(ds, method="separate")
ds.update(knmi)

# %%
# create simulation
sim = nlmod.gwf.sim(ds)

# create time discretisation
tdis = nlmod.gwf.tdis(ds, sim)

# create groundwater flow model
gwf = nlmod.gwf.gwf(ds, sim)

# create ims
ims = nlmod.gwf.ims(sim)

# Create discretization
nlmod.gwf.dis(ds, gwf)

rch = nlmod.gwf.rch(ds, gwf)

# %% plot recharge
f, ax = nlmod.plot.get_map(extent)
nlmod.plot.da(ds["recharge"].sel(time=ds.time[-3]), ds=ds, ax=ax)

f, ax = nlmod.plot.get_map(extent)
indices = np.unique(ds["rch_name"], return_inverse=True)[1].reshape(
    ds["rch_name"].shape
)
da = xr.DataArray(indices, dims=ds["rch_name"].dims, coords=ds["rch_name"].coords)
nlmod.plot.da(da, ds=ds, ax=ax)
