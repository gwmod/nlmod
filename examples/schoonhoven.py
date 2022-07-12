# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:07:14 2022

@author: ruben
"""

# %% import packages
import os
import rioxarray
import rasterio
import numpy as np
import pandas as pd
from rasterstats import zonal_stats

import flopy

import logging
import nlmod

# toon informatie bij het aanroepen van functies
logging.basicConfig(level=logging.INFO)

# %% model settings
model_name = "Schoonhoven"
model_ws = os.path.join("models", model_name)
cachedir = os.path.join(model_ws, "cache")
extent = [116500, 120000, 439000, 442000]
time = pd.date_range("2015", "2022", freq="MS")

# %% downlaod regis and geotop and combine in a layer model
regis = nlmod.read.regis.get_regis(extent, cachedir=cachedir, cachename="layers.nc")

# %% download ahn
ahn_file = nlmod.read.ahn.get_ahn_within_extent(extent)
ahn = rioxarray.open_rasterio(ahn_file.open(), mask_and_scale=True)[0]
fname_ahn = os.path.join(cachedir, "ahn.tif")
ahn.rio.to_raster(fname_ahn)

# %% download layer 'waterdeel' from bgt
bgt = nlmod.read.bgt.get_bgt(extent)
# get the minimum surface level in 1 meter around surface water levels
stats = zonal_stats(bgt.geometry.buffer(1.0), fname_ahn, stats="min")
bgt["ahn_min"] = [x["min"] for x in stats]

# %% create a model dataset
ds = nlmod.mdims.get_empty_model_ds(model_name, model_ws)
ds = nlmod.mdims.set_model_ds_time(ds, time=time)
ds = nlmod.mdims.update_model_ds_from_ml_layer_ds(
    ds, regis, add_northsea=False, keep_vars=["x", "y"]
)

# determine the median surface height
transform = nlmod.mdims.resample.get_dataset_transform(ds)
shape = (len(ds.y), len(ds.x))
resampling = rasterio.enums.Resampling.average
ds["ahn_mean"] = ahn.rio.reproject(
    transform=transform,
    dst_crs=28992,
    shape=shape,
    resampling=resampling,
    nodata=np.NaN,
)

# %% add knmi recharge to the model datasets
knmi_ds = nlmod.read.knmi.get_recharge(ds, cachedir=cachedir, cachename="recharge.nc")
ds.update(knmi_ds)

# %% create modflow packages
sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(ds)

# Create discretization
nlmod.mfpackages.dis_from_model_ds(ds, gwf)

# create node property flow
nlmod.mfpackages.npf_from_model_ds(ds, gwf)

# Create the initial conditions package
nlmod.mfpackages.ic_from_model_ds(ds, gwf, starting_head=0.0)

# Create the output control package
nlmod.mfpackages.oc_from_model_ds(ds, gwf)

# create recharge package
rch = nlmod.mfpackages.rch_from_model_ds(ds, gwf)

# create storagee package
sto = nlmod.mfpackages.sto_from_model_ds(ds, gwf)


# add drains for the surface water
bgt_grid = nlmod.mdims.gdf2grid(bgt, ml=gwf).set_index("cellid")
# for now, remove items without a level
bgt_grid = bgt_grid[~np.isnan(bgt_grid["ahn_min"])]
bgt_grid["stage"] = bgt_grid["ahn_min"]
bgt_grid["rbot"] = bgt_grid["ahn_min"] - 0.5
# use a resistance of 1 meter
bgt_grid["cond"] = bgt_grid.area / 1.0
if True:
    # Model the river Lek as a river with a stage of 0.5 m NAP
    # bgt.plot('bronhouder', legend=True)
    mask = bgt_grid["bronhouder"] == "L0002"
    lek = bgt_grid[mask]
    bgt_grid = bgt_grid[~mask]
    lek["stage"] = 0.5
    lek["rbot"] = -3.0
    spd = nlmod.mfpackages.surface_water.build_spd(lek, "RIV", ds)
    riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data={0: spd})
spd = nlmod.mfpackages.surface_water.build_spd(bgt_grid, "DRN", ds)
drn = flopy.mf6.ModflowGwfdrn(gwf, stress_period_data={0: spd})

# %% run model
nlmod.util.write_and_run_model(gwf, ds)

# %% get the head
head = nlmod.util.get_heads_dataarray(ds)
head[0][0].plot()

# %% plot the average head
f, ax = nlmod.visualise.plots.get_map(extent)
da = head.sel(layer="HLc").mean("time")
qm = ax.pcolormesh(da.x, da.y, da)
nlmod.visualise.plots.colorbar_inside(qm)
bgt.plot(ax=ax, edgecolor="k", facecolor="none")

# %% plot a time series at a certains location
head.interp(x=118052, y=440239, method="nearest").plot.line(hue="layer", size=(10))
