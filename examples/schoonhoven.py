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
from shapely.geometry import Point

import flopy

import logging
import nlmod
# toon informatie bij het aanroepen van functies
logging.basicConfig(level=logging.INFO)

# %% model settings
model_name = 'Schoonhoven'
model_ws = os.path.join('models', model_name)
figdir, cachedir = nlmod.util.get_model_dirs(model_ws)
extent = [116500, 120000, 439000, 442000]
time = pd.date_range('2015', '2022', freq='MS')

# %% download ahn
fname_ahn = os.path.join(cachedir, 'ahn.tif')
if not os.path.isfile(fname_ahn):
    ahn_file = nlmod.read.ahn.get_ahn_within_extent(extent)
    ahn = rioxarray.open_rasterio(ahn_file.open(), mask_and_scale=True)[0]
    ahn.rio.to_raster(fname_ahn)

# %% download layer 'waterdeel' from bgt
bgt = nlmod.read.bgt.get_bgt(extent)
# get the minimum surface level in 1 meter around surface water levels
stats = zonal_stats(bgt.geometry.buffer(1.0), fname_ahn, stats='min')
bgt['ahn_min'] = [x['min'] for x in stats]

# %% downlaod regis
regis = nlmod.read.get_regis(extent, cachedir=cachedir, cachename='regis.nc')

# %% create a grid and create nessecary data
ds = nlmod.read.regis.to_grid(regis, delr=100., delc=100.)
ds = nlmod.mbase.set_ds_attrs(ds, model_name, model_ws)

# %% make a disv-grid (or not, by commenting out next line)
# ds = nlmod.mgrid.refine(ds)
refinement_features = [(bgt[bgt['bronhouder'] == 'L0002'], 2)]
ds = nlmod.mgrid.refine(ds, refinement_features=refinement_features)

# %%
ds = nlmod.mlayers.complete_ds(ds)

# %% add information about time
ds = nlmod.mdims.set_model_ds_time(ds, time=time)

if False:
    # determine the median surface height
    transform = nlmod.resample.get_dataset_transform(ds)
    shape = (len(ds.y), len(ds.x))
    resampling = rasterio.enums.Resampling.average
    ds['ahn_mean'] = ahn.rio.reproject(transform=transform, dst_crs=28992,
                                       shape=shape, resampling=resampling,
                                       nodata=np.NaN)

# %% add knmi recharge to the model datasets
knmi_ds = nlmod.read.knmi.get_recharge(ds, cachedir=cachedir,
                                       cachename='recharge.nc')
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
mg = nlmod.mgrid.modelgrid_from_model_ds(ds)
gi = flopy.utils.GridIntersect(mg, method='vertex')
bgt_grid = nlmod.mdims.gdf2grid(bgt, ix=gi).set_index('cellid')
# for now, remove items without a level
bgt_grid = bgt_grid[~np.isnan(bgt_grid['ahn_min'])]
bgt_grid['stage'] = bgt_grid['ahn_min']
bgt_grid['rbot'] = bgt_grid['ahn_min'] - 0.5
# use a resistance of 1 meter
bgt_grid['cond'] = bgt_grid.area / 1.0
if True:
    # Model the river Lek as a river with a stage of 0.5 m NAP
    #bgt.plot('bronhouder', legend=True)
    mask = bgt_grid['bronhouder'] == 'L0002'
    lek = bgt_grid[mask]
    bgt_grid = bgt_grid[~mask]
    lek['stage'] = 0.0
    lek['rbot'] = -3.0
    spd = nlmod.mfpackages.surface_water.build_spd(lek, 'RIV', ds)
    riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data={0: spd})
spd = nlmod.mfpackages.surface_water.build_spd(bgt_grid, 'DRN', ds)
drn = flopy.mf6.ModflowGwfdrn(gwf, stress_period_data={0: spd})

# %% run model
nlmod.util.write_and_run_model(gwf, ds)

# %% get the head
head = nlmod.util.get_heads_dataarray(ds)

# %% plot the average head
f, ax = nlmod.plot.get_map(extent)
pc = nlmod.plot.da(head.sel(layer='HLc').mean('time'), ds=ds, edgecolor='k')
nlmod.plot.colorbar_inside(pc)
bgt.plot(ax=ax, edgecolor='k', facecolor='none')

# %% plot a time series at a certain location
if ds.gridtype == 'vertex':
    icelld2 = gi.intersect(Point(118052, 440239))['cellids'][0]
    head_point = head[:, :, icelld2]
else:
    head_point = head.interp(x=118052, y=440239, method='nearest')
head_point.plot.line(hue='layer', size=(10))

# %% plot some properties of the first layer
layer = 'HLc'
f, axes = nlmod.plot.get_map(extent, nrows=2, ncols=2)
nlmod.plot.da(ds['idomain'].sel(layer=layer), ds=ds, ax=axes[0, 0])
nlmod.plot.da(ds['kh'].sel(layer=layer), ds=ds, ax=axes[1, 0])
nlmod.plot.da(ds['top'], ds=ds, ax=axes[0, 1])
nlmod.plot.da(ds['botm'].sel(layer=layer), ds=ds, ax=axes[1, 1])
