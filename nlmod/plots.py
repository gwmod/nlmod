# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 21:32:49 2021

@author: oebbe
"""

from nlmod import surface_water
import matplotlib.pyplot as plt

def model_ds_surface_water(model_ds):
    model_ds['extent'] = [95000., 105000., 494000., 500000.]
    opp_water = surface_water.get_gdf_opp_water(model_ds)
    
    fig, ax = plt.subplots()
    opp_water.plot(ax=ax)
    