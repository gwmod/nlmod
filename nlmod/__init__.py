# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 12:13:44 2021.

@author: oebbe
"""

import os

NLMOD_DATADIR = os.path.join(os.path.dirname(__file__), "data")

from . import dcs, dims, gis, gwf, modpath, plot, read, sim, util
from .dims import base, get_ds, grid, layers, resample, time, to_model_ds
from .version import __version__
