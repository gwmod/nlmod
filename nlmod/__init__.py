# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 12:13:44 2021.

@author: oebbe
"""

import os

NLMOD_DATADIR = os.path.join(os.path.dirname(__file__), "data")

from . import gwf, dims, modpath, read, sim, util, plot, gis
from .dims import get_ds, base, grid, layers, time, resample, to_model_ds
from .version import __version__
