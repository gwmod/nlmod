# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 12:13:44 2021.

@author: oebbe
"""

import os

from . import gwf, mdims, modpath, read, sim, util, visualise
from .mdims import *
from .version import __version__
from .visualise import plots as plot

NLMOD_DATADIR = os.path.join(os.path.dirname(__file__), "data")
