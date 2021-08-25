# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 12:13:44 2021.

@author: oebbe
"""

import os

from . import mdims, mfpackages, plots, read, util
from .version import __version__

NLMOD_DATADIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','data')
