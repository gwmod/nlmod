# -*- coding: utf-8 -*-
"""Created on Thu Jan  7 12:13:44 2021.

@author: oebbe
"""

import os

from . import mdims, mfpackages, read, util, visualise
from .mdims import mbase, mgrid, mlayers, mtime, resample
from .version import __version__

NLMOD_DATADIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
