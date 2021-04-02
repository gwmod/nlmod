# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:13:44 2021

@author: oebbe
"""

from . import (util, plots)


from . import mfpackages
from . import mdims
from . import read

from .version import __version__

import os

nlmod_datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','data')