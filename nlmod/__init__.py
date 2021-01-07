# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:13:44 2021

@author: oebbe
"""

from . import (create_model, mtime, mgrid, recharge, surface_water, util, well,
               mfpackages, regis, geotop, northsea)
import os

nlmod_datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','data')