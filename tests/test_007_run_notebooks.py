# -*- coding: utf-8 -*-
"""
run notebooks in the examples directory
"""
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

tst_dir = os.path.dirname(os.path.realpath(__file__))
nbdir = os.path.join(tst_dir, '..', 'examples')
    
def _run_notebook(nbdir, fname):
    
    fname_nb = os.path.join(nbdir, fname)
    with open(fname_nb) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    out = ep.preprocess(nb, {'metadata': {'path': nbdir}})
    
    return out


def test_run_notebook_01_basic_model():
    _run_notebook(nbdir, '01_basic_model.ipynb')
    
   
def test_run_notebook_02_surface_water():
    _run_notebook(nbdir, '02_surface_water.ipynb')
   
    
def test_run_notebook_03_local_grid_refinement():
    _run_notebook(nbdir, '03_local_grid_refinement.ipynb')

def test_run_notebook_04_modifying_layermodels():
    _run_notebook(nbdir, '04_modifying_layermodels.ipynb')