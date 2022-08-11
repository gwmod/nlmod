import os
import sys
import flopy
import numbers
import numpy as np
import pandas as pd
import geopandas as gpd

import logging
logger = logging.getLogger(__name__)


def xy_to_cid(xy, model_ds):
    
    cid = (np.abs(model_ds.x - xy[0])+np.abs(model_ds.y -xy[1])).argmin().item()
    
    return cid

def xy_to_nodes(xy_list, mpf, model_ds, layer=0):
    
    if isinstance(layer, numbers.Number):
        layer = [layer] * len(xy_list)
    
    nodes = []
    for i, xy in enumerate(xy_list):
        cid = xy_to_cid(xy, model_ds)
        if mpf.ib[layer[i], cid] > 0:
            node = layer[i] * mpf.ib.shape[1] + cid
            nodes.append(node)
            
    return nodes

def package_to_nodes(gwf, package_name, mpf):
    
    gwf_package = gwf.get_package(package_name)
    if not hasattr(gwf_package, 'stress_period_data'):
        raise TypeError('only package with stress period data can be used')
    
    pkg_cid = gwf_package.stress_period_data.array[0]['cellid']
    nodes = []
    for cid in pkg_cid:
        if mpf.ib[cid[0], cid[1]] > 0:
            node = cid[0] * mpf.ib.shape[1] + cid[1]
            nodes.append(node)
    
    return nodes

def mpf(gwf, exe_name=None):
    
    # check if the save flows parameter is set in the npf package
    npf = gwf.get_package('npf')
    if not npf.save_flows.array:
        raise ValueError('the save_flows option of the npf package should be True not None')
    
    # check if the tdis has a start_time (this gives an error for some reason)
    if gwf.simulation.tdis.start_date_time.array is not None:
        raise ValueError('modpath cannot handle this for some reason')

    # get executable
    if exe_name is None:
        exe_name = os.path.join(
            os.path.dirname(__file__), "..", "bin", 'mp7')
        
        if sys.platform.startswith("win"):
            exe_name += ".exe"
    
    # create mpf model
    mpf = flopy.modpath.Modpath7(modelname="mp7_" + gwf.name + "_f",
                                 flowmodel=gwf,
                                 exe_name=exe_name,
                                 model_ws=gwf.model_ws,
                                 verbose=True)
    
    return mpf
    
def bas(mpf, porosity=0.3):
    
    mpfbas = flopy.modpath.Modpath7Bas(mpf,  
                                       porosity=porosity)
    
    return mpfbas
    
    
def remove_output(mpf):
    
    mpffiles = [
        mpf.name + ".mppth",
        mpf.name + ".timeseries",
        mpf.name + ".mpend",
    ]

    # remove output
    for f in mpffiles:
        fname = os.path.join(mpf.model_ws, f)
        if os.path.exists(fname):
            os.remove(fname)
            print(f"removed '{f}'")
        else:
            print(f"could not find '{f}'")
            


def load_pathline_data(mpf=None, model_ws=None,
                       model_name=None, return_df=False, return_gdf=False):

    if mpf is None:
        fpth = os.path.join(model_ws, f'mp7_gwf_{model_name}_f.mppth')
    else:
        fpth = os.path.join(mpf.model_ws, mpf.name + '.mppth')
    p = flopy.utils.PathlineFile(fpth, verbose=False)
    if (not return_df) and (not return_gdf):
        return p._data
    elif return_df and (not return_gdf):
        pdf = pd.DataFrame(p._data)
        return pdf
    elif return_gdf and (not return_df):
        pdf = pd.DataFrame(p._data)
        geom = gpd.points_from_xy(pdf['x'], pdf['y'])
        pgdf = gpd.GeoDataFrame(pdf, geometry=geom)
        return pgdf
    else:
        raise ValueError("'return_df' and 'return_gdf' are both True, while only one can be True")

def pg_from_fdt(nodes, divisions=3):
    
    logger.info(f'particle group with {divisions**2} particle per cell face, {6*divisions**2} particles per cell')
    sd = flopy.modpath.FaceDataType(drape=0,
                                    verticaldivisions1=divisions,
                                    horizontaldivisions1=divisions,
                                    verticaldivisions2=divisions,
                                    horizontaldivisions2=divisions,
                                    verticaldivisions3=divisions,
                                    horizontaldivisions3=divisions,
                                    verticaldivisions4=divisions,
                                    horizontaldivisions4=divisions,
                                    rowdivisions5=divisions,
                                    columndivisions5=divisions,
                                    rowdivisions6=divisions,
                                    columndivisions6=divisions)
    
    p = flopy.modpath.NodeParticleData(subdivisiondata=sd, 
                                       nodes=nodes)
    
    pg = flopy.modpath.ParticleGroupNodeTemplate(particledata=p)
    
    return pg


def sim(mpf, pg, gwf, ref_time=None, stoptime=None):

    if stoptime is None:
        stoptimeoption = 'extend'
    else:
        stoptimeoption = 'specified'
    
    
    
    if ref_time is None:
        ref_time = (gwf.simulation.tdis.nper.array-1, #stress period
                    gwf.simulation.tdis.data_list[-1].array[-1][1]-1, # timestep
                    1.0)
    
    mpsim = flopy.modpath.Modpath7Sim(mpf,
                                      simulationtype="combined",
                                      trackingdirection='backward',
                                      weaksinkoption="pass_through",
                                      weaksourceoption="pass_through",
                                      referencetime=ref_time,
                                      stoptimeoption=stoptimeoption,
                                      stoptime=stoptime,
                                      particlegroups=pg)
    
    return mpsim