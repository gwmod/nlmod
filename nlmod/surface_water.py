# -*- coding: utf-8 -*-
"""
functions to add surface water to a mf model using the ghb package.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from flopy.utils import GridIntersect

from tqdm import tqdm

import nlmod
from . import mgrid, util


def get_gdf_surface_water(model_ds):
    """ read a shapefile with surface water as a geodataframe, cut by the 
    extent of the model.


    Parameters
    ----------
    model_ds : xr.DataSet
        dataset containing relevant model information

    Returns
    -------
    gdf_opp_water : GeoDataframe
        surface water geodataframe.

    """
    # laad bestanden in
    fname = os.path.join(nlmod.nlmod_datadir, r'opp_water.shp')
    gdf_swater = gpd.read_file(fname)
    gdf_swater = util.gdf_within_extent(gdf_swater, model_ds.extent)

    return gdf_swater


def get_general_head_boundary(model_ds,
                              modelgrid, da_name,
                              cachedir=None,
                              use_cache=False,
                              verbose=False):
    """ Get general head boundary from surface water geodataframe

    Parameters
    ----------
    model_ds : xr.DataSet
        dataset containing relevant model grid information
    modelgrid : flopy grid
        model grid.
    da_name : str
        name of the polygon shapes, name is used to store data arrays in 
        model_ds
    cachedir : str, optional
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached ghb data is used. The default is False.
    verbose : bool, optional
        print additional information to the screen. The default is False.

    Returns
    -------
    model_ds : xr.DataSet
        dataset with spatial model data including the ghb rasters

    """
    model_ds = util.get_cache_netcdf(use_cache, cachedir, 'ghb_model_ds.nc',
                                     surface_water_to_model_dataset,
                                     model_ds, verbose=verbose,
                                     modelgrid=modelgrid, da_name=da_name)

    return model_ds


def surface_water_to_model_dataset(model_ds, modelgrid, da_name):
    """ create 3 data-arrays from the shapefile with surface water:
    - area: with the area of the shape in the cell
    - cond: with the conductance based on the area and bweerstand column in shapefile
    - peil: with the surface water lvl based on the peil column in the shapefile

    Parameters
    ----------
    model_ds : xr.DataSet
        xarray with model data
    modelgrid : flopy grid
        model grid.
    da_name : str
        name of the polygon shapes, name is used to store data arrays in 
        model_ds

    Returns
    -------
    model_ds : xarray.Dataset
        dataset with modelgrid data. Has 

    """
    gdf = get_gdf_surface_water(model_ds)

    area = xr.zeros_like(model_ds['top'])
    cond = xr.zeros_like(model_ds['top'])
    peil = xr.zeros_like(model_ds['top'])
    for i, row in gdf.iterrows():
        area_pol = mgrid.polygon_to_area(modelgrid, row['geometry'],
                                         xr.ones_like(model_ds['top']),
                                         model_ds.gridtype)
        cond = xr.where(area_pol > area, area_pol / row['bweerstand'], cond)
        peil = xr.where(area_pol > area, row['peil'], peil)
        area = xr.where(area_pol > area, area_pol, area)

    model_ds_out = util.get_model_ds_empty(model_ds)
    model_ds_out[f'{da_name}_area'] = area
    model_ds_out[f'{da_name}_cond'] = cond
    model_ds_out[f'{da_name}_peil'] = peil

    return model_ds_out


def gdf2grid(gdf, ml, method="vertex", **kwargs):
    """
    Intersect a geodataframe with a model grid.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame that needs to be cut by the grid. The GeoDataFrame can
        consist of multiple types (Point, LineString, Polygon and the Multi-
        variants).
    ml : flopy.modflow.Modflow or flopy.mf6.ModflowGwf
        The flopy model that defines the grid.
    method : string, optional
        Method passed to the GridIntersect-class. The default is None, which
        makes GridIntersect choose the best method.
    **kwargs : keyword arguments
        keyword arguments are passed to the intersect_*-methods.

    Returns
    -------
    geopandas.GeoDataFrame
        The GeoDataFrame with the geometries per grid-cell.

    """
    ix = GridIntersect(ml.modelgrid, method=method)
    shps = []
    for _, shp in tqdm(gdf.iterrows(), total=gdf.shape[0],
                       desc="Intersecting with grid"):

        r = ix.intersect(shp.geometry, **kwargs)

        for i in range(r.shape[0]):
            shpn = shp.copy()
            shpn['cellid'] = r['cellids'][i]
            shpn.geometry = r['ixshapes'][i]
            shps.append(shpn)
    return gpd.GeoDataFrame(shps)


def aggregate_surface_water(gdf, method, model_ds=None):
    """Aggregate surface water features.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing surfacewater polygons per grid cell.
        Must contain columns 'stage' (waterlevel), 
        'c0' (bottom resistance), and 'botm' (bottom elevation)
    method : str, optional
        "area_weighted" for area-weighted params, 
        "max_area" for max area params
        "de_lange" for De Lange formula for conductance
    model_ds : xarray.DataSet, optional
        DataSet containing model layer information (only required for 
        method='de_lange')

    Returns
    -------
    celldata : pd.DataFrame
        DataFrame with aggregated surface water parameters per grid cell
    """

    required_cols = {"stage", "c0", "botm"}
    missing_cols = required_cols.difference(gdf.columns)
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Post process intersection result
    gr = gdf.groupby(by="cellid")
    celldata = pd.DataFrame(index=gr.groups.keys())

    for cid, group in tqdm(gr, desc="Aggregate surface water data"):

        stage, cond, rbot = get_surfacewater_params(
            group, method, cid=cid, model_ds=model_ds)

        celldata.loc[cid, "stage"] = stage
        celldata.loc[cid, "cond"] = cond
        celldata.loc[cid, "rbot"] = rbot
        celldata.loc[cid, "area"] = group.area.sum()

    return celldata


def distribute_cond_over_lays(cond, cellid, rivbot, laytop, laybot,
                              idomain=None, kh=None, stage=None):

    if isinstance(rivbot, np.ndarray) or isinstance(rivbot, xr.DataArray):
        rivbot = float(rivbot[cellid])
    if len(laybot.shape) == 3:
        # the grid is structured grid
        laytop = laytop[cellid[0], cellid[1]]
        laybot = laybot[:, cellid[0], cellid[1]]
        if idomain is not None:
            idomain = idomain[:, cellid[0], cellid[1]]
        if kh is not None:
            kh = kh[:, cellid[0], cellid[1]]
    elif len(laybot.shape) == 2:
        # the grid is a vertex grid
        laytop = laytop[cellid]
        laybot = laybot[:, cellid]
        if idomain is not None:
            idomain = idomain[:, cellid]
        if kh is not None:
            kh = kh[:, cellid]

    if stage is None or isinstance(stage, str):
        lays = np.arange(int(np.sum(rivbot < laybot)) + 1)
    elif np.isfinite(stage):
        lays = np.arange(int(np.sum(stage < laybot)),
                         int(np.sum(rivbot < laybot)) + 1)
    else:
        lays = np.arange(int(np.sum(rivbot < laybot)) + 1)
    if idomain is not None:
        # only distribute conductance over active layers
        lays = lays[idomain.values[lays] > 0]
    topbot = np.hstack((laytop, laybot))
    topbot[topbot < rivbot] = rivbot
    d = -1 * np.diff(topbot)
    if kh is not None:
        kd = kh * d
    else:
        kd = d
    if np.all(kd <= 0):
        # when for some reason the kd is 0 in all layers (for example when the
        # river bottom is above all the layers), add to the first active layer
        if idomain is not None:
            first_active = np.where(idomain == 1)[0][0]
        else:
            first_active = 0
        lays = [first_active]
        kd[first_active] = 1.
    conds = cond * kd[lays] / np.sum(kd[lays])
    return np.array(lays), np.array(conds)


def build_spd(celldata, pkg, model_ds, verbose=False):
    """Build stress period data for package (RIV, DRN, GHB).

    Parameters
    ----------
    celldata : geopandas.GeoDataFrame
        GeoDataFrame containing data. Cellid must be the index,
        and must have columns  
    pkg : str
        Modflow package: RIV, DRN or GHB
    model_ds : xarray.DataSet
        DataSet containing model layer information
    verbose : bool, optional
        print warnings if True, default is False

    Returns
    -------
    spd : list
        list containing stress period data: 
        - RIV: [(cellid), stage, cond, rbot]
        - DRN: [(cellid), elev, cond]
        - GHB: [(cellid), elev, cond]
    """

    spd = []
    errors = {}

    for cellid, row in tqdm(celldata.iterrows(),
                            total=celldata.index.size,
                            desc=f"Building stress period data {pkg}:"):
        # rbot
        if "rbot" in row.index:
            rbot = row["rbot"]
            if np.isnan(rbot):
                if verbose:
                    print(f"WARNING!: Cell {cellid} skipped because 'rbot' "
                          "is NaN")
                errors[cellid] = "rbot is NaN"
                continue
        elif pkg == "RIV":
            raise ValueError("Column 'rbot' required for building "
                             "RIV package!")
        else:
            rbot = np.nan

        # stage
        stage = row["stage"]

        if np.isnan(stage):
            if verbose:
                print(f"WARNING: Cell {cellid} skipped because stage is NaN!")
            errors[cellid] = "stage is NaN"
            continue

        if (stage < rbot) and np.isfinite(rbot):
            if verbose:
                print(f"WARNING: stage below bottom elevation in {cellid}, "
                      "stage reset to rbot!")
            stage = rbot

        # conductance
        cond = row["cond"]

        # check value
        if np.isnan(cond):
            if verbose:
                print(f"{cellid}: Conductance is NaN! Info: area={row.area:.2f} "
                      f"len={row.len_estimate:.2f}, BL={row['rbot']}")
            errors[cellid] = "cond is NaN"
            continue

        if cond < 0:
            if verbose:
                print(f"{cellid}, Conductance is negative!, area={row.area:.2f}, "
                      f"len={row.len_estimate:.2f}, BL={row['rbot']}")
            errors[cellid] = "cond is negative"
            continue

        # if surface water penetrates multiple layers:
        lays, conds = distribute_cond_over_lays(cond,
                                                cellid,
                                                rbot,
                                                model_ds.top,
                                                model_ds.bot,
                                                model_ds.idomain,
                                                model_ds.kh,
                                                stage)
        # write SPD
        for lay, cond in zip(lays, conds):
            cid = (lay,) + cellid
            if pkg == "RIV":
                spd.append([cid, stage, cond, rbot])
            elif pkg in ["DRN", "GHB"]:
                spd.append([cid, stage, cond])

    print(f"Skipped {len(errors.keys())} cells because of "
          "missing/erroneous data!")
    return spd


def get_surfacewater_params(group, method, cid=None, model_ds=None,
                            delange_params=None):

    if method == "area_weighted":
        # stage
        stage = agg_area_weighted(group, "stage")
        # cond
        c0 = agg_area_weighted(group, "c0")
        cond = group.area.sum() / c0
        # rbot
        rbot = group["botm"].min()

    elif method == "max_area":
        # stage
        stage = agg_max_area(group, "stage")
        # cond
        c0 = agg_max_area(group, "c0")
        cond = group.area.sum() / c0
        # rbot
        rbot = group["BL"].min()

    elif method == "de_lange":

        # get additional requisite parameters
        if delange_params is None:
            delange_params = {}

        # defaults
        c1 = delange_params.pop("c1", 0.0)
        N = delange_params.pop("N", 1e-3)

        # stage
        stage = agg_area_weighted(group, "stage")

        # cond
        c0 = agg_area_weighted(group, "c0")
        _, _, cond = agg_de_lange(group, cid, model_ds, c1=c1, c0=c0, N=N)

        # rbot
        rbot = group["botm"].min()

    else:
        raise ValueError(f"Method '{method}' not recognized!")

    return stage, cond, rbot


def get_subsurface_params_by_cellid(model_ds, cid):
    r, c = cid
    A = model_ds.delr * model_ds.delc  # cell area
    laytop = model_ds['top'].isel(x=c, y=r).data
    laybot = model_ds['bot'].isel(x=c, y=r).data
    kv = model_ds['kv'].isel(x=c, y=r).data
    kh = model_ds['kh'].isel(x=c, y=r).data
    thickness = model_ds["thickness"].isel(x=c, y=r).data
    return A, laytop, laybot, kh, kv, thickness


def agg_max_area(gdf, col):
    return gdf.loc[gdf.area.idxmax(), col]


def agg_area_weighted(gdf, col):
    nanmask = gdf[col].isna()
    aw = ((gdf.area * gdf[col]).sum(skipna=True) /
          gdf.loc[~nanmask].area.sum())
    return aw


def agg_de_lange(group, cid, model_ds, c1=0.0, c0=1.0, N=1e-3,
                 crad_positive=True):

    (A, laytop, laybot, kh, kv, thickness) = \
        get_subsurface_params_by_cellid(model_ds, cid)

    rbot = group["botm"].min()

    # select active layers
    active = thickness > 0
    laybot = laybot[active]
    kh = kh[active]
    kv = kv[active]
    thickness = thickness[active]

    # layer thickn.
    H0 = laytop - laybot[laybot < rbot][0]
    ilay = 0
    rlay = np.where(laybot < rbot)[0][0]

    # equivalent hydraulic conductivities
    H = thickness[ilay:rlay + 1]
    kv = kv[ilay:rlay + 1]
    kh = kh[ilay:rlay + 1]
    kveq = np.sum(H) / np.sum(H / kv)
    kheq = np.sum(H * kh) / np.sum(H)

    # length
    len_est = estimate_polygon_length(group)
    li = len_est.sum()
    # correction if group contains multiple shapes
    # but covers whole cell
    if group.area.sum() == A:
        li = A / np.max([model_ds.delr, model_ds.delc])

    # width
    B = group.area.sum(skipna=True) / li

    # mean water level
    p = group.loc[group.area.idxmax(), "stage"]  # waterlevel

    # calculate params
    pstar, cstar, cond = de_lange_eqns(
        A, H0, kveq, kheq, c1, li, B, c0, p, N, crad_positive=crad_positive)

    return pstar, cstar, cond


def de_lange_eqns(A, H0, kv, kh, c1, li, Bin, c0, p, N, crad_positive=True):
    """Calculates the conductance according to De Lange

    Parameters
    ----------
    A : float
        celoppervlak (m2)
    H0 : float
        doorstroomde dikte (m)
    kv : float
        verticale doorlotendheid (m/d)
    kh : float
        horizontale doorlatendheid (m/d)
    c1 : float
        deklaagweerstand (d)
    li : float
        lengte van de waterlopen (m)
    Bin : float
        bodembreedte (m)
    c0 : float
        slootbodemweerstand (d)
    p : float
        water peil
    N : float
        grondwateraanvulling
    crad_positive: bool, optional
        whether to allow negative crad values. If True, crad will be set to 0
        if it is negative.

    Returns
    -------
    float
        Conductance (m2/d)

    """
    if li > 1e-3 and Bin > 1e-3 and A > 1e-3:
        Bcor = max(Bin, 1e-3)  # has no effect
        L = A / li - Bcor
        y = c1 + H0 / kv

        labdaL = np.sqrt(y * kh * H0)
        if L > 1e-3:
            xL = L / (2 * labdaL)
            FL = xL * coth(xL)
        else:
            FL = 0.0

        labdaB = np.sqrt(y * kh * H0 * c0 / (y + c0))
        xB = Bcor / (2 * labdaB)
        FB = xB * coth(xB)

        CL = (c0 + y) * FL + (c0 * L / Bcor) * FB
        if CL == 0.0:
            CB = 1.0
        else:
            CB = (c1 + c0 + H0 / kv) / (CL - c0 * L / Bcor) * CL

        # volgens Kees Maas mag deze ook < 0 zijn...
        # er miste ook een correctie in de log voor anisotropie
        # Crad = max(0., L / (np.pi * np.sqrt(kv * kh))
        #            * np.log(4 * H0 / (np.pi * Bcor)))
        crad = radial_resistance(L, Bcor, H0, kh, kv)
        if crad_positive:
            crad = max([0.0, crad])

        # Conductance
        pSl = Bcor * li / A
        if pSl >= 1.0 - 1e-10:
            Wp = 1 / (pSl / CB) + crad - c1
        else:
            Wp = 1 / ((1. - pSl) / CL + pSl / CB) + crad - c1
        cond = A / Wp

        # cstar, pstar
        cLstar = CL + crad

        pstar = p + N * (cLstar - y) * (y + c0) * L / (Bcor * cLstar + L * y)
        cstar = cLstar * (c0 + y) * (Bcor + L) / (Bcor * cLstar + L * y)

        return pstar, cstar, cond
    else:
        return 0., 0., 0.


def radial_resistance(L, B, H, kh, kv):
    return (L / (np.pi * np.sqrt(kh * kv)) *
            np.log(4 * H * np.sqrt(kh) / (np.pi * B * np.sqrt(kv))))


def coth(x):
    return 1.0 / np.tanh(x)


def estimate_polygon_length(gdf):
    # estimate length from polygon (for shapefactor > 4)
    shape_factor = gdf.length / np.sqrt(gdf.area)

    len_est1 = (gdf.length - np.sqrt(gdf.length**2 - 16 * gdf.area)) / 4
    len_est2 = (gdf.length + np.sqrt(gdf.length**2 - 16 * gdf.area)) / 4
    len_est = pd.concat([len_est1, len_est2], axis=1).max(axis=1)

    # estimate length from minimum rotated rectangle (for shapefactor < 4)
    min_rect = gdf.geometry.apply(lambda g: g.minimum_rotated_rectangle)
    xy = min_rect.apply(lambda g: np.sqrt(
        (np.array(g.exterior.xy[0]) - np.array(g.exterior.xy[0][0]))**2 +
        (np.array(g.exterior.xy[1]) - np.array(g.exterior.xy[1][0]))**2))
    len_est3 = xy.apply(lambda a: np.partition(a.flatten(), -2)[-2])

    # update length estimate where shape factor is lower than 4
    len_est.loc[shape_factor < 4] = len_est3.loc[shape_factor < 4]

    return len_est
