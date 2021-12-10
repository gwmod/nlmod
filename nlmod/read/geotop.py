
import logging
import os

import nlmod
import numpy as np
import datetime as dt
import pandas as pd
import xarray as xr

from .. import mdims, cache
from . import regis

logger = logging.getLogger(__name__)

@cache.cache_netcdf
def get_geotop(extent, delr, delc,
               regis_ds, regis_layer='HLc'):
    """get a model layer dataset for modflow from geotop within a certain
    extent and grid.

    if regis_ds and regis_layer are defined the geotop model is only created
    to replace this regis_layer in a regis layer model.


    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    regis_ds: xarray.DataSet
        regis dataset used to cut geotop to the same x and y coördinates
    regis_layer: str, optional
        layer of regis dataset that will be filled with geotop. The default is
        'HLc'.

    Returns
    -------
    geotop_ds: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid
    """
    # check extent
    extent2, nrow, ncol = regis.fit_extent_to_regis(extent, delr, delc)
    for coord1, coord2 in zip(extent, extent2):
        if coord1 != coord2:
            raise ValueError('extent not fitted to regis please fit to regis first, use the nlmod.regis.fit_extent_to_regis function')

    geotop_url = r'http://www.dinodata.nl/opendap/GeoTOP/geotop.nc'
    geotop_ds_raw1 = get_geotop_raw_within_extent(extent, geotop_url)

    litho_translate_df = pd.read_csv(os.path.join(nlmod.NLMOD_DATADIR,
                                                  'geotop',
                                                  'litho_eenheden.csv'),
                                     index_col=0)

    geo_eenheid_translate_df = pd.read_csv(os.path.join(nlmod.NLMOD_DATADIR,
                                                        'geotop',
                                                        'geo_eenheden.csv'),
                                           index_col=0,
                                           keep_default_na=False)

    geotop_ds_raw = convert_geotop_to_ml_layers(geotop_ds_raw1,
                                                regis_ds=regis_ds,
                                                regis_layer=regis_layer,
                                                litho_translate_df=litho_translate_df,
                                                geo_eenheid_translate_df=geo_eenheid_translate_df)

    logger.info('resample geotop data to structured modelgrid')
    geotop_ds = mdims.resample_dataset_to_structured_grid(geotop_ds_raw,
                                                         extent,
                                                         delr, delc)
    geotop_ds.attrs['extent'] = extent
    geotop_ds.attrs['delr'] = delr
    geotop_ds.attrs['delc'] = delc
    geotop_ds.attrs['gridtype'] = 'structured'

    for datavar in geotop_ds:
        geotop_ds[datavar].attrs['source'] = 'Geotop'
        geotop_ds[datavar].attrs['url'] = geotop_url
        geotop_ds[datavar].attrs['date'] = dt.datetime.now().strftime('%Y%m%d')
        if datavar in ['top', 'bot']:
            geotop_ds[datavar].attrs['units'] = 'mNAP'
        elif datavar in ['kh', 'kv']:
            geotop_ds[datavar].attrs['units'] = 'm/day'

    return geotop_ds


def get_geotop_raw_within_extent(extent, url):
    """Get a slice of the geotop netcdf url within the extent, set the x and y
    coordinates to match the cell centers and keep only the strat and lithok
    data variables.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    url : str
        url of geotop netcdf file

    Returns
    -------
    geotop_ds_raw : xarray Dataset
        slices geotop netcdf.
    """

    geotop_ds_raw = xr.open_dataset(url)

    # set x and y dimensions to cell center
    for dim in ['x', 'y']:
        old_dim = geotop_ds_raw[dim].values
        geotop_ds_raw[dim] = old_dim + (old_dim[1] - old_dim[0]) / 2

    # slice extent
    geotop_ds_raw = geotop_ds_raw.sel(x=slice(extent[0], extent[1]),
                                      y=slice(extent[2], extent[3]))
    geotop_ds_raw = geotop_ds_raw[['strat', 'lithok']]

    return geotop_ds_raw


def convert_geotop_to_ml_layers(geotop_ds_raw1, regis_ds=None, regis_layer=None,
                                litho_translate_df=None,
                                geo_eenheid_translate_df=None):
    """ does the following steps to obtain model layers based on geotop:
        1. slice by regis layer (if not None)
        2. compute kh from lithoklasse
        3. create a layer model based on geo-eenheden

    Parameters
    ----------
    geotop_ds_raw1: xr.Dataset
        dataset with geotop netcdf
    regis_ds: xarray.DataSet
        regis dataset used to cut geotop to the same x and y coördinates    
    regis_layer: str, optional
        layer of regis dataset that will be filled with geotop 
    litho_translate_df: pandas.DataFrame
        horizontal conductance (kh)
    geo_eenheid_translate_df: pandas.DataFrame
        dictionary to translate geo_eenheid to a geo name   

    Returns
    -------
    geotop_ds_raw: xarray.DataSet
        geotop dataset with added horizontal conductance

    """

    # stap 1
    if (regis_ds is not None) and (regis_layer is not None):
        logger.info(f'slice geotop with regis layer {regis_layer}')
        top_rl = regis_ds['top'].sel(layer=regis_layer)
        bot_rl = regis_ds['bot'].sel(layer=regis_layer)

        geotop_ds_raw = geotop_ds_raw1.sel(z=slice(np.floor(bot_rl.min().data),
                                                   np.ceil(top_rl.max().data)))

    # stap 2 maak kh matrix a.d.v. lithoklasse
    logger.info('create kh matrix from lithoklasse and csv file')
    kh_from_litho = xr.zeros_like(geotop_ds_raw.lithok)
    for i, row in litho_translate_df.iterrows():
        kh_from_litho = xr.where(geotop_ds_raw.lithok == i,
                                 row['hor_conductance_default'],
                                 kh_from_litho)
    geotop_ds_raw['kh_from_litho'] = kh_from_litho

    # stap 3 maak een laag per geo-eenheid
    geotop_ds_mod = get_top_bot_from_geo_eenheid(geotop_ds_raw,
                                                 geo_eenheid_translate_df)

    return geotop_ds_mod


def get_top_bot_from_geo_eenheid(geotop_ds_raw, geo_eenheid_translate_df):
    """get top, bottom and kh of each geo-eenheid in geotop dataset.

    Parameters
    ----------
    geotop_ds_raw: xr.DataSet
        geotop dataset with added horizontal conductance
    geo_eenheid_translate_df: pandas.DataFrame
        dictionary to translate geo_eenheid to a geo name

    Returns
    -------
    geotop_ds_mod: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid

    Note
    ----
    the 'geo_eenheid' >6000 are 'stroombanen' these are difficult to add because
    they can occur above and below any other 'geo_eenheid' therefore they are
    added to the geo_eenheid below the stroombaan.
    """

    # vindt alle geo-eenheden in model_extent
    geo_eenheden = np.unique(geotop_ds_raw.strat.data)
    geo_eenheden = geo_eenheden[np.isfinite(geo_eenheden)]
    stroombaan_eenheden = geo_eenheden[geo_eenheden < 5999]
    geo_eenheden = geo_eenheden[geo_eenheden < 5999]

    # geo eenheid 2000 zit boven 1130
    if (2000. in geo_eenheden) and (1130. in geo_eenheden):
        geo_eenheden[(geo_eenheden == 2000.) + (geo_eenheden == 1130.)] = [2000., 1130.]

    geo_names = [geo_eenheid_translate_df.loc[float(geo_eenh), 'Code (lagenmodel en boringen)'] for geo_eenh in geo_eenheden]

    # fill top and bot
    top = np.ones((geotop_ds_raw.y.shape[0], geotop_ds_raw.x.shape[0], len(geo_names))) * np.nan
    bot = np.ones((geotop_ds_raw.y.shape[0], geotop_ds_raw.x.shape[0], len(geo_names))) * np.nan
    lay = 0
    logger.info('creating top and bot per geo eenheid')
    for geo_eenheid in geo_eenheden:
        logger.info(geo_eenheid)

        mask = geotop_ds_raw.strat == geo_eenheid
        geo_z = xr.where(mask, geotop_ds_raw.z, np.nan)

        top[:, :, lay] = geo_z.max(dim='z').T + 0.5
        bot[:, :, lay] = geo_z.min(dim='z').T

        lay += 1

    geotop_ds_mod = add_stroombanen_and_get_kh(geotop_ds_raw, top, bot,
                                               geo_names)

    geotop_ds_mod.attrs['stroombanen'] = stroombaan_eenheden

    return geotop_ds_mod


def add_stroombanen_and_get_kh(geotop_ds_raw, top, bot, geo_names):
    """add stroombanen to tops and bots of geo_eenheden, also computes kh per
    geo_eenheid. Kh is computed by taking the average of all kh's of a
    geo_eenheid within a cell (e.g. if one geo_eenheid has a thickness of 1,5m
    in a certain cell the kh of the cell is calculated as the mean of the 3
    cells in geotop)

    Parameters
    ----------
    geotop_ds_raw: xr.DataSet
        geotop dataset with added horizontal conductance
    top: np.array
        raster with top of each geo_eenheid, shape(nlay,nrow,ncol)
    bot: np.array
        raster with bottom of each geo_eenheid, shape(nlay,nrow,ncol)
    geo_names: list of str
        names of each geo_eenheid

    Returns
    -------
    geotop_ds_mod: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid
    """
    kh = np.ones((geotop_ds_raw.y.shape[0], geotop_ds_raw.x.shape[0], len(geo_names))) * np.nan
    thickness = np.ones((geotop_ds_raw.y.shape[0], geotop_ds_raw.x.shape[0], len(geo_names))) * np.nan
    z = xr.ones_like(geotop_ds_raw.lithok) * geotop_ds_raw.z
    logger.info('adding stroombanen to top and bot of each layer')
    logger.info('get kh for each layer')

    for lay in range(top.shape[2]):
        logger.info(geo_names[lay])
        if lay == 0:
            top[:, :, 0] = np.nanmax(top, axis=2)
        else:
            top[:, :, lay] = bot[:, :, lay - 1]
        bot[:, :, lay] = np.where(np.isnan(bot[:, :, lay]), top[:, :, lay], bot[:, :, lay])
        thickness[:, :, lay] = (top[:, :, lay] - bot[:, :, lay])

        # check which geotop voxels are within the range of the layer
        bool_z = xr.zeros_like(z)
        for i in range(z.z.shape[0]):
            bool_z[:, :, i] = np.where((z[:, :, i] >= bot[:, :, lay].T) * (z[:, :, i] < top[:, :, lay].T), True, False)

        kh_geo = xr.where(bool_z, geotop_ds_raw['kh_from_litho'], np.nan)
        kh[:, :, lay] = kh_geo.mean(dim='z').T

    da_top = xr.DataArray(data=top, dims=('y', 'x', 'layer'),
                          coords={'y': geotop_ds_raw.y, 'x': geotop_ds_raw.x,
                                  'layer': geo_names})
    da_bot = xr.DataArray(data=bot, dims=('y', 'x', 'layer'),
                          coords={'y': geotop_ds_raw.y, 'x': geotop_ds_raw.x,
                                  'layer': geo_names})
    da_kh = xr.DataArray(data=kh, dims=('y', 'x', 'layer'),
                         coords={'y': geotop_ds_raw.y, 'x': geotop_ds_raw.x,
                                 'layer': geo_names})
    da_thick = xr.DataArray(data=thickness, dims=('y', 'x', 'layer'),
                            coords={'y': geotop_ds_raw.y, 'x': geotop_ds_raw.x,
                                    'layer': geo_names})

    geotop_ds_mod = xr.Dataset()

    geotop_ds_mod['top'] = da_top
    geotop_ds_mod['bot'] = da_bot
    geotop_ds_mod['kh'] = da_kh
    geotop_ds_mod['kv'] = geotop_ds_mod['kh'] * .25
    geotop_ds_mod['thickness'] = da_thick

    return geotop_ds_mod
