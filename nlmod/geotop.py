
import xarray as xr
import os
import pandas as pd
import numpy as np
import nlmod

from . import mgrid


def get_geotop_dataset(extent, delr, delc,
                       regis_ds, regis_layer='HLc',
                       cachedir=None,
                       use_cache=False,
                       verbose=False):
    """ get a model layer dataset for modflow from geotop within a certain 
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
    cachedir : str
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached resampled regis dataset is used.
        The default is False.
    verbose : bool, optional
        print additional information. The default is False.

    Returns
    -------
    geotop_ds: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid

    """

    geotop_ds_raw1 = get_geotop_raw_within_extent(extent)

    litho_translate_df = pd.read_csv(os.path.join(nlmod.nlmod_datadir,
                                                  'geotop',
                                                  'litho_eenheden.csv'),
                                     index_col=0)

    geo_eenheid_translate_df = pd.read_csv(os.path.join(nlmod.nlmod_datadir,
                                                        'geotop',
                                                        'geo_eenheden.csv'),
                                           index_col=0,
                                           keep_default_na=False)

    geotop_ds_raw = convert_geotop_to_ml_layers(geotop_ds_raw1,
                                                regis_ds=regis_ds,
                                                regis_layer=regis_layer,
                                                litho_translate_df=litho_translate_df,
                                                geo_eenheid_translate_df=geo_eenheid_translate_df,
                                                verbose=verbose)

    geotop_ds = mgrid.get_ml_layer_dataset_struc(raw_ds=geotop_ds_raw,
                                                 extent=extent,
                                                 delr=delr, delc=delc,
                                                 cachedir=cachedir,
                                                 fname_netcdf='geotop_ds.nc',
                                                 use_cache=use_cache,
                                                 verbose=verbose)

    return geotop_ds


def get_geotop_raw_within_extent(extent):
    """ Get a slice of the geotop netcdf url within the extent and only the
    strat and lithok data variables.


    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)

    Returns
    -------
    geotop_ds_raw : xarray Dataset
        slices geotop netcdf.

    """

    url = r'http://www.dinodata.nl/opendap/GeoTOP/geotop.nc'
    geotop_ds_raw = xr.open_dataset(url).sel(x=slice(extent[0], extent[1]),
                                             y=slice(extent[2], extent[3]))
    geotop_ds_raw = geotop_ds_raw[['strat', 'lithok']]

    return geotop_ds_raw


def convert_geotop_to_ml_layers(geotop_ds_raw1, regis_ds=None, regis_layer=None,
                                litho_translate_df=None,
                                geo_eenheid_translate_df=None,
                                verbose=False):
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
    verbose : bool, optional
        print additional information. default is False

    Returns
    -------
    geotop_ds_raw: xarray.DataSet
        geotop dataset with added horizontal conductance

    """

    # stap 1
    if (regis_ds is not None) and (regis_layer is not None):
        if verbose:
            print(f'slice geotop with regis layer {regis_layer}')
        top_rl = regis_ds['top'].sel(layer=regis_layer)
        bot_rl = regis_ds['bot'].sel(layer=regis_layer)

        geotop_ds_raw = geotop_ds_raw1.sel(z=slice(np.floor(bot_rl.min().data),
                                                   np.ceil(top_rl.max().data)))

    # stap 2 maak kh matrix a.d.v. lithoklasse
    if verbose:
        print('create kh matrix from lithoklasse and csv file')
    kh_from_litho = xr.zeros_like(geotop_ds_raw.lithok)
    for i, row in litho_translate_df.iterrows():
        kh_from_litho = xr.where(geotop_ds_raw.lithok == i,
                                 row['hor_conductance_default'],
                                 kh_from_litho)
    geotop_ds_raw['kh_from_litho'] = kh_from_litho

    # stap 3 maak een laag per geo-eenheid
    geotop_ds_mod = get_top_bot_from_geo_eenheid(geotop_ds_raw,
                                                 geo_eenheid_translate_df,
                                                 verbose=verbose)

    return geotop_ds_mod


def get_top_bot_from_geo_eenheid(geotop_ds_raw, geo_eenheid_translate_df,
                                 verbose=False):
    """ get top, bottom and kh of each geo-eenheid in geotop dataset.

    Parameters
    ----------
    geotop_ds_raw: xr.DataSet
        geotop dataset with added horizontal conductance
    geo_eenheid_translate_df: pandas.DataFrame
        dictionary to translate geo_eenheid to a geo name
    verbose : bool, optional
        print additional information. default is False

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
    if verbose:
        print('creating top and bot per geo eenheid')
    for geo_eenheid in geo_eenheden:
        if verbose:
            print(geo_eenheid)

        mask = geotop_ds_raw.strat == geo_eenheid
        geo_z = xr.where(mask, geotop_ds_raw.z, np.nan)

        top[:, :, lay] = geo_z.max(dim='z').T + 0.5
        bot[:, :, lay] = geo_z.min(dim='z').T

        lay += 1

    geotop_ds_mod = add_stroombanen_and_get_kh(geotop_ds_raw, top, bot,
                                               geo_names,
                                               verbose=verbose)

    geotop_ds_mod.attrs['stroombanen'] = stroombaan_eenheden

    return geotop_ds_mod


def add_stroombanen_and_get_kh(geotop_ds_raw, top, bot, geo_names, verbose=False):
    """ add stroombanen to tops and bots of geo_eenheden, also computes kh per 
    geo_eenheid. Kh is computed by taking the average of all kh's of a geo_eenheid
    within a cell (e.g. if one geo_eenheid has a thickness of 1,5m in a certain
    cell the kh of the cell is calculated as the mean of the 3 cells in geotop)

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
    verbose : bool, optional
        print additional information. default is False

    Returns
    -------
    geotop_ds_mod: xr.DataSet
        geotop dataset with top, bot, kh and kv per geo_eenheid

    """
    kh = np.ones((geotop_ds_raw.y.shape[0], geotop_ds_raw.x.shape[0], len(geo_names))) * np.nan
    thickness = np.ones((geotop_ds_raw.y.shape[0], geotop_ds_raw.x.shape[0], len(geo_names))) * np.nan
    z = xr.ones_like(geotop_ds_raw.lithok) * geotop_ds_raw.z
    if verbose:
        print('adding stroombanen to top and bot of each layer')
        print('get kh for each layer')

    for lay in range(top.shape[2]):
        if verbose:
            print(geo_names[lay])
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
