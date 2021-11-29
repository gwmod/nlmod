# -*- coding: utf-8 -*-
"""function to project regis, or a combination of regis and geotop, data on a
modelgrid."""
import logging

import numpy as np
import datetime as dt
import xarray as xr
from scipy.interpolate import griddata

from .. import mdims, cache
from . import geotop

logger = logging.getLogger(__name__)


@cache.cache_netcdf
def get_combined_layer_models(extent, delr, delc,
                              regis_botm_layer=b'AKc',
                              use_regis=True, use_geotop=True,
                              remove_nan_layers=True):
    """combine layer models into a single layer model.

    Possibilities so far include:
        - use_regis -> full model based on regis
        - use_regis and use_geotop -> holoceen of REGIS is filled with geotop


    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    regis_botm_layer : binary str, optional
        regis layer that is used as the bottom of the model. This layer is
        included in the model. the Default is b'AKc' which is the bottom
        layer of regis. call nlmod.regis.get_layer_names() to get a list of
        regis names.
    use_regis : bool, optional
        True if part of the layer model should be REGIS. The default is True.
    use_geotop : bool, optional
        True if part of the layer model should be geotop. The default is True.
    remove_nan_layers : bool, optional
        if True regis and geotop layers with only nans are removed from the
        model. if False nan layers are kept which might be usefull if you want
        to keep some layers that exist in other models. The default is True.

    Raises
    ------
    ValueError
        if an invalid combination of layers is used.

    Returns
    -------
    combined_ds : xarray dataset
        combination of layer models.
    """

    if use_regis:
        regis_ds = get_regis_dataset(extent, delr, delc, regis_botm_layer)
    else:
        raise ValueError('layer models without REGIS not supported')

    if use_geotop:
        geotop_ds = geotop.get_geotop_dataset(extent, delr, delc, regis_ds)

    if use_regis and use_geotop:
        regis_geotop_ds = add_geotop_to_regis_hlc(regis_ds, geotop_ds)

        combined_ds = regis_geotop_ds
    elif use_regis:
        combined_ds = regis_ds
    else:
        raise ValueError('combination of model layers not supported')

    if remove_nan_layers:
        nlay, lay_sel = get_non_nan_layers(combined_ds)
        combined_ds = combined_ds.sel(layer=lay_sel)
        logger.info(f'removing {nlay} nan layers from the model')

    return combined_ds


def get_regis_dataset(extent, delr, delc, botm_layer=b'AKc'):
    """get a regis dataset projected on the modelgrid.

    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    botm_layer : binary str, optional
        regis layer that is used as the bottom of the model. This layer is
        included in the model. the Default is b'AKc' which is the bottom
        layer of regis. call nlmod.regis.get_layer_names() to get a list of
        regis names.

    Returns
    -------
    regis_ds : xarray dataset
        dataset with regis data projected on the modelgrid.
    """
    # check extent
    extent2, nrow, ncol = fit_extent_to_regis(extent, delr, delc)
    for coord1, coord2 in zip(extent, extent2):
        if coord1 != coord2:
            raise ValueError(
                'extent not fitted to regis please fit to regis first, use the nlmod.regis.fit_extent_to_regis function')

    # get local regis dataset
    regis_url = 'http://www.dinodata.nl:80/opendap/REGIS/REGIS.nc'

    regis_ds_raw = xr.open_dataset(regis_url, decode_times=False)

    # set x and y dimensions to cell center
    regis_ds_raw['x'] = regis_ds_raw.x_bounds.mean('bounds')
    regis_ds_raw['y'] = regis_ds_raw.y_bounds.mean('bounds')

    # slice extent
    regis_ds_raw = regis_ds_raw.sel(x=slice(extent[0], extent[1]),
                                    y=slice(extent[2], extent[3]))

    # slice layers
    if isinstance(botm_layer, str):
        botm_layer = botm_layer.encode('utf-8')

    layer_no = np.where((regis_ds_raw.layer == botm_layer).values)[0][0]
    regis_ds_raw = regis_ds_raw.sel(layer=regis_ds_raw.layer[:layer_no + 1])

    # slice data vars
    regis_ds_raw = regis_ds_raw[['top', 'bottom', 'kD', 'c', 'kh', 'kv']]
    regis_ds_raw = regis_ds_raw.rename_vars({'bottom': 'bot'})

    # rename layers
    regis_ds_raw = regis_ds_raw.rename({'layer': 'layer_old'})
    regis_ds_raw.coords['layer'] = regis_ds_raw.layer_old.astype(
        str)  # could also use assign_coords
    regis_ds_raw2 = regis_ds_raw.swap_dims({'layer_old': 'layer'})

    # convert regis dataset to grid
    logger.info('resample regis data to structured modelgrid')
    regis_ds = mdims.resample_dataset_to_structured_grid(regis_ds_raw2,
                                                            extent,
                                                            delr, delc)
    regis_ds.attrs['extent'] = extent
    regis_ds.attrs['delr'] = delr
    regis_ds.attrs['delc'] = delc
    regis_ds.attrs['gridtype'] = 'structured'

    for datavar in regis_ds:
        regis_ds[datavar].attrs['source'] = 'REGIS'
        regis_ds[datavar].attrs['url'] = regis_url
        regis_ds[datavar].attrs['date'] = dt.datetime.now().strftime('%Y%m%d')

    return regis_ds


def add_geotop_to_regis_hlc(regis_ds, geotop_ds,
                            float_correction=0.001):
    """Combine geotop and regis in such a way that the holoceen in Regis is
    replaced by the geo_eenheden of geotop.

    Parameters
    ----------
    regis_ds: xarray.DataSet
        regis dataset
    geotop_ds: xarray.DataSet
        geotop dataset
    float_correction: float
        due to floating point precision some floating point numbers that are
        the same are not recognised as the same. Therefore this correction is
        used.

    Returns
    -------
    regis_geotop_ds: xr.DataSet
        combined dataset
    """
    regis_geotop_ds = xr.Dataset()

    # find holoceen (remove all layers above Holoceen)
    layer_no = np.where((regis_ds.layer == 'HLc').values)[0][0]
    new_layers = np.append(geotop_ds.layer.data,
                           regis_ds.layer.data[layer_no + 1:].astype('<U8')).astype('O')

    top = xr.DataArray(dims=('layer', 'y', 'x'),
                       coords={'y': geotop_ds.y, 'x': geotop_ds.x,
                               'layer': new_layers})
    bot = xr.DataArray(dims=('layer', 'y', 'x'),
                       coords={'y': geotop_ds.y, 'x': geotop_ds.x,
                               'layer': new_layers})
    kh = xr.DataArray(dims=('layer', 'y', 'x'),
                      coords={'y': geotop_ds.y, 'x': geotop_ds.x,
                              'layer': new_layers})
    kv = xr.DataArray(dims=('layer', 'y', 'x'),
                      coords={'y': geotop_ds.y, 'x': geotop_ds.x,
                              'layer': new_layers})

    # haal overlap tussen geotop en regis weg
    logger.info('cut geotop layer based on regis holoceen')
    for lay in range(geotop_ds.dims['layer']):
        # Alle geotop cellen die onder de onderkant van het holoceen liggen worden inactief
        mask1 = geotop_ds['top'][lay] <= (
            regis_ds['bot'][layer_no] - float_correction)
        geotop_ds['top'][lay] = xr.where(mask1, np.nan, geotop_ds['top'][lay])
        geotop_ds['bot'][lay] = xr.where(mask1, np.nan, geotop_ds['bot'][lay])
        geotop_ds['kh'][lay] = xr.where(mask1, np.nan, geotop_ds['kh'][lay])
        geotop_ds['kv'][lay] = xr.where(mask1, np.nan, geotop_ds['kv'][lay])

        # Alle geotop cellen waarvan de bodem onder de onderkant van het holoceen ligt, krijgen als bodem de onderkant van het holoceen
        mask2 = geotop_ds['bot'][lay] < regis_ds['bot'][layer_no]
        geotop_ds['bot'][lay] = xr.where(
            mask2 * (~mask1), regis_ds['bot'][layer_no], geotop_ds['bot'][lay])

        # Alle geotop cellen die boven de bovenkant van het holoceen liggen worden inactief
        mask3 = geotop_ds['bot'][lay] >= (
            regis_ds['top'][layer_no] - float_correction)
        geotop_ds['top'][lay] = xr.where(mask3, np.nan, geotop_ds['top'][lay])
        geotop_ds['bot'][lay] = xr.where(mask3, np.nan, geotop_ds['bot'][lay])
        geotop_ds['kh'][lay] = xr.where(mask3, np.nan, geotop_ds['kh'][lay])
        geotop_ds['kv'][lay] = xr.where(mask3, np.nan, geotop_ds['kv'][lay])

        # Alle geotop cellen waarvan de top boven de top van het holoceen ligt, krijgen als top het holoceen van regis
        mask4 = geotop_ds['top'][lay] >= regis_ds['top'][layer_no]
        geotop_ds['top'][lay] = xr.where(
            mask4 * (~mask3), regis_ds['top'][layer_no], geotop_ds['top'][lay])

        # overal waar holoceen inactief is, wordt geotop ook inactief
        mask5 = regis_ds['bot'][layer_no].isnull()
        geotop_ds['top'][lay] = xr.where(mask5, np.nan, geotop_ds['top'][lay])
        geotop_ds['bot'][lay] = xr.where(mask5, np.nan, geotop_ds['bot'][lay])
        geotop_ds['kh'][lay] = xr.where(mask5, np.nan, geotop_ds['kh'][lay])
        geotop_ds['kv'][lay] = xr.where(mask5, np.nan, geotop_ds['kv'][lay])
        if (mask2 * (~mask1)).sum() > 0:
            logger.info(f'regis holoceen snijdt door laag {geotop_ds.layer[lay].values}')

    top[:len(geotop_ds.layer), :, :] = geotop_ds['top'].data
    top[len(geotop_ds.layer):, :, :] = regis_ds['top'].data[layer_no + 1:]

    bot[:len(geotop_ds.layer), :, :] = geotop_ds['bot'].data
    bot[len(geotop_ds.layer):, :, :] = regis_ds['bot'].data[layer_no + 1:]

    kh[:len(geotop_ds.layer), :, :] = geotop_ds['kh'].data
    kh[len(geotop_ds.layer):, :, :] = regis_ds['kh'].data[layer_no + 1:]

    kv[:len(geotop_ds.layer), :, :] = geotop_ds['kv'].data
    kv[len(geotop_ds.layer):, :, :] = regis_ds['kv'].data[layer_no + 1:]

    regis_geotop_ds['top'] = top
    regis_geotop_ds['bot'] = bot
    regis_geotop_ds['kh'] = kh
    regis_geotop_ds['kv'] = kv

    _ = [regis_geotop_ds.attrs.update({key: item})
         for key, item in regis_ds.attrs.items()]

    # maak top, bot, kh en kv nan waar de laagdikte 0 is
    mask = (regis_geotop_ds['top'] - regis_geotop_ds['bot']) < float_correction
    for key in ['top', 'bot', 'kh', 'kv']:
        regis_geotop_ds[key] = xr.where(mask, np.nan, regis_geotop_ds[key])
        regis_geotop_ds[key].attrs['source'] = 'REGIS/geotop'
        regis_geotop_ds[key].attrs['regis_url'] = regis_ds[key].url
        regis_geotop_ds[key].attrs['geotop_url'] = geotop_ds[key].url
        regis_geotop_ds[key].attrs['date'] = dt.datetime.now().strftime('%Y%m%d')

    return regis_geotop_ds


def fit_extent_to_regis(extent, delr, delc, cs_regis=100.):
    """redifine extent and calculate the number of rows and columns.

    The extent will be redefined so that the borders of the grid (xmin, xmax,
    ymin, ymax) correspond with the borders of the regis grid.

    Parameters
    ----------
    extent : list, tuple or np.array
        original extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    cs_regis : int or float, optional
        cell size of regis grid. The default is 100..

    Returns
    -------
    extent : list, tuple or np.array
        adjusted extent
    nrow : int
        number of rows.
    ncol : int
        number of columns.
    """
    if isinstance(extent, list):
        extent = extent.copy()
    elif isinstance(extent, (tuple, np.ndarray)):
        extent = list(extent)
    else:
        raise TypeError(
            f'expected extent of type list, tuple or np.ndarray, got {type(extent)}')

    logger.info(f'redefining current extent: {extent}, fit to regis raster')

    for d in [delr, delc]:
        if float(d) not in [10., 20., 25., 50., 100., 200., 400., 500., 800.]:
            raise NotImplementedError(f'you probably cannot run the model with this '
                                      f'cellsize -> {delc, delr}')

    # if xmin ends with 100 do nothing, otherwise fit xmin to regis cell border
    if extent[0] % cs_regis != 0:
        extent[0] -= extent[0] % cs_regis

    # get number of columns
    ncol = int(np.ceil((extent[1] - extent[0]) / delr))
    extent[1] = extent[0] + (ncol * delr)  # round xmax up to close grid

    # if ymin ends with 100 do nothing, otherwise fit ymin to regis cell border
    if extent[2] % cs_regis != 0:
        extent[2] -= extent[2] % cs_regis

    nrow = int(np.ceil((extent[3] - extent[2]) / delc))  # get number of rows
    extent[3] = extent[2] + (nrow * delc)  # round ymax up to close grid

    logger.info(f'new extent is {extent} model has {nrow} rows and {ncol} columns')

    return extent, nrow, ncol


def get_non_nan_layers(raw_layer_mod, data_var='bot'):
    """get number and name of layers based on the number of non-nan layers.

    Parameters
    ----------
    raw_layer_mod : xarray.Dataset
        dataset with raw layer model from regis or geotop.
    data_var : str
        data var that is used to check if layer mod contains nan values

    Returns
    -------
    nlay : int
        number of active layers within regis_ds_raw.
    lay_sel : list of str
        names of the active layers.
    """
    logger.info('find active layers in raw layer model')

    bot_raw_all = raw_layer_mod[data_var]
    lay_sel = []
    for lay in bot_raw_all.layer.data:
        if not bot_raw_all.sel(layer=lay).isnull().all():
            lay_sel.append(lay)
    nlay = len(lay_sel)

    logger.info(f'there are {nlay} active layers within the extent')

    return nlay, lay_sel


def get_layer_names():
    """get all the available regis layer names.

    Returns
    -------
    layer_names : np.array
        array with names of all the regis layers.
    """

    regis_url = 'http://www.dinodata.nl:80/opendap/REGIS/REGIS.nc'
    layer_names = xr.open_dataset(regis_url).layer.values

    return layer_names


def extrapolate_regis(regis_ds):
    """Fill missing data in layermodel based on nearest interpolation.

    Used for ensuring layer model contains data everywhere. Useful for
    filling in data beneath the sea for coastal groundwater models.

    Parameters
    ----------
    regis_ds : xarray.DataSet
        REGIS DataSet

    Returns
    -------
    regis_ds : xarray.DataSet
        filled REGIS layermodel with nearest interpolation
    """
    # fill layermodel with nearest interpolation (usually for filling in data
    # under the North Sea)
    mask = np.isnan(regis_ds['top']).all('layer')
    if not np.any(mask):
        # all of the model are is inside
        logger.info("No missing data to extrapolate")
        return regis_ds
    x, y = np.meshgrid(regis_ds.x, regis_ds.y)
    points = (x[~mask], y[~mask])
    xi = (x[mask], y[mask])
    for key in list(regis_ds.keys()):
        data = regis_ds[key].data
        for lay in range(len(regis_ds.layer)):
            values = data[lay][~mask]
            data[lay][mask] = griddata(points, values, xi, method='nearest')
    return regis_ds
