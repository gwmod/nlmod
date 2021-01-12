# -*- coding: utf-8 -*-
"""
function to project regis, or a combination of regis and geotop, data on a 
modelgrid.
"""
import numpy as np
import xarray as xr

from nlmod import mgrid, geotop, util


def get_layer_models(extent, delr, delc,
                     use_regis=True, use_geotop=True,
                     cachedir=None,
                     fname_netcdf='combined_layer_ds.nc',
                     use_cache=False, verbose=False):
    """ get a layer model from regis and/or geotop

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
    use_regis : bool, optional
        True if part of the layer model should be REGIS. The default is True.
    use_geotop : bool, optional
        True if part of the layer model should be geotop. The default is True.
    cachedir : str
        directory to store cached values, if None a temporary directory is
        used. default is None
    fname_netcdf : str, optional
        name of the cached netcdf file. The default is 'combined_layer_ds.nc'.
    use_cache : bool, optional
        if True the cached resampled regis dataset is used.
        The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    combined_ds : xarary dataset
        layer model dataset.

    """

    combined_ds = util.get_cache_netcdf(use_cache, cachedir, fname_netcdf,
                                        get_combined_layer_models,
                                        verbose=verbose,
                                        extent=extent,
                                        delr=delr, delc=delc,
                                        use_regis=use_regis,
                                        use_geotop=use_geotop)

    return combined_ds


def get_combined_layer_models(extent, delr, delc,
                              use_regis=True, use_geotop=True,
                              cachedir=None, use_cache=False, 
                              verbose=False):
    """ combine layer models into a single layer model. 

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
    use_regis : bool, optional
        True if part of the layer model should be REGIS. The default is True.
    use_geotop : bool, optional
        True if part of the layer model should be geotop. The default is True.
    cachedir : str
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached resampled regis dataset is used.
        The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

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
        regis_ds = get_regis_dataset(extent, delr, delc,
                                     cachedir, use_cache=use_cache,
                                     verbose=verbose)
    else:
        raise ValueError('layer models without REGIS not supported')

    if use_geotop:
        geotop_ds = geotop.get_geotop_dataset(extent, delr, delc, regis_ds,
                                              cachedir=cachedir,
                                              use_cache=use_cache,
                                              verbose=verbose)

    if use_regis and use_geotop:
        regis_geotop_ds = add_geotop_to_regis_hlc(regis_ds, geotop_ds,
                                                  verbose=verbose)

        combined_ds = regis_geotop_ds
    elif use_regis:
        combined_ds = regis_ds
    else:
        raise ValueError('combination of model layers not supported')

    return combined_ds


def get_regis_dataset(extent, delr, delc, cachedir, use_cache=False,
                      verbose=False):
    """ get a regis dataset projected on the modelgrid


    Parameters
    ----------
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    cachedir : str
        directory to store cached values, if None a temporary directory is
        used. default is None
    use_cache : bool, optional
        if True the cached resampled regis dataset is used.
        The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    regis_ds : xarray dataset
        dataset with regis data projected on the modelgrid.

    """
    # get local regis dataset
    regis_url = 'http://www.dinodata.nl:80/opendap/REGIS/REGIS.nc'
    regis_ds_raw = xr.open_dataset(regis_url).sel(x=slice(extent[0], extent[1]),
                                                  y=slice(extent[2], extent[3]))
    regis_ds_raw = regis_ds_raw[['top', 'bottom', 'kD', 'c', 'kh', 'kv']]

    regis_ds_raw = regis_ds_raw.rename({'layer': 'layer_old'})
    regis_ds_raw.coords['layer'] = regis_ds_raw.layer_old.astype(str)  # could also use assign_coords
    regis_ds_raw2 = regis_ds_raw.swap_dims({'layer_old': 'layer'})

    # convert regis dataset to grid
    regis_ds = mgrid.get_ml_layer_dataset_struc(raw_ds=regis_ds_raw2,
                                                extent=extent,
                                                delr=delr, delc=delc,
                                                cachedir=cachedir,
                                                fname_netcdf='regis_ds.nc',
                                                use_cache=use_cache,
                                                verbose=verbose)

    return regis_ds


def add_geotop_to_regis_hlc(regis_ds, geotop_ds, 
                            float_correction=0.001,
                            verbose=False):
    """ Combine geotop and regis in such a way that the holoceen in Regis is
    replaced by the geo_eenheden of geotop.

    Parameters
    ----------
    regis_ds: xarray.DataSet
        regis dataset
    geotop_ds: xarray.DataSet
        regis dataset
    float_correction: float
        due to floating point precision some floating point numbers that are
        the same are not recognised as the same. Therefore this correction is
        used.
    verbose : bool, optional
        print additional information. default is False

    Returns
    -------
    regis_geotop_ds: xr.DataSet
        combined dataset  


    """
    regis_geotop_ds = xr.Dataset()

    new_layers = np.append(geotop_ds.layer.data, regis_ds.layer.data[1:].astype('<U8')).astype('O')

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
    if verbose:
        print('cut geotop layer based on regis holoceen')
    for lay in range(geotop_ds.dims['layer']):
        # Alle geotop cellen die onder de onderkant van het holoceen liggen worden inactief
        mask1 = geotop_ds['top'][lay] <= (regis_ds['bottom'][0] - float_correction)
        geotop_ds['top'][lay] = xr.where(mask1, np.nan, geotop_ds['top'][lay])
        geotop_ds['bottom'][lay] = xr.where(mask1, np.nan, geotop_ds['bottom'][lay])

        # Alle geotop cellen waarvan de bodem onder de onderkant van het holoceen ligt, krijgen als bodem de onderkant van het holoceen
        mask2 = geotop_ds['bottom'][lay] < regis_ds['bottom'][0]
        geotop_ds['bottom'][lay] = xr.where(mask2 * (~mask1), regis_ds['bottom'][0], geotop_ds['bottom'][lay])

        # Alle geotop cellen die boven de bovenkant van het holoceen liggen worden inactief
        mask3 = geotop_ds['bottom'][lay] >= (regis_ds['top'][0] - float_correction)
        geotop_ds['top'][lay] = xr.where(mask3, np.nan, geotop_ds['top'][lay])
        geotop_ds['bottom'][lay] = xr.where(mask3, np.nan, geotop_ds['bottom'][lay])

        # Alle geotop cellen waarvan de top boven de top van het holoceen ligt, krijgen als top het holoceen van regis
        mask4 = geotop_ds['top'][lay] >= regis_ds['top'][0]
        geotop_ds['top'][lay] = xr.where(mask4 * (~mask3), regis_ds['top'][0], geotop_ds['top'][lay])

        # overal waar holoceen inactief is, wordt geotop ook inactief
        mask5 = regis_ds['bottom'][0].isnull()
        geotop_ds['top'][lay] = xr.where(mask5, np.nan, geotop_ds['top'][lay])
        geotop_ds['bottom'][lay] = xr.where(mask5, np.nan, geotop_ds['bottom'][lay])
        if verbose:
            if (mask2 * (~mask1)).sum() > 0:
                print(f'regis holoceen snijdt door laag {geotop_ds.layer[lay].values}')

    top[:len(geotop_ds.layer), :, :] = geotop_ds['top'].data
    top[len(geotop_ds.layer):, :, :] = regis_ds['top'].data[1:]

    bot[:len(geotop_ds.layer), :, :] = geotop_ds['bottom'].data
    bot[len(geotop_ds.layer):, :, :] = regis_ds['bottom'].data[1:]

    kh[:len(geotop_ds.layer), :, :] = geotop_ds['kh'].data
    kh[len(geotop_ds.layer):, :, :] = regis_ds['kh'].data[1:]

    kv[:len(geotop_ds.layer), :, :] = geotop_ds['kv'].data
    kv[len(geotop_ds.layer):, :, :] = regis_ds['kv'].data[1:]

    regis_geotop_ds['top'] = top
    regis_geotop_ds['bottom'] = bot
    regis_geotop_ds['kh'] = kh
    regis_geotop_ds['kv'] = kv

    _ = [regis_geotop_ds.attrs.update({key: item})
                 for key, item in regis_ds.attrs.items()]

    # maak bottom nan waar de laagdikte 0 is
    regis_geotop_ds['bottom'] = xr.where((regis_geotop_ds['top'] - regis_geotop_ds['bottom']) < float_correction,
                                          np.nan,
                                          regis_geotop_ds['bottom'])

    return regis_geotop_ds


def fit_extent_to_regis(extent, delr, delc, cs_regis=100.,
                        verbose=False):
    """
    redifine extent and calculate the number of rows and columns.

    The extent will be redefined so that the borders os the grid (xmin, xmax, 
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
    if verbose:
        print(f'redefining current extent: {extent}, fit to regis raster')

    for d in [delr, delc]:
        if float(d) not in [10., 20., 25., 50., 100., 200., 400., 500., 800.]:
            print(f'you probably cannot run the model with this '
                  f'cellsize -> {delc, delr}')

    # if extents ends with 50 do nothing, otherwise rescale extent to fit regis
    if extent[0] % cs_regis == 0 or not extent[0] % (0.5 * cs_regis) == 0:
        extent[0] -= extent[0] % 100
        extent[0] = extent[0] - 0.5 * cs_regis
    # get number of columns
    ncol = int(np.ceil((extent[1] - extent[0]) / delr))
    extent[1] = extent[0] + (ncol * delr)  # round x1 up to close grid

    # round y0 down to next 50 necessary for regis
    if extent[2] % cs_regis == 0 or not extent[2] % (0.5 * cs_regis) == 0:
        extent[2] -= extent[2] % 100
        extent[2] = extent[2] - 0.5 * cs_regis
    nrow = int(np.ceil((extent[3] - extent[2]) / delc))  # get number of rows
    extent[3] = extent[2] + (nrow * delc)  # round y1 up to close grid

    if verbose:
        print(
            f'new extent is {extent} model has {nrow} rows and {ncol} columns')

    return extent, nrow, ncol
