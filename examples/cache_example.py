import numpy as np
import xarray as xr
import nlmod

@nlmod.cache.cache_netcdf
def func_to_create_a_dataset(number):
    """ create a dataarray as an example for the caching method 
    
    Parameters
    ----------
    number : int, float
        values in data array
    
    Returns
    -------
    da : xr.DataArray    
    """
    arr = np.ones(100) * number
    
    da = xr.DataArray(arr, dims=('x'),
                      coords={'x':np.linspace(1,100,100)})
    ds = xr.Dataset()
    ds['test']=da
    
    return ds