import xarray as xr
import numpy as np
from typing import Callable, Union, List, Tuple, Dict, Any


def slice_data(
        data: Union[xr.DataArray, List[xr.DataArray], Tuple[xr.DataArray]],
        period: Union[List[str], Tuple[str, str]]
    ) -> Any:
        """Recursively slices DataArrays across time for single arrays, lists, tuples, or dicts."""
        time_slice = slice(str(period[0]), str(period[1]))
        
        if isinstance(data, xr.DataArray):
            return data.sel(time=time_slice)
        elif isinstance(data, (list, tuple)):
            # Retain original container type (list or tuple)
            return type(data)(slice_data(v, period) for v in data)
        else:
            raise TypeError(f"Unsupported data container type: {type(data)}")
        

def unchunk_time(data: Union[xr.DataArray, List[xr.DataArray], Tuple[xr.DataArray]]) -> Any:
        """Recursively unchunks DataArrays across time for single arrays, lists, or tuples."""
        if isinstance(data, xr.DataArray):
            return data.chunk({'time': -1})
        elif isinstance(data, (list, tuple)):
            # Retain original container type (list or tuple)
            return type(data)(unchunk_time(v) for v in data)
        else:
            raise TypeError(f"Unsupported data container type: {type(data)}")
        

def get_quantile_of_value(da, value, dim='time'):
    """
    Estimates the empirical quantile (0.0 to 1.0) of a given absolute value 
    along a specific dimension(s) in an xarray DataArray.
    """
    # 1. Create a mask to ignore NaN values (e.g., ocean pixels in land data)
    valid_mask = da.notnull()

    # 2. Count the total number of valid data points along the dimension
    n_total = valid_mask.sum(dim=dim)
    
    # 3. Count how many valid values are less than or equal to the target value
    #    (Using .where(valid_mask, False) ensures NaNs aren't accidentally counted)
    n_less_equal = (da <= value).where(valid_mask, False).sum(dim=dim)
    
    # 4. Calculate the quantile (proportion of values <= target)
    quantile = n_less_equal / n_total
    
    # Preserve original attributes for traceability 
    quantile.name = 'estimated_quantile'
    quantile.attrs['description'] = f'Quantile of value {value} along dim {dim}'
    
    return quantile


def correct_threshold(model_data: xr.DataArray, 
                        reference_data: xr.DataArray, 
                        thresh: float, 
                        delta_mode: str = '+', ):

    # Get the quantile in data_reference associated to the specified threshold
    if isinstance(thresh, str):
        threshold = float(thresh.split(' ')[0])
    else:
        threshold = thresh
    q_th = get_quantile_of_value(reference_data, threshold, )
    # Get the value in model_data corresponding to the quantile q_th associated to the specified "threshold" in data_reference
    th_q = xr.apply_ufunc(
        np.nanquantile,
        model_data,
        q_th,
        input_core_dims=[['time'], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )

    # In case threshold is out of range, return the actual threshold corrected by the distance to the max or min value
    ref_max = reference_data.max(dim='time')
    ref_min = reference_data.min(dim='time')
    if delta_mode == '+':
        th_q = xr.where((q_th == 1), th_q+(threshold-ref_max), th_q)
        th_q = xr.where((q_th == 0), th_q+(threshold-ref_min), th_q)
    elif delta_mode == '*':
        th_q = xr.where((q_th == 1), th_q*(threshold/ref_max), th_q)
        th_q = xr.where((q_th == 0), th_q*(threshold/ref_min), th_q)
    return th_q




def apply_delta_scaling(ind_hist: xr.DataArray, # Broadcasted historical reference
                        ind_hist0: xr.DataArray, # Historical model data
                        ind_fut0: xr.DataArray = None, # Future model data
                        delta_mode: str = '+',
                        delta_factor_limit: float = None,
                        delta_factor_percentage: bool = True,
                        compute: bool = True) -> xr.DataArray:
    
    """Applies additive or multiplicative delta scaling to future projections."""
    if delta_mode == '+':
        delta = ind_fut0 - ind_hist0
        ind_fut = ind_hist + delta
    elif delta_mode == '*':
        delta = ind_fut0 / ind_hist0
        if delta_factor_limit is not None:
            delta = delta.where(delta < delta_factor_limit, delta_factor_limit)
            print(f"For some locations, the Delta Change Factor has been limited to {delta_factor_limit}x.")
        ind_fut = ind_hist * delta
        if delta_factor_percentage:
            delta = (delta -1)*100
    else:
        raise ValueError("Delta mode must be '+' for additive or '*' for multiplicative scaling.")
    
    return ind_fut, delta