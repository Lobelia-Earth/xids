

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from dask.diagnostics.progress import ProgressBar
from typing import Callable, Union, List, Tuple, Dict, Any
import inspect


class IDS:
    def __init__(
        self,
        hist_period: Union[List[str], Tuple[str, str]],
        fut_period: Union[List[List[str]], List[Tuple[str, str]]]
    ):
        self.hist_period = hist_period
        self.fut_period = [fut_period] if isinstance(fut_period[0], str) else fut_period

    def unchunk_time(self, data: Union[xr.DataArray, List[xr.DataArray], Tuple[xr.DataArray]]) -> Any:
        """Recursively unchunks DataArrays across time for single arrays, lists, or tuples."""
        if isinstance(data, xr.DataArray):
            return data.chunk({'time': -1})
        elif isinstance(data, (list, tuple)):
            # Retain original container type (list or tuple)
            return type(data)(self.unchunk_time(v) for v in data)
        else:
            raise TypeError(f"Unsupported data container type: {type(data)}")


    def _slice_data(
        self, 
        data: Union[xr.DataArray, List[xr.DataArray], Tuple[xr.DataArray]],
        period: Union[List[str], Tuple[str, str]]
    ) -> Any:
        """Recursively slices DataArrays across time for single arrays, lists, tuples, or dicts."""
        time_slice = slice(str(period[0]), str(period[1]))
        
        if isinstance(data, xr.DataArray):
            return data.sel(time=time_slice)
        elif isinstance(data, (list, tuple)):
            # Retain original container type (list or tuple)
            return type(data)(self._slice_data(v, period) for v in data)
        else:
            raise TypeError(f"Unsupported data container type: {type(data)}")

    def _eval_indicator(
        self,
        func: Callable,
        data: Union[xr.DataArray, List[xr.DataArray], Tuple[xr.DataArray]],
        freq: str = 'YS',
        aggregation: str = 'mean',
        dim_aggr: str = 'time',
        *args: tuple,
        **kwargs: dict,
    ) -> xr.DataArray:

        # 1. Single DataArray -> pass as 1st positional argument
        if isinstance(data, xr.DataArray):
            result = func(data, *args, freq = freq, **kwargs)
        # 2. List or Tuple -> unpack positionally (*sliced_data)
        elif isinstance(data, (list, tuple)):
            result = func(*data, *args,freq = freq, **kwargs)
        else:
            raise ValueError("Unsupported input data format.")

        if aggregation == 'mean':
            return result.mean(dim_aggr)
        elif aggregation is None:
            return result
    
    def _add_attributes(self, data: xr.DataArray, short_name: str = None, long_name: str = None, units: str = None, delta_mode: str = None, delta_factor_percentage: bool = True) -> xr.DataArray:
        """Adds attributes to a DataArray."""
        if delta_mode == '+':
            delta_extra = 'Absolute change in '
        elif delta_mode == '*':
            delta_extra = 'Relative change in '
            if delta_factor_percentage:
                units = '%'
            else:
                units = '1'  # Dimensionless for relative change
        else:
            delta_extra = ''
        if short_name:
            data.name = f'{delta_extra} {short_name}'
        if long_name:
            data.attrs['long_name'] = f'{delta_extra} {long_name}'
        if units:
            data.attrs['units'] = units
        return data
    

    def _get_quantile_of_value(self, da, value, dim='time'):
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

    def correct_threshold(self,
                          model_data: xr.DataArray, 
                          reference_data: xr.DataArray, 
                          thresh: float, 
                          delta_mode: str = '+', ):

        # Get the quantile in data_reference associated to the specified threshold
        if isinstance(thresh, str):
            threshold = float(thresh.split(' ')[0])
        else:
            threshold = thresh
        q_th = self._get_quantile_of_value(reference_data, threshold, )
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



    
    def apply_delta_scaling(self,
                            ind_hist: xr.DataArray, # Broadcasted historical reference
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

    def compute_indicator(
        self,
        indicator_func: Callable,
        reference_data: Union[xr.DataArray, List[xr.DataArray], Tuple[xr.DataArray]],
        model_data: Union[xr.DataArray, List[xr.DataArray], Tuple[xr.DataArray]],
        *args,
        delta_mode: str = '+',
        freq: str = 'YS',
        correct_threshold: bool = False,
        aggregation: str = 'mean',
        short_name: str = None,
        long_name: str = None,
        units: str = None,
        compute: bool = True,
        delta_factor_limit: float = None,
        delta_factor_percentage: bool = True,
        **kwargs
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """Computes bias-corrected indicators supporting positional tuple/list input variables."""

        # Ensure data has the time dimension unchunked
        model_data = self.unchunk_time(model_data)
        reference_data = self.unchunk_time(reference_data)

        with ProgressBar():

            if correct_threshold:
                thresh_corrected = self.correct_threshold(self._slice_data(model_data, self.hist_period), self._slice_data(reference_data, self.hist_period), **kwargs)
                kwargs_model = {'thresh': thresh_corrected}
            else:
                kwargs_model = kwargs
            kwargs_ref = kwargs
            
            # 1. Historical computations
            ind_ref = self._eval_indicator(
                indicator_func, self._slice_data(reference_data, self.hist_period), *args, freq=freq, aggregation=aggregation, **kwargs_ref,
            )
            ind_hist0 = self._eval_indicator(
                indicator_func, self._slice_data(model_data, self.hist_period), *args, freq=freq, aggregation=aggregation, **kwargs_model, 
            )
            ind_hist = ind_ref.broadcast_like(ind_hist0)

            if compute:
                ind_ref = ind_ref.compute()
                ind_hist = ind_hist.compute()

            # 2. Future delta-scaling computations
            fut_list = []
            delta_list = []
            for f_pi in self.fut_period:
                ind_fut0 = self._eval_indicator(
                    indicator_func, self._slice_data(model_data, f_pi), *args, **kwargs_model, freq=freq, aggregation=aggregation
                )
                if compute:
                    ind_fut0 = ind_fut0.compute()

                ind_fut, delta = self.apply_delta_scaling(ind_hist, ind_hist0, ind_fut0, delta_mode=delta_mode, delta_factor_limit=delta_factor_limit, compute=compute)
                f_pi_str = f"{f_pi[0]}-{f_pi[1]}"
                
                ind_fut = ind_fut.assign_coords(period=f_pi_str).expand_dims('period')
                fut_list.append(ind_fut)

                delta_fut = delta.assign_coords(period=f_pi_str).expand_dims('period')
                delta_list.append(delta_fut)

            ind_fut_combined = xr.concat(fut_list, dim='period') if len(fut_list) > 1 else fut_list[0]
            delta_combined = xr.concat(delta_list, dim='period') if len(delta_list) > 1 else delta_list[0]

            # Add period coordinate to historical indicators
            h_pi_str = f"{self.hist_period[0]}-{self.hist_period[1]}"
            ind_ref = ind_ref.assign_coords(period=h_pi_str).expand_dims('period')
            ind_hist = ind_hist.assign_coords(period=h_pi_str).expand_dims('period')

            if compute:
                ind_ref = ind_ref.compute()
                ind_hist = ind_hist.compute()
                delta_combined = delta_combined.compute()
                ind_fut_combined = ind_fut_combined.compute()

            if (short_name is not None) | (long_name is not None) | (units is not None):
                ind_ref = self._add_attributes(ind_ref, short_name=short_name, long_name=long_name, units=units)
                ind_hist = self._add_attributes(ind_hist, short_name=short_name, long_name=long_name, units=units)
                ind_fut_combined = self._add_attributes(ind_fut_combined, short_name=short_name, long_name=long_name, units=units)
                delta_combined = self._add_attributes(delta_combined, short_name=short_name, long_name=long_name, units=units, delta_mode = delta_mode, delta_factor_percentage = delta_factor_percentage)

            return ind_ref, ind_hist, ind_fut_combined, delta_combined