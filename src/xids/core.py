

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from dask.diagnostics.progress import ProgressBar
from typing import Callable, Union, List, Tuple, Dict, Any
import inspect
from xids import utils


class IDS:
    def __init__(
        self,
        hist_period: Union[List[str], Tuple[str, str]],
        fut_period: Union[List[List[str]], List[Tuple[str, str]]]
    ):
        self.hist_period = hist_period
        self.fut_period = [fut_period] if isinstance(fut_period[0], str) else fut_period



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
        model_data = utils.unchunk_time(model_data)
        reference_data = utils.unchunk_time(reference_data)

        with ProgressBar():

            if correct_threshold:
                thresh_corrected = utils.correct_threshold(utils.slice_data(model_data, self.hist_period), utils.slice_data(reference_data, self.hist_period), **kwargs)
                kwargs_model = {'thresh': thresh_corrected}
            else:
                kwargs_model = kwargs
            kwargs_ref = kwargs
            
            # 1. Historical computations
            ind_ref = self._eval_indicator(
                indicator_func, utils.slice_data(reference_data, self.hist_period), *args, freq=freq, aggregation=aggregation, **kwargs_ref,
            )
            ind_hist0 = self._eval_indicator(
                indicator_func, utils.slice_data(model_data, self.hist_period), *args, freq=freq, aggregation=aggregation, **kwargs_model,
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
                    indicator_func, utils.slice_data(model_data, f_pi), *args, **kwargs_model, freq=freq, aggregation=aggregation
                )
                if compute:
                    ind_fut0 = ind_fut0.compute()

                ind_fut, delta = utils.apply_delta_scaling(ind_hist, ind_hist0, ind_fut0, delta_mode=delta_mode, delta_factor_limit=delta_factor_limit, compute=compute)
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