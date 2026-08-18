import numpy as np
import xarray as xr
import xclim
from typing import Callable, Union, List, Tuple, Dict, Any
from xids import utils


def get_windpowerdensity(sfcWind: xr.DataArray, freq: str = 'YS') -> xr.DataArray:
    power = 3
    wpd = 0.5*1.225*(sfcWind**power).resample(time=freq).mean()
    return wpd


def get_percentile_value(data: xr.DataArray, quantile: float = 0.9, freq: str = 'YS') -> xr.DataArray:
    data_resampled = data.resample(time=freq)
    percentile_value = data_resampled.quantile(q=quantile)
    return percentile_value

def get_drought_sev(spi: xr.DataArray, threshold: float = -1, freq: str = 'YS') -> xr.DataArray:
    # Check xclim.indices.generic.spell_mask to later compute severity mean, max or sum
    spi_drought = xr.where(spi < threshold, spi, 0).resample(time=freq).sum()
    return spi_drought

def get_drought_dur(spi: xr.DataArray, threshold: float = -1, freq: str = 'YS') -> xr.DataArray:
    # Check xclim.indices.generic.spell_length_statistics instead
    spi_drought = xr.where(spi < threshold, 1, 0).resample(time=freq).sum()
    return spi_drought
