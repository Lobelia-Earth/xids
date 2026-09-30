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

def get_dtrx(tasmax: xr.DataArray, tasmin: xr.DataArray, freq: str = '1YS') -> xr.DataArray:
    dtr = tasmax - tasmin
    return dtr.resample(time=freq).max()


def get_cooldd(tasmin: xr.DataArray, tas: xr.DataArray, tasmax: xr.DataArray, thresh: Union[float, List[float], List[xr.DataArray]], freq = '1YS') -> xr.DataArray:
    # Spinoni et al. 2018
    if isinstance(thresh, list):  # Check if thresholds is a list
        thresh_tasmin, thresh_tas, thresh_tasmax = thresh
    else:
        thresh_tasmin = thresh
        thresh_tas = thresh
        thresh_tasmax = thresh
    cdd_data1 = 0*tas
    cdd_data2 = (tasmax - thresh_tasmax) / 4
    cdd_data3 = ((tasmax - thresh_tasmax) / 2) - ((thresh_tasmin-tasmin) / 4)
    cdd_data4 = (tas - thresh_tas)

    cdd_data = cdd_data1
    cdd_data = xr.where((tas <= thresh_tas) & (thresh_tasmax < tasmax), cdd_data2, cdd_data)
    cdd_data = xr.where((tasmin < thresh_tasmin) & (thresh_tas < tas), cdd_data3, cdd_data)
    cdd_data = xr.where((tasmin >= thresh_tasmin), cdd_data4, cdd_data)

    return cdd_data.resample(time=freq).sum()
