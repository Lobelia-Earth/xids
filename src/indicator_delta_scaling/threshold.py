from typing import Literal

import numpy as np
import xarray as xr
from xclim.core.calendar import percentile_doy

DeltaMode= Literal['absolute', 'relatice']


# @keep_spatial_chunking
def get_quantile_threshold(da: xr.DataArray, quantile: float):
    """Get threshold based on quantile over time and ensure same chunking as input `da`.

    Parameters
    ----------
    da: xarray.DataArray
        Data to compute the threshold from
    quantile: float
        Quantile to be computed overthe time dimension

    Returns
    -------
    xarray.DataArray, [same units as da]
        Pixel based threshold map
    """
    return da.quantile(q=quantile, dim="time", keep_attrs=True, skipna=True).chunk(
        longitude=-1, latitude=-1
    )


# @keep_spatial_chunking
def get_doy_quantile_threshold(da: xr.DataArray, quantile: float, window=5):
    """Get threshold per dayofyear for `quantile` and a rolling `window`.

    Method 7 of Hyndman and Fan (1996) is used to compute the quantiles by providing alpha
    and beta = 1 to the percentile_doy function. This is equivalent to the default of
    xr.DataArray.quantile.

    Parameters
    ----------
    da: xarray.DataArray
        Data to compute the threshold from, must contain a `time` dimension.
    quantile: float
        Quantile to be computed over the time dimension
    window: int
        Rolling window size in days to compute the quantile over.

    Returns
    -------
    xarray.DataArray, [same units as da]
        Per day of year `quantile`.
    """
    return percentile_doy(
        da.chunk(time=-1), per=quantile * 100, window=window, alpha=1, beta=1
    ).isel(percentiles=0, drop=True)


def _get_quantile(values: np.ndarray, threshold: float):
    """Get the according quantile for a threshold within an array of values.

    First find the position of the closest value to the threshold. If there is more than one
    exact match, the function finds the first and last position of the exact matches
    and returns the middle position's quantile. Otherwise, compute the quantiles
    left and right of the threshold and linearly interpolate to the thresholds position.
    """
    sorted_array = np.sort(values)
    if threshold < sorted_array[0]:
        return 0
    if threshold >= sorted_array[-1]:
        return 1

    n = values.size - 1
    first_pos = np.searchsorted(sorted_array, threshold, side="left")
    last_pos = np.searchsorted(sorted_array, threshold, side="right")
    if first_pos != last_pos:
        return (first_pos + last_pos - 1) / 2 / n

    left_quantile = (first_pos - 1) / n
    right_quantile = first_pos / n

    return np.interp(
        threshold,
        [sorted_array[first_pos - 1], sorted_array[first_pos]],
        [left_quantile, right_quantile],
    )


def get_quantile(da: xr.DataArray, threshold: float) -> xr.DataArray:
    """Get the according quantile for a threshold over the time dimension.

    Parameters
    ----------
    da: xarray.DataArray
        Timeseries data to compute quantiles for a `threshold`
    threshold: float
        Absolute threshold to compute the quantile for
    Returns
    -------
    xarray.DataArray
        Quantile for each pixel representing the `threshold`
    """
    return xr.apply_ufunc(
        _get_quantile,
        da,
        threshold,
        input_core_dims=[["time"], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def map_quantiles(da: xr.DataArray, quantiles: xr.DataArray) -> xr.DataArray:
    """Compute the value corresponding to the given quantile over the time dimension.

    Parameters
    ----------
    da: xarray.DataArray
        Timeseries data to map quantile to
    quantiles: xarray.DataArray
        Map of quantiles to compute the values for, must be have same dimensions as `da` except for the time dimension
    Returns
    -------
    xarray.DataArray
        Values corresponding to the quantiles for each pixel, same dimensions as `quantiles`
    """
    mask = quantiles.notnull()
    quantiles = quantiles.where(mask, 0)
    values = xr.apply_ufunc(
        np.nanquantile,
        da,
        quantiles,
        input_core_dims=[["time"], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    return values.where(mask)


def get_quantile_correction(da: xr.DataArray, quantiles: xr.DataArray):
    """Compute minimum or maximum for quantiles 0 or 1 respectively.

    Parameters
    ----------
    da: xarray.DataArray
        Reference timeseries data to compute the quantile correction for
    quantiles: xarray.DataArray
        Map of quantiles


    Returns
    -------
    xarray.DataArray
        Minimum or maximum value for quantiles 0 or 1 respectively.
    """
    return xr.where(
        quantiles == 0, da.min("time"), xr.where(quantiles == 1, da.max("time"), np.nan)
    )


def apply_quantile_correction(
    threshold_map: xr.DataArray,
    correction: xr.DataArray,
    threshold: float,
    mode:DeltaMode,
):
    """Apply the correction to the threshold map.

    Parameters
    ----------
    threshold_map: xarray.DataArray
        Map of thresholds to apply the correction to.
    correction: xarray.DataArray
        Map of corrections for the thresholds, must be same dimensions as `threshold_map`
    threshold: float
        Absolute threshold to apply the correction to.
    mode: DeltaMode
        Mode to apply the correction additively or multiplicatively.

    Returns
    -------
    xarray.DataArray
        Threshold map with the correction applied, same dimensions as `threshold_map`.
    """
    if mode == "absolute":
        return threshold_map.where(
            correction.isnull(),
            threshold_map + (threshold - correction),
        )

    elif mode == "relative":
        return threshold_map.where(
            correction.isnull(),
            threshold_map * (threshold / correction),
        )


TYPE_TO_FUNC = {
    "quantile": get_quantile_threshold,
    "doy_quantile": get_doy_quantile_threshold,
}


def get_absolute_threshold(
    ref: xr.DataArray,
    hist: xr.DataArray,
    threshold_value: float,
    threshold_units: str,
    threshold_mode: DeltaMode,
) -> xr.DataArray:
    quantiles = get_quantile(ref, threshold_value)
    correction = get_quantile_correction(ref, quantiles)
    threshold_q = map_quantiles(hist, quantiles)

    return (
        apply_quantile_correction(
            threshold_q,
            correction,
            threshold_value,
            threshold_mode,
        )
        # For wet_spell_max_length the threshold is in mm but the data comes in as mm day-1
        # So we force the correct units.
        .assign_attrs(units=threshold_units)
    )

