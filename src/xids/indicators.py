import numpy as np
import xarray as xr
import xclim

def get_windpowerdensity(wind, freq = 'YS'):
    power = 3
    wind_power = 0.5*1.225*(wind**power).resample(time=freq).mean()
    return wind_power


def get_fwiXp(tas, pr, wind, hurs, lat, quantile=0.9, freq = 'YS'):
    indices = xclim.indices.fire.cffwis_indices(tas, pr, wind, hurs, lat)
    fwi = indices[-1]
    fwiXp = fwi.resample(time=freq).quantile(q=quantile)
    return fwiXp