'''
This script is intended to calculate the entropy deficit from Met Office UM 
ensemble data rather than ECMWF reanalysis climatologies.
'''
import metpy.calc as mpc
import numpy as np, xarray as xr
from metpy.units import units
import iris


def get_endfc(scheme, en=0, ofile=True):
    '''
    Calculate Entropy deficit chi_m defined by Tang and Emanuel (2012b)
    Save the result in form of Xarray.Dataset to an nc file.
    
    Link: https://journals.ametsoc.org/doi/full/10.1175/BAMS-D-11-00165.1
    '''
    cp = 1005.7 # specific heat at constant pressure for dry air (J * kg^-1* k^-1)
    rd = 287.05 # gas constant for dry air (J * kg^-1 * k^-1)
    rv = 461.51 # gas constant for water vapor (J * kg^-1 * k^-1)
    lv = 2.555e6 # latent heat of vaporization (J * kg^-1)
    
    fn = "/nobackup/eeajo/{0}/em{1}/".format(scheme,en)
    
    sst = iris.load_cube("{0}umnsaa_pa012.*".format(fn),"surface_temperature").data[0] # K 
    sstc = sst - 273.15 # °C
    slp = iris.load_cube("{0}umnsaa_pa012.*".format(fn),"air_pressure_at_sea_level").data[0] # Pa
    
    q = iris.load_cube("{0}umnsaa_pd012.*".format(fn),"specific_humidity").data
    tk = iris.load_cube("{0}umnsaa_pd012.*".format(fn),"air_temperature").data # temperatures (K)
    p = iris.load_cube("{0}umnsaa_pd012.*".format(fn),"air_pressure").data # temperatures (K)
    
    tc = tk - 273.15 # temperatures (°C)
    rh = np.array(mpc.relative_humidity_from_specific_humidity(pressure = p * units("Pa"),
                                                      temperature = tk * units("K"),
                                                      specific_humidity = q * units("kg/kg")))
    
    # ------------------------------------
    # Saturation moist entropy at sea surface temperature
    e_surf_sat = 610.94*np.exp((17.625*sstc)/(243.04+sstc)) # saturated vapor pressure at SST (Pa)
    mr_surf_sat = 0.622*e_surf_sat/(slp-e_surf_sat) # saturated mixing ratio at SST
    s_surf_sat = cp*np.log(sst)-rd*np.log(slp-e_surf_sat)+lv*mr_surf_sat/sst # J * kg^-1* k^-1
    
    # ------------------------------------
    # Moist entropy for boundary layer (950 hPa)
    tk_bl = np.mean(tk[0:20],axis=0) # temperature in BL, mean
    tc_bl = np.mean(tc[0:20],axis=0) # temperature (degC) in BL, mean
    p_bl = np.mean(p[0:20],axis=0) # pressure in BL
    q_bl = np.mean(q[0:20],axis=0)
    e_bl_sat = 610.94*np.exp((17.625*tc_bl)/(243.04+tc_bl)) # saturated vapor pressure for boundary layer (Pa)
    rh_bl = np.mean(rh,axis=0) # relative humidity for boundary layer
    e_bl = e_bl_sat*rh_bl # vapor pressure for boundary layer (Pa)
    mr_bl = 0.622*e_bl/(p_bl-e_bl) # mixing ratio for boundary layer
    s_bl = cp*np.log(tk_bl)-rd*np.log(p_bl-e_bl)+lv*mr_bl/tk_bl-rv*mr_bl*np.log(rh_bl) # J * kg^-1* k^-1
    
    # ------------------------------------
    # Saturation moist entropy at mid-level (600hPa) in the inner core of the TC
    # here I use ~725 to 675 hPa, or ~ 2.6 to 3 km altitude
    tk_m = np.mean(tk[26:29],axis=0) # temperature at mid-level (k)
    tc_m = np.mean(tc[26:29],axis=0) # temperature at mid-level (°C)
    p_m = np.mean(p[26:29],axis=0) # pressure at mid-level
    e_m_sat = 610.94*np.exp((17.625*tc_m)/(243.04+tc_m)) # saturated vapor pressure at mid-level (Pa)
    mr_m_sat = 0.622*e_m_sat/(p_m-e_m_sat) # saturated mixing ratio for boundary layer
    s_m_sat = cp*np.log(tk_m)-rd*np.log(p_m-e_m_sat)+lv*mr_m_sat/tk_m # J * kg^-1* k^-1
    
    # ------------------------------------
    # Moist entropy at mid-level (600hPa) in the environment
    q_m = np.mean(q[26:29],axis=0)
    rh_m = np.mean(rh[26:29],axis=0) # relative humidity for mid-level
    e_m = e_m_sat*rh_m # vapor pressure for mid-level (Pa)
    mr_m = 0.622*e_m/(p_m-e_m) # mixing ratio for mid-level
    #mr_m = q_m/(1-q_m)
    s_m = cp*np.log(tk_m)-rd*np.log(p_m-e_m)+lv*mr_m/tk_m-rv*mr_m*np.log(rh_m) # J * kg^-1* k^-1
    
    # ------------------------------------
    # Entropy deficit
    chi_m = (s_m_sat-s_m)/(s_surf_sat-s_bl)
    
    return(chi_m)
    # wrap into an xarray.DataArray
    chi_m = xr.DataArray(chi_m,
                   attrs=dict(long_name='Entropy Deficit', units='J * kg^-1* k^-1')
                   )
    # wrap into an xarray.Dataset
    ds = xr.Dataset(dict(chi_m = chi_m),
                    attrs={'chi_m':'Entropy Deficit'})
    
    if ofile is None:
        return ds
    else: # save ds to ofile
        nc_dtype = {'dtype': 'float32'}
        encoding = dict(chi_m = nc_dtype)
        ds.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')


def main():
    scheme="original_cblop3"; en=0
    outfile = '/nobackup/eeajo/{0}/entropy_deficit{1}.nc'.format(scheme,en)
    get_endfc(scheme,en,ofile = outfile)


if __name__ == '__main__':
    main()
