#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:23:51 2020

@author: pablo
"""


from astropy.io import fits
import numpy as np 
from matplotlib import pyplot as plt



class Pipe3Dssp(object):
    """This class provides information regarding the simple stellar population
    models used in MANGA"""
    
    ssps_path = '/home/pablo/obs_data/MANGA/Pipe3D/SSPs/gsd01_156.fits'        
    ssp_properties_path = '/home/pablo/obs_data/MANGA/Pipe3D/SSPs/BASE.gsd01'
    
    def __init__(self):
     
        hdul = fits.open(self.ssps_path)
        
        norm = np.ones(156)
        for i in range(156):
            norm[i] = hdul[0].header['NORM'+str(i)]
            
        self.ssp_SED = hdul[0].data*norm[:, np.newaxis] #Lsun
        
        wl0 = hdul[0].header['CRVAL1']
        deltawl = hdul[0].header['CDELT1']
        self.wl = np.arange(wl0, wl0+deltawl*self.ssp_SED.shape[1], deltawl)
        
        
    def ssp_ages(self, mode='individual'):
        """
        'individual' mode returns one value per ssp (156)
        'age' mode returns the age bins of all ssp's (39)        
        """
        modes = {'individual':np.arange(0, 156), 'age':np.arange(0,39)}
        self.ssp_age= np.loadtxt(self.ssp_properties_path, usecols=(1))
        return self.ssp_age[modes[mode]]
    
    def ssp_metallicity(self):
        self.ssp_met= np.loadtxt(self.ssp_properties_path, usecols=(2))
        return self.ssp_met
    
    def ssp_alive_stellar_mass(self, mode='individual'):
        self.ssp_alive_mass = np.loadtxt(self.ssp_properties_path, usecols=(4))
        self.ssp_alive_mass = self.ssp_alive_mass.reshape(4,39)
        
        if mode=='individual':
            self.ssp_alive_mass = self.ssp_alive_mass.flatten()
        elif mode=='age':
            self.ssp_alive_mass = np.mean(self.ssp_alive_mass, axis=0)
        elif mode=='metallicity':
            self.ssp_alive_mass = np.mean(self.ssp_alive_mass, axis=1)            
            
        return self.ssp_alive_mass
    
    def ssp_initial_mass_lum_ratio(self, mode='individual', wl=5635.):        
        wl_pt = np.where(self.wl==wl)[0][0]
        lum = self.ssp_SED[:, wl_pt]
        initial_lum = lum.reshape(4, 39)[:, 0]                
        
        self.init_mass_to_lum = np.ones((4, 39))/initial_lum[:, np.newaxis]
                
        if mode=='individual':
            self.init_mass_to_lum = self.init_mass_to_lum.flatten()
        elif mode=='age':
            self.init_mass_to_lum = np.mean(self.init_mass_to_lum, axis=0)
        elif mode=='metallicity':
            self.init_mass_to_lum = np.mean(self.init_mass_to_lum, axis=1)            
            
        return self.init_mass_to_lum
    
    def ssp_present_mass_lum_ratio(self, wl=5635.):
        wl_pt = np.where(self.wl==wl)[0][0]
        lum = self.ssp_SED[:, wl_pt]        
        self.mass_to_lum = self.ssp_alive_stellar_mass()/lum
        return self.mass_to_lum
    
        
# Example----------------------------------------------------------------------        
if __name__ == '__main__':
    
    pipssp = Pipe3Dssp()
    
    plt.figure()
    plt.plot(np.log10(pipssp.ssp_initial_mass_lum_ratio()))
    plt.plot(np.log10(pipssp.ssp_present_mass_lum_ratio()))
    
    hdul = fits.open('/home/pablo/obs_data/MANGA/Pipe3D/SSPs/gsd01_156.fits')
    ssps= hdul[0].data
    norm0 = hdul[0].header['NORM0']
    
    
    ssp_age, ssp_mass= np.loadtxt('/home/pablo/obs_data/MANGA/Pipe3D/SSPs/BASE.gsd01',
                              usecols=(1,4), unpack =True)

    wl0 = hdul[0].header['CRVAL1']
    deltawl = hdul[0].header['CDELT1']

    wl = np.arange(wl0, wl0+deltawl*ssps.shape[1], deltawl)

    wl_pt = np.where(wl==5635.)[0][0]
    
    # plt.plot(wl, ssps[8]*norm0)

    lum_mass_relation = ssps[:, wl_pt]

    # plt.plot(ssp_mass/lum_mass_relation, '-o')
    # plt.plot(1/ssps[:, 1000])

    
# ...