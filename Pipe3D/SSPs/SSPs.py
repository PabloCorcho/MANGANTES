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
    """
    This class provides the simple stellar population models that Pipe3D 
        used with MANGA IFU data.      
        
   A total of 156 SSP's with 39 ages and 4 metallicities:
           - 2 ages x 4 metallicities in Granada+ Geneva tracks 
           - 13 ages x 4 Z in Granada+Padova track
           - 24 ages x 4 Zs in Vazdekiz/Falcon-Barroso/MILES SSP models which 
           /Users/cid match the 40 ages we are working with in the CALIFA fits.
    
    Mstars grabbed from file ssp_mass_UN_mu1.3_v9.1.ElCid, and for young ages 
    Mstars = (t/3,631)^-0.05284.
    
    Stellar Evolution = Padova 2000
    
    IMF = UN = salpeter
    
    Each SSP when born had 1 Msun.
    """
    
    ssps_path = '/home/pablo/MANGANTES/Pipe3D/SSPs/gsd01_156.fits'        
    ssp_properties_path = '/home/pablo/MANGANTES/Pipe3D/SSPs/fits_like_properties.dat'
    
    
    def __init__(self):
     
        hdul = fits.open(self.ssps_path)
        
        self.norm = np.ones(156) # Lsun/Msun (a.k.a M/L ratio)
        for i in range(156):
            self.norm[i] = hdul[0].header['NORM'+str(i)]
            
        self.ssp_SED = hdul[0].data*self.norm[:, np.newaxis] #Lsun/AA/Msun (F_lambda)
        
        wl0 = hdul[0].header['CRVAL1']
        deltawl = hdul[0].header['CDELT1']
        self.wl = np.arange(wl0, wl0+deltawl*self.ssp_SED.shape[1], deltawl)
        
        
    def ssp_ages(self, mode='individual'):
        """
        'individual' mode returns one value per ssp (156)
        'age' mode returns the age bins of all ssp's (39)

        WARNING: 'individual' ages are not ordered. 
        They are stored in the same way as the Pipe3D outputs at SFH cubes. 
        """
        self.ssp_age= np.loadtxt(self.ssp_properties_path, usecols=(0))
        if mode == 'individual':
            return self.ssp_age
        elif mode == 'age':
            return np.unique(self.ssp_age)
    
    def ssp_metallicity(self,  mode='individual'):        
        """Idem ssp_ages method"""
        self.ssp_met= np.loadtxt(self.ssp_properties_path, usecols=(1))
        if mode == 'individual':
            return self.ssp_met
        elif mode == 'age':
            return np.unique(self.ssp_met)
    
    def ssp_alive_stellar_mass(self, mode='individual'):
        self.ssp_alive_mass = np.loadtxt(self.ssp_properties_path,
                                                         usecols=(2), unpack=True)
        self.ssp_alive_mass = self.ssp_alive_mass.reshape(4,39)
        
        if mode=='individual':
            self.ssp_alive_mass = self.ssp_alive_mass.flatten()
        elif mode=='age':
            self.ssp_alive_mass = np.mean(self.ssp_alive_mass, axis=0)
        elif mode=='metallicity':
            self.ssp_alive_mass = np.mean(self.ssp_alive_mass, axis=1)            
            
        return self.ssp_alive_mass
    
    def compute_ssp_initial_mass_lum_ratio(self, mode='individual', wl=[4470, 6470]):        
        wl_pt = np.where((self.wl>wl[0])&(self.wl<wl[-1]))[0]
        lum = np.mean(self.ssp_SED[:, wl_pt], axis=1)
        initial_lum = lum.reshape(4, 39)[:, 0]                
        
        self.init_mass_to_lum = np.ones((4, 39))/initial_lum[:, np.newaxis]
                
        if mode=='individual':
            self.init_mass_to_lum = self.init_mass_to_lum.flatten()
        elif mode=='age':
            self.init_mass_to_lum = np.mean(self.init_mass_to_lum, axis=0)
        elif mode=='metallicity':
            self.init_mass_to_lum = np.mean(self.init_mass_to_lum, axis=1)            
            
        return self.init_mass_to_lum
    
    def ssp_present_mass_lum_ratio(self, mode='individual'):        
        if mode == 'individual':
            self.mass_to_lum = 1/self.norm        
            return self.mass_to_lum 
        else: 
            raise NameError('MODE NOT DEVELOPED YET')
    
    def compute_ssp_present_mass_lum_ratio(self, mode='individual', wl=[4470, 6470]):
        wl_pt = np.where((self.wl>wl[0])&(self.wl<wl[-1]))[0]
        lum = np.mean(self.ssp_SED[:, wl_pt], axis=1)
        
        # self.mass_to_lum = self.ssp_alive_stellar_mass()/lum
        self.mass_to_lum = 1/lum
        if mode=='individual':
            self.mass_to_lum = self.mass_to_lum.flatten()
        elif mode=='age':
            self.mass_to_lum = np.mean(self.mass_to_lum, axis=0)
        elif mode=='metallicity':
            self.mass_to_lum = np.mean(self.mass_to_lum, axis=1)                        
        return self.mass_to_lum
    
        
# Example----------------------------------------------------------------------        
if __name__ == '__main__':
    print("""
          Pipe3D-SSPs
          --> Self-consistency tests
          """)
    pipssp = Pipe3Dssp()
    
    plt.figure()
    plt.scatter(np.log10(pipssp.ssp_ages()),
        np.abs(1/pipssp.norm-pipssp.compute_ssp_present_mass_lum_ratio(wl=[5549, 5551]))/pipssp.ssp_present_mass_lum_ratio(), 
        c=pipssp.ssp_metallicity(), marker='*')
    plt.ylabel('abs(M/L(HEADER) - M/L(COMPUTED))/M/L(HEADER)')
    # plt.scatter(np.log10(pipssp.ssp_ages()),
    #             np.log10(pipssp.ssp_present_mass_lum_ratio()),
    #             c=pipssp.ssp_metallicity())
    # plt.scatter(np.log10(pipssp.ssp_ages()),
    #             np.log10(pipssp.ssp_present_mass_lum_ratio(wl=[0, 1e4])),
    #             c=pipssp.ssp_metallicity(), marker='^')
    
    plt.colorbar()
    
    plt.figure()
    plt.scatter(np.log10(pipssp.ssp_ages()), pipssp.ssp_alive_stellar_mass(), 
                c=pipssp.ssp_metallicity())
    plt.colorbar()




# Created by Mr Krtxo...
