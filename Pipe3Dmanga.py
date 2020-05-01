#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:18:23 2020

@author: pablo
"""


# from MANGA import MANGA_remote as MANGA
from MANGA import MANGA as MANGA
from Pipe3D.SSPs.SSPs import Pipe3Dssp
from astropy.io import fits

import numpy as np 
from matplotlib import pyplot as plt


class Pipe3Dmanga(MANGA, Pipe3Dssp):
    
    pipe3dcatalog_path = '/home/pablo/obs_data/MANGA/Pipe3D/manga.Pipe3D-v2_4_3.fits'
    pipe3dcubes_path = '/home/pablo/obs_data/MANGA/Pipe3D/cubes'        
    
    def __init__(self, **kwargs):
        MANGA.__init__(self, **kwargs)
        Pipe3Dssp.__init__(self)
        
        self.pipe3d_path = self.pipe3dcubes_path+'/manga-'+self.plate+\
                                    '-'+self.ifudesign+'.Pipe3D.cube.fits'
                                    
        self.pipe3dcube = fits.open(self.pipe3d_path)    
        
        self.catalog = fits.open(self.pipe3dcatalog_path)[1]
        self.cat_entry = np.where(self.catalog.data['mangaid']=='manga-'+\
                                  self.plate+'-'+self.ifudesign)[0][0]           
        
    def pipe3d_redshift(self):
        return self.catalog.data['redshift'][self.cat_entry]
        
    def pipe3d_tot_lgm(self):
        return self.catalog.data['log_Mass'][self.cat_entry]
        
    def pipe3d_luminnosity_distance(self, unit='Mpc'):
        units ={'Mpc':1, 'Kpc':1e3, 'pc':1e6, 'cm':3.086e+24}
        return self.catalog.data['DL'][self.cat_entry]*units[unit]
    
    def pipe3d_angular_diameter_distance(self, unit='Kpc'):        
        units ={'Mpc':1, 'Kpc':1e3, 'pc':1e6, 'cm':3.086e+24}
        return self.pipe3d_luminnosity_distance()/(1+self.redshift)**2 *units[unit]
    
    
    def get_SFHweights(self, mode='individual'):
        """ 
        This method provides the relative weights of each single stellar 
        population within the considered SSP-library to the flux intensity at
        5635AA for each individual spaxel within the original MaNGA cube.
        In addition, the relative weights for each considered individual SSP 
        we include the corresponding weights for the 39 ages (averaged by 
        metallicity) and the 4 metallicities (averaged by age).
        """
        
        modes = {'individual':np.arange(0,156), 'age':np.arange(156, 195),
                 'metallicity':np.arange(195, 199)}
        
        return self.pipe3dcube[2].data[modes[mode], :, :]
        
    def compute_ssp_masses(self, mode='individual', wl_ref=5635):
        """
        There are two possible modes to compute the masses:
            - 'individual' mode uses the 156 ssps and their corresponding 
               weights.
            - 'age' mode uses metallicity-averaged weights (39).
        """
        
        # Conversion of (rest-frame) flux into luminosity
        self.flux = self.get_flux(unit='Lsun')
        self.wavelength = self.get_wavelength()
        self.redshift = self.pipe3d_redshift()
        
        self.wavelength_to_rest_frame()
        
        flux_ref = np.argmin(np.abs(self.wavelength-wl_ref))
        
        self.angular_diameter_distance = self.pipe3d_angular_diameter_distance()
        self.spaxel_area = self.get_spaxel_area()
        
        try:
            self.luminosity = self.flux_to_luminosity()
        except:            
            self.luminosity_distance = self.pipe3d_luminnosity_distance(unit='cm')
            self.luminosity = self.flux_to_luminosity()
        
                                
        ssp_mass_to_lum = self.ssp_initial_mass_lum_ratio(mode)
        ssp_alive_stellar_mass = self.ssp_alive_stellar_mass(mode)      
        ssp_weights = self.get_SFHweights(mode)        
                
        self.ssp_masses = (
               self.luminosity[flux_ref, np.newaxis, :, :] * self.flux_units                                
                    *ssp_weights *ssp_mass_to_lum[:, np.newaxis, np.newaxis]
                    * ssp_alive_stellar_mass[:, np.newaxis, np.newaxis]  )                  
        
        
    def compute_SFH(self, mode='individual', today=14e9):        
        
        try: 
            self.ssp_masses                    
        except:
            self.compute_ssp_masses(mode)
            
        if mode=='individual':
            self.total_ssp_mass = np.sum(
                        self.ssp_masses.reshape(4, 39, 
                                      self.ssp_masses.shape[1], 
                                      self.ssp_masses.shape[2]), 
                                             axis=0)
        elif mode=='age':
            self.total_ssp_mass = self.ssp_masses
            
            
        ages = self.ssp_ages(mode='age') # 39 different ages
        ages =  ages[::-1] #from old to young
                        
        self.total_ssp_mass = self.total_ssp_mass[::-1, :, :]
        
        
        self.stellar_mass_history = np.cumsum(
            self.total_ssp_mass, axis=0) 
        
        self.time_bins = ages[0] - ages 
        self.time = (self.time_bins[1:]+self.time_bins[:-1])/2
        
        self.star_formation_history = (
            self.stellar_mass_history[1:]-self.stellar_mass_history[:-1])/(
            self.time_bins[1:, np.newaxis, np.newaxis]-\
            self.time_bins[:-1, np.newaxis, np.newaxis])
                
        self.stellar_mass_history = (
                                     self.stellar_mass_history[1:, :, :]+\
                                     self.stellar_mass_history[0:-1, : , :]
                                     )/2
        
        self.specific_star_formation_history = self.star_formation_history/(
                                                    self.stellar_mass_history)        
        
    def integrated_mass_history(self):
        return np.sum(self.stellar_mass_history, axis=(1,2))
    
    def integrated_star_formation_history(self):
        return np.sum(self.star_formation_history, axis=(1,2))
    
    def mass_to_density(self, mass_array):
        return mass_array/self.spaxel_area/4/np.pi        
        
# =============================================================================
# Example        
# =============================================================================
if __name__ == '__main__':
    
    galaxy = Pipe3Dmanga(plate=10001, ifudesign=12701)
    galaxy.compute_SFH(mode='individual')
    
    print('Catalog mass: ', galaxy.pipe3d_tot_lgm())
    print('Derived total mass: ', 
          np.log10(np.sum(galaxy.stellar_mass_history[-1])))
    
    SFH = galaxy.star_formation_history
    sSFH = galaxy.specific_star_formation_history
    M_history = galaxy.stellar_mass_history
    
    time = galaxy.time
    # time_sfh = (time[0:-1] + time[1:])/2
    
    
    plt.figure()
    plt.semilogy(np.log10(time), galaxy.integrated_star_formation_history()/galaxy.integrated_mass_history(),
                 '.-')
    
    last_100My = np.where(time>14e9)[0]    
    M_today = galaxy.stellar_mass_history[-1]        
    plt.figure()
    plt.hist(np.log10(np.mean(sSFH[last_100My, :, :], axis=0)).flatten(), bins=30, density=True,
             histtype='step')
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cm = ax.imshow(np.log10(np.mean(sSFH[last_100My, :, :], axis=0)),
               cmap='jet_r')
    fig.colorbar(cm, ax.inset_axes([.95, .0, .05, 1]))
    ax = fig.add_subplot(122)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cm = ax.imshow(np.log10(M_today),
               cmap='gist_earth')
    fig.colorbar(cm, ax.inset_axes([.95, .0, .05, 1]))
    
    plt.figure()
    
    # plt.hist2d(np.log10(M_today.flatten()), np.log10(sSFR.flatten()))
    