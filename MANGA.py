#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:43:51 2020

@author: pablo
"""
from astropy.io import fits

import numpy as np 
from matplotlib import pyplot as plt

from Pipe3D.SSPs.SSPs import Pipe3Dssp


class MANGA(object):
    
    IFUcubes_path = '/home/pablo/obs_data/MANGA/cubes'        
    fiber_angle_diameter = 0.5/3600 *np.pi/180 # rad    
    fiber_solid_angle = 2*np.pi*(1 - np.cos(fiber_angle_diameter/2))
    
    def __init__(self, **kwargs):
        
        self.plate = str(kwargs['plate'])
        self.ifudesign = str(kwargs['ifudesign'])
        
        path_to_cube = self.IFUcubes_path+'/manga-'+self.plate+\
                                    '-'+self.ifudesign+'-LINCUBE.fits'
                                 
        self.IFUcube = fits.open(path_to_cube)                                    
        
    def get_flux(self, unit='erg'):
        """ 
        This method provides the tridimensional flux cube (float 32) in 
        1e-17 erg/s units. The desired flux energy units are kept in 
        'flux_units' for preventing numerical issues. 
        """
        
        units = {'erg':1e-17, 'Lsun':1e-17/3.828e33} #unit/s/cm2/AA/spxl        
        self.flux_units = units[unit]
        return self.IFUcube[1].data

    def get_wavelength(self, unit='AA'):
        units = {'AA':1}
        return self.IFUcube[6].data*units[unit] 
    
    def get_photo_image(self, band):
        bands = {'g':12, 'r':13, 'i':14, 'z':15}
        return self.IFUcube[bands[band]].data
        
    def flux_to_luminosity(self):
        """
        This method converts flux to luminosity when the distance to the 
        object is provided. CAVEAT: Luminosity has same energy units as flux.
        """
        self.luminosity = 4* np.pi* self.luminosity_distance**2*self.flux
        return self.luminosity

    def wavelength_to_rest_frame(self):
        self.wavelength = self.wavelength/(1+self.redshift)

    def get_spaxel_size(self):
        self.spaxel_diameter=self.angular_diameter_distance*self.fiber_angle_diameter
        return self.spaxel_diameter
    
    def get_spaxel_area(self):
        self.spaxel_area=np.pi * self.get_spaxel_size()**2
        return self.spaxel_area
        
        
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
        
        try:
            self.luminosity = self.flux_to_luminosity()
        except:            
            self.luminosity_distance = self.pipe3d_luminnosity_distance(unit='cm')
            self.luminosity = self.flux_to_luminosity()
        
        self.angular_diameter_distance = self.pipe3d_angular_diameter_distance()
        spaxel_area = self.get_spaxel_area()
                
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
            self.mass_at_time = np.sum(
                        self.ssp_masses.reshape(4, 39, 
                                      self.ssp_masses.shape[1], 
                                      self.ssp_masses.shape[2]), 
                                             axis=0)
        elif mode=='age':
            self.mass_at_time = self.ssp_masses
            
            
        ages = self.ssp_ages(mode='age') # 39 different ages
        self.time =  ages[::-1] #from old to young
                        
        self.mass_at_time = self.mass_at_time[::-1, :, :]
        
        
        self.stellar_mass_history = np.cumsum(
            self.mass_at_time, axis=0) 
        
        self.star_formation_history = -np.diff(self.stellar_mass_history,
                      axis=0)/np.diff(self.time)[:, np.newaxis, np.newaxis]
        
        self.specific_star_formation_history = self.star_formation_history/(
        self.stellar_mass_history[1:, :, :]/2 + self.stellar_mass_history[0:-1, : , :]/2)
        self.sfh_times = (self.time[0:-1] + self.time[1:])/2
        
        
# =============================================================================
# Example        
# =============================================================================
if __name__ == '__main__':
    
    manga_galaxy = Pipe3Dmanga(plate=9002, ifudesign=9101)
    manga_galaxy.compute_SFH()
    
    print('Catalog mass: ', manga_galaxy.pipe3d_tot_lgm())
    print('Derived total mass: ', 
          np.log10(np.sum(manga_galaxy.stellar_mass_history[-1])))
    
    SFH = manga_galaxy.star_formation_history
    sSFH = manga_galaxy.specific_star_formation_history
    M_history = manga_galaxy.stellar_mass_history    
    M_history = (M_history[1:, :, :] + M_history[0:-1, : , :])/2
    time = manga_galaxy.time
    time_shfr = (time[0:-1] + time[1:])/2
    
    plt.figure()
    plt.semilogy(np.log10(time_shfr), np.sum(SFH,axis=(1,2))/np.sum(M_history,
                                                        axis=(1,2)), '.-')
    
    
    M_today = manga_galaxy.stellar_mass_history[-1]        
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cm = ax.imshow(np.log10(np.mean(sSFH[-5:-1, :, :], axis=0)),
               cmap='jet_r')
    fig.colorbar(cm, ax.inset_axes([.95, .0, .05, 1]))
    ax = fig.add_subplot(122)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cm = ax.imshow(np.log10(M_today),
               cmap='gist_earth')
    fig.colorbar(cm, ax.inset_axes([.95, .0, .05, 1]))
    
    plt.figure()
    
    # plt.hist2d(np.log10(M_today.flatten()), np.log10(sSFR.flatten()))
    