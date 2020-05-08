#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:18:23 2020

@author: pablo
"""


from MANGA import MANGA_remote as MANGA
# from MANGA import MANGA as MANGA
from Pipe3D.SSPs.SSPs import Pipe3Dssp
from astropy.io import fits

import numpy as np 
from matplotlib import pyplot as plt


class Pipe3Dmanga(Pipe3Dssp):
    
    fiber_angle_diameter = 0.5/3600 *np.pi/180 # rad            
    pipe3dcatalog_path = '/home/pablo/obs_data/MANGA/Pipe3D/manga.Pipe3D-v2_4_3.fits'
    pipe3dcubes_path = '/home/pablo/obs_data/MANGA/Pipe3D/cubes'        
    
    def __init__(self, **kwargs):
        # MANGA.__init__(self, **kwargs)
        
        self.plate = str(kwargs['plate'])
        self.ifudesign = str(kwargs['ifudesign'])
        
        Pipe3Dssp.__init__(self) # Load SSP properties
        
        self.pipe3d_path = self.pipe3dcubes_path+'/manga-'+self.plate+\
                                    '-'+self.ifudesign+'.Pipe3D.cube.fits'
                                    
        self.pipe3dcube = fits.open(self.pipe3d_path)    
        
        self.catalog = fits.open(self.pipe3dcatalog_path)[1]
        self.cat_entry = np.where(self.catalog.data['mangaid']=='manga-'+\
                                  self.plate+'-'+self.ifudesign)[0][0]           


# =============================================================================
# FLUX MAPS 
# =============================================================================
  
    def get_median_flux(self, unit='erg'):
        units = {'erg':1e-16, 'Lsun':1e-16/3.828e33}        
        self.flux_units = units[unit]
        return self.pipe3dcube[1].data[3] 
    
    def get_Vcontinuum_flux(self, unit='erg'):
        units = {'erg':1e-16, 'Lsun':1e-16/3.828e33}        
        self.flux_units = units[unit]
        return self.pipe3dcube[1].data[0]
    
    def flux_to_luminosity(self):
        """
        This method converts flux to luminosity when the distance to the 
        object is provided. CAVEAT: Luminosity has same energy units as flux.
        """
        self.luminosity = 4* np.pi* self.luminosity_distance**2*self.flux
        return self.luminosity

# =============================================================================
# DISTANCES    
# =============================================================================
    
    def pipe3d_luminosity_distance(self, unit='Mpc'):
        units ={'Mpc':1, 'Kpc':1e3, 'pc':1e6, 'cm':3.086e+24}
        return self.catalog.data['DL'][self.cat_entry]*units[unit]

    def get_spaxel_size(self):
        self.spaxel_diameter=self.angular_diameter_distance*self.fiber_angle_diameter
        return self.spaxel_diameter
    
    def get_spaxel_area(self):
        self.spaxel_area=np.pi * self.get_spaxel_size()**2
        return self.spaxel_area

    def get_solid_angle(self):
        try: 
            self.redshift
        except:
            self.redshift = self.pipe3d_redshift()
            
        # w = 4*np.pi*self.fiber_angle_diameter**2/(1+self.redshift)**2
        w = 4*np.pi*.5**2/(1+self.redshift)**2
        return w
        
    def pipe3d_angular_diameter_distance(self, unit='Kpc'):        
        units ={'Mpc':1, 'Kpc':1e3, 'pc':1e6, 'cm':3.086e+24}
        return self.pipe3d_luminosity_distance()/(1+self.redshift)**2 *units[unit]

    
    def get_segmentation(self):
        self.segmentation = self.pipe3dcube[1].data[1]        
        
        n_bins = len(np.unique(self.segmentation))
        bins = np.arange(0, n_bins) 
        
        self.segmentation_cube = np.zeros((n_bins, self.segmentation.shape[0], 
                                                   self.segmentation.shape[1]),
                                          dtype=bool)        
        for i in range(n_bins):
            self.segmentation_cube[i, :, :] = self.segmentation == bins[i]
    
        
# =============================================================================
# CATALOG PROPERTIES & MAPS   
# =============================================================================
    def pipe3d_redshift(self):
        return self.catalog.data['redshift'][self.cat_entry]
        
    def get_surface_density(self, dust_corr=True):
        """
        Stellar Mass density per pixel without dust correction
        --> log10(Msun/spaxels^2) 	
        
        *CAVEAT*: This map has not considered mass loss due to stellar death.        
        """        
        if dust_corr:
            return self.pipe3dcube[1].data[19]
        else:
            return self.pipe3dcube[1].data[18]
    
    def pipe3d_tot_lgm(self):
        return self.catalog.data['log_Mass'][self.cat_entry]
        
    def get_luminosity_weighted_age(self):
        return self.pipe3dcube[1].data[5]    
    def get_mass_weighted_age(self):
        return self.pipe3dcube[1].data[6]   
            
# =============================================================================
# STAR FORMATION HISTORIES    
# =============================================================================
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
        self.flux = self.get_median_flux(unit='Lsun')
        self.redshift = self.pipe3d_redshift()
        
        self.angular_diameter_distance = self.pipe3d_angular_diameter_distance()
        self.spaxel_area = self.get_spaxel_area()
        
        try:
            self.luminosity = self.flux_to_luminosity()
        except:            
            self.luminosity_distance = self.pipe3d_luminosity_distance(unit='cm')
            self.luminosity = self.flux_to_luminosity()
        
                                
        ssp_mass_to_lum = self.ssp_present_mass_lum_ratio(mode)
        ssp_alive_stellar_mass = self.ssp_alive_stellar_mass(mode)      
        ssp_weights = self.get_SFHweights(mode)        
                
        self.ssp_masses = (
               self.luminosity[np.newaxis, :, :] * self.flux_units                                
                    *ssp_weights *ssp_mass_to_lum[:, np.newaxis, np.newaxis]
                    *ssp_alive_stellar_mass[:, np.newaxis, np.newaxis]  
                           )                  
        
    def compute_luminosity_weighted_age(self):
        ages = self.ssp_ages(mode='individual')
        ssp_weights = self.get_SFHweights()        
        lum_weighted_age = np.sum(np.log10(ages[:, np.newaxis, np.newaxis]) * ssp_weights, axis=0)
        return lum_weighted_age
    
    def compute_mass_weighted_age(self):
        ages = self.ssp_ages(mode='individual')
        ssp_mass_to_lum = self.ssp_present_mass_lum_ratio(mode='individual')
        ssp_weights = self.get_SFHweights()        
        
        all_weights = np.sum(ssp_weights, axis=0)
        mask = all_weights==0               
        
        mass_weighted_age = np.sum(
        np.log10(ages[:, np.newaxis, np.newaxis])*ssp_weights*ssp_mass_to_lum[:, np.newaxis, np.newaxis], 
        axis=0)/np.sum(ssp_weights*ssp_mass_to_lum[:, np.newaxis, np.newaxis], axis=0)
        mass_weighted_age[mask] = 0
        
        return mass_weighted_age
                
    def compute_SFH(self, mode='individual', today=14e9):        
        
        try: 
            self.ssp_masses                    
        except:
            self.compute_ssp_masses(mode)
        
        ages = self.ssp_ages(mode='individual')            
        
        sort_ages = np.argsort(ages)
        
        sorted_ssp_masses = self.ssp_masses[sort_ages, :, :]
        
        sorted_ssp_masses = sorted_ssp_masses.reshape(39, 4, 
                                      self.ssp_masses.shape[1], 
                                      self.ssp_masses.shape[2])
        
        self.total_ssp_mass = np.sum(sorted_ssp_masses, axis=1)
        
            
        
        ages =  np.unique(ages[sort_ages])[::-1] #from old to young
        
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
        
    def compute_binned_SFH(self, mode='individual', today=14e9):        
        self.get_segmentation()
        
        try: 
            self.ssp_masses                    
        except:
            self.compute_ssp_masses(mode)
        
        ages = self.ssp_ages(mode='individual')            
        
        sort_ages = np.argsort(ages)
        
        sorted_ssp_masses = self.ssp_masses[sort_ages, :, :]
        
        sorted_ssp_masses = sorted_ssp_masses.reshape(39, 4, 
                                      self.ssp_masses.shape[1], 
                                      self.ssp_masses.shape[2])
        
        self.total_ssp_mass = np.sum(sorted_ssp_masses, axis=1)
            
        ages =  np.unique(ages[sort_ages])[::-1] #from old to young
        self.time_bins = ages[0] - ages 
        
        ages =  ages[::-1] #from old to young
                        
        self.total_ssp_mass = self.total_ssp_mass[::-1, :, :]
        
        area = self.get_spaxel_area()
        
        self.stellar_mass_history = np.cumsum(
            self.total_ssp_mass, axis=0) 
        
        self.binned_mass_history = np.zeros((self.segmentation_cube.shape[0],
                                                 ages.size))
        self.bin_area = np.zeros(self.segmentation_cube.shape[0])
        
        for i in range(self.segmentation_cube.shape[0]):
            n_spaxels = len( self.segmentation_cube[i, :, :][
                self.segmentation_cube[i, :, :] == True])
            
            self.binned_mass_history[i, :] = np.sum(
                self.stellar_mass_history[:, self.segmentation_cube[i, :, :]], 
                axis=(1))
            self.bin_area[i] = area*n_spaxels
            
            
        
        
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
    
    galaxy = Pipe3Dmanga(plate=9510, ifudesign=12703)
    galaxy.compute_SFH(mode='individual')
    galaxy.compute_binned_SFH()
    
    print('Catalog mass: ', galaxy.pipe3d_tot_lgm())
    print('Derived total mass: ', 
          np.log10(np.sum(galaxy.stellar_mass_history[-1])))
    
    SFH = galaxy.star_formation_history
    sSFH = galaxy.specific_star_formation_history
    M_history = galaxy.stellar_mass_history
    
    time = galaxy.time
    

    
    last_100My = np.where(time>14e9)[0]    
    M_today = galaxy.stellar_mass_history[-1]        
    
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cm = ax.imshow(np.log10(np.mean(sSFH[last_100My, :, :], axis=0)),
               cmap='jet_r')
    fig.colorbar(cm, orientation='horizontal')
    ax = fig.add_subplot(132)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cm = ax.imshow(np.log10(M_today),
               cmap='gist_earth')
    fig.colorbar(cm, orientation='horizontal')
    
      
    integrated_mass = np.sum(galaxy.binned_mass_history, axis=0)
    integrated_sfh = np.diff(integrated_mass)/np.diff(galaxy.time_bins)
    sfh_time = (galaxy.time_bins[1:] + galaxy.time_bins[:-1])/2
    
    plt.figure()
    plt.subplot(211)
    plt.semilogy(np.log10(galaxy.time_bins), integrated_mass)
    plt.subplot(212)    
    plt.semilogy(np.log10(sfh_time), integrated_sfh)
  
    
# =============================================================================
#     Surface mass density
# =============================================================================
    
    cat_surface_density = galaxy.get_surface_density(dust_corr=False)
    
    plt.figure(figsize=(9,7))    
    plt.subplot(131)
    plt.title(r'Computed $\log(M_\odot/spxl)$', fontsize=7)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(np.log10(M_today), cmap='gist_earth',
               vmin=np.nanpercentile(cat_surface_density, 30),
               vmax=np.nanpercentile(cat_surface_density, 95))
    plt.colorbar(orientation='horizontal')
    
    plt.subplot(132)    
    plt.title(r'Pipe3D $\log(M_\odot/spxl)$', fontsize=7)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(cat_surface_density, 
               cmap='gist_earth',
               vmin=np.nanpercentile(cat_surface_density, 30),
               vmax=np.nanpercentile(cat_surface_density, 95))
    plt.colorbar(orientation='horizontal')
    
    plt.subplot(133)
    plt.title(r'Residuals', fontsize=7)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(np.log10(M_today)-galaxy.get_surface_density(dust_corr=False),
               cmap='inferno')
    plt.colorbar(orientation='horizontal')
    plt.savefig('example/surface_mass_density_test.png', bbox_inches='tight')
    
# =============================================================================
#     Luminosity weighted age
# =============================================================================
    
    cat_lum_weighted_age = galaxy.get_luminosity_weighted_age()
    
    plt.figure(figsize=(9,7))    
    plt.subplot(131)
    plt.title(r'Computed luminosity-weighted age $\log(\tau/yr)$', fontsize=7)
    plt.imshow(galaxy.compute_luminosity_weighted_age(), 
                vmin=np.percentile(cat_lum_weighted_age[cat_lum_weighted_age>0], 5),
                vmax=np.percentile(cat_lum_weighted_age[cat_lum_weighted_age>0], 95),
                cmap='jet')
    plt.colorbar(orientation='horizontal')
    plt.subplot(132)
    plt.title(r'Pipe3D luminosity-weighted age $\log(\tau/yr)$', fontsize=7)
    plt.imshow(galaxy.get_luminosity_weighted_age(), 
                vmin=np.percentile(cat_lum_weighted_age[cat_lum_weighted_age>0], 5),
                vmax=np.percentile(cat_lum_weighted_age[cat_lum_weighted_age>0], 95),
                cmap='jet'
                )
    plt.colorbar(orientation='horizontal')
    plt.subplot(133)
    plt.title('Residuals', fontsize=7)
    plt.imshow(galaxy.get_luminosity_weighted_age()- galaxy.compute_luminosity_weighted_age(),
                cmap='inferno')
    plt.colorbar(orientation='horizontal')
    plt.savefig('example/lum_weighted_age_test.png', bbox_inches='tight')
# =============================================================================
#     Mass weighted age
# =============================================================================
    
    cat_ass_weighted_age = galaxy.get_mass_weighted_age()
    plt.figure(figsize=(9,7))    
    plt.subplot(131)
    plt.title(r'Computed mass-weighted age $\log(\tau/yr)$', fontsize=7)
    plt.imshow(galaxy.compute_mass_weighted_age(), 
                vmin=np.nanpercentile(cat_ass_weighted_age[cat_ass_weighted_age>0], 5),
                vmax=np.nanpercentile(cat_ass_weighted_age[cat_ass_weighted_age>0], 95),
                cmap='jet')
    plt.colorbar(orientation='horizontal')
    plt.subplot(132)
    plt.title(r'Pipe3D mass-weighted age $\log(\tau/yr)$', fontsize=7)
    plt.imshow(galaxy.get_mass_weighted_age(), 
                vmin=np.nanpercentile(cat_ass_weighted_age[cat_ass_weighted_age>0], 5),
                vmax=np.nanpercentile(cat_ass_weighted_age[cat_ass_weighted_age>0], 95),
                cmap='jet'
                )
    plt.colorbar(orientation='horizontal')
    plt.subplot(133)
    
    residuals = galaxy.get_mass_weighted_age() - galaxy.compute_mass_weighted_age()
    
    plt.title('Residuals', fontsize=7)
    plt.imshow(residuals,
                cmap='inferno', vmin=np.percentile(residuals, 10), vmax=np.percentile(residuals, 90))
    plt.colorbar(orientation='horizontal')
    plt.savefig('example/mass_weighted_age_test.png', bbox_inches='tight')
    
    
    # plt.hist2d(np.log10(M_today.flatten()), np.log10(sSFR.flatten()))
    