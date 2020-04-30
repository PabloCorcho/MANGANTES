#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:43:51 2020

@author: pablo
"""

from marvin.tools.cube import Cube
from astropy.io import fits
import numpy as np 

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

class MANGA_remote(object):
    
    fiber_angle_diameter = 0.5/3600 *np.pi/180 # rad    
    fiber_solid_angle = 2*np.pi*(1 - np.cos(fiber_angle_diameter/2))
    
    def __init__(self, **kwargs):
        
        self.plate = str(kwargs['plate'])
        self.ifudesign = str(kwargs['ifudesign'])
        
        print('Accessing remotely to cube...[This step may take some time]')
        self.IFUcube = Cube(plateifu=self.plate+'-'+self.ifudesign)                                    
        
    def get_flux(self, unit='erg'):
        """ 
        This method provides the tridimensional flux cube (float 32) in 
        1e-17 erg/s units. The desired flux energy units are kept in 
        'flux_units' for preventing numerical issues. 
        """        
        units = {'erg':1e-17, 'Lsun':1e-17/3.828e33} #unit/s/cm2/AA/spxl                
        self.flux_units = units[unit]
        print('Downloading flux...[This step may take some time]')
        return self.IFUcube.flux.value

    def get_wavelength(self, unit='AA'):
        units = {'AA':1}
        return self.IFUcube.flux.wavelength.value*units[unit] 
    
    def get_photo_image(self, band):
        # bands = {'g':12, 'r':13, 'i':14, 'z':15}
        # return self.IFUcube[bands[band]].data
        return print(' \'get_photo_image\' not implemented')
        
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
        
