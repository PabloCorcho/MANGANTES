#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:36:34 2020

@author: pablo
"""


import numpy as np
from astropy.io import fits 
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d

ssps_path = '/home/pablo/MANGANTES/Pipe3D/SSPs/gsd01_156.fits'        
ssp_alive_stars_mass_path = '/home/pablo/MANGANTES/Pipe3D/SSPs/BASE.gsd01'


ages_mass, met_mass, masses = np.loadtxt(ssp_alive_stars_mass_path, usecols=(1,2,4), unpack=True)

# ages_mass = ages_mass.reshape(4, 39)
# met_mass = met_mass.reshape(4, 39)
# masses = masses.reshape(4, 39)
sort_ages = np.argsort(ages_mass)

ages_mass = ages_mass[sort_ages]
met_mass = met_mass[sort_ages]
masses = masses[sort_ages]

f = interp2d(ages_mass, met_mass, masses)

hdul = fits.open(ssps_path)

names = []

for i in range(156):
    names.append(hdul[0].header['NAME'+str(i)])
     

metallicities = np.zeros((156))    
ages = np.zeros((156))    
alive_mass = np.zeros((156))    
for i, name in enumerate(names):
    metallicities[i] = float('0.'+name[-11:-4])
    ages[i] = float(name[9:-16])*1e9    
    print(np.argsort(np.abs(ages_mass-ages[i])))
    print(np.argsort(np.abs(met_mass-metallicities[i])))
    alive_mass[i] = f(ages[i], metallicities[i])    
    # alive_mass[i] = masses[np.where((ages_massages[i])&(met_mass==metallicities[i]))[0]]
    
plt.figure()
plt.semilogx(ages, alive_mass, 'o')
plt.semilogx(ages_mass, masses, '.')


with open('fits_like_properties.dat', 'w') as file:
    file.write('# SSP fundamental properties\n'+
               '# ages [yr]    metallicities     alive stellar mass [Msun]\n'
               )    
    for i in range(alive_mass.size):
        file.write('{:}   {:}   {:}\n'.format(ages[i], metallicities[i], 
                                            alive_mass[i])
                   )
    



