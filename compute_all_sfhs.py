#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:09:09 2020

@author: pablo
"""

import numpy as np 
from Pipe3Dmanga import Pipe3Dmanga
from astropy.io import fits

from matplotlib import pyplot as plt

catalog_path= '/home/pablo/obs_data/MANGA/Pipe3D/manga.Pipe3D-v2_4_3.fits'
hdul = fits.open(catalog_path)
ifudesigns = np.array(hdul[1].data['ifudsgn'], dtype=str)
plates = np.array(hdul[1].data['plate'], dtype=str)

results = '/home/pablo/MANGA_SFHs'

sfh_fig = plt.figure(figsize=(10,10))
sfh_ax = sfh_fig.add_subplot(111)

bin_mass = np.linspace(3, 9, 21)
bin_ssfr = np.linspace(-13, -5)

mid_mass = (bin_mass[0:-1] + bin_mass[1:])/2
mid_ssfr = (bin_ssfr[0:-1] + bin_ssfr[1:])/2

all_histograms = 0

for ith in range(3):
        
    galaxy = Pipe3Dmanga(plate=plates[ith], ifudesign=ifudesigns[ith])
    galaxy.compute_SFH(mode='individual')
        
    SFH = galaxy.star_formation_history
    sSFH = galaxy.specific_star_formation_history
    MH = galaxy.stellar_mass_history
    time = galaxy.sfh_times
    
    
    last_100My = np.where(time<1e8)[0]    
    
    mean_SFR = np.mean(SFH[last_100My, :, :], axis=0)
    mean_sSFR = np.mean(sSFH[last_100My, :, :], axis=0)
    
    print(galaxy.pipe3d_tot_lgm())
    print(np.log10(np.sum(galaxy.stellar_mass_history[-1])))
    
    
    fig = plt.figure()
    
    ax = fig.add_subplot(121)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    cm = ax.imshow(np.log10(mean_sSFR),
                cmap='jet_r')
    fig.colorbar(cm, ax.inset_axes([.95, .0, .05, 1]))
    
    ax = fig.add_subplot(122)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    cm = ax.imshow(np.log10(MH[-1, :, :]),
                cmap='gist_earth')
    fig.colorbar(cm, ax.inset_axes([.95, .0, .05, 1]))
    
    fig.savefig(results+'/today_values/'+ 'manga-'+ str(plates[ith])+'-'+\
                            str(ifudesigns[ith])+'.png', bbox_inches='tight')
    
    H, _, _ = np.histogram2d(np.log10(MH[-1, :, :]).flatten(),
                 np.log10(mean_sSFR).flatten(),
                 bins=[bin_mass, bin_ssfr])
    sfh_ax.loglog(time, np.sum(SFH,axis=(1,2))/np.sum(MH[0:-1,:,:], axis=(1,2)), '.-')

    all_histograms += H
sfh_fig.savefig(results+'/all_sfhs.pdf', bbox_inches='tight')    


plt.figure()
plt.imshow(np.log10(all_histograms.T), origin='lower', cmap='inferno',
            extent=(mid_mass[0], mid_mass[-1], mid_ssfr[0], mid_ssfr[-1]))

# plt.contourf(mid_mass, mid_ssfr, np.log10(all_histograms.T), cmap='inferno',
#              levels=20)
plt.colorbar()    