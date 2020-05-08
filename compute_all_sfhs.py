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
from scipy.interpolate import interp1d

catalog_path= '/home/pablo/obs_data/MANGA/Pipe3D/manga.Pipe3D-v2_4_3.fits'
hdul = fits.open(catalog_path)
ifudesigns = np.array(hdul[1].data['ifudsgn'], dtype=str)
plates = np.array(hdul[1].data['plate'], dtype=str)

results = '/home/pablo/Results_MANGA'

sfh_fig = plt.figure()
sfh_ax = sfh_fig.add_subplot(111)

ssfr_fig = plt.figure()
ssfr_ax = ssfr_fig.add_subplot(111)


bin_mass = np.linspace(3, 10, 31)
bin_ssfr = np.linspace(-15, -5, 41)
bin_sfr = np.linspace(-7, 3, 41)

mid_mass = (bin_mass[0:-1] + bin_mass[1:])/2
mid_ssfr = (bin_ssfr[0:-1] + bin_ssfr[1:])/2
mid_sfr = (bin_sfr[0:-1] + bin_sfr[1:])/2

dlgm = (bin_mass[1:]- bin_mass[:-1])
dlgssfr = (bin_ssfr[1:] - bin_ssfr[:-1])
dlgsfr = (bin_sfr[1:]- bin_sfr[:-1])

all_histograms_sigma_sfr = 0
all_histograms_sigma_ssfr = 0
all_sSFRs = 0

# for ith in range(100):
for ith in range(plates.size):
    print('\n galaxy --> {}  \n'.format(ith))        
    try:
        
        
        galaxy = Pipe3Dmanga(plate=plates[ith], ifudesign=ifudesigns[ith])
        galaxy.compute_binned_SFH(mode='individual')
        
        
        MH = galaxy.binned_mass_history
        bin_size = galaxy.bin_area 
        time = galaxy.time_bins
        
        SFR = (MH[:, 1:]-MH[:, :-1])/(time[np.newaxis, 1:]-time[np.newaxis, :-1])
        sSFR = SFR / (MH[:, 1:]*0.5 +MH[:, :-1]*0.5)
        mid_time = (time[1:]+time[:-1])/2
        
        last_100My = np.where(mid_time>np.max(mid_time)-1e8)[0]
        
        mean_SFR = np.mean(SFR[:, last_100My], axis=1)
        mean_sSFR = np.mean(sSFR[:, last_100My], axis=1)
        
        print(galaxy.pipe3d_tot_lgm())
        print(np.log10(np.sum(MH[:, -1])))
        
        H_sigma_sfr, _ , _ = np.histogram2d(np.log10(MH[:, -1]/bin_size), np.log10(mean_SFR),
                     bins=[bin_mass, bin_sfr])
        
        H_sigma_ssfr, _ , _ = np.histogram2d(np.log10(MH[:, -1]/bin_size), np.log10(mean_sSFR),
                     bins=[bin_mass, bin_ssfr])
        
        H_ssfr , _ = np.histogram(np.log10(mean_sSFR).flatten(), bins=bin_ssfr, 
                                  density=True)
        
          
        ssfr_ax.semilogy(mid_ssfr, H_ssfr, 'k-', alpha=0.05)
        
    
        all_histograms_sigma_sfr += H_sigma_sfr
        all_histograms_sigma_ssfr += H_sigma_ssfr
        all_sSFRs += H_ssfr
    except:
        print('Failure')
    
sfh_fig.savefig(results+'/all_sfhs.pdf', bbox_inches='tight')    
ssfr_fig.savefig(results+'/all_ssfr.pdf', bbox_inches='tight')    


np.savetxt('/home/pablo/Results_MANGA/plane_M_ssfr.dat', all_histograms_sigma_ssfr)
np.savetxt('/home/pablo/Results_MANGA/plane_M_sfr.dat', all_histograms_sigma_sfr)

plt.figure()
plt.imshow(np.log10(all_histograms_sigma_ssfr.T), origin='lower', cmap='inferno',
            extent=(mid_mass[0], mid_mass[-1], mid_ssfr[0], mid_ssfr[-1]))
plt.colorbar()
plt.xlabel(r'$\log(\Sigma)$')
plt.ylabel(r'$\log(sSFR/yr)$')
plt.savefig('/home/pablo/Results_MANGA/resolved_sSFR_M_plane.pdf', bbox_inches='tight')

plt.figure()
plt.imshow(np.log10(all_histograms_sigma_sfr.T), origin='lower', cmap='inferno',
            extent=(mid_mass[0], mid_mass[-1], mid_sfr[0], mid_sfr[-1]))
plt.colorbar()
plt.xlabel(r'$\log(\Sigma)$')
plt.ylabel(r'$\log(SFR [M_\odot/yr])$')
plt.savefig('/home/pablo/Results_MANGA/resolved_SFR_M_plane.pdf', bbox_inches='tight')


norm_ssfr = np.sum(all_histograms_sigma_ssfr*dlgssfr[np.newaxis, :], axis=1)
conditional_M_ssfr = all_histograms_sigma_ssfr/norm_ssfr[:, np.newaxis]
# plt.contourf(mid_mass, mid_ssfr, np.log10(all_histograms.T), cmap='inferno',
#              levels=20)

fig, axs = plt.subplots(nrows=5, ncols=6, sharex=True, sharey=True, figsize=(10,10))
for ith in range(30):    
    ax = axs.flatten()[ith]
    ax.annotate(r'$\Sigma_* = {:.2}$'.format(mid_mass[ith]), xy=(.05, .85), 
                xycoords='axes fraction')
    ax.annotate(r'{:d}'.format(int(norm_ssfr[ith])), xy=(.05, .75), 
                xycoords='axes fraction', fontsize=9, color='b')
    ax.semilogy(mid_ssfr, conditional_M_ssfr[ith, :])
    ax.set_ylim(1e-3, 1)
    ax.set_xlim(-15.1, -6.1)
    # ax.set_xlabel(r'$\log(sSFR/yr)$')
    # ax.set_ylabel(r'$dp(sSFR|M_*)d\log(sSFR/yr)$')
fig.savefig('/home/pablo/Results_MANGA/conditional_probability.pdf')