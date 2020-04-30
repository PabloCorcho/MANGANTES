#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:31:44 2020

@author: pablo
"""


from astropy.io import fits
import numpy as np 
from matplotlib import pyplot as plt


catalog_path= '/home/pablo/obs_data/MANGA/Pipe3D/manga.Pipe3D-v2_4_3.fits'

hdul = fits.open(catalog_path)


web_path = 'https://dr15.sdss.org/sas/dr15/manga/spectro/pipe3d/v2_4_3/2.4.3/'

mangaids = np.array(hdul[1].data['mangaid'], dtype=str)
plate = np.array(hdul[1].data['plate'], dtype=str)


path_to_cubes = []
for ith in range(len(mangaids)):
    path_to_cubes.append(web_path + plate[ith] +'/'+ mangaids[ith]+ '.Pipe3D.cube.fits.gz')


with open('/home/pablo/obs_data/MANGA/Pipe3D/manga_pip3_urls.txt', 'w') as f:
    for i in range(len(path_to_cubes)):
        f.write(path_to_cubes[i]+'\n')


original_cube_path = 'https://dr15.sdss.org/sas/dr15/manga/spectro/redux/v2_4_3/'

original_cube_path_extension = '-LINCUBE.fits.gz'
original_path_to_cubes = []

for ith in range(len(mangaids)):
    original_path_to_cubes.append(original_cube_path + plate[ith] +'/stack/'+ \
                                  mangaids[ith]+ original_cube_path_extension)
with open('/home/pablo/obs_data/MANGA/manga_urls.txt', 'w') as f:
    for i in range(len(original_path_to_cubes)):
        f.write(original_path_to_cubes[i]+'\n')
