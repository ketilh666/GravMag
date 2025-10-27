# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 08:27:51 2025

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

# Mag inversion stuff
from gravmag import mag
from gravmag.inversion import map_inversion
from gravmag.modeling import map_modeling
from khio.grid import write_esri_grid
from gravmag.common import gridder

#--------------------------
# Folders
#--------------------------

block = False

pkl = '../data/eq_source/pkl/'
png = 'png_eq_source/'
if not os.path.isdir(png): os.mkdir(png)
if not os.path.isdir(pkl): os.mkdir(pkl)

#--------------------------
# Read data
#--------------------------

models = {}
datas = {}
synts = {}
synt_lists = {}

led_list = [0, 20, 60]
for led in led_list:

    fname = f'LARGE_TMA_Edge_Mirrored_led_{led}.pkl'
    print(f'fname = {fname}')
    with open (pkl + fname, 'rb') as fid:
        [datas[led], models[led], synts[led], synt_lists[led]] = pickle.load(fid)

#--------------------------
# Plot comparison
#--------------------------

led = led_list[0]
z_list = [synt_lists[led][jj].z[0][0,0] for jj in range(len(synt_lists[led]))]

cmap = 'jet'
scl = 1e-3
data = datas[led]
model = models[led]
xtnt = scl*np.array([data.y.min(), data.y.max(), 
                     data.x.min(), data.x.max()])
# Make symmetric coloscale
vmax = 300.0
vmin = -vmax

xr1, xr2 = data.y[0], data.y[-1]
yr1, yr2 = data.x[0], data.x[-1]
rect_x = scl*np.array([xr1, xr2, xr2, xr1, xr1])
rect_y = scl*np.array([yr1, yr1, yr2, yr2, yr1])

ncol, nrow = 3,3
fig, axs = plt.subplots(nrow, ncol, figsize=(14,12))

led1 = led_list[2]
synt_list1 = synt_lists[led1]
for ii, jj in enumerate([0,3,5]):
    
    st1 = synt_list1[jj]
    ax = axs.ravel()[ii]
    im = ax.imshow(st1.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('TMA [nT]')
    ax.set_title(f'led={led1}, z={z_list[jj]}m')

led = led_list[1]
synt_list = synt_lists[led]
for ii, jj in enumerate([0,3,5]):
    
    st1 = synt_list1[jj]
    st = synt_list[jj]
    ax = axs.ravel()[ii+1*ncol]
    im = ax.imshow(st.tma.T-st1.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('TMA [nT]')
    ax.set_title(f'Diff led={led1}, led={led}, z={z_list[jj]}m')

led = led_list[0]
synt_list = synt_lists[led]
for ii, jj in enumerate([0,3,5]):
    
    st1 = synt_list1[jj]
    st = synt_list[jj]
    ax = axs.ravel()[ii+2*ncol]
    im = ax.imshow(st.tma.T-st1.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('TMA [nT]')
    ax.set_title(f'Diff led={led1}, led={led}, z={z_list[jj]}m')

for ax in axs.ravel():
    ax.plot(rect_x, rect_y, 'm-')
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

z_top, z_base = np.mean(model.z[0]), np.mean(model.z[1])
inc_data = 2
inc_mod  = 2*inc_data
fig.suptitle(f'Comparison Equivalent source redatuming (z_top={z_top:.0f}m, z_base={z_base:.0f}m, inc_mod={inc_mod}, inc_data={inc_data}, led={led})')
fig.tight_layout(pad=1.0)
fig.savefig(png + 'Difference_Edge_Tapers.png')

plt.show(block=block)