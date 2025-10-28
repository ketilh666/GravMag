# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 13:13:39 2025

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

# Mag inversion stuff
from gravmag import mag
from gravmag import ftg
from gravmag.earth import MagneticBackgroundField
from gravmag.inversion import map_inversion
from gravmag.modeling import map_modeling
from gravmag.meta import MapData
from khio.grid import write_esri_grid
from khio.grid import write_irap_grid
from cube import CubeData

#-------------------------
# Folders
#-------------------------

block = False

pkl = '../data/gm_urg/'
png = 'png_urg/'
if not os.path.isdir(png): os.mkdir(png)

#-------------------------
# Read input data
#-------------------------

# TMA data
fname_gm = 'LdF_GravMag_for_Inversion.pkl'
with open (pkl + fname_gm, 'rb') as fid:
    gm = pickle.load(fid)

# Inversion runs with z positive down
z_list = [-gm.z[0].T]
g_list = [gm.grd[jj].T for jj in range(len(gm.grd))]
data = MapData(gm.y, gm.x, z_list, g_list)
data.label = gm.label
data.label_z = [gm.label_z[0]]
# Get rid of the regional data
data.grd = data.grd[0:3]
data.label = data.label[0:3]

fname_csem = 'LdF_CSEM_Resistance_Grids.pkl'
with open (pkl + fname_csem, 'rb') as fid:
    csem = pickle.load(fid)

# Define model geometry for inversion
mod_geom = MapData(csem.y, csem.x, [0,0])
ix0 = int((mod_geom.x[0] - data.x[0])/mod_geom.dx) 
iy0 = int((mod_geom.y[0] - data.y[0])/mod_geom.dy) 
ny, nx = mod_geom.ny, mod_geom.nx
mod_geom.z = [None, None]
zwrk = -gm.z[1].T
mod_geom.z[0] = zwrk[iy0:iy0+ny, ix0:ix0+nx]
mod_geom.z[1] = 5500.0*np.ones_like(mod_geom.z[0]) 
mod_geom.label_z = ['Top_Basement', 'Curie_Depth']

# Which dataset to run?
kkk = 0
# kkk = 1
data.tma = data.grd[kkk]
lab_gm = data.label[kkk]

#-------------------------
# Band decomposition
#-------------------------

# Blakely top and base from power spectrum
zz_nofilt  = data.power_depth(k_min=0, k_max=0.05*np.pi/data.dx, kplot=True )
zz_nofilt['fig'].savefig(png + f'All_Data_Power_Depth_{lab_gm}.png')

# Band decomposition
k_nyq = np.pi/data.dx
rk = [1/8, 1/4, 1/2] # Rel Nyquist
nband = len(rk)

data_filt = [None for r in rk]
data_band = [None for r in rk]
zz = [None for r in rk]
   
kplot_filt = False
kplot_band = False

# High-cut filter
ltap =  5 # Pre   fft taper
ntap = -1 # Post ifft taper
for jj in range(nband):
    data_filt[jj] = data.bandpass(0, rk[jj]*k_nyq, ltap=ltap, ntap=ntap, kplot=False)
    if kplot_filt:
        jnk = data_filt[jj].power_depth(k_min=0, k_max=rk[jj]*k_nyq, kplot=True)

# First band is the base layer
jj = 0
data_band[jj] = MapData(data.x, data.y, data.z)
data_band[jj].tma = data_filt[jj].tma
if kplot_band:
    zz[jj] = data_band[jj].power_depth(k_min=0.0*k_nyq, k_max=rk[jj]*k_nyq, kplot=True)

# Band 1 to nband-1
for jj in range(1, nband):
    data_band[jj] = MapData(data.x, data.y, data.z)
    data_band[jj].tma = data_filt[jj].tma - data_filt[jj-1].tma
    if kplot_band:
        zz[jj] = data_band[jj].power_depth(k_min=rk[jj-1]*k_nyq, k_max=rk[jj]*k_nyq, kplot=True)

#----------------------------
# Dump to pickle file
#----------------------------

B0, inc, dec = 43856.7, 63.8, -1.69 # B0 in nT, assuming RTP
earth = MagneticBackgroundField(B0, inc, dec)
mod_geom.vt = earth.vt
mod_geom.B0, mod_geom.inc, mod_geom.dec = B0, inc, dec

fname = f'GM_Data_Band_Decomp_{lab_gm}.pkl'
with open (pkl + fname, 'wb') as fid:
    pickle.dump([data, data_band, data_filt, mod_geom], fid)

#---------------------------
# QC plotting
#---------------------------

vmin_list = [-70, -30]
vmax_list = [ 70,  15]

cmap = 'jet'
scl = 1e-3
xtnt = scl*np.array([data.y.min(), data.y.max(), data.x.min(), data.x.max()])
# Make symmetric coloscale
vmax = vmax_list[kkk]
vmin = vmin_list[kkk]

# Hi cut filtered
fig, axs = plt.subplots(2,2, figsize=(12,7))
title_list = [f'{lab_gm} all data'] + [f'Low Pass {lab_gm} {rk[ii]}*k_nyq' for ii in range(len(rk))]
for jj, dd in enumerate([data] + data_filt):
    ax = axs.ravel()[jj]
    im = ax.imshow(dd.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)    
    cb = ax.figure.colorbar(im, ax=ax)
    # cb.set_label('TMA [nT]')
    ax.set_title(title_list[jj])
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'GM_data_filtered_{lab_gm}.png')

# Band decomp
fig, axs = plt.subplots(2,2, figsize=(12,7))
title_list = [f'{lab_gm} all data'] + [f'{lab_gm} band_{ii}' for ii in range(len(rk))]
for jj, dd in enumerate([data] + data_band):
    ax = axs.ravel()[jj]
    im = ax.imshow(dd.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)    
    cb = ax.figure.colorbar(im, ax=ax)
    # cb.set_label('TMA [nT]')
    ax.set_title(title_list[jj])
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'GM_data_bands_{lab_gm}.png')

# Band decomp
fig, axs = plt.subplots(2,2, figsize=(12,7))
title_list = [f'{lab_gm} all data'] + [f'{lab_gm} band_{ii}' for ii in range(len(rk))]
for jj, dd in enumerate([data] + data_band):
    ax = axs.ravel()[jj]
    im = ax.imshow(dd.tma.T, origin='lower', extent=xtnt, cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax)
    # cb.set_label('TMA [nT]')
    ax.set_title(title_list[jj])
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'GM_data_bands_{lab_gm}_indep_scl.png')

plt.show(block=block)





















