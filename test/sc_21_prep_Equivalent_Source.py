# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:36:09 2025

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import khio.xyz as xyz

# Mag inversion stuff
from gravmag.meta import MapData
from gravmag.earth import MagneticBackgroundField
from gravmag.meta import xy_taper
# from gravmag.common import gridder

#----------------------
# Folders
#----------------------

xyz_dir = '../data/eq_source/xyz/'
fname = 'LARGE_Delta_T_Alti230m.XYZ'

pkl = '../data/eq_source/pkl/'
png = 'png_eq_source/'
if not os.path.isdir(png): os.mkdir(png)
if not os.path.isdir(pkl): os.mkdir(pkl)

block = False

#-------------------------
# Read mag data
#-------------------------

wrk, hd = xyz.read_xyz(xyz_dir+fname)
n_east = int(np.where(np.diff(wrk[:,0]) < 0)[0][0] + 1)
n_nort = int(np.round(wrk.shape[0]/n_east))

nn = n_east*n_nort
print(f'CHECKING: n_east, n_nort, nn-wrk.shape[0] = {n_east:.0f}, {n_nort:.0f}, {nn-wrk.shape[0]:.0f}')

# On regular grid
g_east  = wrk[:,0].reshape(n_nort, n_east)
g_nort  = wrk[:,1].reshape(n_nort, n_east)
east, nort = g_east[0,:], g_nort[:,0]
tma = wrk[:,2].reshape(n_nort, n_east)

# Flight altitude
zalt = -240*np.ones_like(tma)

#---------------------------
# Create MapData objects
#    x = Northing
#    y = Easting
#    z is positive down
#---------------------------

# Data
data = MapData(nort, east, zalt.T)
data.tma = tma.T 

# Model geometry
inc = 1
n_east_mod = int((data.ny-1)/inc + 1)
n_nort_mod = int((data.nx-1)/inc + 1)
z_seabed =   230.0*np.ones((n_east_mod, n_nort_mod))
z_top  = 10000 # Top  Magnetic layer
z_base = 18000 # Base Magnetic layer
mod_geom = MapData(nort[::inc], east[::inc], [z_top, z_base])

# Magnetic background field
B0, inc, dec = 52000, 90.0, 0.0 # B0 in nT, assuming RTP
earth = MagneticBackgroundField(B0, inc, dec)
mod_geom.vt = earth.vt
mod_geom.B0, mod_geom.inc, mod_geom.dec = B0, inc, dec

#------------------------------
# Band decomposition 
#------------------------------

# Top and base from power spectrum (Blakely, 1996)
zz_nofilt  = data.power_depth(k_min=0, k_max=0.1*np.pi/data.dx, kplot=True )
zz_nofilt['fig'].savefig(png + 'All_Data_Power_Depth.png')

# Band decomposition
k_nyq = np.pi/data.dx
rk = [1/4, 2/4, 3/4] # Rel Nyquist wavenumber
nband = len(rk)

data_filt = [None for r in rk]
data_band = [None for r in rk]
zz = [None for r in rk]
   
kplot_filt = False
kplot_band = False

# High-cut filter
ltap = 5  # Pre   fft taper
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

ntap_tma = 0
data.tma = xy_taper(data.tma, ntap_tma)

ctap = str(ntap_tma)
with open (pkl + f'LARGE_TMA_Data.pkl', 'wb') as fid:
    pickle.dump([data, data_band, data_filt, mod_geom], fid)

#---------------------------
# QC plotting
#---------------------------

cmap = 'jet'
scl = 1e-3
xtnt = scl*np.array([data.y.min(), data.y.max(), data.x.min(), data.x.max()])
# Symmetric coloscale
vmax = np.max(np.abs(data.tma))
vmin = -vmax

# Input data
fig, axs = plt.subplots(1,2, figsize=(8,4))

ax = axs.ravel()[0]
ax.scatter(scl*wrk[:,0], scl*wrk[:,1], c=wrk[:,2], 
           vmin=vmin, vmax=vmax, cmap=cmap)
ax.scatter(scl*wrk[:n_east//10 ,0], scl*wrk[:n_east//10 ,1], c='r', marker='o')
ax.set_title('TMA (xyz)')

ax = axs.ravel()[1]
im = ax.imshow(data.tma.T, origin='lower', extent=xtnt, 
               vmin=vmin, vmax=vmax, cmap=cmap)
ax.set_title('TMA (grid)')

for ax in axs.ravel():
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')
    
fig.tight_layout(pad=1.0)
fig.savefig(png + 'Data_QC.png')

# Hi cut filtered
fig, axs = plt.subplots(2,2, figsize=(8,7))
title_list = ['TMA all data'] + [f'TMA {rk[ii]}*k_nyq' for ii in range(len(rk))]
for jj, dd in enumerate([data] + data_filt):
    ax = axs.ravel()[jj]
    im = ax.imshow(dd.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)    
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label('TMA [nT]')
    ax.set_title(title_list[jj])
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + 'Data_filtered.png')

# Band decomp
fig, axs = plt.subplots(2,2, figsize=(8,7))
title_list = ['TMA all data'] + [f'TMA band_{ii}' for ii in range(len(rk))]
for jj, dd in enumerate([data] + data_band):
    ax = axs.ravel()[jj]
    im = ax.imshow(dd.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)    
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label('TMA [nT]')
    ax.set_title(title_list[jj])
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + 'Data_bands.png')

plt.show(block=block)