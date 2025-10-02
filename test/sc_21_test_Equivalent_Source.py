# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:36:09 2025

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

# Mag inversion stuff
from gravmag import mag
from gravmag.earth import MagneticBackgroundField
from gravmag.inversion import map_inversion
from gravmag.modeling import map_modeling
from gravmag.meta import MapData
from gravmag.meta import xy_taper
from khio.grid import write_esri_grid
from khio.grid import write_irap_grid
from gravmag.common import gridder

#-----------------------
# Folders
#-----------------------

pkl = '../data/pkl_eq_source/'
png = f'png_eq_source/'
if not os.path.isdir(png): os.mkdir(png)
if not os.path.isdir(pkl): os.mkdir(pkl)

write_grid = False
write_pkl  = False

block = False

#-----------------------------------------
# Read data
#-----------------------------------------

fname_in = f'LARGE_TMA_Data_taper_0.pkl'
print(f'Input data file: {pkl + fname_in}')
with open (pkl + fname_in, 'rb') as fid:
    [data, data_band, data_filt, mod_geom] = pickle.load(fid)
    
    # Move source layer
    mod_geom.z[0] += 4000.0
    mod_geom.z[1] += 4000.0

#------------------------------
# Run inversion 
#------------------------------

eps = 1e-6
lam = 1e-3

model = mod_geom
inc_data = 2
inc_mod  = 2*inc_data

data_inv, ver = data, '' # Use ful spectrum

# Assumt RTP (and 1D)
# green = mag.green_rtp
green = mag.green_1d
args = [eps]
gf_max = 1e10  # Controls the subdivision in chunks
niter  = 0    # Run linear inversion only

tic = time.perf_counter()
inver, synt = map_inversion(green, data_inv, model, *args,
                            lam=lam, gf_max=gf_max, nnn=1, 
                            inc_data=inc_data, inc_mod=inc_mod,
                            resamp=False, verbose=1)

toc = time.perf_counter()
print(f'Time inversion: {toc-tic:.1f} sec')
print(f'RMS error: {synt.rms_err:.3f}')

#-------------------------------------------------------
#  Restore the original model sampling (for plotting)
#-------------------------------------------------------

xi = inver.gx.ravel()
yi = inver.gy.ravel()
vi = inver.magn.ravel()
model.magn = gridder(xi, yi, vi, model.gx, model.gy)

#-----------------------------------
# Forward modeling at new datums
#-----------------------------------

z_list = [0, 1200, 2400, 3600, 4800, 6000]
geom = MapData(data.x, data.y, z_list)

tic = time.perf_counter()
synt_list = map_modeling(green, geom, model, *args, 
                         gf_max=1e9, nnn=1,  
                         inc_data=inc_data, inc_mod=2, 
                         resamp=True, snap=True, verbose=1)

toc = time.perf_counter()
print(f'Time modeling: {toc-tic:.1f} sec')
print(f'n_datum: {len(synt_list)}')

#----------------------------
# Dump to pickle file
#----------------------------

if write_pkl:
    with open (pkl + f'LARGE_TMA_Redat_inc_mod_{inc_mod}{ver}.pkl', 'wb') as fid:
        pickle.dump([data_inv, inver, synt, synt_list], fid)
        
    # with open (pkl + f'LARGE_TMA_Redat_inc_mod_{inc_mod}{ver}.pkl', 'rb') as fid:
    #     [data_inv, inver, synt, synt_list] = pickle.load(fid)

#---------------------------
# QC plotting
#---------------------------

# Resample for plotting
# inc_res = int(inc_mod*mod_geom.dx/data.dx)
# inver_res = inver.resample(inc_res, do_all=True, verbose=3)
inver_res = inver

cmap = 'jet'
scl = 1e-3
xtnt = scl*np.array([data.y.min(), data.y.max(), data.x.min(), data.x.max()])
# Make symmetric coloscale
vmax = 300.0
vmin = -vmax

fig, axs = plt.subplots(3,3, figsize=(14,12))

ax = axs.ravel()[0]
im = ax.imshow(data_inv.tma.T, origin='lower', extent=xtnt, 
               vmin=vmin, vmax=vmax, cmap=cmap)
cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('TMA [nT]')
ax.set_title('TMA [nT]')

ax = axs.ravel()[1]
# im = ax.imshow(inver_res.magn.T, origin='lower', extent=xtnt)
im = ax.imshow(model.magn.T, origin='lower', extent=xtnt)
cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('M [A/m]')
ax.set_title('Magnetization [A/m]')

ax = axs.ravel()[2]
im = ax.imshow(synt.tma.T, origin='lower', extent=xtnt, 
               vmin=vmin, vmax=vmax, cmap=cmap)
cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('TMA [nT]')
ax.set_title('Synt TMA [nT]')

for jj, st in enumerate(synt_list):
    ax = axs.ravel()[jj+3]
    im = ax.imshow(st.tma.T, origin='lower', extent=xtnt, 
                   vmin=vmin, vmax=vmax, cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('TMA [nT]')
    ax.set_title(f'Synt TMA z={st.z[0][0,0]}m [nT]')

for ax in axs.ravel():
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

z_top, z_base = np.mean(model.z[0]), np.mean(model.z[1])
fig.suptitle(f'Equivalent source redatuming (z_top={z_top:.0f}m, z_base={z_base:.0f}m, inc_mod={inc_mod}, inc_data={inc_data})')
fig.tight_layout(pad=1.0)
fig.savefig(png + f'Equi_Source_inc_mod_{inc_mod}_inc_data_{inc_data}{ver}.png')

#-------------------------------
# Dump results to esri grids
#-------------------------------

if write_grid:
    
    esri = f'esri_incmod{inc_mod}/'
    if not os.path.isdir(esri): os.mkdir(esri)

    # Save figure together with data
    fig.savefig(esri + f'Equi_Source_inc_mod_{inc_mod}{ver}.png')

    ierr=write_esri_grid(esri+'data_tma.esri', 
                         data_inv.y, data_inv.x, data_inv.tma.T, verbose=1)

    ierr=write_esri_grid(esri+f'equivalent_source_incmod{inc_mod}.esri', 
                         inver_res.y, inver_res.x, inver_res.magn.T, verbose=1)

    ierr=write_esri_grid(esri+f'synt_tma+230m_incmod{inc_mod}.esri', 
                         synt.y, synt.x, synt.tma.T, verbose=1)

    for jj, st in enumerate(synt_list):
        ierr=write_esri_grid(esri+f'synt_tma_{st.z[0][0,0]:.0f}m_incmod{inc_mod}.esri', 
                             st.y, st.x, st.tma.T, verbose=1)
        print(st.z[0][0,0], ierr)

plt.show(block=block)