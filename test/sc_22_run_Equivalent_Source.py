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

# import khio.grid as grid
# import khio.xyz as xyz

# Mag inversion stuff
from gravmag import mag
from gravmag.inversion import map_inversion
from gravmag.modeling import map_modeling
from gravmag.meta import MapData
from khio.grid import write_esri_grid
from gravmag.common import gridder

#--------------------------
# Folders
#--------------------------

pkl = '../data/eq_source/pkl/'
png = 'png_eq_source/'
if not os.path.isdir(png): os.mkdir(png)
if not os.path.isdir(pkl): os.mkdir(pkl)

write_grid = False
write_pkl  = True

#-----------------------------------------------------
# Read data and model geometry
# At this point, the model only defines geometry
#-----------------------------------------------------

fname_in = 'LARGE_TMA_Data_taper_0.pkl'
print(f'Input data file: {pkl + fname_in}')
with open (pkl + fname_in, 'rb') as fid:
    [data, data_band, data_filt, model] = pickle.load(fid)
    
    model.z[0] += 4000.0
    # model.z[1] += 4000.0

#--------------------------------------------------------------------
# Expand edges and mirror data mirroring to suppress edge effects
#--------------------------------------------------------------------

led = 20                             # Data  extension
lem = int(led*(data.dx/model.dx)) # Model extension

data_ext = data.mirror_edges(led, verbose=1, kplot=False)
model_ext = model.mirror_edges(lem, verbose=1)

#------------------------------
# Run inversion 
#------------------------------

eps = 1e-6
lam = 1e-3

inc_data = 2
inc_mod  = 2*inc_data

# Assumt RTP
# green = mag.green_rtp  # Slow 3D function
green = mag.green_1d    # Fast 1D function
args = [eps]

gf_max = 1e12  # Controls subdivision in chunks (make it huge)
niter  = 0     # Run linear inversion only

tic = time.perf_counter()
inver_ext, synt_ext = map_inversion(green, data_ext, model_ext, *args,
                                    lam=lam, gf_max=gf_max, nnn=1, 
                                    inc_data=inc_data, inc_mod=inc_mod,
                                    resamp=False, verbose=3)

toc = time.perf_counter()
print(f'Time inversion: {toc-tic:.1f} sec')
print(f'RMS error: {synt_ext.rms_err:.3f}')

#-------------------------------------------------------
#  Restore the original model sampling (for plotting)
#-------------------------------------------------------

xi = inver_ext.gx.ravel()
yi = inver_ext.gy.ravel()
vi = inver_ext.magn.ravel()
model_ext.magn = gridder(xi, yi, vi, model_ext.gx, model_ext.gy)

#-----------------------------------
# Forward modeling at new datums
#-----------------------------------

z_list = [0, 1200, 2400, 3600, 4800, 6000]
geom_ext = MapData(data_ext.x, data_ext.y, z_list)

tic = time.perf_counter()
synt_list_ext = map_modeling(green, geom_ext, inver_ext, *args, 
                             gf_max=gf_max, nnn=1,  
                             inc_data=inc_data, inc_mod=1, 
                             resamp=True, snap=True, verbose=1)

toc = time.perf_counter()
print(f'Time modeling: {toc-tic:.1f} sec')
print(f'n_datum: {len(synt_list_ext)}')

#------------------------------------------------------
# Remove added edge nodes before output and plotting
#------------------------------------------------------

remove_edges = True
if remove_edges:
    
    model = model_ext.remove_edges(verbose=1)
    synt = synt_ext.remove_edges(led, verbose=1)
    
    synt_list = [None for st in synt_list_ext]
    for jj, st in enumerate(synt_list_ext):
        synt_list[jj] = st.remove_edges(led, verbose=1)
            
else:
    model = model_ext
    synt = synt_ext
    synt_list = synt_list_ext

#----------------------------
# Dump to pickle file
#----------------------------

if write_pkl:
    
    fname_ut = f'LARGE_TMA_Edge_Mirrored_led_{led}.pkl'
    with open (pkl + fname_ut, 'wb') as fid:
        pickle.dump([data, model, synt, synt_list], fid)
        
    with open (pkl + fname_ut, 'rb') as fid:
        [data, model, synt, synt_list] = pickle.load(fid)

#---------------------------
# QC plotting
#---------------------------

cmap = 'jet'
scl = 1e-3
xtnt = scl*np.array([data.y.min(), data.y.max(), 
                     data.x.min(), data.x.max()])
# Make symmetric coloscale
vmax = 300.0
vmin = -vmax

xr1, xr2 = data.y[0], data.y[-1]
yr1, yr2 = data.x[0], data.x[-1]
rect_x = scl*np.array([xr1, xr2, xr2, xr1, xr1])
rect_y = scl*np.array([yr1, yr1, yr2, yr2, yr1])

fig, axs = plt.subplots(3,3, figsize=(14,12))

ax = axs.ravel()[0]
im = ax.imshow(data.tma.T, origin='lower', extent=xtnt, 
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
    ax.plot(rect_x, rect_y, 'm-')
    ax.axis('scaled')
    ax.set_xlabel('y (Easting) [km]')
    ax.set_ylabel('x (Northing) [km]')

z_top, z_base = np.mean(model.z[0]), np.mean(model.z[1])
fig.suptitle(f'Equivalent source redatuming (z_top={z_top:.0f}m, z_base={z_base:.0f}m, inc_mod={inc_mod}, inc_data={inc_data}, led={led})')
fig.tight_layout(pad=1.0)
fig.savefig(png + f'Edge_Tapered_led_{led}.png')

# DUmp results to esri grid
if write_grid:
    
    esri = f'esri_led_{led}/'
    if not os.path.isdir(esri): os.mkdir(esri)

    # Save figure together with data
    fig.savefig(esri + f'Edge_Tapered_led_{led}.png')

    ierr=write_esri_grid(esri+'data_tma.esri', 
                         data.y, data.x, data.tma.T, verbose=1)

    ierr=write_esri_grid(esri+f'equivalent_source_led_{led}.esri', 
                         model.y, model.x, model.magn.T, verbose=1)

    ierr=write_esri_grid(esri+f'synt_tma+240m_led_{led}.esri', 
                         synt.y, synt.x, synt.tma.T, verbose=1)

    for jj, st in enumerate(synt_list):
        ierr=write_esri_grid(esri+f'synt_tma_{st.z[0][0,0]:.0f}m_led_{led}.esri', 
                             st.y, st.x, st.tma.T, verbose=1)
        print(st.z[0][0,0], ierr)

plt.show(block=False)