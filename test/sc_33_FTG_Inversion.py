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
from gravmag import ftg
from gravmag.inversion import map_inversion

#-----------------------------------------
# Folders
#-----------------------------------------

block = False

pkl = '../data/gm_urg/'
png = 'png_urg/'
if not os.path.isdir(png): os.mkdir(png)

# Well locations for plotting
y_well = np.array([1056385.35]) # Easting
x_well = np.array([6881116.65]) # Northing
name_well = ['EPS-1']

#-----------------------------------------
# Read data
#-----------------------------------------

# Read TMA data
lab_gm = 'LdF_boug_tz'
fname_data = f'GM_Data_Band_Decomp_{lab_gm}.pkl'
with open (pkl + fname_data, 'rb') as fid:
    [data, data_band, data_filt, mod_geom] = pickle.load(fid)

#------------------------------
# Run inversion 
#------------------------------

eps = 1e-6
lam = 1e-3

kband = 0
# kband = 1

data.tma, ver = data_band[kband].tma, f'band_{kband}'
model = mod_geom

if   kband == 0:
    model.z[0] = 1000.0*np.ones_like(model.gx)
    model.z[1] = 7500.0*np.ones_like(model.gx)
    inc_data = 1
    inc_mod  = 12

elif kband == 1:
    model.z[0] = 1000.0*np.ones_like(model.gx)
    model.z[1] = 2500.0*np.ones_like(model.gx)
    inc_data = 1
    inc_mod  = 8

args = [eps]
green = ftg.green
gf_max = 1e9  # Controls the subdivision in chunks

tic = time.perf_counter()
inver, synt = map_inversion(green, data, model, *args, 
                            lam=lam, gf_max=gf_max, 
                            inc_data=inc_data, inc_mod=inc_mod,
                            resamp=False, verbose=1)

toc = time.perf_counter()
print(f'Time inversion: {toc-tic:.1f} sec')
print(f'RMS error: {synt.rms_err:.3f}')

#----------------------------
# Dump to pickle file
#----------------------------

fname_ut = f'GM_Inversion_{lab_gm}_inc_{inc_data}_{inc_mod}_{ver}.pkl'
with open (pkl + fname_ut, 'wb') as fid:
    pickle.dump([data, inver, synt], fid)

#---------------------------
# QC plotting
#---------------------------

# Resample for plotting
# inc_res = int(inc_mod*mod_geom.dx/data.dx)
# inver_res = inver.resample(inc_res, do_all=True, verbose=3)
inver_r = inver

scl = 1e-3
xtnt_d = scl*np.array([data.y[0], data.y[-1], data.x[0], data.x[-1]])
vmin_d, vmax_d = -35, 15

# Plot data
fig, axs = plt.subplots(1,2, figsize=(14, 4))

ax = axs.ravel()[0]
im = ax.imshow(data.tma.T, origin='lower', extent=xtnt_d,
               vmin=vmin_d, vmax=vmax_d) 
cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Gzz [Eo]')
ax.set_title(f'Data {lab_gm}')

ax = axs.ravel()[1]
im = ax.imshow(synt.tma.T, origin='lower', extent=xtnt_d,
               vmin=vmin_d, vmax=vmax_d) 
cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Gzz [Eo]')
ax.set_title(f'Synt {lab_gm}')

for ax in axs.ravel():
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'GM_Data_and_Synt_{lab_gm}_inc_{inc_data}_{inc_mod}_{ver}.png')

# PLot inversion
xtnt_i = scl*np.array([inver_r.y[0], inver_r.y[-1], inver_r.x[0], inver_r.x[-1]])

# PLot inversion
fig, ax = plt.subplots(1, figsize=(7.5, 4))
# interp = 'none'
interp = 'bicubic'
vmin_i, vmax_i = -200, 100

im = ax.imshow(inver_r.magn.T, origin='lower', extent=xtnt_i, 
               vmin=vmin_i, vmax=vmax_i, interpolation=interp)
cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('drho [kg/m3]')
ax.scatter(scl*y_well, scl*x_well, marker='o', c='r')
for name in name_well: ax.text(scl*y_well, scl*x_well, name)
ax.set_title('Density anomaly [kg/m3]')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'GM_Inversion_{lab_gm}_inc_{inc_data}_{inc_mod}_{ver}_{interp}.png')

#-------------------------
# PLot EM for comparison
#-------------------------

fname_csem = 'LdF_CSEM_Resistance_Grids.pkl'
# fname_mgi = 'All_Models.pkl'
with open (pkl + fname_csem, 'rb') as fid:
    csem = pickle.load(fid)

xtnt_csem = scl*np.array([csem.x[0], csem.x[-1], csem.y[0], csem.y[-1]])

# EM inversion
fig, axs = plt.subplots(1,2, figsize=(14, 4))
for ii, jj in enumerate([2, 3]):
    interp = 'none'
    # interp = 'bicubic'
    
    ax = axs.ravel()[ii]
    im = ax.imshow(csem.grd[jj], origin='lower', extent=xtnt_csem, 
                   interpolation=interp)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('vp [m/s]')
    ax.scatter(scl*y_well, scl*x_well, marker='o', c='r')
    for name in name_well: ax.text(scl*y_well, scl*x_well, name)
    ax.set_title(f'{csem.label[jj]}')    
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'LdF_Conductance.png')

plt.show(block=False)
