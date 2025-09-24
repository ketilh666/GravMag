# -*- coding: utf-8 -*-
"""
Test the inversion with magnetic anomalies at two depths:
 o Shallow: Mound-like anomalies
 o Deep: Ridge-like anomaly

Created on Thu Jan  7 13:10:03 2021
@author: kehok@equinor.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import time
import os

from gravmag.inversion import map_inversion
from gravmag import mag
from gravmag.meta import MapData
from gravmag.inversion import image_stack

#------------------------------
# Run inversion test
#------------------------------

block = False

pkl = '../Data/pkl/'
png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

ktest  = 1 # The one and only 2-layer test
fname1 = pkl+'test'+str(ktest)+'_tma_data_for_inv_2layer.pkl'
with open(fname1,'rb') as fid:
    data1, mod_geom1 = pickle.load(fid)

# 2-layer case: Make a plot for reference only
zz_nofilt  = data1.power_depth(k_min=0, k_max=0.4*np.pi/data1.dx, kplot=True )

# New order
k_nyq = np.pi/data1.dx
rk = [0.15, 0.30, 0.50] # Rel Nyquist
nband = len(rk)
lam_dict = {0: 1e-3, 1: 1e-3, 2: 1e-3}
inc_mod  = [4, 2, 1]
inc_data = [4, 2, 1]

zz_jnk, data_filt = [], []    
# High-cut filter
for jj in range(nband):
    data_filt.append(data1.bandpass(0, rk[jj]*k_nyq, kplot=False))
    zz_jnk.append(data_filt[jj].power_depth(k_min=0, k_max=rk[jj]*k_nyq, kplot=True))

# split in bands and esimate depths
zz = [np.nan for jj in range(len(rk))]
data_band = []
for jj in range(nband):
    data_band.append(MapData(data1.x, data1.y, data1.z))
        
# First band is the base layer
jj = 0
data_band[jj].tma = data_filt[jj].tma
zz[jj] = data_band[jj].power_depth(k_min=0.0*k_nyq, k_max=rk[jj]*k_nyq, kplot=True)
# Band 1 to nband-1
for jj in range(1, nband):
    data_band[jj].tma = data_filt[jj].tma - data_filt[jj-1].tma
    zz[jj] = data_band[jj].power_depth(k_min=rk[jj-1]*k_nyq, k_max=rk[jj]*k_nyq, kplot=True)

# Init output model objects
mod_list = [np.nan for jj in range(nband)]
#mod_list[0] = MapData(mod_geom1.x, mod_geom1.y, [0,1500])
#mod_list[1] = MapData(mod_geom1.x, mod_geom1.y, [0,1000])
#mod_list[2] = MapData(mod_geom1.x, mod_geom1.y, [0, 500])
mod_list[0] = MapData(mod_geom1.x, mod_geom1.y, [1000,1500])
mod_list[1] = MapData(mod_geom1.x, mod_geom1.y, [   0,1000])
mod_list[2] = MapData(mod_geom1.x, mod_geom1.y, [   0, 500])

# Run inversion
func_grn = mag.green
func_jac = mag.jacobi
inc, dec = 90, 0
vt_e = vt_m = mod_geom1.vt
eps = 1e-6

tic = time.perf_counter()
inver = [np.nan for jj in range(nband)]
synt  = [np.nan for jj in range(nband)]
for jj in range(0,nband):
    print('### Submodel {}'.format(jj))
    lam = lam_dict[jj]
    gf_max = 1e12
    mod_list[jj].vt = vt_e

    inver[jj], synt[jj] = map_inversion(func_grn, data_band[jj], mod_list[jj], 
                                        vt_e, vt_m, eps, lam=lam_dict[jj],
                                        func_jac=func_jac, gf_max=gf_max, nnn=1, 
                                        inc_data=inc_data[jj], inc_mod=inc_mod[jj],
                                        verbose=1)

toc = time.perf_counter()
time_inv = toc - tic
print('Time inversion: {}'.format(time_inv))

# Stack partial inversions
inv_stk = image_stack(inver[0:])

# PLot data and inversion result
for jj in range(len(rk)):
    
    clim_on = False
    dmin, dmax = [-10,-60, -100], [5, 30, 300]
    mmin, mmax = [-6,-2, -1, -5], [3, 1, 2, 3]
    
    data, kh, sd = data_band[jj], inver[jj], synt[jj]
    data.tma = np.real(data.tma)
    fig, (ax, bx, cx) = plt.subplots(1,3,figsize=(18,6)) 
    zr, delz = data.z[0][0,0], kh.z[1][0,0] - kh.z[0][0,0] 
    fig.suptitle('Band {}: dx=dy={}, z2-z1={}, zr={}'.format(jj, kh.dx, delz, zr), fontsize=14)

    xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
    im = ax.imshow(data.tma.T, origin='lower', extent=xtnt, cmap=cm.jet, interpolation='bicubic')
    if clim_on: cm.ScalarMappable.set_clim(im,vmin=dmin[jj],vmax=dmax[jj])
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
    ax.set_xlim(data.y[0], data.y[-1])
    ax.set_ylim(data.x[0], data.x[-1])
    ax.set_ylabel('northing x [m]')
    ax.set_xlabel('easting y [m]')
    ax.set_title('Mag data [nT]: inc={}deg, dec={}deg'.format(inc, dec))
 
    xtnt2 = [kh.y[0], kh.y[-1], kh.x[0], kh.x[-1]]
    im = bx.imshow(kh.magn.T, origin='lower', extent=xtnt2, cmap=cm.jet, interpolation='bicubic')
    if clim_on: cm.ScalarMappable.set_clim(im,vmin=mmin[jj],vmax=mmax[jj])
    cb = bx.figure.colorbar(im, ax=bx, shrink=0.9) 
    bx.set_xlim(data.y[0], data.y[-1])
    bx.set_ylim(data.x[0], data.x[-1])
    bx.set_ylabel('northing x [m]')
    bx.set_xlabel('easting y [m]')
    bx.set_title('Mag inversion [A/m]: inc={}deg, dec={}deg'.format(inc, dec))

    im = cx.imshow(sd.tma.T, origin='lower', extent=xtnt, cmap=cm.jet, interpolation='bicubic')
    if clim_on: cm.ScalarMappable.set_clim(im,vmin=dmin[jj],vmax=dmax[jj])
    cb = ax.figure.colorbar(im, ax=cx, shrink=0.9) 
    ax.set_xlim(sd.y[0], sd.y[-1])
    ax.set_ylim(sd.x[0], sd.x[-1])
    ax.set_ylabel('northing x [m]')
    ax.set_xlabel('easting y [m]')
    ax.set_title('Synt data [nT]: inc={}deg, dec={}deg'.format(inc, dec))

    fig.savefig(png+'test'+str(ktest)+'_band'+str(jj)+'_inv_2layer_varz1.png')
    
# plot vertical slice
kh = inv_stk
fig, (ax, bx) = plt.subplots(2,1,figsize=(5,8)) 

ix = data.nx//2
ax.plot(data_band[0].y, data_band[0].tma[:,ix], 'r-', label=' 0-15% Nyq')
ax.plot(data_band[1].y, data_band[1].tma[:,ix], 'b-', label='15-30% Nyq')
ax.plot(data_band[2].y, data_band[2].tma[:,ix], 'k-', label='30-60% Nyq')
ax.set_xlim(-6000, 6000)
ax.set_ylabel('depth z [m]')
ax.set_xlabel('easting y [m]')
ax.set_title('Mag data bands: x = {}m'.format(data_band[1].x[ix]))
ax.legend()

xtnt2 = [kh.y[0], kh.y[-1], kh.z[-1], kh.z[0]]
jx = inv_stk.nx//2
im = bx.imshow(kh.magn[:,:,jx], origin='upper', extent=xtnt2, 
               aspect=5, interpolation='bicubic', cmap=cm.jet)
if clim_on: cm.ScalarMappable.set_clim(im,vmin=mmin[-1],vmax=mmax[-1])
cb = bx.figure.colorbar(im, ax=bx, shrink=0.7) 
bx.set_xlim(-6000, 6000)
bx.set_ylim(kh.z[-1], 0)
bx.set_ylabel('depth z [m]')
bx.set_xlabel('easting y [m]')
bx.set_title('Mag inv stack: x = {}m'.format(kh.x[jx]))

fig.savefig(png+'test'+str(ktest)+'_stack_inv_2layer_varz1.png')

plt.show(block=block)    
