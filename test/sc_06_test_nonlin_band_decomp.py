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
from gravmag.common import plot_gauss_newton

#------------------------------
# Run inversion test
#------------------------------

block = True

pkl = '../Data/pkl/'
png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

ktest = 1 # The one and only 2-layer test

# Input from files
#fname0 = 'pkl/'+'test'+str(ktest)+'_tma_data_for_inv_1layer.pkl'
#with open(fname0,'rb') as fid:
#    data0, mod_geom0 = pickle.load(fid)
fname1 = pkl + 'test'+str(ktest)+'_tma_data_for_inv_2layer.pkl'
with open(fname1,'rb') as fid:
    data1, mod_geom1 = pickle.load(fid)

## 1-layer 
#zz0 = data0.power_depth(k_min=0, k_max=0.3*np.pi/data0.dx, kplot=True)

# 2-layer case
zz_nofilt  = data1.power_depth(k_min=0, k_max=0.4*np.pi/data1.dx, kplot=True )

#rk = [0.60, 0.40, 0.20]
rk = [0.60, 0.30, 0.15]
k_nyq = np.pi/data1.dx
nband = len(rk)

zz_jnk, data_filt = [], []    
# High-cut filter
for jj in range(len(rk)):
    data_filt.append(data1.bandpass(0, rk[jj]*k_nyq, kplot=False))
    zz_jnk.append(data_filt[jj].power_depth(k_min=0, k_max=rk[jj]*k_nyq, kplot=False))

# split in bands and esimate depths
zz, data_band = [], []    
for jj in range(len(rk)-1):
    data_band.append(MapData(data1.x, data1.y, data1.z))
    data_band[jj].tma = data_filt[jj].tma - data_filt[jj+1].tma
    zz.append(data_band[jj].power_depth(k_min=rk[jj+1]*k_nyq, k_max=rk[jj]*k_nyq, kplot=True))
# Last is the base layer:
jj = len(rk)-1
data_band.append(MapData(data1.x, data1.y, data1.z))
data_band[jj].tma = data_filt[jj].tma
zz.append(data_band[jj].power_depth(k_min=0.01*k_nyq, k_max=rk[jj]*k_nyq, kplot=True))

# Init output model objects
mod_list = []
mod_list.append(MapData(mod_geom1.x, mod_geom1.y, [0, 500- 50]))
mod_list.append(MapData(mod_geom1.x, mod_geom1.y, [0,1000- 50]))
mod_list.append(MapData(mod_geom1.x, mod_geom1.y, [0,1500- 50]))

# Run inversion
niter = 3
func_grn = mag.green
func_jac = mag.jacobi
inc, dec = 90, 0
vt_e = vt_m = mod_geom1.vt
eps = 1e-6
lam = 1e-3

inver = [np.nan for jj in range(nband)]
synt  = [np.nan for jj in range(nband)]

tic = time.perf_counter()
for jj in range(nband):
    print('### Submodel {}'.format(jj))
    mod_list[jj].vt = vt_m
    inver[jj], synt[jj] = map_inversion(func_grn, data_band[jj], mod_list[jj], vt_e, vt_m, eps, 
                          func_jac=func_jac, gf_max=1e12, nnn=1, lam=lam, 
                          niter=niter, verbose=1)

toc = time.perf_counter()
time_inv = toc - tic
print('Time inversion: {}'.format(time_inv))

# Stack partial inversions
inv_stk = image_stack(inver[0:])

# PLot data and inversion result
for jj in range(nband):   
    figs = plot_gauss_newton(inver[jj], synt[jj], data_band[jj])
    figs[0].savefig(png + 'test'+str(ktest)+'_band'+str(jj)+'_inv_2layer_nonlin.png')
    
# plot vertical slice
kh = inv_stk
fig, (ax, bx) = plt.subplots(2,1,figsize=(5,8)) 
clim_on = False

ix = data1.nx//2
ax.plot(data_band[0].y, data_band[0].tma[:,ix], 'k-', label='30-60% Nyq')
ax.plot(data_band[1].y, data_band[1].tma[:,ix], 'b-', label='15-30% Nyq')
ax.plot(data_band[2].y, data_band[2].tma[:,ix], 'r-', label=' 0-15% Nyq')
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

fig.savefig(png + 'test'+str(ktest)+'_stack_inv_2layer_nonlin.png')

plt.show(block=block)  
    
