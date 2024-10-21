# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:39:43 2021

@author: kehok
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

import gravmag.mag as mag
import gravmag.grav as grav
from gravmag.meta import MapData
from gravmag.inversion import map_inversion

#------------------------
# setup
#------------------------

block = True

pkl = '../Data/pkl/'
png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

print('sc_02_ ...')

# Marquardt-Levenberg regularization weights:
lam_dict = {0: 1e-7, 1: 1e-5, 2: 1e-3, 3: 1e-1}

test_nan = False
k_frst, k_last = 0, 0
for ktest in range(k_frst,k_last+1):
    print('### ktest = {}'.format(ktest))
    
    fname_test = pkl+'test'+str(ktest)+'_tma_data_for_inversion.pkl'
    with open(fname_test,'rb') as fid:
        data, mod_geom = pickle.load(fid) 
        
    # Put in some nans (to check if nversion still works)
    if test_nan:
        data.tma[0,1:20]  = data.tma[0:10,0]  = np.nan
        data.tma[-1,-20:] = data.tma[-10:,-1] = np.nan
        for jj in range(len(mod_geom.z)):
            mod_geom.z[jj][0,1:20]  = mod_geom.z[jj][0:10,0]  = np.nan
            mod_geom.z[jj][-1,-20:] = mod_geom.z[jj][-10:,-1] = np.nan
        
    func_grn = mag.green
    func_jac = mag.jacobi
    vt_e = vt_m = mod_geom.vt
    eps = 1e-7
    #lam = lam_dict[ktest]
    lam = 1e-7
    gf_max = 1e12
    tic = time.perf_counter()
    kh, synt = map_inversion(func_grn, data, mod_geom, vt_e, vt_m, eps, 
                             func_jac=func_jac, gf_max=gf_max, nnn=1, 
                             lam=lam, verbose=1, inc_mod=1)
    
    toc = time.perf_counter()
    time_mi = toc-tic
    print('Test {}: time = {}s'.format(ktest, toc-tic))         
    
    #-------------------------------
    #  PLot results
    #-------------------------------

    # PLot data and inversion result
    fig, (ax, bx) = plt.subplots(1,2,figsize=(12,6)) 
    zr, delz = data.z[0][0,0], kh.z[1][0,0] - kh.z[0][0,0] 
    inc, dec = mod_geom.inc, mod_geom.dec
    fig.suptitle('Test {}: dx=dy={}, z2-z1={}, zr={}'.format(ktest, kh.dx, delz, zr), fontsize=14)

    xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
    im = ax.imshow(data.tma.T, origin='lower', extent=xtnt, interpolation='bicubic')
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
    ax.set_xlim(data.y[0], data.y[-1])
    ax.set_ylim(data.x[0], data.x[-1])
    ax.set_ylabel('northing x [m]')
    ax.set_xlabel('easting y [m]')
    ax.set_title('Mag GF: inc={}deg, dec={}deg'.format(inc, dec))
 
    xtnt2 = [kh.y[0], kh.y[-1], kh.x[0], kh.x[-1]]
    im = bx.imshow(kh.magn.T, origin='lower', extent=xtnt2, interpolation='bicubic')
    cb = bx.figure.colorbar(im, ax=bx, shrink=0.9) 
    bx.set_xlim(data.y[0], data.y[-1])
    bx.set_ylim(data.x[0], data.x[-1])
    bx.set_ylabel('northing x [m]')
    bx.set_xlabel('easting y [m]')
    bx.set_title('Mag inv: inc={}deg, dec={}deg'.format(inc, dec))
    
    fig.savefig(png + 'gscl_250_inversion_test_'+str(ktest)+'.png')
    
    plt.show(block=block)