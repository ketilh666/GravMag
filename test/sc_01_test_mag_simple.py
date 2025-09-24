# -*- coding: utf-8 -*-
"""
Test script for magnetic inversion of simple single anomaly

Created on Thu Jan  7 11:04:08 2021
@author: kehok@equinor.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import time
import os

from gravmag.earth import MagneticBackgroundField
from gravmag.meta import MapData
from gravmag.inversion import marq_leven
from gravmag.common import load_test_model
from gravmag.mag import green
from gravmag.mag import to_nT

#---------------------------------------------
# Run tests
#---------------------------------------------

block = False

pkl = '../Data/pkl/'
png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

ntest = 1
for ktest in range(0,ntest):
    print(ktest)
    
    gscl = 250. # sale of the problem (grid spacing)
    if   ktest == 0: inc, dec, zr = 90,   0,  -2*gscl
    elif ktest == 1: inc, dec, zr = 90,   0,  -4*gscl
    elif ktest == 2: inc, dec, zr = 90,   0,  -8*gscl
    elif ktest == 3: inc, dec, zr = 90,   0, -16*gscl
    elif ktest == 4: inc, dec, zr = 75, -15,  -4*gscl
    elif ktest == 5: inc, dec, zr = 45, -25,  -4*gscl

    gf_compute = True
    lam = 1e-32         # Marquardt-Levenberg parameter

    # Functions to comput Green's function and Jacobian:
    #func = green_ij

    # Load the test model        
    model = load_test_model(gscl=gscl, inc=inc, dec=dec)

    #  Data mesh
    nx, ny = 80+1, 64+1
    xh, yh = gscl*((nx-1)//2), gscl*((ny-1)//2)
    data = MapData(np.linspace(-xh, xh, nx), np.linspace(-yh, yh, ny), zr)

    # Vectors for the iterator function
    vr = np.vstack([data.gx.flatten(), data.gy.flatten(), data.z[0].flatten()]).T
    vm_1 = np.vstack([model.gx.flatten(), model.gy.flatten(), model.z[0].flatten()]).T
    vm_2 = np.vstack([model.gx.flatten(), model.gy.flatten(), model.z[1].flatten()]).T
    vt_e, vt_m = model.vt, model.vt
    
    # Compute Green's function or read back from file
    tic = time.perf_counter()
    eps = 1e-32
    ds = model.dx*model.dy
    fname_pkl = pkl+'gscl_'+str(int(gscl))+'_test'+str(ktest)+'_JesusSaves.pkl'
    if gf_compute: 
        AA = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
        with open(fname_pkl,'wb') as fid:
            pickle.dump(AA, fid) 
    else: 
        with open(fname_pkl,'rb') as fid:
            AA = pickle.load(fid) 
    toc = time.perf_counter()
    time_gf = toc - tic
    
    # Make the synt data
    mm = model.mag.reshape(model.nx*model.ny,1)
    dd = AA.dot(mm)
    data.tma = to_nT*dd.reshape(data.ny, data.nx)

#CUT        fname_test = pkl+'test'+str(ktest)+'_tma_data.pkl'
#CUT        with open(fname_test,'wb') as fid:
#CUT            pickle.dump([data, model], fid) 

    # Initialize the inversion output model object:
    inver = MapData(model.x, model.y, model.z)
    
    # Initial value for M is zero => first iter is the linear inversion:
    tic = time.perf_counter()
    mm, rank, cond = marq_leven(AA, dd, lam)
    inver.magn = mm.reshape(inver.ny, inver.nx)
    toc = time.perf_counter()
    time_inv = toc - tic

    # Synt data in inverted model
    mm2 = inver.magn.reshape(model.nx*model.ny,1)
    dd2 = AA.dot(mm)
    synt_tma = to_nT*dd2.reshape(data.ny, data.nx)

    # Just for printing and potting
    ATA = AA.T.dot(AA)
    JJ = np.diag(ATA.diagonal())
    rank0 = np.linalg.matrix_rank(ATA)

    ngf = vr.shape[0]*vm_1.shape[0]
    print(f'ngf = {ngf}, time_gf = {time_gf:.1f}s, time_inv = {time_inv:.3f}s')
    print(f'Test {ktest}: rank0, rank, cond = {rank0}, {rank}, {cond:.1f}')

    #-------------------------------
    #  PLot results
    #-------------------------------
    
    mmax = [5,3.5,2.5,1.9,5,5]
    
    interp = 'bicubic'
    interp = 'None'
    # PLot data and inversion result
    fig, (ax, bx, cx, sx) = plt.subplots(1,4,figsize=(18,5)) 
    fig.suptitle('Test {}: dx=dy={}, z2-z1={}, zr={}'.format(ktest, gscl, 2*gscl, zr), fontsize=14)

    xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
    im = ax.imshow(data.tma.T, origin='lower', extent=xtnt, interpolation=interp)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
#    ax.set_xlim(data.y[0], data.y[-1])
#    ax.set_ylim(data.x[0], data.x[-1])
    ax.set_xlim(inver.y[0], inver.y[-1])
    ax.set_ylim(inver.x[0], inver.x[-1])
    ax.set_ylabel('northing x [m]')
    ax.set_xlabel('easting y [m]')
    ax.set_title('Mag GF: inc={}deg, dec={}deg'.format(inc, dec))
 
    xtnt2 = [inver.y[0], inver.y[-1], inver.x[0], inver.x[-1]]
    im = bx.imshow(inver.magn.T, origin='lower', extent=xtnt2, interpolation=interp)
    cm.ScalarMappable.set_clim(im,vmin=0,vmax=mmax[ktest])
    cb = bx.figure.colorbar(im, ax=bx, shrink=0.9) 
    bx.set_xlim(inver.y[0], inver.y[-1])
    bx.set_ylim(inver.x[0], inver.x[-1])
    bx.set_ylabel('northing x [m]')
    bx.set_xlabel('easting y [m]')
    bx.set_title('Mag inv: inc={}deg, dec={}deg'.format(inc, dec))
    
    # PLot ATA + lam*JJ matrix
    nm, wrk= len(mm), ATA+lam*JJ
    im = cx.imshow(wrk, origin='upper', cmap=cm.magma)
    cm.ScalarMappable.set_clim(im)
    cbar = cx.figure.colorbar(im, ax=cx, shrink=0.8); 
    cx.set_title('Test {}: ATA + lam*diag(ATA)'.format(ktest))

    im = sx.imshow(synt_tma.T, origin='lower', extent=xtnt, interpolation=interp)
    cb = sx.figure.colorbar(im, ax=sx, shrink=0.9) 
#    sx.set_xlim(data.y[0], data.y[-1])
#    sx.set_ylim(data.x[0], data.x[-1])
    sx.set_xlim(inver.y[0], inver.y[-1])
    sx.set_ylim(inver.x[0], inver.x[-1])
    sx.set_ylabel('northing x [m]')
    sx.set_xlabel('easting y [m]')
    sx.set_title('Mag GF: inc={}deg, dec={}deg'.format(inc, dec))

    fig.savefig(png+'gscl_'+str(int(gscl))+'_test_'+str(ktest)+'zoom_.png')
    plt.show(block=block)        
