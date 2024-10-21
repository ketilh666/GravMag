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
#from gravmag.common import green_iter
#from gravmag.mag import green_ij, jacobi_ij
from gravmag.mag import green, jacobi
from gravmag.mag import to_nT
from gravmag.common import plot_gauss_newton

#---------------------------------------------
# Run tests
#---------------------------------------------

block = True

pkl = '../Data/pkl/'
png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

k_frst, k_last = 1, 1
for ktest in range(k_frst,k_last+1):
    
    #-----------------------------------------
    #   Get parameters
    #-----------------------------------------
    
    print('### Test {} ###'.format(ktest))
    
    gscl = 250. # sale of the problem (grid spacing)
    if   ktest == 0: inc, dec, zr = 90,   0,  -4*gscl
    elif ktest == 1: inc, dec, zr = 90,   0,  -8*gscl
    elif ktest == 2: inc, dec, zr = 90,   0, -12*gscl
    elif ktest == 3: inc, dec, zr = 90,   0, -16*gscl
    elif ktest == 4: inc, dec, zr = 75, -15,  -4*gscl
    elif ktest == 5: inc, dec, zr = 45, -25,  -4*gscl

    niter =  8           # Number of iterations in non-lin GN inversion
    lam = 1e-6           # Marquardt-Levenberg parameter
    z_shift_ini = -50.0  # Shift of art model for base
    split = False

    # Functions to comput Green's function and Jacobian:
    #func_grn, func_jac = green_ij, jacobi_ij

    #------------------------------------------
    # Forward model synthetic data
    #------------------------------------------

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
    vt_e, vt_m = model.vt_e, model.vt_m
    
    # Compute Green's function 
    tic = time.perf_counter()
    eps = 1e-32
    ds = model.dx*model.dy
    #AA = ds*green_iter(func_grn, vr, vm_1, vm_2, vt_e, vt_m, eps)
    AA = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
    toc = time.perf_counter()
    time_gf = toc - tic
    
    # Make the synt data
    mm = model.mag.reshape(model.nx*model.ny,1)
    dd = AA.dot(mm)

    #---------------------------------------------------------------------
    # Ibitialize stuff for inversion
    #---------------------------------------------------------------------

    # Initialize the inversion output objects
    synt  = MapData(data.x, data.y, data.z)
    inver = MapData(model.x, model.y, model.z)
    inver.vt_e, inver.vt_m = model.vt_e, model.vt_m 
    # Initial value for base of source layer
    inver.z[1] = inver.z[1] + z_shift_ini 
    
    # Initialize lists for gathering iterations
    magn_it = [None for ii in range(niter+1)]   # Inverted magnetization
    base_it = [None for ii in range(niter+1)]   # Inverted base source layer
    synt_it = [None for ii in range(niter+1)]   # Synt data from current model
    rank_it = [None for ii in range(niter+1)]   # Rank of pseudo inverse
    synt.rms_err = [None for ii in range(niter+1)]   # RMS error of current model
    
    # Compute once and for all:
    ds = inver.dx*inver.dy
    gx_flat, gy_flat = inver.gx.flatten(), inver.gy.flatten()
    vm_1 = np.vstack([gx_flat, gy_flat, inver.z[0].flatten()]).T
    vr   = np.vstack([data.gx.flatten(), data.gy.flatten(), data.z[0].flatten()]).T
    vt_e, vt_m = inver.vt_e, inver.vt_m

    #---------------------------------------------
    #   Linear inversion: fixed z2
    #---------------------------------------------

    # First iter is the linear inversion (initial value for M is zero)
    tic = time.perf_counter()
    it = 0
    print('Iteration {}: Linear inversion'.format(it))
    base_it[it] = inver.z[1].reshape(-1,1)
    vm_2 = np.vstack([gx_flat, gy_flat, inver.z[1].flatten()]).T
    LL = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
    magn_it[it], rank_it[it] = marq_leven(LL, dd, lam)
    base_it[it] = inver.z[1].reshape(-1,1) # Not updated, same is initial
    
    #----------------------------------------------------
    # Non-linear inversion: Joint update of mag and z2
    #----------------------------------------------------
    
    # Non-linear GN inversion: Joint mag and zbase update
    nh = magn_it[0].shape[0]
    for it in range(niter):
        
        print('Iteration {}: Non-linear inversion'.format(it+1))
        # Compute data residual for current model:
        vm_2 = np.vstack([gx_flat, gy_flat, base_it[it].flatten()]).T
        LL = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
        synt_it[it] = LL.dot(magn_it[it])
        deld = dd - synt_it[it]
        synt.rms_err[it] = np.sqrt(np.sum(deld**2)/np.sum(dd**2))
        
        # Compute Jacobain matrix
        smag = magn_it[it]
        KK = ds*jacobi(vr, smag, vm_2, vt_e, vt_m, eps)
        JJ = np.hstack((LL, KK)) # The full Jacobian
        
        # Compute model update (mag and zb)
        delm, rank_it[it+1] = marq_leven(JJ, deld, lam)
        magn_it[it+1] = magn_it[it] + delm[:nh]
        base_it[it+1] = base_it[it] + delm[nh:] 

    #-------------------------------------------------------
    #  Data residual and rms error after last iteration
    #-------------------------------------------------------

    # Synt data and error from last iteration:
    it = niter
    vm_2 = np.vstack([gx_flat, gy_flat, base_it[it].flatten()]).T
    LL = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
    synt_it[it] = LL.dot(magn_it[it])
    deld = dd - synt_it[it]
    synt.rms_err[it] = np.sqrt(np.sum(deld**2)/np.sum(dd**2))

    # Timing
    toc = time.perf_counter()
    time_inv = toc - tic
    print('Time inversion: {}'.format(time_inv))

    #-------------------------------------------------------
    # Reshape and plot last update
    #-------------------------------------------------------

    # The data
    data.tma = to_nT*dd.reshape(data.ny, data.nx)

    # Get the first and last for plotting:
    inver.mag0 = magn_it[ 0].reshape(inver.ny, inver.nx)
    inver.magn = magn_it[-1].reshape(inver.ny, inver.nx)
    inver.zb0 = base_it[ 0].reshape(inver.ny, inver.nx)
    inver.zbn = base_it[-1].reshape(inver.ny, inver.nx)
    synt.tma0 = to_nT*synt_it[ 0].reshape(synt.ny, synt.nx) 
    synt.tma  = to_nT*synt_it[-1].reshape(synt.ny, synt.nx)
    
    #-------------------------------
    #  PLot results
    #-------------------------------
    
    head = 'Test {}:'.format(ktest)
    figs = plot_gauss_newton(inver, synt, data, head=head, interp='bicubic')
    figs[0].savefig(png + 'gn_' + 'test_{}'.format(ktest) + '_inversion_0.png')
    figs[1].savefig(png + 'gn_' + 'test_{}'.format(ktest) + '_relerr_0.png')
    
    plt.show(block=block)  
