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
#from gravmag.common import green_iter
from gravmag.meta import MapData
from gravmag.inversion import map_inversion
from gravmag.common import plot_gauss_newton

#----------------------------
# Setup
#----------------------------

block = True

pkl = '../Data/pkl/'
png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

print('sc_02_ ...')

# Marquardt-Levenberg regularization weights:
lam_dict = {0: 1e-6, 1: 1e-4, 2: 1e-1, 3: 1e-1}

k_frst, k_last = 1, 1
for ktest in range(k_frst,k_last+1):
    
    print('### Test {} ###'.format(ktest))

    fname_test = pkl + 'test'+str(ktest)+'_tma_data_for_inversion.pkl'
    with open(fname_test,'rb') as fid:
        data, model = pickle.load(fid) 
        
    niter =  5           # Number of iterations in non-lin GN inversion
    lam = 1e-6           # Marquardt-Levenberg parameter
    z_shift_ini =   50.0 # Shift of art model for base

    # Initialize the mod_inision output object
    mod_ini = MapData(model.x, model.y, model.z)
    mod_ini.vt = model.vt
    mod_ini.vt_e, mod_ini.vt_m = model.vt, model.vt 
    # Initial value for z2
    mod_ini.z[1] = mod_ini.z[1] + z_shift_ini 
    
    func_grn = mag.green
    func_jac = mag.jacobi
    vt_e = vt_m = mod_ini.vt
    eps = 1e-3
    #lam = lam_dict[ktest]
    lam = 1e-6
    tic = time.perf_counter()
    # vt_e, vt_m, eps are picked up by *args on the other side:
    inver, synt = map_inversion(func_grn, data, mod_ini, vt_e, vt_m, eps, 
                                func_jac=func_jac, niter=niter, gf_max=1e12,
                                nnn=1, lam=lam, verbose=1)
    toc = time.perf_counter()
    time_gn = toc-tic

    print('Test {}: time = {}s'.format(ktest, toc-tic))         
    
    #-------------------------------
    #  PLot results
    #-------------------------------

    head = 'Test {}:'.format(ktest)
    interp = 'bicubic'
    figs = plot_gauss_newton(inver, synt, data, head=head, interp=interp)
    figs[0].savefig(png + 'gn_' + 'test_{}'.format(ktest) + '_multi_gn_lam1e-6.png')
    figs[1].savefig(png + 'gn_' + 'test_{}'.format(ktest) + '_multi_RE_lam1e-6.png')

    plt.show(block=block)  