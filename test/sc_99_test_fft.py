# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:00:27 2020

@author: kehok
"""

#--------------------------------------
#  Test scipy fft functions
#     fft, ifft
#     fft2, ifft2
#     fftshift
#     fftfreq
#--------------------------------------

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import pickle
import os

block = False

pkl = '../Data/pkl/'
png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

for ktest in range(0,3):
    
    fname_test = pkl + 'test'+str(ktest)+'_tma_data_for_inversion.pkl'
    with open(fname_test,'rb') as fid:
        data, mod_geom = pickle.load(fid) 
    
    # fft pars
    nkx, nky = data.nx, data.ny
    dkx, dky = 2*np.pi/(data.nx*data.dx), 2*np.pi/(data.ny*data.dy) 
 
    tma_k_raw = fft.fft2(data.tma)
    tma_k = fft.fftshift(tma_k_raw)
    
    # Wavenumbers    
    kxarr_raw = 2*np.pi*fft.fftfreq(nkx, data.dx)
    kyarr_raw = 2*np.pi*fft.fftfreq(nky, data.dy)
    kxarr = fft.fftshift(kxarr_raw)
    kyarr = fft.fftshift(kyarr_raw)
    
    # Wavelengths
    lamx, lamy = 2*np.pi/kxarr, 2*np.pi/kyarr
        
    # Grids
    gkx, gky = np.meshgrid(kxarr, kyarr)
    gkr = np.sqrt(gkx**2 + gky**2)
    glamr = 2*np.pi/gkr

    # Inverse fft
    tma_x = fft.ifft2(tma_k_raw)

    # PLot data and inversion result
    fig, axs = plt.subplots(1, 3 ,figsize=(16, 5))
    ax, bx, cx = axs.ravel()
    zr, delz = data.z[0][0,0], mod_geom.z[1][0,0] - mod_geom.z[0][0,0] 
    inc, dec = mod_geom.inc, mod_geom.dec
    fig.suptitle('Test {}: dx=dy={}, z2-z1={}, zr={}'.format(ktest, mod_geom.dx, delz, zr), fontsize=14)
    
    xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
    im = ax.imshow(np.abs(tma_x.T), origin='lower', extent=xtnt)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
    ax.set_xlim(data.y[0], data.y[-1])
    ax.set_ylim(data.x[0], data.x[-1])
    ax.set_ylabel('northing x [m]')
    ax.set_xlabel('easting y [m]')
    ax.set_title('x-space: inc={}deg, dec={}deg'.format(inc, dec))
     
    xtnt2 = [kyarr[0], kyarr[-1], kxarr[0], kxarr[-1]]
    im = bx.imshow(np.abs(tma_k.T), origin='lower', extent=xtnt2)
    cb = bx.figure.colorbar(im, ax=bx, shrink=0.9) 
    #bx.set_xlim(data.y[0], data.y[-1])
    #bx.set_ylim(data.x[0], data.x[-1])
    bx.set_ylabel('kx [1/m]')
    bx.set_xlabel('ky [1/m]')
    bx.set_title('k-space: inc={}deg, dec={}deg'.format(inc, dec))
    
    cx.scatter(gkr.flatten(), np.abs(tma_k.flatten()))
    cx.set_ylabel('Magnitude')
    cx.set_xlabel('Radial wavenumber [1/m]')
    cx.set_xlim(0,0.01)
    
    fig.tight_layout(pad=1.)
    fig.savefig(png + 'fft_test_'+str(ktest)+'.png')
    
    plt.show(block=block)  