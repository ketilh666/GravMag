# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:00:27 2020

@author: kehok
"""

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import pickle

#--------------------------------------
#  Useful scipy fft functions
#     fft, ifft
#     fft2, ifft2
#     fftshift
#     fftfreq
#--------------------------------------

def bandpass(data, k_locu, k_hicu, **kwargs):
    """ Bndpass filter gravmag data in kx,ky space. 
    
    The same k_min and k_max are applied in bot x- and y-directions.
    
    NB! So far only the high-cut filter has been implemented.
    
    Parameters
    ----------
    data: object
    k_locu: float, low-cut  wavenumber
    k_hicu: float: high-cut wavenumber
    
    kwargs
    ------
    ltap: int. Taper length of wavenumber filter
    square: bool, square filter? default is False
    kplot: bool, QC plot?
    
    Returns
    -------
    tma_x_bp: array of floats, shape=(data.ny, data.nx)
    
    Programmed:
        Ketil Hokstad, 24. January 2020 (Matlab)
        Ketil Hokstad, 16. December 2020    
    """

    #--------------------------------------
    #  Useful scipy fft functions
    #     fft, ifft
    #     fft2, ifft2
    #     fftshift
    #     fftfreq
    #--------------------------------------

    # Get the kwargs
    square = kwargs.get('square',False)
    ltap = kwargs.get('ltap', 1)
    kplot = kwargs.get('kplot', False)

    # fft pars
    nkx, nky = data.nx, data.ny
    dkx, dky = 2*np.pi/(data.nx*data.dx), 2*np.pi/(data.ny*data.dy) 
    kx_nyq, ky_nyq = np.pi/data.dx, np.pi/data.dy

    # Wavenumber arrays
    kxarr = 2*np.pi*fft.fftfreq(nkx, data.dx)
    kyarr = 2*np.pi*fft.fftfreq(nky, data.dy)

    # High corner wavenumbers, same in x- and y-directions
    dkr = np.maximum(dkx, dky)
    k_hico = k_hicu - ltap*dkr

    # Forward FFT
    tma_x = data.tma.copy()
    tma_k = fft.fft2(tma_x)

    if square: 
        print('bandpass: Dont be square')
        return 0.0
#        # Corner and cut indices
#        jx_hico = np.int(np.round(k_hico/dkx)) 
#        jy_hico = np.int(np.round(k_hico/dky)) 
#        jx_hicu = np.int(np.round(k_hicu/dkx)) 
#        jy_hicu = np.int(np.round(k_hicu/dky)) 
#        # Nyquist indices
#        jx_nyq = np.int(np.round(nkx/2))
#        jy_nyq = np.int(np.round(nky/2))
#        # Square filter
#        tma_k_bp = tma_k.copy()
#        tma_k_bp[:,jx_hicu:jx_nyq+1] = 0.0
#        tma_k_bp[:,-jx_nyq:-jx_hicu] = 0.0
#        tma_k_bp[jy_hicu:jy_nyq+1,:] = 0.0
#        tma_k_bp[-jy_nyq:-jy_hicu,:] = 0.0    
    else:   
        # Radial filter
        gkx, gky = np.meshgrid(kxarr, kyarr)
        gkr = np.sqrt(gkx**2 + gky**2)
        wgt = np.ones_like(gkr)
        # Taper
        iarr = np.array([ii for ii in range(ltap)], dtype=float)
        tap = 0.5*(1+np.cos(np.pi*(iarr+1)/(ltap+1)))
        for ii in range(ltap):
            kr = k_hico + ii*dkr
            wgt[gkr>kr] = tap[ii]
        # Kill
        wgt[gkr > k_hicu] = 0.0    
        tma_k_bp = tma_k.copy()*wgt
        print('bandpass: ltap, k_hico/k_hicu = {}, {}'.format(ltap, k_hico/k_hicu))
   
    # TODO: Low-cut filtering
        
    # Inverse FFT
    tma_x_bp = fft.ifft2(tma_k_bp)
        
    #-------------------------------------
    #  QC plots
    #-------------------------------------
    
    if kplot:
    
        fig, axs = plt.subplots(2,2,figsize=(10,9))
        fig.suptitle('(kx, ky) bandpass filter: k_locu = {}, k_hicu = {}'.format(k_locu, k_hicu), fontsize=14)
        
        ax = axs[0][0]
        xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
        im = ax.imshow(np.abs(tma_x.T), origin='lower', extent=xtnt)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
        ax.set_xlim(data.y[0], data.y[-1])
        ax.set_ylim(data.x[0], data.x[-1])
        ax.set_ylabel('northing x [m]')
        ax.set_xlabel('easting y [m]')
        ax.set_title('x-space: input')
         
        ax = axs[0][1]
        xtnt2 = [kyarr[0], kyarr[-1], kxarr[0], kxarr[-1]]
        im = ax.imshow(np.abs(tma_k.T), origin='lower')
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
        ax.set_ylabel('y-index []')
        ax.set_xlabel('x-index []')
        ax.set_title('k-space: input')
    
        ax = axs[1][0]
        xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
        im = ax.imshow(np.abs(tma_x_bp.T), origin='lower', extent=xtnt)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
        ax.set_xlim(data.y[0], data.y[-1])
        ax.set_ylim(data.x[0], data.x[-1])
        ax.set_ylabel('northing x [m]')
        ax.set_xlabel('easting y [m]')
        ax.set_title('x-space: input')
    
        ax = axs[1][1]
        xtnt2 = [kyarr[0], kyarr[-1], kxarr[0], kxarr[-1]]
        im = ax.imshow(np.abs(tma_k_bp.T), origin='lower')
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
        ax.set_ylabel('y-index []')
        ax.set_xlabel('x-index []')
        ax.set_title('k-space: output')
            
    return tma_x_bp

#--------------------------------------------------------
#  Power spectrum analysis for top and base
#---------------------------------------------------------


    # FOR testing: DELETE LATER
    ktest = 1
    fname_test = 'pkl/'+'test'+str(ktest)+'_tma_data_for_inversion.pkl'
    with open(fname_test,'rb') as fid:
        data, mod_geom = pickle.load(fid)
        
    lam_nyq=1000
    
    # fft pars
    nkx, nky = data.nx, data.ny
    dkx, dky = 2*np.pi/(data.nx*data.dx), 2*np.pi/(data.ny*data.dy) 

    # Wavenumber arrays
    kxarr = 2*np.pi*fft.fftfreq(nkx, data.dx)
    kyarr = 2*np.pi*fft.fftfreq(nky, data.dy)
    gkx, gky = np.meshgrid(kxarr, kyarr)

    # Radial wavenumbers
    gkr = np.sqrt(gkx**2 + gky**2)

    # High cut and high corner wavenumbers
    kx_hicu = ky_hicu = 2*np.pi/lam_nyq
    kx_hico, ky_hico = 0.95*kx_hicu, 0.95*ky_hicu 

    # Corner and cut indices
    jx_hico = np.int(np.round(kx_hico/dkx)) 
    jy_hico = np.int(np.round(ky_hico/dky)) 
    jx_hicu = np.int(np.round(kx_hicu/dkx)) 
    jy_hicu = np.int(np.round(ky_hicu/dky)) 
    
    # Nyquist indices
    jx_nyq = np.int(np.round(nkx/2))
    jy_nyq = np.int(np.round(nky/2))

    # Forward FFT
    tma_x = data.tma.copy()
    tma_k = fft.fft2(tma_x)

    # Lowpass filter
    tma_k_nan = tma_k.copy()
    tma_k_nan[:,jx_hicu:jx_nyq+1] = np.nan
    tma_k_nan[:,-jx_nyq:-jx_hicu] = np.nan
    tma_k_nan[jy_hicu:jy_nyq+1,:] = np.nan
    tma_k_nan[-jy_nyq:-jy_hicu,:] = np.nan
    
    # TODO: Taper from loco to locu
    
    # Replace nans by zeros before inverse FFT
    ind = np.isnan(tma_k_nan)
    tma_k_bp = tma_k_nan.copy()
    tma_k_bp[ind] = 0.0
    
    # Inverse FFT
    tma_x_bp = fft.ifft2(tma_k_bp)
    
    # Power spectrum (for top and base estimation)
    tma_p = (np.abs(tma_k)**2)/((data.nx*data.ny)**2)
    tma_p_nan = (np.abs(tma_k_nan)**2)/((data.nx*data.ny)**2)
      
    # Top and base magnetic layer for given pass band
    
    #-------------------------------------
    #  QC plots
    #-------------------------------------
    
    fig, axs = plt.subplots(2,3,figsize=(12,6))
    #inc, dec = mod_geom.inc, mod_geom.dec
    #fig.suptitle('Test {}: dx=dy={}, z2-z1={}, zr={}'.format(ktest, mod_geom.dx, delz, zr), fontsize=14)
    
    ax = axs[0][0]
    xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
    im = ax.imshow(np.abs(tma_x.T), origin='lower', extent=xtnt)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
    ax.set_xlim(data.y[0], data.y[-1])
    ax.set_ylim(data.x[0], data.x[-1])
    ax.set_ylabel('northing x [m]')
    ax.set_xlabel('easting y [m]')
    #ax.set_title('x-space: inc={}deg, dec={}deg'.format(inc, dec))
     
    ax = axs[0][1]
    xtnt2 = [kyarr[0], kyarr[-1], kxarr[0], kxarr[-1]]
    im = ax.imshow(np.abs(tma_k.T), origin='lower', extent=xtnt2)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
    #ax.set_xlim(data.y[0], data.y[-1])
    #ax.set_ylim(data.x[0], data.x[-1])
    ax.set_ylabel('kx [1/m]')
    ax.set_xlabel('ky [1/m]')
    #ax.set_title('k-space: inc={}deg, dec={}deg'.format(inc, dec))

    ax = axs[1][1]
    xtnt2 = [kyarr[0], kyarr[-1], kxarr[0], kxarr[-1]]
    im = ax.imshow(np.abs(tma_k_bp.T), origin='lower', extent=xtnt2)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
    #ax.set_xlim(data.y[0], data.y[-1])
    #ax.set_ylim(data.x[0], data.x[-1])
    ax.set_ylabel('kx [1/m]')
    ax.set_xlabel('ky [1/m]')
    #ax.set_title('k-space: inc={}deg, dec={}deg'.format(inc, dec))
    
    ax = axs[1][0]
    xtnt = [data.y[0], data.y[-1], data.x[0], data.x[-1]]
    im = ax.imshow(np.abs(tma_x_bp.T), origin='lower', extent=xtnt)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
    ax.set_xlim(data.y[0], data.y[-1])
    ax.set_ylim(data.x[0], data.x[-1])
    ax.set_ylabel('northing x [m]')
    ax.set_xlabel('easting y [m]')
    #ax.set_title('x-space: inc={}deg, dec={}deg'.format(inc, dec))
    
    ax = axs[0][2]
    ax.scatter(gkr[1:,1:].flatten(), np.log(np.sqrt(tma_p[1:,1:])) , c='b')
    ax.scatter(gkr[1:,1:].flatten(), np.log(np.sqrt(tma_p_nan[1:,1:])) , c='r')
    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Radial wavenumber [1/m]')
    #ax.set_xlim(0,0.01)

    ax = axs[1][2]
    ax.scatter(gkr[1:,1:].flatten(), np.log(np.sqrt(tma_p[1:,1:])/gkr[1:,1:]) , c='b')
    ax.scatter(gkr[1:,1:].flatten(), np.log(np.sqrt(tma_p_nan[1:,1:])/gkr[1:,1:]) , c='r')
    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Radial wavenumber [1/m]')
    #ax.set_xlim(0,0.01)

    fig.savefig('png/fft_test_'+str(ktest)+'.png')
    
    plt.show()    
    
    
