# -*- coding: utf-8 -*-
"""
Some functions used for different purposes
 o green_iter: Green's function loop using itertools.product
 o load_test_model: Just for testing the GF computation
 
Created on Wed Dec  9 14:23:38 2020
@author: kehok@equinor.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product

from gravmag.earth import MagneticBackgroundField
from gravmag.meta import MapData    

#-----------------------------------------------------------------------
#   Common Green's function iteration
#-----------------------------------------------------------------------
               
def green_iter(func, vr, vm_1, vm_2, *args):
    """Compute gravmag Green's function - the iteration loop.
 
    The nested loop over anomaly positions and recording positions is
    performed using itertools.product from the itertools package. 
    
    The function depends only on geometry, i.e. x, y, z of magnetic 
    anomaly and receiver, respectively. 
    
    The parameter list on calling the function must fit that of func()
    
    Parameters
    ----------
    func: Function object 
        Called by itertools.product to compute ij element of Green's function
    vr: float, array of 3C vectors, shape=(nr,3)
        Co-ordinates of the observation points
    vm_1 float, array of 3C vectors, shape=(nm,3)
        Horizontal coordinates and top of anomaly points
    vm_2: float, array of 3C vectors, shape=(nm,3)
        Horizontal coordinates and base of anomaly points
        
    args (passed on to func)
    ----
    vt_e: float, 3C vector, shape=(1,3)
        Direction of earth magnetic background field (tangent vector)
    vt_m: float, 3C vector, shape=(1,3)
        Direction of magnetization, currently va=vt
    eps: float, stabilization (typically eps=1e-12)


    Returns
    -------
    AA: float, array, shape=(nr, nm)
        The Greens function A (which is a matrix in this case)
        Data can be obtained by dd = dx*dy**AA.dor(mm)

    Programmed: 
        KetilH, 9. December 2020 
    """ 

    nr = np.max(vr.shape)
    nm = np.max(vm_1.shape)
             
    #   Loop over all data point and model points
    AA = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space
            AA[jj,ii] = func(vr[jj,:], vm_1[ii,:], vm_2[ii,:], *args)

#    # Iterate
#    AA = np.array([func(vr[jj,:], vm_1[ii,:], vm_2[ii,:], *args) 
#    for jj, ii in product(range(nr), range(nm))]).reshape(nr,nm)

    return AA

#----------------------------------------------------------------
#   Gridder function
#----------------------------------------------------------------

def gridder(xi, yi, vi, gx, gy, **kwargs):
    """ Data interpolation, utilizing stuff from numpy/scipy
        
        I made this function because of the awkward way the input
        to griddata is specified
        
        Parameters
        ----------
        xi, yi: floats, (x, y) coordinates of input
        vi: floats. Values at (xi,yi)
        gx, gy: floats, output (x,y) locations
        
        kwargs
        ------
        fill_value: Values to fill in utside the convex hull (default is nan)
        method: 'linear', 'nearest' or  'cubic', optional
        
        
        Returns
        -------
        gval: float: The interpolated values at (gx, gy)
        
    """

    from scipy.interpolate import griddata

    fill_value = kwargs.get('fill_value', np.nan)
    method = kwargs.get('method', 'linear')

    ind = np.isfinite(vi)
    pts = np.array((xi[ind], yi[ind]), dtype=float).T
    val = np.array(vi[ind], dtype=float)    
    gval = griddata(pts,val,(gx,gy),method=method, fill_value=fill_value) 

    return gval

#---------------------------------------------------------------
#  Simple curve smooting
#---------------------------------------------------------------

def smooth(uu, lf):
    """Simple 1D smoothing filter
    Parameters
    ----------
    uu: array of float, input curve
    lf: int, filter klength
    
    Returns
    -------
    vv: array of float, filtered curve
    """ 
    
    if lf==0:
        
        # Do nothing
        vv = uu
    
    else:   
        
        # Lenght must be odd
        if lf % 2 == 0: lf += 1
        
        # Replace nans
        ind1 = np.argmin(uu!=uu)
        uu[0:ind1] = uu[ind1] 
        
        # Filter
        ww = np.convolve(uu,np.ones(lf,dtype=int),'valid')/lf
        rr = np.arange(1,lf-1,2)
        
        # Fix the edges
        ww_left = np.cumsum(uu[:lf-1])[::2]/rr
        ww_rght = (np.cumsum(uu[:-lf:-1])[::2]/rr)[::-1]
        
        vv = np.concatenate([ww_left, ww, ww_rght])
        
        # Put back the nans
        uu[0:ind1] = np.nan 
        vv[0:ind1] = np.nan 
    
    return vv

#----------------------------------------------------------------
# PLot inversion results
#----------------------------------------------------------------

def plot_gauss_newton(inver, synt, data, **kwargs):
    "PLot result from Gauss Newton"    
        
    # Backward compatibility
    try:    mag0 = inver.mag0
    except: mag0 = inver.mag
    try:    magn = inver.magn
    except: magn = inver.mag
    try:    zb0  = inver.zb0
    except: zb0  = inver.z[1]
    try:    zbn  = inver.zbn
    except: zbn  = inver.z[1]
    
    # Get the kwargs
    niter = kwargs.get('niter', len(synt.rms_err)-1)
    scl_up = 1.00
    mmin = kwargs.get('mmin', np.min(scl_up*magn)) # Model parameter range
    mmax = kwargs.get('mmax', np.max(scl_up*magn)) # Model parameter range
    zmin = kwargs.get('zmin', np.min(scl_up*zbn))  # Depth range
    zmax = kwargs.get('zmax', np.max(scl_up*zbn))  # Depth range
    prop_name = kwargs.get('prop_name','NRM')
    interp = kwargs.get('interp', 'none')
    scl = kwargs.get('scl', 1e-3)
    cmap = kwargs.get('cmap',cm.viridis)

    # Grav or mag?
    head = kwargs.get('head', 'Magnetic inversion')
    if   head.lower()[0:3] == 'gzz':
        anom = np.nan
        resid = np.nan
        alab, plab = 'gzz [Eo]', 'Density [kg/m3]'
    elif head.lower()[0] == 'g':
        anom = np.nan
        resid = np.nan
        alab, plab = 'gz [mGal]', 'Density [kg/m3]'
    else:
        anom  = data.tma
        resid = data.tma - synt.tma # Data residual
        alab, plab = 'TMA [nT]', 'NRM [A/m]'
        
    amin = kwargs.get('amin', np.min(1.05*anom)) # Data range
    amax = kwargs.get('amax', np.max(1.05*anom)) # Data range

    # figure list
    figs = []

    # plot the input data and inversion results
    if niter: fig, axs = plt.subplots(2,3,figsize=(15,8))
    else:     fig, axs = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(head)

    # z pos down
    xtnt  = [scl*data.y[0], scl*data.y[-1], scl*data.x[0], scl*data.x[-1]]
    xtnt2 = [scl*inver.y[0], scl*inver.y[-1], scl*inver.x[0], scl*inver.x[-1]]
    
    print(xtnt)
    ax = axs.ravel()[0]
    im = ax.imshow(anom.T, origin='lower', extent=xtnt, 
                   cmap=cmap, interpolation=interp)        
    cm.ScalarMappable.set_clim(im,vmin=amin,vmax=amax)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(alab)
    ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
    ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
    ax.set_ylabel('northing x [km]')
    ax.set_xlabel('easting y [km]')
    ax.set_title('Data input')
 
    ax = axs.ravel()[1]
    im = ax.imshow(mag0.T, origin='lower', extent=xtnt2, 
                   cmap=cmap, interpolation=interp)
    cm.ScalarMappable.set_clim(im,vmin=mmin,vmax=mmax)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(plab)
    ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
    ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
    ax.set_ylabel('northing x [km]')
    ax.set_xlabel('easting y [km]')
    ax.set_title('{} linear inversion'.format(prop_name))
    
    ax = axs.ravel()[2]
    im = ax.imshow(zb0.T, origin='lower', extent=xtnt2, 
                   cmap=cmap, interpolation=interp)
    cm.ScalarMappable.set_clim(im,vmin=zmin,vmax=zmax)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label('Base [m]')
    ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
    ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
    ax.set_ylabel('northing x [km]')
    ax.set_xlabel('easting y [km]')
    ax.set_title('z_base initial')

    # The rest is relevant only for iterative GN inversion:
    if niter:    

        ax = axs.ravel()[3]
        im = ax.imshow(resid.T, origin='lower', extent=xtnt, 
                       cmap=cmap, interpolation=interp)
        cm.ScalarMappable.set_clim(im,vmin=amin,vmax=amax)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label(alab)
        ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
        ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
        ax.set_ylabel('northing x [km]')
        ax.set_xlabel('easting y [km]')
        ax.set_title('Residual iter {}'.format(niter))
     
        ax = axs.ravel()[4]
        im = ax.imshow(inver.magn.T, origin='lower', extent=xtnt2, 
                       cmap=cmap, interpolation=interp)
        cm.ScalarMappable.set_clim(im,vmin=mmin,vmax=mmax)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label(plab)
        ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
        ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
        ax.set_ylabel('northing x [km]')
        ax.set_xlabel('easting y [km]')
        ax.set_title('{} GN iter {}'.format(prop_name, niter))
        
        ax = axs.ravel()[5]
        im = ax.imshow(inver.zbn.T, origin='lower', extent=xtnt2, 
                       cmap=cmap, interpolation=interp)
        cm.ScalarMappable.set_clim(im,vmin=zmin,vmax=zmax)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label('Base [m]')
        ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
        ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
        ax.set_ylabel('northing x [km]')
        ax.set_xlabel('easting y [km]')
        ax.set_title('z_base GN iter {}'.format(niter))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    figs.append(fig)

    # PLot the rms error
    if niter:
        it = [ii for ii in range(len(synt.rms_err))]
        figs.append(plt.figure())
        plt.plot(it[1:], synt.rms_err[1:],'r-')
        plt.xlabel('GN iteration []')
        plt.ylabel('Rel RMS misfit []')
        plt.suptitle('{} Rel RMS error'.format(head))    
                   
    return figs

#-------------------------------------------------------------------
#   Plot data decomposed in bands: Bands may have different dx,dy
#------------------------------------------------------------------

def plot_data_bands(data, data_band, **kwargs):
    """Plot data decompoed into wavenumber bands """

    scl = kwargs.get('scl', 1e-3)
    interp = kwargs.get('interp', 'none')
    cmap = kwargs.get('cmap',cm.viridis)
    
    nband = len(data_band)

    # Grav or mag?
    head = kwargs.get('head', 'Magnetic')
    if   head.lower()[0:3] == 'gzz':
        anom = np.nan
        resid = np.nan
        alab, plab = 'gzz [Eo]', 'Density [kg/m3]'
    elif head.lower()[0] == 'g':
        anom = np.nan
        resid = np.nan
        alab, plab = 'gz [mGal]', 'Density [kg/m3]'
    else:
        anom  = data.tma
        anom_band = [data_band[jj].tma for jj in range(nband)]
        alab = 'TMA [nT]'
        
    # Mac/min values for colorbar
    amin = kwargs.get('amin', np.min(1.05*anom)) # Data range
    amax = kwargs.get('amax', np.max(1.05*anom)) # Data range
    
    # Figure layout
    if nband <= 2:
        nrow, ncol = 1, nband+1
    elif nband <= 5:
        nrow, ncol = 2, (nband+1+1)//2
    
    # plot the input data and inversion results
    fig, axs = plt.subplots(nrow,ncol,figsize=(5*ncol,5*nrow)) 

    # z pos down
    xtnt  = [scl*data.y[0], scl*data.y[-1], scl*data.x[0], scl*data.x[-1]]
    #print(xtnt)
    
    # Plot bands
    for kkk in range(nband):
        jrow, jcol = kkk//ncol, kkk%ncol
        if nrow == 1: ax = axs[jcol]
        else: ax = axs[jrow, jcol] 
        print(kkk, jrow, jcol)
        im = ax.imshow(anom_band[kkk].T, origin='lower', extent=xtnt, 
                       interpolation=interp, cmap=cmap)        
        cm.ScalarMappable.set_clim(im,vmin=amin,vmax=amax)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label(alab)
        ax.set_xlim(scl*data.y[0], scl*data.y[-1])
        ax.set_ylim(scl*data.x[0], scl*data.x[-1])
        ax.set_ylabel('northing x [km]')
        ax.set_xlabel('easting y [km]')
        ax.set_title('{} band {} ({}% Nyq)'.format(head, kkk, np.round(100*data_band[kkk].rk)))
        
    # PLot full wavenumber range data:
    kkk = nband
    jrow, jcol = kkk//ncol, kkk%ncol
    if nrow == 1: ax = axs[jcol]
    else: ax = axs[jrow, jcol] 
    print(kkk, jrow, jcol)
    im = ax.imshow(anom.T, origin='lower', extent=xtnt, 
                   interpolation=interp, cmap=cmap)        
    cm.ScalarMappable.set_clim(im,vmin=amin,vmax=amax)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(alab)
    ax.set_xlim(scl*data.y[0], scl*data.y[-1])
    ax.set_ylim(scl*data.x[0], scl*data.x[-1])
    ax.set_ylabel('northing x [km]')
    ax.set_xlabel('easting y [km]')
    ax.set_title('{} all wavenumbers'.format(head))
    
    return fig

#-----------------------------------------------------------
#  Plot a vertical slice from the stacked inversion cube
#-----------------------------------------------------------

def plot_vertical_slice(data_band, kh, **kwargs):
    """ PLot vertical slice from stack of wavenumber band inversions """
    
    nband = len(data_band)
    scl = kwargs.get('scl', 1e-3)
    xl = kwargs.get('xl', None)
    yl = kwargs.get('yl', None)
    rk = kwargs.get('rk', [data_band[kk].rk for kk in range(nband)])
    cmap = kwargs.get('cmap',cm.viridis)
    head = kwargs.get('head','Magnetic')
    
    scl_up = 1.00
    amin = kwargs.get('amin', None) # Data range
    amax = kwargs.get('amax', None) # Data range
    mmin = kwargs.get('mmin', None) # Model parameter range
    mmax = kwargs.get('mmax', None) # Model parameter range
    zmin = kwargs.get('zmin', np.min(kh.z)) # Depth range
    zmax = kwargs.get('zmax', np.max(kh.z)) # Depth range

    # Must give a line location for either xl or yl    
    if (not xl) and (not yl):
        print('plot_vertical_slice: Must give xl or yl')
        return None
    if xl and yl:
        print('plot_vertical_slice: Must give xl or yl, not both')
        return None
    
    # Get the line location in the cube
    if xl:
        ix = np.int(np.round((xl - data_band[0].x[0])/data_band[0].dx))
        jx = np.int(np.round((xl- kh.x[0])/kh.dx))

    elif yl:
        iy = np.int(np.round((yl - data_band[0].y[0])/data_band[0].dy))
        jy = np.int(np.round((yl- kh.y[0])/kh.dy))

    # PLot stack    
    col = ['r','b','k','m','c','g','y']
    fig, (ax, bx) = plt.subplots(2,1,figsize=(6,9)) 
    
    if xl:
        # PLot a data slice along y-direction at x=xl:
        for kk in range(nband):
            ax.plot(scl*data_band[kk].y, data_band[kk].tma[:,ix], col[kk]+'-', 
                    label='{}% Nyq'.format(np.int(100*rk[kk])))
        ax.set_xlim(scl*data_band[0].y[0], scl*data_band[0].y[-1])
        ax.set_xlabel('easting y [km]')
        ax.set_title('Data bands at x = {:.1f}km'.format(scl*data_band[1].x[ix]))
        ax.legend()
    
        # PLot a model slice along y-direction at x=xl:
        xtnt2 = [scl*kh.y[0], scl*kh.y[-1], kh.z[-1], kh.z[0]]
        im = bx.imshow(kh.magn[:,:,jx], origin='upper', extent=xtnt2, 
                       aspect='auto', interpolation='bicubic', cmap=cmap)
        cm.ScalarMappable.set_clim(im,vmin=mmin,vmax=mmax)
        cb = bx.figure.colorbar(im, ax=bx, shrink=0.7, orientation='horizontal') 
        bx.set_xlim(scl*kh.y[0], scl*kh.y[-1])
        bx.set_xlabel('easting y [km]')
        bx.set_title('{} inversion stack at x = {:.1f}km'.format(head, scl*kh.x[jx]))

    elif yl:
        # PLot a data slice along y-direction at x=xl:
        for kk in range(nband):
            ax.plot(scl*data_band[kk].x, data_band[kk].tma[iy,:], col[kk]+'-', 
                    label='{}% Nyq'.format(np.int(100*rk[kk])))
        ax.set_xlim(scl*data_band[0].x[0], scl*data_band[0].x[-1])
        ax.set_xlabel('easting y [km]')
        ax.set_title('Data bands at y = {:.1f}km'.format(scl*data_band[1].y[iy]))
        ax.legend()
    
        # PLot a model slice along y-direction at x=xl:
        xtnt2 = [scl*kh.x[0], scl*kh.x[-1], kh.z[-1], kh.z[0]]
        im = bx.imshow(kh.magn[:,jy,:], origin='upper', extent=xtnt2, 
                       aspect='auto', interpolation='bicubic', cmap=cmap)
        cm.ScalarMappable.set_clim(im,vmin=mmin,vmax=mmax)
        cb = bx.figure.colorbar(im, ax=bx, shrink=0.7, orientation='horizontal') 
        bx.set_xlim(scl*kh.x[0], scl*kh.x[-1])
        bx.set_xlabel('easting y [km]')
        bx.set_title('{} inversion stack at y = {:.1f}km'.format(head, scl*kh.y[jy]))

    # Some common stuff
    ax.set_ylim(amin, amax)
    ax.set_ylabel('TMA [nT]')
    bx.set_ylim(zmax, zmin)
    bx.set_ylabel('depth z [m]')
    cb.set_label('NRM [A/m]')

    return fig

#-----------------------------------------------------------------
#   Make a simple diffractor model for testing
#-----------------------------------------------------------------
        
def load_test_model(**kwargs):
    """Load a simple model for testing """
    
    # kwargs
    gscl = kwargs.get('gscl', 25.0)  # grid scale (grid size)
    inc = kwargs.get('inc', 90.0)  # Inclination
    dec = kwargs.get('dec',  0.0)  # Declination
    mag = kwargs.get('mag', 10.0)  # Magnetization anomaly
    rho = kwargs.get('rho',100.0)  # Density anomaly
    
    # Earth background field
    B0 = 1e-9*52000 # Back ground field in Tesla
    bgf = MagneticBackgroundField(B0, inc, dec)
    
    # Model mesh: Top of anomaly is at z=0
    dx, dy = 1.0*gscl, 1.0*gscl
    nx, ny = 12+1, 8+1
    xh, yh = gscl*((nx-1)//2), gscl*((ny-1)//2)
    xm, ym = np.linspace(-xh, xh, nx), np.linspace(-yh, yh, ny)
    model = MapData(xm, ym, [0, 2*gscl])
    
    # Location of the anomaly is at the center:
    ixc, iyc = (model.nx-1)//2, (model.ny-1)//2
    model.x0, model.y0 = model.x[ixc], model.y[iyc]
    
    # Magnetic anomaly
    model.mag = np.zeros_like(model.gx)
    model.mag[iyc, ixc] = mag
    model.vt = bgf.vt.copy()
    model.vt_e, model.vt_m = model.vt, model.vt

    # Density anomaly
    model.rho = np.zeros_like(model.gx)
    model.rho[iyc, ixc] = rho

    return model

    
