# -*- coding: utf-8 -*-
"""
Set up geometry

Created on Tue Dec  8 07:43:15 2020
@author: kehok@equinor.com
"""

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import RegularGridInterpolator

# Define for convenience 
NoneType = type(None)
class MapData:
    """Set up the geometry, including np.meshgrid(x,y)
    
    Parameters
    ----------
    x: array of floats, shape=(nx)
    y: array of floats, shape=(ny)
    z: list of float or array of floats, [shape=(ny, nx)]
    
    args:
    -----
    grd: grid or list of grids, shape=(ny,nx)
    
    Returns
    -------
    self: object

    Programmed: 
        KetilH,   6. December 2020 
        KetilH,  24. February 2021
        KetilH,  17. August 2022
    """
    
    def __init__(self, x, y, z, *args):
    
        self.nx, self.ny = len(x), len(y)
        self.dx, self.dy = x[1]-x[0], y[1]-y[0]
        
        self.x, self.y = x, y
        self.gx, self.gy = np.meshgrid(x, y) 
       
        # Optional: list of maps with shape (ny,nx)
            
        # z must be alist
        if not isinstance(z,list): z=[z]
        
        # Horizons
        self.z = []
        for jj in range(len(z)):
            if   isinstance(z[jj], float): 
                self.z.append(z[jj]*np.ones_like(self.gx))
            elif isinstance(z[jj], int):
                self.z.append(float(z [jj])*np.ones_like(self.gx))
            else:
                self.z.append(z[jj])
        
        # Optional: Attribute maps
        if len(args)>0: 
            grd = args[0]
            if not isinstance(grd,list): grd = [grd]
            self.grd = grd  
               
        # Optional: Labels for the attributes
        if len(args)>1: 
            label= args[1]
            if not isinstance(label,list): label = [label]
            self.label = label  
        
        # Optional: Labels for the attributes
        if len(args)>2: 
            unit= args[2]
            if not isinstance(unit,list): unit = [unit]
            self.unit = unit  

    def __repr__(self):
        return str(self.__dict__)

    #------------------------------------------------------
    #   Class method: Bandpass filter
    #------------------------------------------------------
        
    def bandpass(self, k_locu, k_hicu, **kwargs):
        """ Bandpass filter gravmag data in kx,ky space. 
        
        The same k_min and k_max are applied in bot x- and y-directions.
        
        NB! So far only the high-cut filter has been implemented.
        
        Parameters
        ----------
        k_locu: float, low-cut  wavenumber
        k_hicu: float: high-cut wavenumber
        
        kwargs
        ------
        square: bool, optional (default False). Radial or square k-filter
        ltap: int, optional (default is ltap=3). Filter edge taper length
        ntap: int, optional (default is ntap=5). Taper of map edges afet ifft
        kplot: bool, QC plot?
        
        Returns
        -------
        data: self filtered
        
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
        ltap = kwargs.get('ltap', 3)
        ntap = kwargs.get('ntap', 5)
        kplot = kwargs.get('kplot', False)
    
        # fft pars
        nkx, nky = self.nx, self.ny
        dkx, dky = 2*np.pi/(self.nx*self.dx), 2*np.pi/(self.ny*self.dy) 
        kx_nyq, ky_nyq = np.pi/self.dx, np.pi/self.dy
    
        # Wavenumber arrays
        kxarr = 2*np.pi*fft.fftfreq(nkx, self.dx)
        kyarr = 2*np.pi*fft.fftfreq(nky, self.dy)
    
        # High corner wavenumbers, same in x- and y-directions
        dkr = np.maximum(dkx, dky)
        k_hico = k_hicu - ltap*dkr
        k_loco = 0.0
    
    
        print('meta.bandpass:')
        print(' o ltap, k_hico/k_hicu = {}, {}'.format(ltap, k_hico/k_hicu))
        print(' o ntap = {}'.format(ntap))
    
        # Forward FFT
        #tma_x = self.tma.copy()
        tma_x = edge_taper(self.tma.copy(), ltap)
        tma_k = fft.fft2(tma_x)
    
        if square: 
            print('meta.bandpass: Dont be square')
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
       
        # TODO: Low-cut filtering           
        
        
        # Inverse FFT: kill tiny imag part 
        tma_x_bp = np.real(fft.ifft2(tma_k_bp))
        
        # Taper the edge
        tma_x_bp = xy_taper(tma_x_bp, ntap)
        
        # QC plot
        if kplot:
        
            fig, axs = plt.subplots(2,2,figsize=(10,9))
            fig.suptitle('(kx, ky) bandpass filter: k_locu/k_nyq = {}, k_hicu/k_nyq = {}'.format(k_locu/kx_nyq, k_hicu/kx_nyq), fontsize=14)
            
            ax = axs[0][0]
            xtnt = [self.y[0], self.y[-1], self.x[0], self.x[-1]]
            im = ax.imshow(tma_x.T, origin='lower', extent=xtnt)
            cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
            ax.set_xlim(self.y[0], self.y[-1])
            ax.set_ylim(self.x[0], self.x[-1])
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
            xtnt = [self.y[0], self.y[-1], self.x[0], self.x[-1]]
            im = ax.imshow(tma_x_bp.T, origin='lower', extent=xtnt)
            cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
            ax.set_xlim(self.y[0], self.y[-1])
            ax.set_ylim(self.x[0], self.x[-1])
            ax.set_ylabel('northing x [m]')
            ax.set_xlabel('easting y [m]')
            ax.set_title('x-space: output')
        
            ax = axs[1][1]
            xtnt2 = [kyarr[0], kyarr[-1], kxarr[0], kxarr[-1]]
            im = ax.imshow(np.abs(tma_k_bp.T), origin='lower')
            cb = ax.figure.colorbar(im, ax=ax, shrink=0.9) 
            ax.set_ylabel('y-index []')
            ax.set_xlabel('x-index []')
            ax.set_title('k-space: output')
                
        # Output object
        data = MapData(self.x, self.y, self.z)
        data.tma = tma_x_bp
        data.k_hicu, data.k_hico = k_hicu, k_hico
        data.k_locu, data.k_loco = k_locu, k_loco
        data.ltap = ltap
        if kplot:
            data.fig = fig
            
        return data

    #------------------------------------------------------
    #   Class method: Top and base of anomalous layer
    #------------------------------------------------------
        
    def power_depth(self, **kwargs):
        """ Bndpass filter gravmag data in kx,ky space. 
        
        Estimate depth to top and base of anomalous layer using
        the method presented by Blakely (1995), CHapter 11.
            
        Parameters
        ----------
         
        kwargs
        ------
        k_min: float, optional (default is 0.5*k_nyq)
            Smallest wavenumber used to compute z_base
        k_max: float, optional (default is 0.5*k_nyq)
            Largest  wavenumber used to compute z_base
        kplot: bool, QC plot?
        
        Returns
        -------
        dd: dict with elements
            'top': float. Top of layer
            'cnt': float. Center of layer
            'base': float. Base of layer
            'thick': float. Thickness of layer
            'base': float. Base of layer
        
        Programmed:
            Ketil Hokstad, 16. December 2020    
        """
        
        # Get the kwargs
        kplot = kwargs.get('kplot', False)
        # Min and max wabvenumbers used in power spectral analysis
        k_min = kwargs.get('k_min', 0.5*np.pi/self.dx)
        k_max = kwargs.get('k_max', 0.5*np.pi/self.dx)
        
        # fft pars
        nkx, nky = self.nx, self.ny
        dkx, dky = 2*np.pi/(self.nx*self.dx), 2*np.pi/(self.ny*self.dy) 
        kx_nyq, ky_nyq = np.pi/self.dx, np.pi/self.dy
    
        # Wavenumber arrays
        kxarr = 2*np.pi*fft.fftfreq(nkx, self.dx)
        kyarr = 2*np.pi*fft.fftfreq(nky, self.dy)
        gkx, gky = np.meshgrid(kxarr, kyarr)
        gkr = np.sqrt(gkx**2 + gky**2)
    
        # Need to pick 1 for later:
        k_nyq = kx_nyq
    
        # Forward FFT
        tma_x = self.tma.copy()
        tma_k = fft.fft2(tma_x)
        
        # Power spectrum
        tma_p = np.abs(tma_k)/(nkx*nky)
        # Get rid of zero power
        ind_nz = (tma_p > 1.0e-64) & (gkr > 0.0)
        tma_p_nz = tma_p[ind_nz]
        gkr_nz   = gkr[ind_nz]
        # For later comparison
        ind0 = (gkr_nz > 0) 
        X0  = gkr_nz[ind0]
        Y01  = np.log(tma_p_nz[ind0])
        Y02  = np.log(tma_p_nz[ind0]/X0)
        
        # Receiver depth
        #zr = self.z[0][0,0]
        zr = 0.0
        
        # Estimate top from high wavenumbers
        ind1 = (gkr_nz >= k_max) & (gkr_nz <= k_nyq)
        X1  = gkr_nz[ind1]
        Y1  = np.log(tma_p_nz[ind1])
        print(X1.shape, Y1.shape)
        a1, b1, r1, jk1, jnk2 = linregress(X1, Y1) # y=ax+b
        xl1 = np.array([np.min(X1), np.max(X1)], dtype=float)
        yl1 = a1*xl1 + b1
        z_top = -a1 + zr
        
        # Estimate center and base from low wavenumbers
        ind2 = (gkr_nz >= k_min) & (gkr_nz < k_max)
        X2 = gkr_nz[ind2]
        Y2 = np.log(tma_p_nz[ind2]/X2)
        print(X2.shape, Y2.shape)
        a2, b2, r2, jnk1, jnk2 = linregress(X2, Y2) # y=ax+b
        xl2 = np.array([np.min(X2), np.max(X2)], dtype=float)
        yl2 = a2*xl2 + b2
        z_cnt = -a2 + zr
        z_base = 2*z_cnt - z_top
        
        print(f'zr, z_top, z_cnt, z_base = {zr}, {z_top}, {z_cnt}, {z_base}')
        
        # QC plot
        if kplot:
            fig, axs = plt.subplots(1,2,figsize=(12,6))
            #inc, dec = mod_geom.inc, mod_geom.dec
            #fig.suptitle('Test {}: dx=dy={}, z2-z1={}, zr={}'.format(ktest, mod_geom.dx, delz, zr), fontsize=14)
            
            ax = axs[0]
            ax.scatter(X0/k_nyq, Y02 , c='b')
            ax.scatter(X2/k_nyq, Y2 , c='r')
            ax.plot(xl2/k_nyq, yl2, 'k-')
            ax.set_ylabel('log(|P|/|k|)')
            ax.set_xlabel('Normalized radial wavenumber')
            ax.set_xlim(0,np.max(gkr)/k_nyq)
            ax.set_ylim(np.min(Y01), np.max(Y02))
            ax.set_title('z_base = {}'.format(z_base))

            ax = axs[1]
            ax.scatter(X0/k_nyq, Y01 , c='b')
            ax.scatter(X1/k_nyq, Y1 , c='r')
            ax.plot(xl1/k_nyq, yl1, 'k-')
            ax.set_ylabel('log(|P|)')
            ax.set_xlabel('Normalized radial wavenumber')
            ax.set_xlim(0,np.max(gkr)/k_nyq)
            ax.set_ylim(np.min(Y01), np.max(Y02))
            ax.set_title('z_top = {}'.format(z_top))
        
        else:
            fig=0
            
        # Return a dict
        dd = {'top': z_top, 'cnt': z_cnt, 'base': z_base, 
              'thick': z_base-z_top, 'zr': zr, 'fig': fig  }
            
        return dd     

    #------------------------------------------------------
    #   Decimate MapData object to coarser grid
    #------------------------------------------------------

    def decimate(self, *args, **kwargs):
        """Decimate a MapData object to coarser grid.
        
        Output is a new MapData object. THe initial object is unchanged.

        If inc is given: Same increment used in both x and y directions
        If inc_x and inc_y are given: Different increments in x and y
        
        Parameters
        ----------
        
        args
        ----
        inc: int, optional (defualt is 1)
            decimation increment, applied in both x and y directions
                
        kwargs
        ------
        inc: int, optional (defualt is 1)
        inc_x: int, optional (default is inc) decimation increment in x
        inc_y: int: optional (default is inc) decimation increment in y
        do_all: bool, optional (default=False)
            Decimate also the fields not created by __init__
        
        Returns
        -------
        data: self decimated by inc
        
        Programmed:
            Ketil Hokstad, 2. February 2021
        """
        
        # Get args:
        if len(args) == 0: inc = 1
        else: inc = args[0]
        
        # Get the kwargs
        inc = kwargs.get('inc', inc)
        inc_x = kwargs.get('inc_x', inc)
        inc_y = kwargs.get('inc_y', inc)
        do_all = kwargs.get('do_all', False)
        verbose = kwargs.get('verbose', 0)
                        
        # Decimate all the fields created by __init__
        #z = [zi[::inc_y, ::inc_x] for zi in range(self.z)]
        data = MapData(self.x[::inc_x], self.y[::inc_y], 
                       [zi[::inc_y, ::inc_x] for zi in self.z])

        if verbose>0:
            print('MapData.decimate:')
            print(' o inc   = {}'.format(inc))
            print(' o inc_x, inc_y = {}, {}'.format(inc_x, inc_y))
            print(' o before: nx, ny = {}, {}'.format(self.nx, self.ny))
            print(' o after:  mx, my = {}, {}'.format(data.nx, data.ny))
            print(' o do_all = {}'.format(do_all))

        # Decimate other fields:
        if do_all:
            
            fld_list = list(self.__dict__.keys())[9:]
        
            for fld in fld_list:
               
                att = getattr(self, fld)
                
                if isinstance(att, (list, tuple)):
                    att2 = []
                    for ai in att:
                        ai2 = _decim_fld(ai, self.x, self.y, inc_x, inc_y, fld, verbose)
                        att2.append(ai2)
                else:
                    att2 = _decim_fld(att, self.x, self.y, inc_x, inc_y, fld, verbose)

                setattr(data, fld, att2)
                    
        # Output
        return data

    #------------------------------------------------------
    #   Resample MapData object to denser grid
    #------------------------------------------------------

    def resample(self, *args, **kwargs):
        """Resample a MapData object to denser grid.
        
        If inc is given: Same increment used in both x and y directions
        If inc_x and inc_y are given: Different increments in x and y
        
        Parameters
        ----------
        
        args
        ----
        inc: int, optional (defualt is 1)
            decimation increment, applied in both x and y directions
                
        kwargs
        ------
        inc: int, optional (defualt is 1)
        inc_x: int, optional (default is inc) decimation increment in x
        inc_y: int: optional (default is inc) decimation increment in y
        do_all: bool, optional (default=False)
            Decimate also the fields not created by __init__
        
        Returns
        -------
        data: self decimated by inc
        
        Programmed:
            Ketil Hokstad, 2. February 2021
        """
        
        # Get args:
        if len(args) == 0: inc = 1
        else: inc = args[0]
        
        # Get the kwargs
        inc = kwargs.get('inc', inc)
        inc_x = kwargs.get('inc_x', inc)
        inc_y = kwargs.get('inc_y', inc)        
        do_all = kwargs.get('do_all', False)
        verbose = kwargs.get('verbose', 0)
        
        # Decimate all the fields created by __init__
        nz = len(self.z) # List
        nx, ny = self.nx, self.ny # nx, ny of the input
        x = np.linspace(self.x[0], self.x[-1], inc_x*(nx-1)+1)
        y = np.linspace(self.y[0], self.y[-1], inc_y*(ny-1)+1)
        gx, gy = np.meshgrid(x, y)
        mx, my = x.shape[0], y.shape[0]
        
        z = []
        for jj, zi in enumerate(self.z):
            rgi = RegularGridInterpolator((self.x, self.y), zi.T)
            xy_par = list(zip(gx.ravel(),gy.ravel()))
            z.append(rgi(xy_par).reshape(my, mx))
        
        data = MapData(x, y, z)
        
        if verbose>0:
            print('MapData.resample:')
            print(' o inc   = {}'.format(inc))
            print(' o inc_x, inc_y = {}, {}'.format(inc_x, inc_y))
            print(' o before: nx, ny = {}, {}'.format(self.nx, self.ny))
            print(' o after:  mx, my = {}, {}'.format(data.nx, data.ny))
            print(' o do_all = {}'.format(do_all))
            
        # Resample other fields:
        if do_all:
            
            fld_list = list(self.__dict__.keys())[9:]
        
            for fld in fld_list:               
                att = getattr(self, fld)                
                
                if isinstance(att, (list, tuple)):
                    att2 = []
                    for ai in att:
                        ai2 = _resamp_fld(ai, self.x, self.y, data.x, data.y, fld, verbose)
                        att2.append(ai2)
                else:
                    att2 = _resamp_fld(att, self.x, self.y, data.x, data.y, fld, verbose)

                setattr(data, fld, att2)
                
        # Output
        return data

#------------------------------------------------------
#  Map decimation helper function
#------------------------------------------------------

# Local function:
def _decim_fld(att, x, y, inc_x, inc_y, *args):
    """Internal function used by decimate method for regular grids"""

    verbose, fld = False, ''
    if len(args)>0: fld = args[0]
    if len(args)>1: verbose = args[1]

    # Copy scalar data:
    if isinstance(att, (int, float, bool, complex, str, NoneType)):
        if verbose>0: print('  - copy: {}'.format(fld))
        att2 = att
        
    # np.ndarray
    elif isinstance(att, np.ndarray):
      
        # 2D arrays
        if   att.ndim == 2:
            if verbose>0: print('  - resample: {}'.format(fld))
            att2 = att[::inc_y, ::inc_x]       
            
        # 1D arrays (just copy the shit)
        elif att.ndim == 1:
            if verbose>0: print('  - copy: {}'.format(fld))
            att2 = att

    return att2

#------------------------------------------------------
#  Map reampling helper function
#------------------------------------------------------

# Local function:
def _resamp_fld(att, x, y, x2, y2, *args):
    """Internal function used by reasmple method for regular grids"""

    verbose, fld = False, ''
    if len(args)>0: fld = args[0]
    if len(args)>1: verbose = args[1]

    # Copy scalar data:
    if isinstance(att, (int, float, bool, complex, str, NoneType)):
        if verbose>0: print('  - copy: {}'.format(fld))
        att2 = att
        
    # np.ndarray
    elif isinstance(att, np.ndarray):
      
        # 2D arrays
        if   att.ndim == 2:
            
            if verbose>0: print('  - resample: {}'.format(fld))
            rgi = RegularGridInterpolator((x, y), att.T)
            qx, qy = np.meshgrid(x2, y2)
            xy_par = list(zip(qx.ravel(), qy.ravel()))
            att2 = rgi(xy_par).reshape(y2.shape[0], x2.shape[0])
        
        # 1D arrays (just copy the shit)
        elif att.ndim == 1:
            if verbose>0: print('  - copy: {}'.format(fld))
            att2 = att

    return att2

#--------------------------------------------------------
#  Edge taper
#--------------------------------------------------------

def edge_taper(w0, ltap):
    """Cosine edge taper """ 

    iarr = np.array([ii for ii in range(ltap)], dtype=float)
    tap = 0.5*(1-np.cos(np.pi*(iarr+1)/(ltap+1)))
    
    w1 = w0
    ny, nx = w1.shape
    for ii in range(ltap):
        w1[ii,ii:nx-ii] = tap[ii]*w1[ii,ii:nx-ii] 
        w1[ny-1-ii,ii:nx-ii] = tap[ii]*w1[ny-1-ii,ii:nx-ii] 
        w1[ii+1:ny-ii-1,ii] = tap[ii]*w1[ii+1:ny-ii-1,ii] 
        w1[ii+1:ny-ii-1,nx-1-ii] = tap[ii]*w1[ii+1:ny-ii-1,nx-1-ii] 
        
    return w1

#--------------------------------------------------------
#  Map taper
#--------------------------------------------------------

def xy_taper(wm0, ntap):
    """Cosine taper map edges in x-domain """ 
    
    if ntap <= 0:
        wm1 = wm0
        
    else:
        # Edge taper:
        jarr = np.array([jj for jj in range(ntap)], dtype=float)
        xtap = 0.5*(1-np.cos(np.pi*(jarr+1)/(ntap+1)))
        
        xmask = np.ones_like(wm0)
        ny, nx = xmask.shape
        
        for jy in range(ny):
            xmask[jy,0:ntap]   = xtap[:: 1]*xmask[jy,0:ntap]
            xmask[jy,nx-ntap:] = xtap[::-1]*xmask[jy,nx-ntap:]
        
        for jx in range(nx):
            xmask[0:ntap, jx]   = xtap[:: 1]*xmask[0:ntap, jx]
            xmask[ny-ntap:, jx] = xtap[::-1]*xmask[ny-ntap:, jx]
            
        wm1 = xmask*wm0
        
    return wm1

