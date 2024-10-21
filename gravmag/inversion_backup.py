# -*- coding: utf-8 -*-
"""
Functions for gravmag inversion

Created on Mon Dec 14 13:23:21 2020
@author: kehok@equinor.com
"""
import numpy as np
import matplotlib.pyplot as plt
import time

import gravmag.mag as mag
import gravmag. grav as grav
from gravmag.meta import MapData
from gravmag.meta import edge_taper
from gravmag.common import smooth

#-----------------------------------------------------------------------
#  Roll-along map inversion: 
#  Output is one map representing depth interval z1 to z2
#-----------------------------------------------------------------------

def map_inversion(func_grn, data_in, model_in, *args, **kwargs):
    """Roll-along loop over nearest neighbor subgrids (controlled by nnn)  
    
        ---------------------------------
        !   |   |   |   |   |   |   |   |
        ---------------------------------
        !   |   |   | x | x | x |   |   |  x = data aperture
        ---------------------------------
        !   |   |   | x | o | x |   |   |  o = model inverted
        ---------------------------------  
        !   |   |   | x | x | x |   |   |  nnn = 1 in this case
        ---------------------------------
        !   |   |   |   |   |   |   |   |
        ---------------------------------
        !   |   |   |   |   |   |   |   |
        ---------------------------------
    
    TODO: from_nT and to_nT must be replaced by a general datascaling
          to deal with gravity and FTG data. PARTLY DONE 2/2-2021

    Parameters
    ----------
    func_grn: function object
        Function to compute Green's function matrix
    model_in: object
        Model object initiated with geometry and initial base 
        of source layer in model.z[1]
    data_in: object
        data object with mag or grav data
        
    args (passed on to func_grn and func_jac)
    ----
    vt_e: float, 3C vector, shape=(1,3)
        Direction of earth magnetic background field (tangent vector)
    vt_m: float, 3C vector, shape=(1,3)
        Direction of magnetization, currently va=vt
    eps: float, stabilization (typically eps=1e-12)        
        
    kwargs
    ------
    niter: int (default is 0)
        Number of non-linear GN iterations
    func_jac: function object
        Function to compute the Jacobian wrt z2 in GN inversion
    nnn: int, optional (default nnn=1)
        Number of nearest neighbor chunks in roll along buffer
    gf_max: int, optional (default is 1e6)
        Max numer of Green's function elements per chunk in roll-along inversion
    ltap: int. Taper length
    inc_data: int. Data resampling 
    inc_mod: int. Model resamlping
    verbose: int, print shit?
        
    Programmed:
        Ketil Hokstad, 26. September 2018 (Matlab)
        Ketil Hokstad, 14. December 2020    
        Ketil Hokstad, 13. January 2021    
    """
    
    # Get kwargs
    niter = kwargs.get('niter', 0)
    func_jac = kwargs.get('func_jac', False)
    nnn = kwargs.get('nnn', 1)
    gf_max = kwargs.get('gf_max', 1e6)    
    lam = kwargs.get('lam',1e-32)
    verbose = kwargs.get('verbose', 0)
    ltap = kwargs.get('ltap', 0)
    inc_data = kwargs.get('inc_data', 1)
    inc_mod = kwargs.get('inc_mod', 1)
    
    # Data scaling (to/from SI units)
    to_SI   = kwargs.get('to_SI', mag.to_SI)
    from_SI = kwargs.get('from_SI', 1.0/to_SI)    
    
    # Decimate data and model
    data  = data_in.decimate(inc_data, do_all=True, verbose=verbose)
    model = model_in.decimate(inc_mod, do_all=True, verbose=verbose)
    
    # Compute roll along supergrid: NB! nfl refers to the model grid
    rnum = data.dx*data.dy*gf_max
    rden = model.dx*model.dy*(2*nnn+1)**2
    nfl = np.int(np.floor(np.sqrt(np.sqrt(rnum/rden))))

    # Chunks
    nx_chunk = np.int(np.ceil(model.nx/nfl))
    ny_chunk = np.int(np.ceil(model.ny/nfl))

    if verbose>0:
        print('GravMag inversion')
        print(' o gf_max = {:d}'.format(np.int(gf_max)))
        print(' o nnn = {}'.format(nnn))
        print(' o nfl = {:d}'.format(nfl))
        print(' o ltap = {:d}'.format(ltap))
        print(' o nx_chunk  = {:d}'.format(nx_chunk))
        print(' o ny_chunk  = {:d}'.format(ny_chunk))
        print(' o lam  = {}'.format(lam))
        print(' o inc_data = {}'.format(inc_data))
        print(' o inc_mod  = {}'.format(inc_mod))
        print(' o args:')
        for kk, arg in enumerate(args):
            print('   - kk, arg = {}, {}'.format(kk, arg))

    # print('Then return ...')
    # return -1, -1
        
    # Initialize the output model object
    kh = MapData(model.x, model.y, model.z)
    kh.mag0 = nans_like(kh.gx)
    kh.magn = nans_like(kh.gx)
    kh.zb0 = nans_like(kh.gx)
    kh.zbn = nans_like(kh.gx)

    # Initialize synt data object
    synt = MapData(data.x, data.y, data.z)
    synt.tma = nans_like(data.gx)
    
    # loop over model chunks    
    if verbose >0: print('Roll-along inversion:')
    for iyc in range(ny_chunk):
        for ixc in range(nx_chunk):
            if verbose > 0: print(' o iyx, ixc = {}, {}'.format(iyc, ixc))
                        
            indxs = ra_indxs(ixc, iyc, model, data, nfl, nnn)
            jx1, jx2, jy1, jy2, kx1, kx2, ky1, ky2 = indxs
            my, mx = jy2-jy1+1, jx2-jx1+1
            
            
            if verbose > 1: print(' o my, mx = {}, {}'.format(my, mx))
            if verbose > 2:
                print(' o jy1, jy2, ky1, ky2, jx1, jx2, kx1, kx2 = {}, {}, {}, {}, {}, {}, {}, {}'.format(
                          jy1, jy2, ky1, ky2, jx1, jx2, kx1, kx2))
            
            # Filters to remove nan and inf
            jnd = np.isfinite(model.z[0][jy1:jy2+1, jx1:jx2+1])
            knd = np.isfinite(data.tma[ky1:ky2+1, kx1:kx2+1])
            
            # Data space: Rec=data.z[0]
            vr = np.vstack([data.gx[knd], data.gy[knd], data.z[0][knd]]).T
            
            # Taper edges of the data
            wrk = edge_taper(data.tma[ky1:ky2+1, kx1:kx2+1].copy(), ltap)
            dd = to_SI*wrk[knd].reshape(-1,1) 
            
            # Model space: Top=model.z[0], once and for all
            gx_flat = model.gx[jnd]
            gy_flat = model.gy[jnd]
            vm_1 = np.vstack([gx_flat, gy_flat, model.z[0][jnd]]).T
            #vt_e, vt_m = model.vt, model.vt
            ds = model.dx*model.dy
    
            # Lists for gathering iterations
            magn_it = [None for ii in range(niter+1)]   # Inverted magnetization
            base_it = [None for ii in range(niter+1)]   # Inverted base source layer
            synt_it = [None for ii in range(niter+1)]   # Synt data from current model
            rms_err = [None for ii in range(niter+1)]   # RMS error of current model
            rank_it = [None for ii in range(niter+1)]   # Rank of pseudo inverse

            # First iter is the linear inversion (initial value for M is zero)
            it = 0
            print('   - Iteration {}: Linear inversion'.format(it))
            base_it[it] = model.z[1][jnd].reshape(-1,1)
            vm_2 = np.vstack([gx_flat, gy_flat, base_it[it].flatten()]).T
            LL = ds*func_grn(vr, vm_1, vm_2, *args)
            magn_it[it], rank_it[it] = marq_leven(LL, dd, lam)
            
            # Non-linear GN inversion: Joint mag and zbase update
            nh = magn_it[0].shape[0]
            for it in range(niter):
            
                print('   - Iteration {}: Non-linear inversion'.format(it+1))
                # Compute data residual for current model:
                vm_2 = np.vstack([gx_flat, gy_flat, base_it[it].flatten()]).T
                LL = ds*func_grn(vr, vm_1, vm_2, *args)
                synt_it[it] = LL.dot(magn_it[it])
                deld = dd - synt_it[it]
                rms_err[it] = np.sqrt(np.sum(deld**2)/np.sum(dd**2))
                    
                # Compute Jacobain matrix
                smag = magn_it[it]                
                KK = ds*func_jac(vr, smag, vm_2, *args)
                JJ = np.hstack((LL, KK)) # The full Jacobian
                
                # Compute model update (mag and zb)
                delm, rank_it[it+1] = marq_leven(JJ, deld, lam)
                magn_it[it+1] = magn_it[it] + delm[:nh]
                base_it[it+1] = base_it[it] + delm[nh:] 
            
            # Synt data and error from last iteration:
            it = niter
            vm_2 = np.vstack([gx_flat, gy_flat, base_it[it].flatten()]).T
            LL = ds*func_grn(vr, vm_1, vm_2, *args)
            synt_it[it] = LL.dot(magn_it[it])
            deld = dd - synt_it[it]
            rms_err[it] = np.sqrt(np.sum(deld**2)/np.sum(dd**2))
            
            # Get the first and last for plotting:
            kh.mag0[jnd] = magn_it[ 0].flatten()
            kh.magn[jnd] = magn_it[it].flatten()
            kh.zb0[jnd] = base_it[ 0].flatten()
            kh.zbn[jnd] = base_it[it].flatten()
            
            synt.tma[knd] = from_SI*synt_it[it].flatten()
            synt.rms_err = rms_err

    # Resample the output to input grids:
    kh_ut = kh.resample(inc_mod, do_all=True, verbose=verbose)
    synt_ut = synt.resample(inc_data, do_all=True, verbose=verbose)

    return kh_ut, synt_ut

def nans_like(aa):
    """Make an array filled with nans with same shape as aa"""
    bb = np.empty(aa.shape)
    bb.fill(np.nan)
    return bb

def ra_indxs(ixc, iyc, model, data, nfl, nnn):
    """ Compute roll-along indices in model grid and data grid """ 
    
    # Work variables:
    qatx = (model.x[0] - data.x[0])/data.dx
    qaty = (model.y[0] - data.y[0])/data.dy
    ratx, raty = model.dx/data.dx, model.dy/data.dy
    
    # Indices in model grid
    jy1 = iyc*nfl
    jy2 = np.minimum(jy1+nfl-1, model.ny-1)
    jx1 = ixc*nfl
    jx2 = np.minimum(jx1+nfl-1, model.nx-1)
    
    # Indices in data space
    ky1 = np.int(np.rint(qaty + (jy1 - nnn*nfl)*raty))
    ky2 = np.int(np.rint(qaty + (jy2 + nnn*nfl)*raty))
    ky1, ky2 = np.maximum(ky1,0), np.minimum(ky2,data.ny-1)
    kx1 = np.int(np.rint(qatx + (jx1 - nnn*nfl)*ratx))
    kx2 = np.int(np.rint(qatx + (jx2 + nnn*nfl)*ratx))           
    kx1, kx2 = np.maximum(kx1,0), np.minimum(kx2,data.nx-1)
            
#    print('jx1, jx2, kx1, kx2 = {}, {} : {}, {}'.format(jx1, jx2, kx1, kx2))
#    print('xm1, xm2, xd1, xd2 = {}, {} : {}, {}'.format(model.x[jx1], model.x[jx2], data.x[kx1], data.x[kx2]))

    return jx1, jx2, jy1, jy2, kx1, kx2, ky1, ky2

#------------------------------------------------------------------------
#   Marquardt-Levenberg solver
#------------------------------------------------------------------------

def marq_leven(AA, dd, lam):
    """Core Marquardt-Levenberg solution.
    
    Parameters
    ----------
    AA: float, matrix, shape=(nd,nm)
    dd: float, vector, shape=(nd)
    lam: float, for regularization
    
    Returns
    -------
    mm: float, vector, shape=(nm). Solution
    rank: float. Rank of the ATA matrix
    
    Programmed:
        Ketil Hokstad, 13. December 2017 (Matlab)
        Ketil Hokstad,  9. December 2020    
    """

    #print('Core Marquardt-Levenberg solution')
    dd_pc = AA.T.dot(dd)
    ATA = AA.T.dot(AA)
    JJ = np.diag(ATA.diagonal())
    mm, res, rank, s = np.linalg.lstsq(ATA+lam*JJ, dd_pc, rcond=None)
    
    return mm, rank

#-------------------------------------------------------
# Image stack
#-------------------------------------------------------

def image_stack(inv_list, **kwargs):
    """ Stack k-band images form inversion
    
    Parameters
    ----------
    inv_list: List of MapData objects
              The results from multi-wavenumber-band inversions
              
    kwargs
    ------
    dz: float (default is dx/5). Vertical sampling
    method: str, (default is 'old')
        method = 'blocky': Old blocky method
        method = 'linear': Linear interpolation
    lf: int (default is 0). Smoothing filter length for method='linear'
    
    Returns
    -------
    inv_stk: MapData object with magnetization cube
    
    Programmed:
        Ketil Hokstad, November 2020 (method=old)
        Ketil Hokstad, February 2021 (method=linear)
    """
    
    # Get the kwargs
    dz = kwargs.get('dz', 0.2*inv_list[0].dx)
    lf = kwargs.get('lf', 0)
    method = kwargs.get('method', 'old')
    lf = kwargs.get('lf', 0)
    
    zt, zb = np.min(inv_list[-1].z[0]), np.max(inv_list[0].z[1])
    nz = np.int(np.round((zb-zt)/dz)) + 1
    nx, ny = inv_list[0].nx, inv_list[0].ny

    z = np.linspace(zt, zb, nz)
    nb = len(inv_list)
    
    print('image_stack: nx,ny,nb={},{},{}'.format(nx,ny,nb))
    print(' o method={}'.format(method))
    
    # Backward compatibility
    for kk in range(nb):
        try: inv_list[kk].magn = inv_list[kk].mag
        except: pass

    # Initialize model cube
    stk = np.ones([nz, ny, nx], dtype=float)

    if method.lower()[0] == 'l':

        for jj in range(ny):
            for ii in range(nx):
                
                # Order is lowest wavenumber at [0], highest at [-1]
                zp = [inv_list[kk].z[1][jj,ii] for kk in range(nb)] 
                zp = zp + [inv_list[-1].z[0][jj,ii]]
                wp = [inv_list[kk].magn[jj,ii] for kk in range(nb)]
                wp = [inv_list[0].magn[jj,ii]] + wp
                zp = np.array(zp, dtype=float)
                wp = np.array(wp, dtype=float)
                fp = wp.copy()
                for kk in range(2,len(zp)):
                    fp[kk] = np.sum(wp[1:kk+1])                   
                
                # Interpolate and smooth
                uu = np.interp(z,zp[::-1],fp[::-1], left=np.nan, right=fp[0])
                stk[:,jj,ii] = smooth(uu, lf)
                
    else:

        for jj in range(ny):
            for ii in range(nx):
                uu = np.zeros_like(z)
                for kk in range(nb):
                    z0, z1 = inv_list[kk].z[0][jj,ii], inv_list[kk].z[1][jj,ii]
                    k0 =  np.int(np.round((z0-z[0])/dz))
                    k1 =  np.int(np.round((z1-z[0])/dz))
                    #if (ii==0) and (jj==0): print(kk, k0, k1)
#                    stk[k0:k1+1, jj, ii] = stk[k0:k1+1, jj, ii] + inv_list[kk].magn[jj, ii]
                    uu[k0:k1+1] = uu[k0:k1+1] + inv_list[kk].magn[jj, ii]
                    
                # FIll in nan down to seabed and filter
#                stk[0:k0, jj, ii] = np.nan
                uu[0:k0] = np.nan
                stk[:,jj,ii] = smooth(uu, lf)                
                
    # Return object            
    inv_stk = MapData(inv_list[0].x, inv_list[0].y, 0)
    inv_stk.z, inv_stk.nz, inv_stk.dz = z, nz, dz
    inv_stk.magn = stk

    return inv_stk


