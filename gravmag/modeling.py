# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:20:56 2025

@author: KEHOK
"""
import numpy as np
import matplotlib.pyplot as plt
import time

import gravmag.mag as mag
import gravmag. grav as grav
from gravmag.meta import MapData
from gravmag.meta import edge_taper
from gravmag.common import smooth

def map_modeling(func_grn, geom_in, model_in, *args, **kwargs):
    """ Forward modeling of data with magntization model from map inversion.
    
    The purpose of the function is to perform redatuming of magnetic data
    by the equivalent source method. Typical workflow:
        
        1. Map inversion of magnetic data recorded at datum z=zr.
        2. Forward modeling with model from map inversion with
           at datums z=[z0, z1, z2, ..., zn]
    
    Note: The coordinate system is right-handed with positive z direction down.
          Hence, x=Northing, y=Easting. 
          Coordinates are Cartesian with dimension meters.

    Parameters
    ----------
    func_grn: function object
        Function to compute Green's function matrix
    geom: MapData object
        Model object with geometry and depths for redatuming
    model: MapData object
        Magnetization model from map inversion.

    args (passed on to func_grn and func_jac)
    ----
    vt_e: float, 3C vector, shape=(1,3)
        Direction of earth magnetic background field (tangent vector)
    vt_m: float, 3C vector, shape=(1,3)
        Direction of magnetization, currently va=vt
    eps: float, stabilization (typically eps=1e-12)        

    kwargs
    ------
    nnn: int, optional (default nnn=1) NOT USED
        Number of nearest neighbor chunks in roll along buffer
    gf_max: int, optional (default is 1e6) NOT USED
        Max numer of Green's function elements per chunk in roll-along inversion
    ltap: int. Taper length
    inc_data: int. Data resampling 
    inc_mod: int. Model resamlping
    verbose: int, print shit?
    z_list: list or np.array. List of new datums (default is z_list = geom.z)

    Programmed:
        Ketil Hokstad, 6. August 2025    
    """
    
    # Get kwargs
    nnn = kwargs.get('nnn', 1) # NOT USED
    gf_max = kwargs.get('gf_max', 1e6)  # NOT USED    
    verbose = kwargs.get('verbose', 0)
    ltap = kwargs.get('ltap', 0)
    inc_data = kwargs.get('inc_data', 1)
    inc_mod = kwargs.get('inc_mod', 1)
    z_list  = kwargs.get('z_list', [zj[0,0] for zj in geom_in.z])
    
    # geom scaling (to/from SI units)
    to_SI   = kwargs.get('to_SI', mag.to_SI)
    from_SI = kwargs.get('from_SI', 1.0/to_SI)    

    # Decimate geom and model
    geom  = geom_in.decimate(inc_data, do_all=True, verbose=verbose)
    model = model_in.decimate(inc_mod, do_all=True, verbose=verbose)

    # Compute roll along supergrid: NB! nfl refers to the model grid
    # rnum = geom.dx*geom.dy*gf_max
    # rden = model.dx*model.dy*(2*nnn+1)**2
    # nfl = int(np.floor(np.sqrt(np.sqrt(rnum/rden))))

    # Chunks
    # nx_chunk = int(np.ceil(model.nx/nfl))
    # ny_chunk = int(np.ceil(model.ny/nfl))

    if verbose>0:
        print('Map modeling')
        # print(' o gf_max = {:d}'.format(int(gf_max)))
        # print(' o nnn = {}'.format(nnn))
        # print(' o nfl = {:d}'.format(nfl))
        print(' o ltap = {:d}'.format(ltap))
        # print(' o nx_chunk  = {:d}'.format(nx_chunk))
        # print(' o ny_chunk  = {:d}'.format(ny_chunk))
        print(' o inc_data = {}'.format(inc_data))
        print(' o inc_mod  = {}'.format(inc_mod))
        print(' o z_list = {}'.format(z_list))
        print(' o args:')
        for kk, arg in enumerate(args):
            print('   - kk, arg = {}, {}'.format(kk, arg))

    # Filters to remove nan and inf
    jnd = np.isfinite(model.magn)
    knd = np.isfinite(geom.z[0])

    # Model space: Top=model.z[0], once and for all
    gx_flat = model.gx[jnd]
    gy_flat = model.gy[jnd]
    gz_flat1 = model.z[0][jnd]
    gz_flat2 = model.z[1][jnd]
    vm_1 = np.vstack([gx_flat, gy_flat, gz_flat1]).T
    vm_2 = np.vstack([gx_flat, gy_flat, gz_flat2]).T
    ds = model.dx*model.dy

    # Modeling
    synt_ut = [None for zj in z_list]
    for jj, z in enumerate(z_list):
        
        # Data space:
        synt = MapData(geom.x, geom.y, z)
        synt.tma = np.zeros_like(synt.z[0])
        
        vr = np.vstack([synt.gx[knd], 
                        synt.gy[knd], 
                        synt.z[0][knd]]).T
        
        LL = ds*func_grn(vr, vm_1, vm_2, *args)
        tma = LL.dot(model.magn[jnd])
        synt.tma[knd] = from_SI*tma.flatten()
        
        synt_ut[jj] = synt.resample(inc_data, do_all=True, verbose=verbose)

    return synt_ut



