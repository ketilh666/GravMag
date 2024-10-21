# -*- coding: utf-8 -*-
"""
Compute parameters for earth magnetic background magnetic field

Created on Sun Dec  6 15:46:51 2020
@author: kehok@equinor.com
"""

import numpy as np

# Constants:
mu0 = 4*np.pi*1e-7;
d2r, r2d = np.pi/180, 180/np.pi

class MagneticBackgroundField:
    """ Define Earth magnetic background field for a geographical 
    location of interest
    
    Parameters
    ----------
    b0, float
        Background field magnitude [nT]
    inc, float
        Inclination angle [deg]
    dec, float
        Declination angle [deg]
        
    Returns
    -------
    self: object
    
    Programmed: 
        KetilH, 12. December 2017 (Matlab)
        KetilH,  5. December 2020
    """
    
    def __init__(self, b0, inc, dec, **kwargs):
        
        self.b0 = b0 
        self.inc = inc
        self.dec = dec
        
        self.lon = kwargs.get('lon',63.429722222)
        self.lat = kwargs.get('lat',10.393333333)
        
        # Polar and azimuth angles:
        self.theta = (90.0 - self.inc) # polar angle with vertical
        self.phi   = self.dec      # azimuth
        
        # Tangent unit vector to earth magntic field
        tx = np.cos(d2r*inc)*np.cos(d2r*dec) 
        ty = np.cos(d2r*inc)*np.sin(d2r*dec)
        tz = np.sin(d2r*inc)
        self.vt = np.array([tx, ty, tz], dtype=float)
        
         # Normal unit vector in same vertical plane
        rn2 = (tx**2 + ty**2)*(ty)**2  + ((tx)**2 + (ty)**2)**2
        ux =  tx*ty/np.sqrt(rn2)
        uy =  ty*ty/np.sqrt(rn2)
        uz = -(tx**2 + ty**2)/np.sqrt(rn2)
        self.vn = np.array([ux, uy, uz], dtype=float)

        # b0 and h0 vectors:
        self.vb0 = self.b0*self.vt
        self.vh0 = self.vb0/mu0
        self.h0  = self.b0/mu0      # A/m
        
        self.label = 'Earth magnetic field b0 [SI]'
    
    def __repr__(self):
        return str(self.__dict__)

