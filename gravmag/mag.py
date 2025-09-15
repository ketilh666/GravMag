# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:19:11 2020

@author: kehok
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#-----------------------
# Some constants
#-----------------------

mu0 = 4*np.pi*1e-7
to_nT, from_nT = 1.0e9, 1.0e-9
from_SI, to_SI = 1.0e9, 1.0e-9
d2r, r2d = np.pi/180, 180/np.pi

#--------------------------------------------------
# Lambda function to compute scalar product
#--------------------------------------------------

# Scalar product for 3C vectors
dot = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#--------------------------------------------------
#   Compute Greens function matrix
#--------------------------------------------------

def green(vr, vm_1, vm_2, vt_e, vt_m, eps):
    """ Compute the magnetic Green's function matrix.
 
    The function depends only on geometry, i.e. x, y, z of magnetic 
    anomaly and receiver, respectively. 
    
    Parameters
    ----------
    vr: float, array of 3C vectors, shape=(nr,3)
        Co-ordinates of the observation points
    vm_1 float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and top of anomaly points
    vm_2: float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and base of anomaly points
    vt_e: float, 3C vector, shape=3)
        Direction of earth magnetic background field (tangent vector)
    vt_m: float, 3C vector, shape=(3)
        Direction of magnetization, currently va=vt
    eps: float, stabilization
    
    Returns
    -------
    grn: float
        Aray of the magnetic Green's function matrix

    Programmed: 
        Ketil Hokstad, 13. December 2017 (Matlab)
        Ketil Hokstad,  9. December 2020  
        Ketil Hokstad, 13. January  2021  
    """
            
    nr = vr.shape[0]
    nm = vm_2.shape[0]
    
    # Compute Green's function array
    grn = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space
    
            vp = vm_1[ii,:] - vr[jj,:]
            vq = vm_2[ii,:] - vr[jj,:]
        
            p1 = np.sqrt(dot(vp,vp) + eps)
            p2 = p1*p1  
            p3 = p1*p2
            
            q1 = np.sqrt(dot(vq,vq) + eps)
            q2 = q1*q1 
            q3 = q1*q2 
         
            pw1 = -(vt_m[2] + dot(vt_m,vp)/p1)*(vt_e[2] + dot(vt_e,vp)/p1)/((vp[2]+p1)**2) 
            pw2 =  (dot(vt_m,vt_e)/p1 - (dot(vt_m,vp))*(dot(vt_e,vp))/p3)/(vp[2]+p1)
            
            qw1 = -(vt_m[2] + dot(vt_m,vq)/q1)*(vt_e[2] + dot(vt_e,vq)/q1)/((vq[2]+q1)**2) 
            qw2 =  (dot(vt_m,vt_e)/q1 - (dot(vt_m,vq))*(dot(vt_e,vq))/q3)/(vq[2]+q1)
        
            rf  = mu0/(4*np.pi)
            grn[jj,ii] = rf*(qw1 + qw2 - pw1 - pw2)
    
    return grn

#--------------------------------------------------
#   Compute Jacobian matrix wrt z2
#--------------------------------------------------

# Define function
def jacobi(vr, smag, vm, vt_e, vt_m, eps):
    """ Compute magnetic Jacobian matrix wrt z2 for the current model.
 
    The Jacobian wrt base source layer z2 is  J = (dQ/dz2)*M
    
    Parameters
    ----------
    vr: float, array of 3C vectors, shape=(nr,3)
        Co-ordinates of the observation points
    smag: float, array of scalars, shape=nm
        Scalar magnetization of the anomaly
    vm: float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and base of anomaly points
    vt_e: float, 3C vector, shape=3)
        Direction of earth magnetic background field (tangent vector)
    vt_m: float, 3C vector, shape=(3)
        Direction of magnetization, currently va=vt
    eps: float, stabilization
    
    Returns
    -------
    jac_ij: float
        One element of the magnetic Jacobian wrt z2

    Programmed: 
        Ketil Hokstad,  7. January 2020  
    """
    
    nr = vr.shape[0]
    nm = vm.shape[0]
    
    # Compute Jacobian matrix wrt z_base
    jac = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space

            vq = vm[ii,:] - vr[jj,:]
            
            q1 = np.sqrt(dot(vq,vq) + eps)
            q2 = q1*q1 
            q3 = q1*q2 
            q5 = q3*q2
            
            w1 = -dot(vt_m,vt_e)/q3
            w2 = 3*(dot(vt_m,vq))*(dot(vt_e,vq))/q5
        
            rf  = mu0/(4*np.pi)
#            print(jj, ii, smag[ii].shape)
#            print('w1, w2, smag[ii]={}, {}, {}'.format(w1, w2, smag[ii]))
            jac[jj,ii] = rf*(w1 + w2)*smag[ii]
    
    return jac

#--------------------------------------------------
#   Compute Greens function, ONE ELEMENT
#--------------------------------------------------

# Define function
# def green_ij(vr, vm_1, vm_2, vt_e, vt_m, eps):
#     """ Compute one element of the magnetic Green's function matrix.
 
#     OBSOLETE, NOT USED

#     The function depends only on geometry, i.e. x, y, z of magnetic 
#     anomaly and receiver, respectively. 
    
#     Parameters
#     ----------
#     vr: float, 3C vectors, shape=(3)
#         Co-ordinates of the observation points
#     vm_1 float, 3C vector, shape=(3)
#         Horizontal coordinates and top of anomaly points
#     vm_2: float, 3C vector, shape=(3)
#         Horizontal coordinates and base of anomaly points
#     vt_e: float, 3C vector, shape=3)
#         Direction of earth magnetic background field (tangent vector)
#     vt_m: float, 3C vector, shape=(3)
#         Direction of magnetization, currently va=vt
#     eps: float, stabilization
    
#     Returns
#     -------
#     grn_ij: float
#         One element of the magnetic Green's function matrix

#     Programmed: 
#         Ketil Hokstad, 13. December 2017 (Matlab)
#         Ketil Hokstad,  9. December 2020  
#     """
            
#     # Compute Green's function element
#     vp = vm_1 - vr
#     vq = vm_2 - vr

#     p1 = np.sqrt(dot(vp,vp) + eps)
#     p2 = p1*p1  
#     p3 = p1*p2
    
#     q1 = np.sqrt(dot(vq,vq) + eps)
#     q2 = q1*q1 
#     q3 = q1*q2 
 
#     pw1 = -(vt_m[2] + dot(vt_m,vp)/p1)*(vt_e[2] + dot(vt_e,vp)/p1)/((vp[2]+p1)**2) 
#     pw2 =  (dot(vt_m,vt_e)/p1 - (dot(vt_m,vp))*(dot(vt_e,vp))/p3)/(vp[2]+p1)
    
#     qw1 = -(vt_m[2] + dot(vt_m,vq)/q1)*(vt_e[2] + dot(vt_e,vq)/q1)/((vq[2]+q1)**2) 
#     qw2 =  (dot(vt_m,vt_e)/q1 - (dot(vt_m,vq))*(dot(vt_e,vq))/q3)/(vq[2]+q1)

#     rf  = mu0/(4*np.pi)
#     grn_ij = rf*(qw1 + qw2 - pw1 - pw2)
    
#     return grn_ij

# #-----------------------------------------------------------
# #   Compute Jacobian matrix element wrt z2, , ONE ELEMENT
# #-----------------------------------------------------------

# Define function
# def jacobi_ij(vr, smag, vm, vt_e, vt_m, eps):
#     """ Compute one element of the magnetic Jacobian wrt z2 for the current model.

#     OBSOLETE, NOT USED
 
#     The Jacobian wrt base source layer z2 is  J = (dQ/dz2)*M
    
#     Parameters
#     ----------
#     vr: float, 3C vectors, shape=(3)
#         Co-ordinates of the observation points
#     smag: float
#         Scalar magnetization of the anomaly
#     vm: float, 3C vector, shape=(3)
#         Horizontal coordinates and base of anomaly points
#     vt_e: float, 3C vector, shape=3)
#         Direction of earth magnetic background field (tangent vector)
#     vt_m: float, 3C vector, shape=(3)
#         Direction of magnetization, currently va=vt
#     eps: float, stabilization
    
#     Returns
#     -------
#     jac_ij: float
#         One element of the magnetic Jacobian wrt z2

#     Programmed: 
#         Ketil Hokstad,  7. January 2020  
#     """
    
#     # Compute Jacobian element
#     vq = vm - vr
    
#     q1 = np.sqrt(dot(vq,vq) + eps)
#     q2 = q1*q1 
#     q3 = q1*q2 
#     q5 = q3*q2
    
#     w1 = -dot(vt_m,vt_e)/q3
#     w2 = 3*(dot(vt_m,vq))*(dot(vt_e,vq))/q5

#     rf  = mu0/(4*np.pi)
#     jac_ij = rf*(w1 + w2)*smag
    
#     return jac_ij


            

        