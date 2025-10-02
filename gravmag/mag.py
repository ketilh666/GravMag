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

def green(vr, vm_1, vm_2, vt_e, vt_m, eps, **kwargs):
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
    
    kwargs
    ------
    dx: float. Grid spacing of data grid
    dy: float. Grid spacing of data grid
    dx_snp: float. Grid spacing of magnetization model grid
    dy_snp: float. Grid spacing of magnetization model model

    Returns
    -------
    grn: float
        Aray of the magnetic Green's function matrix

    Programmed: 
        Ketil Hokstad, 13. December 2017 (Matlab)
        Ketil Hokstad,  9. December 2020  
        Ketil Hokstad, 13. January  2021  
        Ketil Hokstad, 24. September 2024 (snap to grid)
    """
            
    dy = dx = kwargs.get('dx', 1000.0)
    dy_snp = dx_snp = kwargs.get('dx_snp', 1.0)

    vr_snp = np.ones_like(vr)
    vr_snp[:,0] = dx_snp*np.round(vr[:,0]/dx_snp)
    vr_snp[:,1] = dy_snp*np.round(vr[:,1]/dy_snp)
    vr_snp[:,2] = vr[:,2]

    nr = vr.shape[0]
    nm = vm_2.shape[0]
    
    print(f'mag.green: dx_snp, vt_e, eps = {dx_snp}, {vt_e}, {eps}')

    # Compute Green's function array
    grn = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space
    
            vp = vm_1[ii,:] - vr_snp[jj,:]
            vq = vm_2[ii,:] - vr_snp[jj,:]
        
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
#   Compute Greens function matrix
#--------------------------------------------------

def green_rtp(vr, vm_1, vm_2, eps, **kwargs):
    """ Compute the magnetic Green's function for RTP data. 

    *RTP = Reduction to pole (the magnetic background filed is normal incidence)
    
    This function is faster than the general function bacuase all
    scalar products and calls to the dot(a,b) function are eliminated. 
 
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
    eps: float, stabilization (avoid division by zero)
    
    Returns
    -------
    grn_rtp: float
        Aray of the magnetic Green's function matrix

    kwargs
    ------
    dx: float. Grid spacing of data grid
    dy: float. Grid spacing of data grid
    dx_snp: float. Grid spacing of magnetization model grid
    dy_snp: float. Grid spacing of magnetization model model

    Programmed: 
        Ketil Hokstad, 13. December 2017 (Matlab)
        Ketil Hokstad,  9. December 2020  
        Ketil Hokstad, 13. January  2021  
        Ketil Hokstad, 17. September 2025 (RTP)  
        Ketil Hokstad, 24. September 2024 (snap to grid)
    """
            
    dy = dx = kwargs.get('dx', 1000.0)
    dy_snp = dx_snp = kwargs.get('dx_snp', 1.0)

    vr_snp = np.ones_like(vr)
    vr_snp[:,0] = dx_snp*np.round(vr[:,0]/dx_snp)
    vr_snp[:,1] = dy_snp*np.round(vr[:,1]/dy_snp)
    vr_snp[:,2] = vr[:,2]

    nr = vr.shape[0]
    nm = vm_2.shape[0]
    
    print(f'mag.green_rtp: dx_snp, eps = {dx_snp}, {eps}')

    # Compute Green's function array
    grn_rtp = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space
    
            vp = vm_1[ii,:] - vr_snp[jj,:]
            vq = vm_2[ii,:] - vr_snp[jj,:]
        
            p2 = vp[0]*vp[0] + vp[1]*vp[1] + vp[2]*vp[2] + eps
            p1 = np.sqrt(p2)               # r (distance from receivers to top horizon)
            p3 = p1*p2                     # r**3
            
            q2 = vq[0]*vq[0] + vq[1]*vq[1] + vq[2]*vq[2] + eps
            q1 = np.sqrt(q2)               # q (distance from receivers to base horizon)
            q3 = q1*q2                     # q**3
        
            pw1 = -(1 + vp[2]/p1)*(1 + vp[2]/p1)/((vp[2]+p1)**2) 
            pw2 =  (1/p1 - vp[2]*vp[2]/p3)/(vp[2]+p1)
            
            qw1 = -(1 + vq[2]/q1)*(1 + vq[2]/q1)/((vq[2]+q1)**2) 
            qw2 =  (1/q1 - vq[2]*vq[2]/q3)/(vq[2]+q1)

            rf  = mu0/(4*np.pi)
            grn_rtp[jj,ii] = rf*(qw1 + qw2 - pw1 - pw2)
    
    return grn_rtp

def green_1d(vr, vm_1, vm_2, eps, **kwargs):
    """ Compute the magnetic Green's function for RTP data in the 1D case. 

    *RTP = Reduction to pole (the magnetic background filed is normal incidence)
    
    This function utilize horizontal invariance and look-up tables. 
     
    Parameters
    ----------
    vr: float, array of 3C vectors, shape=(nr,3)
        Co-ordinates of the observation points
    vm_1 float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and top of anomaly points
    vm_2: float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and base of anomaly points
    eps: float, stabilization (avoid division by zero)
    
    Returns
    -------
    grn_1d: float
        Aray of the magnetic Green's function matrix

    kwargs
    ------
    dx: float. Grid spacing of data grid
    dy: float. Grid spacing of data grid
    dx_snp: float. Grid spacing of magnetization model grid
    dy_snp: float. Grid spacing of magnetization model model

    Programmed: 
        Ketil Hokstad, 13. December 2017 (Matlab)
        Ketil Hokstad,  9. December 2020  
        Ketil Hokstad, 13. January  2021  
        Ketil Hokstad, 17. September 2025 (RTP)  
        Ketil Hokstad, 30. September 2024 (snap to grid)
   """
            
    dy = dx = kwargs.get('dx', 1000.0)
    dy_snp = dx_snp = kwargs.get('dx_snp', 1.0)
    
    # Snap to grid
    vr_snp = np.ones_like(vr)
    vr_snp[:,0] = dx_snp*np.round(vr[:,0]/dx_snp)
    vr_snp[:,1] = dy_snp*np.round(vr[:,1]/dy_snp)
    vr_snp[:,2] = vr[:,2]

    # Grid for look-up table
    # dy = dx = np.min(np.abs(np.diff(vr_snp[:,0])))
    # dy = np.max(np.abs(np.diff(vr_snp[:,1])))
    x1, x2 = np.min(vr_snp[:,0]), np.max(vr_snp[:,0]) 
    y1, y2 = np.min(vr_snp[:,1]), np.max(vr_snp[:,1]) 
    nxarr = int(np.ceil((x2-x1)/dx)) + 1
    nyarr = int(np.ceil((y2-y1)/dy)) + 1
    
    z1 = np.nanmean(vm_1[:,2]) - np.nanmean(vr[:,2])
    z2 = np.nanmean(vm_2[:,2]) - np.nanmean(vr[:,2])

    # Compute look-up table
    xarr = np.linspace(0, x2-x1, nxarr)
    yarr = np.linspace(0, y2-y1, nyarr)
    
    grn_tab = np.zeros((nyarr, nxarr), dtype=float)
    for iy in range(nyarr):
        for ix in range(nxarr):
            
            vp = [xarr[ix], yarr[iy], z1]
            vq = [xarr[ix], yarr[iy], z2]
    
            p2 = vp[0]*vp[0] + vp[1]*vp[1] + vp[2]*vp[2] + eps
            p1 = np.sqrt(p2)               # r (distance from receivers to top horizon)
            p3 = p1*p2                     # r**3
            
            q2 = vq[0]*vq[0] + vq[1]*vq[1] + vq[2]*vq[2] + eps
            q1 = np.sqrt(q2)               # q (distance from receivers to base horizon)
            q3 = q1*q2                     # q**3
        
            pw1 = -(1 + vp[2]/p1)*(1 + vp[2]/p1)/((vp[2]+p1)**2) 
            pw2 =  (1/p1 - vp[2]*vp[2]/p3)/(vp[2]+p1)
            
            qw1 = -(1 + vq[2]/q1)*(1 + vq[2]/q1)/((vq[2]+q1)**2) 
            qw2 =  (1/q1 - vq[2]*vq[2]/q3)/(vq[2]+q1)

            rf  = mu0/(4*np.pi)
            grn_tab[iy, ix] = rf*(qw1 + qw2 - pw1 - pw2)
    
    # QC plot GF table
    # fig = plt.figure()
    # xtnt = np.array([xarr[0], xarr[1], yarr[0], yarr[1]])
    # plt.imshow(grn_tab, origin='lower', extent=xtnt)
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.title('G(xr,xm)')
    # fig.savefig('Greens_Function_Table.png')
    
    # Compute greens function using look-up table
    nr = vr.shape[0]
    nm = vm_2.shape[0]
    
    print(f'mag.green_1d: dx, dx_snp, eps = {dx}, {dx_snp}, {eps}')

    # Compute Green's function array
    grn_1d = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
    
        x = vm_1[:,0] - vr_snp[jj,0]
        y = vm_1[:,1] - vr_snp[jj,1]
        ix = np.rint(np.abs(x/dx)).astype(int)
        iy = np.rint(np.abs(y/dy)).astype(int)
        grn_1d[jj,:] = grn_tab[iy, ix]            
            
    return grn_1d

#--------------------------------------------------
#   Compute Jacobian matrix wrt z2
#--------------------------------------------------

# Define function
def jacobi(vr, smag, vm, vt_e, vt_m, eps, **kwargs):
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
    
    kwargs
    ------
    dx: float. Grid spacing of data grid
    dy: float. Grid spacing of data grid
    dx_snp: float. Grid spacing of magnetization model grid
    dy_snp: float. Grid spacing of magnetization model model

    Returns
    -------
    jac_ij: float
        One element of the magnetic Jacobian wrt z2

    Programmed: 
        Ketil Hokstad,  7. January 2020  
        Ketil Hokstad, 24. September 2024 (snap to grid)
    """    
    
    dy = dx = kwargs.get('dx', 1000.0)
    dy_snp = dx_snp = kwargs.get('dx_snp', 1.0)

    vr_snp = np.ones_like(vr)
    vr_snp[:,0] = dx_snp*np.round(vr[:,0]/dx_snp)
    vr_snp[:,1] = dy_snp*np.round(vr[:,1]/dy_snp)
    vr_snp[:,2] = vr[:,2]

    nr = vr.shape[0]
    nm = vm.shape[0]
    
    print(f'mag.jacobi: dx_snp, vt_e, eps = {dx_snp}, {vt_e}, {eps}')

    # Compute Jacobian matrix wrt z_base
    jac = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space

            vq = vm[ii,:] - vr_snp[jj,:]
            
            q1 = np.sqrt(dot(vq,vq) + eps)
            q2 = q1*q1 
            q3 = q1*q2 
            q5 = q3*q2
            
            w1 = -dot(vt_m,vt_e)/q3
            w2 = 3*(dot(vt_m,vq))*(dot(vt_e,vq))/q5
        
            rf  = mu0/(4*np.pi)
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


            

        