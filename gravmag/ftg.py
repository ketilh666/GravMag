import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print('ftg does not exist')

#-----------------------
# Some constants
#-----------------------

gamma = 6.67430e-11 # gravity constant [m**3/(kg s**2)]
to_Eo, from_Eo = 1.0e9, 1.0e-9 # from SI to Eotvos
from_SI, to_SI = 1.0e9, 1.0e-9 # from Eotvos to SI
d2r, r2d = np.pi/180, 180/np.pi

#--------------------------------------------------
# Lambda function to compute scalar product
#--------------------------------------------------

# Scalar product for 3C vectors
dot = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#--------------------------------------------------
#   Compute Greens function matrix
#--------------------------------------------------

def green(vr, vm_1, vm_2, eps):
    """ Compute the Gravity Gzz Green's function matrix.
 
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
        Ketil Hokstad, 15. September 2025 (FTG, Gzz)
    """
            
    nr = vr.shape[0]
    nm = vm_2.shape[0]
    
    # Compute Green's function array
    grn = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space
    
            vp = vm_1[ii,:] - vr[jj,:]
            vq = vm_2[ii,:] - vr[jj,:]
        
            p1 = np.sqrt(dot(vp,vp) + eps) # r (distance from receivers to top horizon)
            p2 = p1*p1                     # r**2
            p3 = p1*p2                     # r**3
            
            q1 = np.sqrt(dot(vq,vq) + eps) # q (distance from receivers to base horizon)
            q2 = q1*q1                     # q**2
            q3 = q1*q2                     # q**3
         
            pw1 = -((1.0 + vp[2]/p1)**2)/((vp[2]+p1)**2) 
            pw2 =  (1.0/p1 - (vp[2]**2)/p3) /(vp[2]+p1)

            qw1 = -((1.0 + vq[2]/q1)**2)/((vq[2]+q1)**2) 
            qw2 =  (1.0/q1 - (vq[2]**2)/q3) /(qp[2]+q1)

            grn[jj,ii] = gamma*(qw1 + qw2 - pw1 - pw2)
    
    return grn
