# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:46:40 2023

@author: kehok
"""


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pickle
import os

# KetilH stuff
from gravmag.inversion import image_stack
from gravmag.common import plot_vertical_slice
from cube import CubeData
# import khio

#----------------------------------------------
#   Crunch through the hole damn shit
#----------------------------------------------

block = False
kplot = False

krun_list =      [1, 1]
nband_run_list = [2, 3]

pkl = '../data/pkl_island/'
excel = '../data/excel_island/'
png = 'png_island/'
if not os.path.isdir(png): os.mkdir(png)

scl = 1e-3

for kkk in range(len(krun_list)):
    
    krun = krun_list[kkk]
    nband_run = nband_run_list[kkk]

    print(f'krun, nband_run = {krun}, {nband_run}')

    #----------------------------------------------
    #  Read inversion result from pickle file
    #----------------------------------------------
    
    # Save the outputs from the inversion to pickle file
    fname = pkl + 'MagInversion_run'+str(krun)+'.pkl'
    with open(fname,'rb') as fid:
        [data_band, mod_list, inver, synt] = pickle.load(fid)
    
    rk = []
    for db in data_band:
        rk.append(db.rk)
    
    #----------------------------------
    #  Read location of power plants
    #----------------------------------
    
    file_excel = 'PowerPlants_Iceland.xlsx'
    with pd.ExcelFile(excel + file_excel) as ff:
        plants = pd.read_excel(ff)
    
    #-----------------------------------------------
    #  Stack wavenumber bands
    #------------------------------------------------
    
    # Stack partial inversions
    nband = len(inver)
    nband_stack = nband_run
    nz = 79
    z = np.linspace(-800, 7000, nz)
    
    stk_lin = image_stack(inver[0:nband_stack], z=z, method='linear', lf=0)
    #stk_blk = image_stack(inver[0:nband_stack], z=z, method='blocky', lf=0)
        
    #-----------------------------------------------
    #  PLot results
    #------------------------------------------------
    
    mmin, mmax = -15, 15
    
    # PLot slice from the stack cube
    xlc = plants.loc[0, 'y_isn']
    fig_0 = plot_vertical_slice(data_band[0:nband_stack], stk_lin, xl=xlc, rk=rk, mmin=mmin, mmax=mmax)
    
    xlc = plants.loc[1, 'y_isn']
    fig_1 = plot_vertical_slice(data_band[0:nband_stack], stk_lin, xl=xlc, rk=rk, mmin=mmin, mmax=mmax)
    
    fig_0.savefig(png + 'Magnetization_Vertical_Slice_HH_run{krun}_{nband_stack}bands.png')
    fig_1.savefig(png + 'Magnetization_Vertical_Slice_NV_run{krun}_{nband_stack}bands.png')
    
    #------------------------------------------------------------
    #   Make a righ-handed cube with z poiting up (like Petrel)
    #------------------------------------------------------------
    
    mag_grd = CubeData(stk_lin.y, stk_lin.x, -stk_lin.z)
    mag_grd.magn = np.transpose(stk_lin.magn, (0,2,1))
    
    # Put in the upper and lower surfaces as horizons
    mag_grd.hor = [None for jj in range(nband_stack+1)]
    mag_grd.hor[0] = inver[0].z[0].T
    for jj in range(nband_stack):
        mag_grd.hor[jj+1] = inver[jj].z[1].T
    
    #------------------------------------
    #  QC plots
    #------------------------------------
    
    vmin = -2
    vmax = 10
    
    zp_list = [-1000, -2000, -3000, -4000, -5000, -6000]
    nl = len(zp_list)
    xtnt = scl*np.array([mag_grd.x[0], mag_grd.x[-1], mag_grd.y[0], mag_grd.y[-1]])
    fig, axs  = plt.subplots(2,nl//2, figsize=(12,8))
    for kk, zp in enumerate(zp_list):
            
        # Interpolation2
        dz = np.diff(mag_grd.z)[0]
        iz = int((zp - mag_grd.z[0])/dz)
        print(iz, mag_grd.z[iz], zp)
    
        ax = axs.ravel()[kk]
        im=ax.imshow(mag_grd.magn[iz,:,:], origin='lower', extent=xtnt, vmin=vmin, vmax=vmax)
        cb = ax.figure.colorbar(im, ax=ax)
        ax.scatter(scl*plants.loc[0,'x_isn'], scl*plants.loc[0, 'y_isn'], c='r', marker='s', label='Hellisheidi')
        ax.scatter(scl*plants.loc[1,'x_isn'], scl*plants.loc[1, 'y_isn'], c='r', marker='d', label='Nesjavellir')
    
        ax.axis('scaled')
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_title(f'Magnetization: z = {zp}')
    
    fig.tight_layout(pad=2.0)
    fig.savefig(png+f'Magnetization_QC_H_run{krun}_{nband_stack}bands.png')
    
    # Vertical slices
    jy_list = [25,50,75,100,125,150,175,200]
    xtnt_v = scl*np.array([mag_grd.x[0], mag_grd.x[-1], mag_grd.z[-1], mag_grd.z[0]])
    fig, axs  = plt.subplots(2,4, figsize=(20,5))
    
    for kk, jy in enumerate(jy_list):
        
        ax = axs.ravel()[kk]
        im = ax.imshow(mag_grd.magn[:,jy,:], origin='upper', 
                       extent=xtnt_v, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x [km]')
        ax.set_ylabel('z [km]')
        lab = scl*mag_grd.y[kk]
        ax.set_title('Magnetization: y={:.1f}km'.format(lab))
        cb = ax.figure.colorbar(im, ax=ax)
        
    fig.tight_layout(pad=2.0)
    fig.savefig(png+f'Magnetization_QC_V_run{krun}_{nband_stack}bands.png')
    
    plt.show()
    
    # Save the outputs from the inversion to pickle file
    fname = f'MagInversion_Stack_run{krun}_{nband_stack}bands.pkl'
    with open(pkl + fname,'wb') as fid:
        pickle.dump([mag_grd], fid)
    
plt.show(block=block)