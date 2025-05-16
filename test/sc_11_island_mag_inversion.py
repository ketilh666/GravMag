# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 08:14:34 2022

@author: kehok
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pickle
import time
import os

# KetilH stuff
# import isnet
from fourier.k_filter import kfilt_1d

# Mag inversion stuff
from gravmag import mag
from gravmag.meta import MapData
from gravmag.common import plot_data_bands
from gravmag.common import plot_gauss_newton
from gravmag.inversion import map_inversion
from gravmag.inversion import image_stack
from gravmag.earth import MagneticBackgroundField
from gravmag.common import plot_vertical_slice
# import khio

#-------------------------
#  Read input data
#-------------------------

block = False
kplot = False

B0 = 52300.  # Used to compute tma from tmi 

pkl = '../data/pkl_island/'
excel = '../data/excel_island/'
png = 'png_island/'
if not os.path.isdir(png): os.mkdir(png)

scl = 1e-3

# Read mag data from pickle file
with open(pkl + 'Magnetic_Grids.pkl','rb') as fid:
    [mdat] = pickle.load(fid)

file_excel = 'PowerPlants_Iceland.xlsx'
with pd.ExcelFile(excel + file_excel) as ff:
    plants = pd.read_excel(ff)

#--------------------------------------------
#  Data and model
#      data.z[0] = receiver altitude
#      model.z[0] = surface elevation
#      flight altitude is 800m according to Sveinborg
#--------------------------------------------

krun = 1
if   krun == 0:  # Aeromag only 
    z_dem = mdat.z[1] # GEBCO
    zr = z_dem - 800.0
elif krun == 1:  # Aeromag only
    z_dem = mdat.z[1] # GEBCO
    zr = np.mean(z_dem) - 800.0

model = MapData(mdat.x, mdat.y, z_dem)
data = MapData(mdat.x, mdat.y, zr)
data.tma =  mdat.tmi_aero - B0
    
# Plot a histogram of elevations
fig=plt.figure()
bins=np.linspace(0, 800, 41)
plt.hist(-z_dem.ravel(), bins=bins, density=True)
plt.xlabel('Elevation above MSL [m]')
plt.ylabel('pdf')
plt.title('Elevation distribution within AOI')
fig.savefig(png+'Elevation_Distribution_in_AOI.png')

#------------------------
#  Inversion paraemters
#------------------------

# Marquardt Levenberg regularization weight
lam_dict = {0: 1e-3, 1: 1e-3, 2: 1e-3, 3: 1e-3}              # Marquardt-Levenberg parameter

# Increments, running inversion on sparse grids for low k
inc_data, inc_mod = [8, 4, 2], [16, 8, 4]
z_add, rk = [7480, 4660, 1980], [0.0625, 0.125, 0.25]

niter = 0     # No Gauss-Newton iter
nband_run = 3 # Run a limited nuber of wavenumber bands
gf_max = 1e7  # Controls the subdivision in chunks

k_nyq = np.pi/data.dx
nband = len(rk)

# Nyquist frequency
k_nyq = np.pi/data.dx

#--------------------------------------------------------
# Wavenumber bands: Put this into a functio or method
#--------------------------------------------------------

# Make a plot for reference only
zz_nofilt  = data.power_depth(k_min=0, k_max=rk[2]*k_nyq, kplot=True)

zz_jnk, data_filt = [], []    
# High-cut filter
for jj in range(nband):
    data_filt.append(data.bandpass(0, rk[jj]*k_nyq, ltap=3, kplot=kplot))
    zz_jnk.append(data_filt[jj].power_depth(k_min=0, k_max=rk[jj]*k_nyq, kplot=kplot))

#------------------------------------------------
# Split in wavenumber ranges and esimate depths
#------------------------------------------------

data_band = []
for jj in range(nband):
    data_band.append(MapData(data.x, data.y, data.z))

zz = [np.nan for jj in range(nband)]

# First band is the base layer
jj = 0
data_band[jj].tma = data_filt[jj].tma
data_band[jj].lam_max = data.dx*(np.min([data.nx, data.ny]) -1)
data_band[jj].lam_min = 2.0*data.dx/rk[0]
data_band[jj].rk = rk[jj]
zz[jj] = data_band[jj].power_depth(k_min=0.0*k_nyq, k_max=rk[jj]*k_nyq, kplot=kplot)

# Band 1 to nband-1
for jj in range(1, nband):
    data_band[jj].tma = data_filt[jj].tma - data_filt[jj-1].tma
    data_band[jj].lam_max = 2.0*data.dx/rk[jj-1] # Longest  wavelength in range
    data_band[jj].lam_min = 2.0*data.dx/rk[jj]   # Shortest wavelength in range  
    data_band[jj].rk = rk[jj]
    zz[jj] = data_band[jj].power_depth(k_min=rk[jj-1]*k_nyq, k_max=rk[jj]*k_nyq, kplot=kplot)

# End of data prep

# QC plot of data bands
scl = 1e-3
fig = plot_data_bands(data, data_band, scl=scl)

for ax in fig.axes:
    ax.scatter(scl*plants.loc[0,'x_isn'], scl*plants.loc[0, 'y_isn'], c='r', label='HH')
    ax.scatter(scl*plants.loc[1,'x_isn'], scl*plants.loc[1, 'y_isn'], c='r', label='NV')

#------------------------------------------------
# Initialize the output model
#------------------------------------------------

# Init output model objects
mod_list = [np.nan for jj in range(nband)]
for jj in range(nband):
    #mod_list[jj] = MapData(model.x, model.y, [model.z[0], model.z[0]+z_add[jj]])
    mod_list[jj] = MapData(model.x, model.y, [model.z[0], data.z[0]+z_add[jj]])

#------------------------------
# Run inversion 
#------------------------------

func_grn = mag.green
func_jac = mag.jacobi
b0, inc, dec = B0, 75.4, -12.8 # B0 in Tesla
#b0, inc, dec = B0, 90, 0 # B0 in Tesla
earth = MagneticBackgroundField(b0, inc, dec)
eps = 1e-6

tic = time.perf_counter()
inver = [np.nan for jj in range(nband)]
synt  = [np.nan for jj in range(nband)]
for jj in range(0,nband_run):
    print('########## Inversion band {} ##########'.format(jj))
    lam = lam_dict[jj]
    mod_list[jj].vt = earth.vt
    vt_e = vt_m = earth.vt
    inver[jj], synt[jj] = map_inversion(func_grn, data_band[jj], mod_list[jj], 
                                        vt_e, vt_m, eps, lam=lam_dict[jj],
                                        func_jac=func_jac, gf_max=gf_max, nnn=1, 
                                        inc_data=inc_data[jj], inc_mod=inc_mod[jj],
                                        niter=niter, verbose=3)

toc = time.perf_counter()
time_inv = toc - tic
print('Time inversion: {} sec'.format(time_inv))

#######################
# Some junk plots
#######################

scl = 1e-3
xtnt = scl*np.array([model.y[0], model.y[-1], model.x[0], model.x[-1]])

# PLot inversion bands
fig, axs = plt.subplots(2,nband_run, figsize=(5*nband_run,10))
for kb in range(0, nband_run):
    ax = axs.ravel()[kb+nband_run]
    im = ax.imshow(inver[kb].magn.T, origin='lower', extent=xtnt)
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label('Magnetization [A/m]')
    ax.axis('scaled')
    ax.set_title(f'Aeromag inversion band[{kb}]')
    ax.scatter(scl*plants.loc[0,'x_isn'], scl*plants.loc[0, 'y_isn'], c='r', label='HH')
    ax.scatter(scl*plants.loc[1,'x_isn'], scl*plants.loc[1, 'y_isn'], c='r', label='NV')
    ax.set_xlabel('easting [km]')
    ax.set_ylabel('northing [km]')

for kb in range(0, nband_run):
    ax = axs.ravel()[kb]
    im = ax.imshow(data_band[kb].tma.T, origin='lower', extent=xtnt)
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label('TMA [nT]')
    ax.axis('scaled')
    ax.set_title(f'Aeromag data band[{kb}]')
    ax.scatter(scl*plants.loc[0,'x_isn'], scl*plants.loc[0, 'y_isn'], c='r', label='HH')
    ax.scatter(scl*plants.loc[1,'x_isn'], scl*plants.loc[1, 'y_isn'], c='r', label='NV')
    ax.set_xlabel('easting [km]')
    ax.set_ylabel('northing [km]')
fig.tight_layout(pad=2.)
fig.savefig(png + f'Aeromag_Data_and_Inversion_run{krun}.png')

#########################

# Save the outputs from the inversion to pickle file
fname = pkl + 'MagInversion_run'+str(krun)+'.pkl'
with open(fname,'wb') as fid:
    pickle.dump([data_band, mod_list, inver, synt], fid)

plt.show(block=block)







