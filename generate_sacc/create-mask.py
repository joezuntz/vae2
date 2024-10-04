#Creates a healpix mask copying Prat & Zuntz et al. to get area of 12300 sqdeg. 
import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os

output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# create a healpix map of zeros
nside = 2048
npix = hp.nside2npix(nside)
binary_mask = np.zeros(npix)
# pixel ids
pix = np.arange(npix)
#position of pixels in RA and Dec, theta=longitude, phi=latitute
theta, phi = hp.pix2ang(nside, pix, nest=False, lonlat=True)

survey = "Y1"
A_full = 41_253.0 # sq deg
A_Y1 = 17935.71831


f_sky = A_Y1/A_full
phi0 = -np.degrees(np.arcsin(2 * A_Y1 / A_full))
phi1 = 0
print(phi0)

theta0, theta1 = 0., 360.  

# phi0, phi1 = -36.61, 0 
# # new sky area:
# # 
    

# pixels within boundaries set to 1
in_mask = (theta<theta1)&(theta>theta0)&(phi<phi1)&(phi>phi0)
binary_mask[in_mask] = 1.
print('The area of the '+survey+'mask is:', hp.nside2pixarea(nside, degrees=True)*binary_mask[in_mask].size)

#write file out
hp.write_map(output_dir+'/lsst_'+survey+'_binary_mask.fits',binary_mask,overwrite=True)
hp.mollview(binary_mask)
# save plot of mask - comment out to save disk space if running in home directory on NERSC
plt.savefig(output_dir+'lsst_'+survey+'_binary_mask.png')


