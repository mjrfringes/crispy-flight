import numpy as np
from astropy.io import fits
from extract import extract1D
from update import update, apply_update

# these are the calibration files when you want to extract
old_mono = 'files/monochromekey_original.fits'
old_calib = 'files/calib_original.fits'

# new file comes in
fname = '/Users/mrizzo/Downloads/det691.60_828.40_20190205T1223.fits'
img = fits.getdata(fname)

# wavelength grid in nm that you want to extract
lam_midpts = np.arange(695.,826.,10.)
# alternatively:
# from crispy.tools.reduction import calculateWaveList
# lam_midpts,lam_endpts = calculateWaveList(par,method='optext')

#### Cube extraction ####
cube = extract1D(img,lam_midpts,old_calib)
fits.writeto('out.fits',cube,overwrite=True)

#### Wavelength update always at 760nm ####
fname = '/Users/mrizzo/Downloads/det760.0_760.0_20190205T1228.fits'
update_760 = fits.getdata(fname)
dx,dy,snr = update(update_760,old_mono)

#### Verify that values are not crazy large ####
Dx = np.nanmean(dx)
Dy = np.nanmean(dy)
print (Dx,Dy)

#### Apply update ####
# pick a name for new calib (for example, use the date/time for review)
# tstring = time.strftime("%Y%m%d-%H%M%S")
new_mono = 'files/monochromekey760_new.fits'
new_calib = 'files/calib_new.fits'
Dx,Dy,sigx,sigy = apply_update(dx,dy,snr,
                               old_mono,
                               old_calib,
                               new_mono,
                               new_calib,
                               fitrot=True)
                               
#### Cube extraction again ####
cube = extract1D(img,lam_midpts,new_calib)
fits.writeto('out_after_calib.fits',cube,overwrite=True)

# and so forth...
