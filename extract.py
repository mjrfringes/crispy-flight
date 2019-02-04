# this is extremely lean version of the IFS Horne extraction software, with only the bare
# minimum needed
# This relies completely on appropriate calibration files built on the ground using crispy

import numpy as np
from astropy.io import fits
from scipy import interpolate

def extract1D(  image,
                lamlist,
                calib_file,
                sum=False,
                delt_y=5
                ):
    '''
    Performs 1D Horne spectral extraction if sum=False, or simply the 1D sum if sum=True.
    
    Parameters
    ----------
    image: 2D array of floats
        Represents the cleaned IFS detector image.
    lamlist: 1D array of desired wavelengths
        Outputs data onto desired wavelength grid. Values are in nanometer
    calib_file: calibration data structure file path
        Fits file that contains all necessary calibration for extraction, 
        including the mapping between pixels and wavelengths, etc. 
        This file can be created by the crispy main software.
    sum: boolean
        If True, this represents a 1D sum in the cross-spectral direction
        If False, this fits a 1D Gaussian in the cross-spectral direction
    dy: int
        Determines the width of the microspectrum in the cross-spectral direction
        
    Returns
    -------
    cube: 3D array of floats
        This is the exctracted IFS datacube
    
    7/19/18:
    @TODO: Add dimensions to all arrays or write memo describing all of that
    @TODO: define what gets hardcoded and not
    
    '''
    
    # number of lenslets (hardcoded for now)
    nlens=108
    
    # load calibration file
    calib = fits.open(calib_file)
    
    # wavelength corresponding to each center pixel for each location along the spectral
    # direction
    lam_indx = calib[0].data
    
    # x & y pixel locations of each psflet for each wavelength
    # nominally this is evaluated at the center of the pixels across the dispersion
    # direction
    xindx = calib[1].data
    yindx = calib[2].data
    
    # array of number of wavelengths for each microspectrum
    # for example, towards the edges of the detector there will be less wavelengths
    # per microspectrum since some of them fall outside the detector 
    nlam = calib[3].data.astype(int)
    
    # Number of maximum wavelengths across the spectral dimension across the entire field
    Nmax = np.amax(nlam)

    # boolean mask for good vs bad psflets
    good = calib[4].data.astype(int)
    
    # standard deviation of the gaussian fits (same size as xindx and yindx)
    sig = calib[5].data
    
    # define arrays of indices to extract the data from the detector
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    
    # allocate space for results
    coefs = np.zeros(tuple([max(Nmax, len(lamlist))] + list(yindx.shape)[:-1]))
    cube = np.zeros((len(lamlist), nlens, nlens))
    xarr, yarr = np.meshgrid(np.arange(Nmax), np.arange(delt_y))
    print xarr.shape,yarr.shape

    # This is the main loop over all lenslets
    for i in range(xindx.shape[0]):
        for j in range(yindx.shape[1]):
            # only do this for a good lenslet that hasn't been flagged
            if good[i, j]:
                # load the values for this microspectrum
                _x = xindx[i, j, :nlam[i, j]]
                _y = yindx[i, j, :nlam[i, j]]
                _sig = sig[i, j, :nlam[i, j]]
                _lam = lam_indx[i, j, :nlam[i, j]]
                # this needs to be done just to get rid of unintended NaNs
                iy = np.nanmean(_y)
                
                if ~np.isnan(iy):
                    # find the index of the bottom line of the microspectrum
                    i1 = int(iy - delt_y / 2.)+1
                    
                    # this constructs a 2D array of pixel distances from the 
                    # center line of the microspectrum
                    dy = _y[xarr[:,:len(_lam)]] - y[i1:i1 + delt_y,int(_x[0]):int(_x[-1]) + 1]
                    
                    # now extract the microspectrum from the image
                    data = image[i1:i1 + delt_y, int(_x[0]):int(_x[-1]) + 1]

                    # if the mode is simple sum, we will not need any weights
                    if sum: weight = 1.
                    else:
                        weight = np.exp(-dy**2 / _sig**2)
                        # normalize weights for each microspectrum column
                        weight /= np.sum(weight,axis=0)[np.newaxis,:]
                    
                    # sum along the cross-spectral direction
                    coefs[:len(_lam), i, j] = np.sum(weight * data, axis=0)
                
                    # normalize if not doing the sum
                    if ~sum:
                        coefs[:len(_lam), i, j] /= np.sum(weight**2, axis=0)
                
                    # now we want to interpolate onto the desired wavelength grid
                    # for this we use a third-order spline interpolation, but we could
                    # test much simpler methods for 1D interpolation
                    # also, if grid is unique, might be able to hard code some of the
                    # interpolation steps to save flops
                    # place results in the cube placeholder
                    tck = interpolate.splrep(_lam, coefs[:len(_lam), i, j], s=0, k=3)
                    cube[:, j, i] = interpolate.splev(lamlist, tck, ext=1)
                else:
                    cube[:, j, i] = np.NaN
            else:
                cube[:, j, i] = np.NaN
                
    # It is possible that an extra flatfielding step will be wanted. If this is desired
    # in this function, then it would be:
    # cube *= flatfield
    # A similar expression would hold for further masking or smoothing or whatever.
    # Of course, a flatfield and/or a mask would need to be supplied to the function.
    
    return cube
