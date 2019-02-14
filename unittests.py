from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from scipy import interpolate

def gauss(x, a, x0, sig,b,c):
    '''
    Simple gaussian function with usual inputs
    '''
    return c+b*x+a*np.exp(-(x-x0)**2/(2.*sig**2))
    


def fit_monochromatic_cube( cube,
                            lamlist,
                            returnAll = False,
                            sigma_guess = 5):
    '''
    Fits an extracted data cube with a gaussian to find the wavelength peak
    
    Parameters
    ----------
    cube: 3D ndarray
        The extracted datacube where all bad pixels are NaNs
    lamlist: 1D array
        List of wavelengths corresponding to the slices of the cube
        Suggested units: nanometers (in which sigma_guess is about 5)
    returnAll: boolean
        If True, return the full results of the curve fit function (popt,pcov)
        If False, return only the central wavelength (Default)
    sigma_guess: float
        Guess at the width of the gaussian fit in same units as lamlist (Default 5)
    '''
    vals = np.nansum(np.nansum(cube,axis=2),axis=1)
    popt, pcov = curve_fit( gauss,
                            lamlist,
                            vals,
                            p0=[np.amax(vals),lamlist[np.argmax(vals)],sigma_guess,0,0]
                            )
    if returnAll: return popt,pcov
    else: return popt[1]


    ## to reconstruct gaussian function, use;
    # gauss(lamlist,*popt)

def plotfit(cube,lamlist):
    vals = np.nansum(np.nansum(cube,axis=2),axis=1)
    popt, pcov = fit_monochromatic_cube(cube,lamlist,returnAll=True)
    curve_vals = gauss(lamlist, *popt)
    print popt[1]
    plt.figure(figsize=(10,10))
    plt.plot(lamlist,curve_vals,label='Fitted')
    plt.plot(lamlist,vals,label='Data')
    plt.legend()

def histograms(dx,dy):
    plt.figure(figsize=(10,10))
    plt.hist(dx[~np.isnan(dx)], bins='auto',label='dx') 
    plt.hist(dy[~np.isnan(dy)], bins='auto',label='dy') 
    plt.legend()


def inspect_update(old_mono,
                   update_image,
                   new_mono,
                   outname=None,
                   outdir=''):
    
    
    fig, ax = plt.subplots(figsize=(15, 15))
    image = fits.getdata(update_image)
    # start by displaying grayscale image in the background
    mean = np.mean(image)
    std = np.std(image)
    norm = mpl.colors.Normalize(vmin=mean, vmax=mean + 5 * std)
    ax.imshow(
        image,
        cmap='Greys',
        norm=norm,
        interpolation='nearest',
        origin='lower')
    
    # open new monochrome and display positions as red circles
    monochromekey = fits.open(new_mono)
    xpos = monochromekey[1].data
    ypos = monochromekey[2].data
    marker = '+'
    plt.plot(xpos.flatten(),ypos.flatten(),marker=marker,linestyle='None',color='crimson',alpha=0.5, label='New wavelength solution')
    
    # open old monochrome and display positions as blue crosses
    monochromekey = fits.open(old_mono)
    xpos = monochromekey[1].data
    ypos = monochromekey[2].data
    marker = 'x'
    plt.plot(xpos.flatten(),ypos.flatten(),marker=marker,linestyle='None',alpha=0.3, label='Old wavelength solution')
    plt.xlim([0,image.shape[0]])
    plt.ylim([0,image.shape[1]])
    plt.legend(fontsize=20)
    if outname is None: 
        if '/' in update_image:
            name = update_image.split('/')[-1].split('.fits')[0]
    else: name = outname
    print(name)
    fig.savefig(outdir+'inspect_update_%s.pdf' % (name), dpi=100)
    fig.savefig(outdir+'inspect_update_%s.png' % (name), dpi=300)
    plt.close(fig)
    
def check_calib(fname, # file name of image used to update
                lam,   # wavelength corresponding to image
                calib_file,
                outdir='', 
                outname='test'):
    '''
    '''
    
    # number of lenslets (hardcoded for now)
    nlens=108
    
    image = fits.getdata(fname)
    
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
#     coefs = np.zeros(tuple([max(Nmax, len(lamlist))] + list(yindx.shape)[:-1]))
#     cube = np.zeros((len(lamlist), nlens, nlens))
#     xarr, yarr = np.meshgrid(np.arange(Nmax), np.arange(delt_y))
#     print xarr.shape,yarr.shape

    xpos = np.zeros((xindx.shape[0],xindx.shape[1]))
    ypos = np.zeros((yindx.shape[0],yindx.shape[1]))
    
    # This is the main loop over all lenslets
    for i in range(xindx.shape[0]):
        for j in range(yindx.shape[1]):
            # only do this for a good lenslet that hasn't been flagged
            if good[i, j]:
                # load the values for this microspectrum
                _x = xindx[i, j, :nlam[i, j]]
                _y = yindx[i, j, :nlam[i, j]]
                _lam = lam_indx[i, j, :nlam[i, j]]
                tck = interpolate.splrep(_lam, _x, s=0, k=3)
                xpos[i,j] = interpolate.splev(lam, tck, ext=1)
                ypos[i,j] = np.nanmean(_y)
                
    fig, ax = plt.subplots(figsize=(15, 15))
    image = fits.getdata(fname)
    # start by displaying grayscale image in the background
    mean = np.mean(image)
    std = np.std(image)
    norm = mpl.colors.Normalize(vmin=mean, vmax=mean + 5 * std)
    ax.imshow(
        image,
        cmap='Greys',
        norm=norm,
        interpolation='nearest',
        origin='lower')

    marker='+'
    plt.plot(xpos.flatten(),ypos.flatten(),marker=marker,linestyle='None',color='crimson',alpha=0.5, label='New wavelength solution')
    plt.xlim([0,image.shape[0]])
    plt.ylim([0,image.shape[1]])
    plt.legend(fontsize=20)
    
    fig.savefig(outdir+'%s.pdf' % outname, dpi=100)
                