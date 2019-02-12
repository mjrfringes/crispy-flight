from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def gauss(x, a, x0, sig,b):
    '''
    Simple gaussian function with usual inputs
    '''
    return b+a*np.exp(-(x-x0)**2/(2.*sig**2))
    


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
                            p0=[np.amax(vals),lamlist[np.argmax(vals)],sigma_guess,0]
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