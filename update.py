import numpy as np
from astropy.io import fits
from scipy import interpolate

def centroid(img):
    '''
    Very simple centroiding function implemented manually
    
    Parameters
    ----------
    img: 2d array
        Image to centroid
        Image can be masked with masked values equal to np.NaN
        
    Returns
    -------
    sx, sy: floats
        centroid coordinates in pixels
    '''
    lx,ly = img.shape
    xgrid = np.arange(lx)
    ygrid = np.arange(ly)
    xgrid,ygrid = np.meshgrid(xgrid, ygrid)
    s = np.nansum(img)
    if np.nonzero(s):
        sx = np.nansum(xgrid * img) / s
        sy = np.nansum(ygrid * img) / s
    else:
        sx = np.NaN
        sy = np.NaN
    return sx,sy
    

def update( data,
            monochromekey_file,
            halfsize=5,
            apdiam=3,
            snrthreshold=10
            ):
    '''
    Determines dx, dy offsets for all lenslets
    global offset is the nanmean of these arrays
    
    Parameters
    ----------
    data: 2d array
        New clean monochromatic data with high SNR psflets
    monochromekey: calibration structure instance for single wavelength
        Contains x, y positions of all lenslets for given wavelength
        Also contains lenslet mask "good"
    halfsize: int
        Half size of search area in pixels
    apdiam: int
        Aperture diameter for photometry
    snrthreshold: int
        Threshold for which psflet centroids will be discarded
    '''
    
    monochromekey = fits.open(monochromekey_file)
    
    # load positions and mask
    x = monochromekey[1].data
    y = monochromekey[2].data
    good = monochromekey[3].data

    # calculate data median and allocate space
    median = np.median(data)
    ysize, xsize = data.shape
    dy = np.zeros_like(x)
    dx = np.zeros_like(x)
    snr = np.zeros_like(x)
    
    # calculate arrays that are useful
    mgrid = np.arange(2*halfsize)
    xgrid,ygrid = np.meshgrid(mgrid,mgrid)
    
    # run centroid routine for all psflets
    for j in range(x.shape[0]):
        for k in range(x.shape[1]):
            if good[j,k]:
                xl = x[j,k]
                yl = y[j,k]
                # define bottom left coordinates of cutout
                xmin = int(xl-halfsize)+1
                ymin = int(yl-halfsize)+1
                
                # ensure cutout is within image
                if ymin>0 and xmin>0 and xmin+2*halfsize<xsize and ymin+2*halfsize<ysize:
                    # define cutout
                    cutout = data[ymin:ymin + 2*halfsize,xmin:xmin + 2*halfsize] - median

                    # here is the new centroiding function: we could change this to something more robust
                    dx[j,k], dy[j,k] = centroid(cutout)

                    # mask used for elementary aperture photometry
                    apmask = (xgrid-dx[j,k])**2 + (ygrid-dy[j,k])**2 < apdiam**2
                    apval = np.nansum(apmask * cutout)

                    # estimate of SNR, only valid for very high fluxes, could do better
                    snr[j,k] = np.abs(apval) / np.sqrt(np.abs(apval) + median)
                    
                    # apply threshold
                    if snr[j,k] < snrthreshold:
                        dy[j,k] = np.NaN
                        dx[j,k] = np.NaN
                    else:
                        dy[j,k] -= y[j,k] - ymin
                        dx[j,k] -= x[j,k] - xmin
            else:
                snr[j,k] = np.NaN
                dy[j,k] = np.NaN
                dx[j,k] = np.NaN
        
    return dx,dy,snr
    
def apply_update(   dx,
                    dy,
                    snr,
                    old_monochromekey_file,
                    old_calib_file,
                    new_monochromekey_file,
                    new_calib_file,
                    fitrot = True
                    ):
    
    # if we don't fit the rotation, then it's a simple mean offset
    if not fitrot:
        Dx = np.nanmean(dx)
        Dy = np.nanmean(dy)
        # update the monochrome
        hdul = fits.open(old_monochromekey_file)
        hdul[1].data += Dx
        hdul[2].data += Dy
            # probably want to write something here in the header
        hdul.writeto(new_monochromekey_file)
        hdul.close()
        print "Created %s" % new_monochromekey_file
        
        # update the calibration file
        hdul = fits.open(old_calib_file)
        # for cross-dispersion axis, this is quite trivial 
        # (assuming microspectra are nice and aligned)
        # for dispersion axis it is a bit more complicated because
        # we need to re-interpolate the grid if the shift is not an integer pixel
        xindx = hdul[1].data.copy()
        nlam = hdul[3].data.astype(int)
        lams = hdul[0].data.copy()
        good = hdul[4].data
        for ix in range(xindx.shape[0]):
            for iy in range(xindx.shape[1]):
                if good[ix,iy]:
                    tck_y = interpolate.splrep(xindx[ix,iy,:nlam[ix,iy]], lams[ix,iy,:nlam[ix,iy]], k=1, s=0)
                    dDx = Dx - np.int(Dx)
                    hdul[0].data[ix,iy,:nlam[ix,iy]] = interpolate.splev(xindx[ix,iy,:nlam[ix,iy]]+dDx, tck_y)
                    hdul[1].data[ix,iy,:nlam[ix,iy]] += np.int(Dx)
                    hdul[2].data[ix,iy,:nlam[ix,iy]] += Dy
                    
        hdul.writeto(new_calib_file)
        hdul.close()
        print "Created %s" % new_calib_file
            
        # calculate standard deviations
        sigx = np.nanstd(dx)/np.sqrt(np.sum(~np.isnan(dx)))
        sigy = np.nanstd(dy)/np.sqrt(np.sum(~np.isnan(dy)))
        
#         export offsets and stddev
        return Dx,Dy,sigx,sigy

    # if we need to fit the rotation, then we need a bit more work
    else:
        hdul = fits.open(old_monochromekey_file)    
        # load positions and mask
        x = hdul[1].data.copy()
        y = hdul[2].data.copy()
        good = hdul[3].data.copy()
        
        # form the clouds of points before and after update
        old = []
        new = []
        
        # need to know the approximate center of rotation for better accuracy
        xc = x[x.shape[0]//2,x.shape[1]//2]
        yc = y[y.shape[0]//2,y.shape[1]//2]
        
        # put point coordinates in the lists
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                if good[j,k] and ~np.isnan(x[j,k]) and ~np.isnan(y[j,k]) and ~np.isnan(dx[j,k]) and ~np.isnan(dy[j,k]):
                    old.append([x[j,k] - xc,y[j,k] - yc])
                    new.append([x[j,k] + dx[j,k] - xc,y[j,k] + dy[j,k] - yc])

        # put them in matrix format to represent a translation and a rotation
        # about the center of the detector
        A_data = []
        for pt in old:
            A_data.append( [-pt[1], pt[0], 1, 0] )
            A_data.append( [ pt[0], pt[1], 0, 1] )

        b_data = []
        for pt in new:
            b_data.append(pt[0])
            b_data.append(pt[1])

        # Solve via least squares (svd also possible if preferred)
        # Solving Ax=b
        A = np.matrix( A_data )
        b = np.matrix( b_data ).T
        results,residuals,rank,sv = np.linalg.lstsq(A, b)
        s,c,tx,ty = np.array(results.T)[0]
        print('s,c,tx,ty:',s,c,tx,ty)

        # record the actual dx, dy that was applied to all lenslets
        Dx = c*(x-xc) - s*(y-yc) + tx + xc - x
        Dy = c*(y-yc) + s*(x-xc) + ty + yc - y
        
        # apply Dx, Dy
        hdul[1].data += Dx
        hdul[2].data += Dy
        
        hdul[0].header['comment'] = 'Previous file: '+old_monochromekey_file
        hdul[0].header['comment'] = 's,c,tx,ty:'+str(s)+str(c)+str(tx)+str(ty)
        
        # calculate standard deviations
        sigx = np.nanstd(Dx)/np.sqrt(np.sum(~np.isnan(Dx)))
        sigy = np.nanstd(Dy)/np.sqrt(np.sum(~np.isnan(Dy)))
        
        # @TODO: need to mask some dots due to rotation?
                    
        # here update some headers and push the updates
        hdul.writeto(new_monochromekey_file)
        hdul.close()
        print "Created %s" % new_monochromekey_file

        # update the calibration file
        hdul = fits.open(old_calib_file)
        # for cross-dispersion axis, this is quite trivial 
        # (assuming microspectra are nice and aligned)
        # for dispersion axis it is a bit more complicated because
        # we need to re-interpolate the grid if the shift is not an integer pixel
        xindx = hdul[1].data.copy()
        nlam = hdul[3].data.astype(int)
        lams = hdul[0].data.copy()
        good = hdul[4].data
        for ix in range(xindx.shape[0]):
            for iy in range(xindx.shape[1]):
                if good[ix,iy]:
                    tck_y = interpolate.splrep(xindx[ix,iy,:nlam[ix,iy]], lams[ix,iy,:nlam[ix,iy]], k=1, s=0)
                    iDx = np.int(Dx[ix,iy])
                    dDx = Dx[ix,iy] - iDx
                    hdul[0].data[ix,iy,:nlam[ix,iy]] = interpolate.splev(xindx[ix,iy,:nlam[ix,iy]]+dDx, tck_y)
                    hdul[1].data[ix,iy,:nlam[ix,iy]] += iDx
                    hdul[2].data[ix,iy,:nlam[ix,iy]] += Dy[ix,iy]
        
        hdul.writeto(new_calib_file)
        hdul.close()
        print "Created %s" % new_calib_file

    
        return Dx,Dy,sigx,sigy
            # rotation matrix coeffs and translation vector
#             return s,c,tx,ty
        
