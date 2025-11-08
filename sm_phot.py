# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:58:58 2022

@author: Seth
"""

import numpy as np
from astropy.io import fits
import os
import glob
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error
import astropy.stats as stat
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder

def mediancombine(filelist):
    '''
    Stacks arrays and takes median along each pixel
    '''
    n = len(filelist)
    first_frame_data = fits.getdata(filelist[0])
    imsize_y, imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y, imsize_x , n)) 
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im
        
    med_frame = np.median(fits_stack, axis = 2)    
    return med_frame

def bias_subtract(filename, path_to_masterbias, outpath):
    '''
    This function takes the file name for a .fits image, and a path to the master bias image
    and subtracts the master bias from the input image, then writes an output file with the new image
    '''
    input_file=fits.getdata(filename)
    input_header=fits.getheader(filename)
    master_bias=fits.getdata(path_to_masterbias)
    output_file=input_file-master_bias
    output_filename = filename.split('\\')[-1]
    fits.writeto(outpath + 'b_' + output_filename, output_file, input_header, overwrite=True) 
    return

def scale_master_dark(path_to_master_dark, desired_exptime):
    '''
    This function will take our master dark frame, find its exposure time, then based on whatever the desired
    exposure time is, will scale our master dark down to match that.
    '''
    master_dark = fits.getdata(path_to_master_dark) # master_dark just gets the master dark data
    dark_exp = fits.getheader(path_to_master_dark)['EXPOSURE'] # dark_exp finds the exposure time of the master dark
    scale = desired_exptime/dark_exp # this scale factor is responsible for scaling our master dark frame to match
                                    # that of our desired exposure time
    return scale*master_dark, dark_exp*scale

def dark_subtract(filename, path_to_master_dark, filepath):
    '''
    This function will take in a bias subtracted flat frame, the master dark, and the bias subtracted file's path
    and subtract the master dark data from the bias subtracted flat frames
    '''
    input_file = fits.getdata(filename)
    input_header = fits.getheader(filename)
    exp_time = fits.getheader(filename)['EXPOSURE']
    scaled_dark, scaled_expt = scale_master_dark(path_to_master_dark, exp_time)
    if scaled_expt != exp_time:
        raise Exception("(replace this with your custom useful error message here)")

    output_file = input_file-scaled_dark
    output_filename = filename.split('\\')[-1]
    fits.writeto(filepath + 'd' + output_filename, output_file, input_header, overwrite=True)
    return

def norm_combine_flats(filelist):
    '''
    Divides flats by their own medians, then takes the median along pixels
    '''
    n = len(filelist)
    first_frame_data = fits.getdata(filelist[0])
    imsize_y, imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y, imsize_x , n))
    
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        norm_im =  im / np.median(im)
        fits_stack[:,:,ii] = norm_im
        
    med_frame = np.median(fits_stack, axis=2)
    return med_frame

def flatfield(filename,master_flat_path,filepath):
    master_flat=fits.getdata(master_flat_path)
    input_file=fits.getdata(filename)
    header=fits.getheader(filename)
    output_file=input_file/master_flat
    output_filename = filename.split('\\')[-1]
    
    fits.writeto(filepath+'f'+output_filename,output_file,header,overwrite=True)
    
def centroid(image,starxy,backgroundxy,size=50):
    '''
    image should be glob of fits image
    star,background should be array-like, shape=2 containing x,y pixel values
    associated with regions containing target star and noise only, respectively
    this function assumes a box width of 50 pixels; assuming shifts are not too close to 25 pixels,
    this should work fine
    returns x,y coords for center of source
    '''
    if type(image)==str:
        img=fits.getdata(image)
    else:
        img=image
    #get box limits
    #and also need 0,0 in img coords: input star coords are box center
    #-> (0,0) = starxs[0],starys[1] since y=0 is at top of img
    width=int(size/2)
    starxs=[starxy[0]-width,starxy[0]+width]
    starys=[starxy[1]-width,starxy[1]+width]
    backxs=[backgroundxy[0]-width,backgroundxy[0]+width]
    backys=[backgroundxy[1]-width,backgroundxy[1]+width]
    #make cutouts using previously defined limits
    starstamp=img[int(starys[0]):int(starys[1]),int(starxs[0]):int(starxs[1])]
    backstamp=img[int(backys[0]):int(backys[1]),int(backxs[0]):int(backxs[1])]
    #define median background noise and 3 sigma
    bgmedian=np.median(backstamp)
    bg3sig=3*np.std(backstamp)
    starstamp-=bgmedian #subtract median noise
    #need to isolate indexes+values where val>3sigma
    zeroed=np.where(starstamp>bg3sig,starstamp,0)#sets pixels to 0 if not >3sigma -> I can just take averages/sums and 
    #these pixels will be ignored as they are now 0
    w_sum=np.nansum(zeroed) #sum of weights
    zeroed = zeroed/w_sum #rescale source pixels prior to averaging
    #for each row/col, the following arrays will give the x/y coord
    #just multiply by zeroed array and sum to get weighted average position
    y2d=[np.repeat(i,repeats=len(zeroed[0]),axis=0) for i in range(len(zeroed[0]))]
    x2d=[np.arange(len(zeroed[0])) for i in range(len(zeroed[0]))]
    xlocal=np.nansum(x2d*zeroed)
    ylocal=np.nansum(y2d*zeroed)
    #these are only relative to the box, so I need to shift them according to the absolute image coords
    xcoord=starxs[0]+xlocal
    ycoord=starys[1]-ylocal
    return xcoord, ycoord

def cshift(im1,im2,starxy,backgroundxy,size=50):
    '''
    im1 is reference image, shifts will be x,y shifts to apply to (img2 + shifts will align)
    star, background xy are same as centroid function, should probably be chosen for im1
    '''
        
    x_im1,y_im1=centroid(im1,starxy,backgroundxy,size=size)
    x_im2,y_im2=centroid(im2,starxy,backgroundxy,size=size)
    xshift= x_im1 - x_im2
    yshift= y_im1 - y_im2
    return xshift,yshift

def roll_and_pad(image,xshift,yshift,pad_size=50):
    '''
    Takes the shifts we calculated earlier and changes the pixel coordinates accordingly.
    Pads with NaNs
    '''
    xshift=round(xshift,0)
    yshift=round(yshift,0)
    rolled=np.roll(np.roll(image,int(yshift),axis=0), int(xshift), axis=1)
    padded=np.pad(rolled,pad_size,mode='constant',constant_values=-1000)
    paddednan=np.where(padded>-999,padded,np.nan)
    return paddednan

def stack_ims(imlist,xshifts,yshifts,path,pad_size=50,obj='',xmax_shift='none',ymax_shift='none'):
    '''
    Takes a set of images, calculates their offsets, aligns them, and stacks them to produce a single image

    Parameters
    ----------
    imlist : TYPE
        DESCRIPTION.
    xshifts : TYPE
        DESCRIPTION.
    yshifts : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    pad_size : TYPE, optional
        DESCRIPTION. The default is 50.
    obj : TYPE, optional
        DESCRIPTION. The default is ''.
    xmax_shift : TYPE, optional
        DESCRIPTION. The default is 'none'.
    ymax_shift : TYPE, optional
        DESCRIPTION. The default is 'none'.

    Returns
    -------
    None.

    '''
    if os.path.isdir(path + 'Stacked_'+obj) == False:
            os.mkdir(path + 'Stacked_'+obj)
            print('\n Making new subdirectory for stacked images:', path + 'Stacked_'+obj+' \n')
    path2=path+'Stacked_'+obj
    ref_sz=fits.getdata(imlist[0])
    hed=fits.getheader(imlist[0])
    filt=fits.getheader(imlist[0])['FILTER']
    ref_pd=np.pad(ref_sz,pad_size,mode='constant',constant_values=-1000)
    image_stack = np.zeros([ref_pd.shape[0],ref_pd.shape[1],len(imlist)])
    for i in range(len(imlist)):
        xshift=xshifts[i]
        yshift=yshifts[i]
        img=fits.getdata(imlist[i])
        img_p=roll_and_pad(img,xshift,yshift,pad_size=pad_size)
        f_name=imlist[i].split('\\')[-1]
        fits.writeto(path+'shifted'+f_name,img_p,overwrite=True)
        image_stack[:,:,i]=img_p
        
    stacked_im=np.nanmedian(image_stack,axis=2)
    
    if (xmax_shift != 'none')&(ymax_shift != 'none'):
        max_x=int(xmax_shift+pad_size)
        max_y=int(ymax_shift+pad_size)
    else:
        max_x=int(np.nanmax(np.absolute(xshifts))+pad_size)#shifts must be int, and need to trim nans too
        max_y=int(np.nanmax(np.absolute(yshifts))+pad_size)
    
    stacked_im=stacked_im[max_y:-max_y,max_x:-max_x]

    fits.writeto(path+'Stacked_'+obj+'/Stacked_image_'+obj+'_'+filt+'.fits',stacked_im,hed,overwrite=True)
    
def stack_image_set(objec,impath,filters,ref_gl,ref_star,ref_back,pad_size=50):
    '''
    takes object name, path to data, filter set, and reference image path, star and background coords
    stacks images of same filter and saves them
    '''
    imgs=glob.glob(impath+'fdb_'+objec+'*.fit')
    imgs.sort()
    ref_im=fits.getdata(ref_gl)
    xoff_all=np.array([])
    yoff_all=np.array([])
    for image in imgs:
        xs1,ys1=cshift(ref_im,image,ref_star,ref_back)
        xoff_all=np.append(xoff_all,int(np.absolute(xs1)))
        yoff_all=np.append(yoff_all,int(np.absolute(ys1)))
    
    big_x=np.nanmax(xoff_all)
    big_y=np.nanmax(xoff_all)
    
    print('Using '+ref_gl+' as reference')
    for filt in filters:
        imgsf=glob.glob(impath+'fdb_'+objec+'*'+filt+'.fit')
        print('Stacking: ',imgsf)
    
        xoffsets=np.array([])
        yoffsets=np.array([])
        for ind in range(len(imgsf)):
            img=fits.getdata(imgsf[ind])
            xoff,yoff=cshift(ref_im,img,ref_star,ref_back)
            xoffsets=np.append(xoffsets,xoff)
            yoffsets=np.append(yoffsets,yoff)
        
        stack_ims(imgsf,xoffsets,yoffsets,impath,pad_size=pad_size,obj=objec,xmax_shift=big_x,ymax_shift=big_y)
        print('Stacked '+filt+' band images')
        
def bg_error_estimate(fitsfile):
    """
    This function will first calculate the background error, which it uses to then finds the total 
    error of the image
    """
    fitsdata = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    
    # What is happening in the next step? Read the docstring for sigma_clip.
    # Answer: removes data more than 3 sigma away from background noise
    filtered_data = sigma_clip(fitsdata, sigma=3.,copy=False)
    
    # Summarize the following steps:
    # Everywhere that was clipped in filtered_data is now given a NaN value
    # Taking the square root gives us the standard error because photons counts are represented 
    # by a Poisson distribution
    # Replaces all NaN values with the median error value
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    print("Writing the background-only error image: ", fitsfile.split('.')[0]+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+"_bgerror.fit", bkg_error, hdr, overwrite=True)
    
    effective_gain = 1.4 # electrons per ADU
    
    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)  
    
    print("Writing the total error image: ", fitsfile.split('.')[0]+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+"_error.fit", error_image, hdr, overwrite=True)
    
    return error_image

def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    This function takes a fits image, a sigma threshold, and the FWHM, identifies sources as being
    nsigma time the median absolute deviation above the noise, centroids the sources and
    returns their pixel coordinates
    """

    # First, check if the region file exists yet, so it doesn't get overwritten
    regionfile = fitsfile.split(".")[0] + ".reg"
     
    if os.path.exists(regionfile) == True:
        print(regionfile, "already exists in this directory. Rename or remove the .reg file and run again.")
        #return    
    
    
    # *** Read in the data from the fits file ***
    image = fits.getdata(fitsfile)
    
    # *** Measure the median absolute standard deviation of the image: ***
    bkg_sigma = mad_std(image)

    # *** Define the parameters for DAOStarFinder ***
    daofind = DAOStarFinder(fwhm=fwhm_value, threshold=nsigma_value*bkg_sigma)
    
    # Apply DAOStarFinder to the image
    sources = daofind(image)
    nstars = len(sources)
    print("Number of stars found in ",fitsfile,":", nstars)
    
    # Define arrays of x-position and y-position
    xpos = np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])
    
    # Write the positions to a .reg file based on the input file name
    if os.path.exists(regionfile) == False:
        f = open(regionfile, 'w') 
        for i in range(0,len(xpos)):
            f.write('circle '+str(xpos[i])+' '+str(ypos[i])+' '+str(fwhm_value)+'\n')
        f.close()
        print("Wrote ", regionfile)
    
    return xpos, ypos # Return the x and y positions of each star as variables

def measurePhotometry(fitsfile, starxy_pos_list, aperture_radius, sky_inner, sky_outer, error_array):
    """
    Takes an image, coordinates for sources, aperture and annulus sizes, and errors, creates a table of aperture
    and annulus counts, background subtracted aperture counts, and propogates error from background noise and
    mean error in background noise to aperture area
    """
    # *** Read in the data from the fits file:
    image = fits.getdata(fitsfile)
    
    starapertures = CircularAperture(starxy_pos_list,r = aperture_radius)
    skyannuli = CircularAnnulus(starxy_pos_list, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    # What is new about the way we're calling aperture_photometry?
    # now we factor in the error derived from the error estimate function
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
        
    # calculates mean background counts/pixel from the annulus area
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    #calculates the total background counts inside the aperture using aperture area and background count density
    bkg_starap_sum = bkg_mean * starapertures.area
    #subtracts background counts from total aperture counts
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    #adds background subtracted aperture counts to table
    phot_table['bg_subtracted_star_counts'] = final_sum
    
    # calculates the uncertainty in background noise per pixel
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    #calculates error in background counts inside aperture
    bkg_sum_err = bkg_mean_err * starapertures.area
    
    # adds total background counts and background mean uncertainty in quadrature and adds to table
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table

def zpcalc(magzp, magzp_err, filtername, dataframe):
    inst_mag = dataframe[str(filtername)+'_inst']
    inst_mag_err = dataframe[str(filtername)+'inst_err']
    magcalc = inst_mag + magzp
    magcalc_err = np.sqrt((magzp_err)**2 + (inst_mag_err)**2)
    dataframe[str(filtername)+'_cal'] = magcalc
    dataframe[str(filtername)+'cal_err'] = magcalc_err