#this script will randomly insert the absorption lines into the spectra in subsamples.
#update in Jul 1. 2018: add line_model_injection_ver01
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import h5py
from scipy import signal
from lowess import lowess
from scipy import interpolate
from random import uniform, gauss


def continuum_fit(wave, flux, error, iteration_num):


    wave_fit = wave
    flux_fit = flux
    error_fit = error
    print('continum_fit_line17, wave_fit\n', wave_fit)
    print('continum_fit_line18, flux_fit\n', flux_fit)
    print('continum_fit_line19, error_fit\n', error_fit)
	
    fit_lo = lowess(wave_fit, flux_fit,f=0.02, iter=1)
    index_overflow = (fit_lo != 0.0)
    flux_fit = flux_fit[index_overflow]
    wave_fit = wave_fit[index_overflow]
    error_fit = error_fit[index_overflow]
    print('continum_fit_line24, fit_lo\n', fit_lo)
    print('continum_fit_line25, index_overflow\n', index_overflow)
    print('continum_fit_line26, flux_fit\n', flux_fit)
    print('continum_fit_line27, wave_fit\n', wave_fit)
    print('continum_fit_line28, error_fit\n', error_fit)


    ddd=signal.medfilt(flux_fit/fit_lo,kernel_size=41) #51
    fit_lo = fit_lo*ddd


    f = interpolate.interp1d(wave_fit, fit_lo)
    fit_final=f(wave)
    print('continum_fit_line36, ddd\n', ddd)
    print('continum_fit_line37, fit_lo\n', fit_lo)
    print('continum_fit_line40, f\n', f)
    print('continum_fit_line40, fit_final\n', fit_final)
		
    return fit_final






def double_gaussian(x, center1, center2, EW1, EW2, sigma1, sigma2, offset):


    scale1 =  -1.0/np.sqrt(2*np.pi*sigma1**2)
    scale2 =  -1.0/np.sqrt(2*np.pi*sigma2**2)

    double_profile =  scale1 * EW1 * np.exp( - (x - center1)**2.0 / (2.0 * sigma1**2.0) ) \
          + scale2 * EW2 * np.exp( - (x - center2)**2.0 / (2.0 * sigma2**2.0) ) + offset

    return double_profile



def double_fit(wave, fnor, zabs, line_list, errornor):
    center1_lower = line_list[0]*(1+zabs-0.1)/(1+zabs)
    center1_upper = line_list[0]*(1+zabs+0.1)/(1+zabs)

    center2_lower = line_list[1]*(1+zabs-0.1)/(1+zabs)
    center2_upper = line_list[1]*(1+zabs+0.1)/(1+zabs)

    print('double_fit_line55, center1_lower\n', center1_lower)
    print('double_fit_line56, center1_upper\n', center1_upper)
    print('double_fit_line58, center2_lower\n', center2_lower)
    print('double_fit_line59, center2_upper\n', center2_upper)	

    EW1_lower = 0.0
    EW1_upper = 10.0

    EW2_lower = 0.0
    EW2_upper = 10.0

    sigma1_lower = 0.0
    sigma1_upper = 5.0

    sigma2_lower = 0.0
    sigma2_upper = 5.0

    offset_lower = -0.05
    offset_upper = 0.05

    popt, pcov = curve_fit(double_gaussian, wave, fnor, sigma = errornor,\
                 bounds=([center1_lower, center2_lower, EW1_lower, EW2_lower, sigma1_lower, sigma2_lower, offset_lower],\
                         [center1_upper, center2_upper, EW1_upper, EW2_upper, sigma1_upper, sigma2_upper, offset_upper]) )
    print('double_fit_line81, popt\n', popt)

    index_err1=(popt[0]+3*popt[4] >= wave) & (popt[0]-3*popt[4] <= wave)
    error1 = 2.355*popt[4]/np.mean((fnor[index_err1]+1)/errornor[index_err1])
    print('double_fit_line88, index_err1\n', index_err1)
    print('double_fit_line89, error1\n', error1)

    index_err2=(popt[1]+3*popt[5] >= wave) & (popt[1]-3*popt[5] <= wave)
    error2 = 2.355*popt[5]/np.mean((fnor[index_err2]+1)/errornor[index_err2])
    print('double_fit_line88, index_err2\n', index_err2)
    print('double_fit_line89, error2\n', error2)
	
    print('this is SNR for measure:', np.mean((fnor[index_err1]+1)/errornor[index_err1]), np.mean((fnor[index_err2]+1)/errornor[index_err2]) )

    return popt, error1, error2





def single_gaussian(x, center1,  EW1, sigma1,  offset):


    scale1 =  -1.0/np.sqrt(2*np.pi*sigma1**2)


    single_profile =  scale1 * EW1 * np.exp( - (x - center1)**2.0 / (2.0 * sigma1**2.0) ) + offset

    return single_profile



def single_fit(wave, fnor, zabs, line_list, errornor):
    center1_lower = line_list[0]*(1+zabs-0.1)/(1+zabs)
    center1_upper = line_list[0]*(1+zabs+0.1)/(1+zabs)


    EW1_lower = 0.0
    EW1_upper = 10.0



    sigma1_lower = 0.0
    sigma1_upper = 5.0


    offset_lower = -0.01
    offset_upper = 0.01

    popt, pcov = curve_fit(double_gaussian, wave, fnor, sigma = errornor,\
                 bounds=([center1_lower, EW1_lower, sigma1_lower, offset_lower],\
                         [center1_upper, EW1_upper, sigma1_upper, offset_upper]) )

    index_err1=(popt[0]+3*popt[2] >= wave) & (popt[0]-3*popt[2] <= wave)
    error1 = 2.355*popt[2]/np.mean((fnor[index_err1]+1)/errornor[index_err1])



    return popt, error1



def noise_generation(wave, flux, error):
    new_flux=np.zeros(wave.size)
    for i in np.arange(wave.size):
        arr=np.random.normal(flux[i], error[i], 1000)
        index_need=np.random.randint(arr.size, size=1)
        new_flux[i]=arr[index_need]

    return new_flux



def gaussian_profile(x, mu, sig):
    div = x - np.roll(x, +1)
    index_profile=(mu+6*sig >= x) & (mu-6*sig <= x)
    profile = -1.0/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))
    scale=np.abs(np.sum(profile[index_profile]*div[index_profile]))
    #print scale
    profile = profile/scale

    return profile


def line_model_injection(wave, flux, zqso, line_list, error):
    #scan the entire wavelength array and find the region where the SNR is high enough.
    index_zero = (error > 0.0)
    error_reomve = error[index_zero]
    flux_remove =  flux[index_zero]
    wave_remove = wave[index_zero]
    SNR_array = flux_remove/error_reomve
    index_region = (SNR_array >= 3.0)
    wave_good = wave_remove[index_region]
    zabs_SNR_lower = wave_good.min()/line_list[0]-1
    zabs_SNR_upper = wave_good.max()/line_list[1]-1


    #set the lyman-alpha limit and zqso limit
    zqso_lower = np.max( [(1+zqso)*1530.0/(line_list[0]-15) -1, 4000.0/(line_list[0]-15) -1]) #zqso*1215.0/(line_list[0]-15) -1
    zqso_upper = np.min([zqso, 8000/(line_list[0]-15) -1])


    zabs_lower = np.max(np.array([zabs_SNR_lower, zqso_lower ]))
    zabs_upper = np.min(np.array([zabs_SNR_upper, zqso_upper ]))
    zabs = uniform(zabs_lower, zabs_upper)

    print('this is the z info:', zqso, zabs)


    #calcualate the width of absorption profile
    line_wave=line_list*(1+zabs)
    #calculate the resolution of lines.
    resolution=2000
    min_reso = np.mean(line_wave/(resolution*2.355))
    SNR_array = np.zeros((1, line_list.shape[0]))



    line_model=np.zeros((wave.size, line_list.size))
    #cacluate the coressponding error for lines
    for i in np.arange(line_list.size):
        index_error = (wave >= line_wave[i]-3*min_reso) & (wave <= line_wave[i]+3*min_reso)
        SNR_array[0, i] = np.mean(flux[index_error]/error[index_error])


    print('this is the minimum SNR: %s ' % SNR_array.min())

    EW_array = np.zeros((1, line_list.shape[0]))
    # generate the FWHM for the line

    FWHM_nal = (1+zabs)*np.min([4.5, np.max([1.0, gauss(1.7, 0.7)])])
    print('this is the FWHM for both lines: %s ' % FWHM_nal)
    SL_array = np.zeros((1, line_list.shape[0]))

    #make sure the EW > 0.5. since min FWHM =1.0
    #min SL = SNR*0.5
    low_lim_line1 = np.max([3, 0.5*SNR_array[0, 0]])
    SL_array[0, 0] = uniform(low_lim_line1, SNR_array[0, 0])

    #make sure the ratio is 1~2 and 2803 line > 0.5
    low_lim = np.max([3, 0.5*SL_array[0, 0],  0.5*SNR_array[0, 1]])
    up_lim = np.min([SL_array[0, 0], SNR_array[0, 1]])
    SL_array[0, 1] = uniform( low_lim, up_lim )

    # generate the EW for the line
    for i in np.arange(line_list.size):
        EW_array[0, i] = SL_array[0, i]*FWHM_nal / ( (1+zabs)*SNR_array[0, i])
        line_model[:,i]=gaussian_profile(wave, line_wave[i], FWHM_nal/2.355)*EW_array[0, i]*(1+zabs)



    abs_model=np.sum(line_model, axis=1)+1
    flux_abs=flux*abs_model

    line_info = np.concatenate((EW_array, SL_array), axis=1)
    FWHM_final =  FWHM_nal/(1+zabs)

    return flux_abs, line_info, SNR_array, zabs, FWHM_final


def line_model_injection_ver01(wave, flux, zqso, line_list, error, SNR_threshold):
    #this version remove the EW > 0.5 constrains.
    #scan the entire wavelength array and find the region where the SNR is high enough.
    index_zero = (error > 0.0)
    error_reomve = error[index_zero]
    flux_remove =  flux[index_zero]
    wave_remove = wave[index_zero]
    SNR_array = flux_remove/error_reomve
    index_region = (SNR_array >= SNR_threshold)
    wave_good = wave_remove[index_region]

    if wave_good.size > 20:
        zabs_SNR_lower = wave_good.min()/line_list[0]-1
        zabs_SNR_upper = wave_good.max()/line_list[1]-1


        #set the CIV limit and zqso limit
        #also the reduce the sky emission
        zqso_lower = np.max( [(1+zqso)*1530.0/(line_list[0]-15) -1, 4000.0/(line_list[0]-15) -1])
        zqso_upper = np.min([zqso, 8000.0/(line_list[0]-15) -1])


        zabs_lower = np.max(np.array([zabs_SNR_lower, zqso_lower ]))
        zabs_upper = np.min(np.array([zabs_SNR_upper, zqso_upper ]))
        zabs = uniform(zabs_lower, zabs_upper)

        print('this is the z info:', zqso, zabs)


        #calcualate the width of absorption profile
        line_wave=line_list*(1+zabs)
        #calculate the resolution of lines.
        resolution=2000
        min_reso = np.mean(line_wave/(resolution*2.355))
        SNR_array = np.zeros((1, line_list.shape[0]))



        line_model=np.zeros((wave.size, line_list.size))
        #cacluate the coressponding error for lines
        for i in np.arange(line_list.size):
            index_error = (wave >= line_wave[i]-3*min_reso) & (wave <= line_wave[i]+3*min_reso)
            SNR_array[0, i] = np.mean(flux[index_error]/error[index_error])


        #print('this is the minimum SNR: %s ' % SNR_array.min() )

        EW_array = np.zeros((1, line_list.shape[0]))
        # generate the FWHM for the line

        FWHM_nal = (1+zabs)*np.min([4.5, np.max([1.0, gauss(1.7, 0.7)])])
        #print('this is the FWHM for both lines: %s ' % FWHM_nal)
        SL_array = np.zeros((1, line_list.shape[0]))

        #low_lim_line1 = np.max([SNR_threshold, SNR_array[0, 0]])
        SL_array[0, 0] = uniform(SNR_threshold, SNR_array[0, 0])

        #make sure the ratio is 1~2 and 2803 line > 0.5
        low_lim = np.max([SNR_threshold, 0.5*SL_array[0, 0]])
        up_lim = np.min([SL_array[0, 0], SNR_array[0, 1]])
        SL_array[0, 1] = uniform( low_lim, up_lim )

        # generate the EW for the line
        for i in np.arange(line_list.size):
            EW_array[0, i] = SL_array[0, i]*FWHM_nal / ( (1+zabs)*SNR_array[0, i])
            line_model[:,i]=gaussian_profile(wave, line_wave[i], FWHM_nal/2.355)*EW_array[0, i]*(1+zabs)



        abs_model=np.sum(line_model, axis=1)+1
        flux_abs=flux*abs_model

        line_info = np.concatenate((EW_array, SL_array), axis=1)
        FWHM_final =  FWHM_nal/(1+zabs)

    else:
        print('cannot insert the absorption lines due to low SNR')
        flux_abs = flux
        line_info = np.zeros((1, 2))
        SNR_array = np.zeros((1, 4))
        zabs = 0.0
        FWHM_final = 0.0
    return flux_abs, line_info, SNR_array, zabs, FWHM_final



def line_model_injection_new(wave, flux, zqso, line_list, error, SNR_threshold):
    #scan the entire wavelength array and find the region where the SNR is high enough.
    index_zero = (error > 0.0)
    error_reomve = error[index_zero]
    flux_remove =  flux[index_zero]
    wave_remove = wave[index_zero]
    SNR_array = flux_remove/error_reomve
    index_region = (SNR_array >= SNR_threshold)
    wave_good = wave_remove[index_region]
    print('line_model_line321, index_zero\n', index_zero)
    print('line_model_line322, error_remove\n', error_reomve)
    print('line_model_line323, flux_remove\n', flux_remove)
    print('line_model_line324, wave_remove\n', wave_remove)
    print('line_model_line325, SNR_array\n', SNR_array)
    print('line_model_line326, index_region\n', index_region)
    print('line_model_line327, wave_good\n', wave_good)

    if wave_good.size > 20:
        #zabs_SNR_lower = wave_good.min()/line_list[0]-1  #MgII
        #zabs_SNR_upper = wave_good.max()/line_list[1]-1  #MgII
        zabs_SNR_lower = wave_good.min()/line_list[0]-1
        zabs_SNR_upper = wave_good.max()/line_list[1]-1
        print('line_model_line337, zabs_SNR_lower\n', zabs_SNR_lower)
        print('line_model_line338, zabs_SNR_upper\n', zabs_SNR_upper)


        #set the CIV limit and zqso limit
        #also the reduce the sky emission
        #zqso_lower = np.max( [(1+zqso)*1530.0/(line_list[0]-15) -1, 4000.0/(line_list[0]-15) -1]) #MgII
        #zqso_upper = np.min([zqso, 8000.0/(line_list[0]-15) -1])								   #MgII
        zqso_lower = np.max( [(1+zqso)*3000.0/(line_list[0]-15) -1, 4000.0/(line_list[0]-15) -1])
        zqso_upper = np.min([zqso, 9200.0/(line_list[0]-15) -1])
        print('line_model_line345, zqso_lower\n', zqso_lower)
        print('line_model_line346, zqso_upper\n', zqso_upper)


        zabs_lower = np.max(np.array([zabs_SNR_lower, zqso_lower ]))
        zabs_upper = np.min(np.array([zabs_SNR_upper, zqso_upper ]))
        print('line_model_line351, zabs_lower\n', zabs_lower)
        print('line_model_line352, zabs_upper\n', zabs_upper)
        zabs = uniform(zabs_lower, zabs_upper)

        print('this is the z info:', zqso, zabs)
        print('line_model_line357, zqso\n', zqso)
        print('line_model_line357, zabs\n', zabs)


        #calcualate the width of absorption profile
        line_wave=line_list*(1+zabs)
        #calculate the resolution of lines.
        #resolution=2000 #MgII
        resolution=4000
        min_reso = np.mean(line_wave/(resolution*2.355))
        SNR_array = np.zeros((1, line_list.shape[0]))
        print('line_model_line363, line_wave\n', line_wave)
        print('line_model_line366, min_reso\n', min_reso)
        print('line_model_line367, SNR_array\n', SNR_array)



        line_model=np.zeros((wave.size, line_list.size))
        #cacluate the coressponding error for lines
        for i in np.arange(line_list.size):
            index_error = (wave >= line_wave[i]-3*min_reso) & (wave <= line_wave[i]+3*min_reso)
            SNR_array[0, i] = np.mean(flux[index_error]/error[index_error])


        #print('this is the minimum SNR: %s ' % SNR_array.min() )

        EW_array = np.zeros((1, line_list.shape[0]))
        print('line_model_line383, EW_array\n', EW_array)
        # generate the FWHM for the line

        FWHM_nal = (1+zabs)*np.min([4.5, np.max([1.0, gauss(1.7, 0.7)])]) #MgII
        #FWHM_nal = (1+zabs)*np.min([3.5, np.max([1.0, gauss(1.2, 0.5)])]) #CaII
        print('line_model_line387, FWHW_nal\n', FWHM_nal)
        #print('this is the FWHM for both lines: %s ' % FWHM_nal)
        SL_array = np.zeros((1, line_list.shape[0]))
        print('line_model_line390, SL_array\n', SL_array)

        #make sure the EW > 0.5. since min FWHM =1.0
        #min SL = SNR*0.5
        low_lim_line1 = np.max([SNR_threshold, 0.5*SNR_array[0, 0]])
        SL_array[0, 0] = uniform(low_lim_line1, SNR_array[0, 0])
        print('line_model_line395, low_lim_line1\n', low_lim_line1)
        print('line_model_line396, SL_array\n', SL_array)

        #make sure the ratio is 1~2 and 2803 line > 0.5
        low_lim = np.max([SNR_threshold, 0.5*SL_array[0, 0],  0.5*SNR_array[0, 1]])
        up_lim = np.min([SL_array[0, 0], SNR_array[0, 1]])
        SL_array[0, 1] = uniform( low_lim, up_lim )
        print('line_model_line401, low_lim\n', low_lim)
        print('line_model_line402, up_lim\n', up_lim)
        print('line_model_line403, SL_array\n', SL_array)

        # generate the EW for the line
        for i in np.arange(line_list.size):
            EW_array[0, i] = SL_array[0, i]*FWHM_nal / ( (1+zabs)*SNR_array[0, i])
            line_model[:,i]=gaussian_profile(wave, line_wave[i], FWHM_nal/2.355)*EW_array[0, i]*(1+zabs)



        abs_model=np.sum(line_model, axis=1)+1
        flux_abs=flux*abs_model
        print('line_model_line415, abs_model\n', abs_model)
        print('line_model_line416, flux_abs\n', flux_abs)

        line_info = np.concatenate((EW_array, SL_array), axis=1)
        FWHM_final =  FWHM_nal/(1+zabs)
        print('line_model_line420, line_info\n', line_info)
        print('line_model_line421, FWHM_final\n', FWHM_final)

    else:
        print('cannot insert the absorption lines due to low SNR')
        flux_abs = flux
        line_info = np.zeros((1, 2))
        SNR_array = np.zeros((1, 4))
        zabs = 0.0
        FWHM_final = 0.0
    return flux_abs, line_info, SNR_array, zabs, FWHM_final
