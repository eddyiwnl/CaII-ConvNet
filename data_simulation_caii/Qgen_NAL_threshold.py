#this script will randomly insert the absorption lines into the spectra in subsamples.
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from loadspec import *
import time
import sys
import glob
import pickle
import h5py
from sklearn.decomposition import IncrementalPCA
from lowess import lowess
from scipy import signal
from absorber_utils import *
from scipy.optimize import curve_fit


z_stat = np.arange(24)*0.2
z_end = np.arange(25)*0.2
z_end = z_end[1:]


wave_window = [3800.00, 9200.0]
dlogw=1.E-4
no_grids=np.log10(wave_window[1]/wave_window[0])/dlogw
print('line25, no_grids', no_grids)
wave=np.arange(no_grids)*dlogw+np.log10(wave_window[0])
print('line28, wave', wave)
wave=10**wave
print('line30, wave', wave)

for bink in np.arange(24):
    #bink = bink + 10
    print('line33, bink',bink)
    matrix_name='./Qgen_fit/Qgen_fit_zbin%s.hdf5' % bink
    with h5py.File(matrix_name,'r') as hf:
        spectra = hf["spectra"][:]
        error = hf["error"][:]
        fit = hf["fit_final"][:]
        info = hf["sp_info"][:]
        norm_value = hf["norm_value"][:]
    print('line38, spectra',spectra)
    print('line39, error',error)
    print('line40, fit',fit)
    print('line41, info',info)
    print('line42, norm_value',norm_value)
    snr_name = './Qgen_SNR/Qgen_snr_zbin%s.hdf5' % bink
    with h5py.File(snr_name,'r') as hf:
        snr_spectra = hf["SNR_info"][:]
    


    #perform redshift exam, only use spectra with ci*(1+z) < 9200 & mg*(1+z) > 3800, need to change this part:
    #MGII joann index_zreject = (2900.0 * (info[:,5]+1) < 4500.0) | ( 1530.0 * (info[:,5]+1) > 8500.0)
    print('line50, snr_spectra',snr_spectra)
    #index_zreject = (2900.0 * (info[:,5]+1) < 4500.0) | ( 1530.0 * (info[:,5]+1) > 9200.0) #acc is 50%
    #index_zreject = (3200.0 * (info[:,5]+1) < 3800.0) | ( 1530.0 * (info[:,5]+1) > 9200.0) #acc is 62%
    index_zreject = (4100.0 * (info[:,5]+1) < 4500.0) | ( 2700.0 * (info[:,5]+1) > 9200.0)
    print('line57, index_zreject',index_zreject)

    index_zkeep = np.logical_not(index_zreject)
    print('line60, index_zkeep',index_zkeep)

    #print(spectra)
    #print(index_zkeep)
    spectra = spectra[index_zkeep, :]
    error = error[index_zkeep, :]
    fit = fit[index_zkeep, :]
    info = info[index_zkeep, :]
    norm_value = norm_value[index_zkeep, :]
    snr_spectra = snr_spectra[index_zkeep, :]
    print('line65, spectra',spectra)
    print('line66, error',error)
    print('line67, fit',fit)
    print('line68, info',info)
    print('line69, norm_value',norm_value)
    


    index_list=np.unique(np.argwhere(np.isnan(snr_spectra))[:,0])
    print('line79, index_list',index_list)

    spectra = np.delete(spectra, index_list, axis=0)
    error =np.delete( error, index_list, axis=0)
    fit = np.delete(fit, index_list, axis=0)
    info = np.delete(info, index_list, axis=0)
    norm_value = np.delete(norm_value, index_list, axis=0)
    snr_spectra = np.delete(snr_spectra, index_list, axis=0) #remove nan element
    print('line82, spectra',spectra)
    print('line83, error',error)
    print('line84, fit',fit)
    print('line85, info',info)
    print('line86, norm_value',norm_value)
    print('line87, snr_spectra',snr_spectra)

    index_snr_keep = (snr_spectra[:,0] >= 5.0)
    print('line95, index_snr_keep',index_snr_keep)

    spectra = spectra[index_snr_keep, :]
    error = error[index_snr_keep, :]
    fit = fit[index_snr_keep, :]
    info = info[index_snr_keep, :]
    norm_value = norm_value[index_snr_keep, :]
    snr_spectra = snr_spectra[index_snr_keep, :]         #keep the snr > 5.0 element
    print('line98, spectra',spectra)
    print('line99, error',error)
    print('line100, fit',fit)
    print('line101, info',info)
    print('line102, norm_value',norm_value)
    print('line103, snr_spectra',snr_spectra)

    spno = spectra.shape[0]
    print('range is %s to %s' % (z_stat[bink], z_end[bink]) )
    print('line 111, total number is %s' % spno)
    if spno >= 10:
        #random label
        true_num = int(round(spno*0.5))
        print ('positive is %s' % true_num)
        label_matrix=np.zeros((spno,1))
        index_ture=np.random.randint(spno,size= true_num)
        label_matrix[index_ture] = 1.0

        zqso = info[:,5]

        spectra_matrix = np.zeros((spno, wave.size))
        abs_info = np.zeros((spno, 8))
        bad_spectra = np.zeros((spno, 1))
        measurement_info = np.zeros((spno, 9))
        print('line116, true_num',true_num)
        print('line118, label_matrix',label_matrix)
        print('line119, index_ture',index_ture)
        print('line120, label_matrix',label_matrix)
        print('line122, zqso',zqso)
        print('line124, spectra_matrix',spectra_matrix)
        print('line125, abs_info',abs_info)
        print('line126, bad_spectra',bad_spectra)
        print('line127, measurement_info',measurement_info)

        for i in np.arange(spno):
           fit_final = fit[i,:]
           error_final = error[i,:]
           #print np.mean(fit_final), np.mean(error_final)
           if (np.mean(fit_final) == 0.0) | (np.mean(error_final) == 0.0):
               bad_spectra[i, 0] = -999.0
               print('this is bad spectra')

           print('line139, fit_final', fit_final)
           print('line140, error_final', error_final)
           print('this is spectra %s, the label is %s' % (i, label_matrix[i]) )
           print('SNR is %s' % snr_spectra[i,0])
           if label_matrix[i] == 1:



               #joann mgII  line_list=np.array([2796.35, 2803.53])
			   #new_fit, line_info, SNR_array, zabs, FWHM_final = \
               #      line_model_injection_new(wave, fit_final, zqso[i], line_list, error_final, 3.0)
               line_list=np.array([3934.77, 3969.59])
               new_fit, line_info, SNR_array, zabs, FWHM_final = line_model_injection_new(wave, fit_final, zqso[i], line_list, error_final, 3.0)
               print('line158, new_fit',new_fit)
               print('line158, line_info',line_info)
               print('line158, SNR_array',SNR_array)
               print('line158, zabs',zabs)
               print('line158, FWHM_final',FWHM_final)

               new_flux = noise_generation(wave, new_fit, error_final)
               print('line165, new_flux',new_flux)
               #print('this is input info:')
               #print(line_info)
               #print('this is SNR:', SNR_array)
               '''
               plt.plot(wave, new_fit)
               plt.plot(wave, new_error)
               plt.show()
               '''


               new_flux_measure = new_flux

               #prevent the overflow
               index_overflow = (fit_final != 0.0)
               fnor = new_flux_measure[index_overflow]/fit_final[index_overflow]
               errornor = error_final[index_overflow]/fit_final[index_overflow]

               wave_rest = wave[index_overflow]/(1+zabs)

               #first, fit the absorption lines and derive the FWHM
               #fit the double curve here:
               region_index = (wave_rest >= line_list.min() - 40.0) & (wave_rest <= line_list.max() + 40.0)
               #region_index = (wave_rest >= line_list.min() - 20.0) & (wave_rest <= line_list.max() + 20.0)

               wave_rest1=wave_rest[region_index]
               fnor1=fnor[region_index]-1
               errornor1 =errornor[region_index]
               print('line180, index_overflow',index_overflow)
               print('line181, fnor',fnor)
               print('line182, errornor',errornor)
               print('line184, wave_rest', wave_rest)
               print('line188, region_index',region_index)
               print('line190, wave_rest1',wave_rest1)
               print('line191, fnor1',fnor1)
               print('line192, errornor1',errornor1)

               popt, err1, err2 = double_fit(wave_rest1, fnor1, zabs, line_list, errornor1) #popt (1, 7)
               print('this is in %s, this is out %s' % (line_info[0, 0], popt[2]) )
               print('this is in %s, this is out %s' % (line_info[0, 1], popt[3]) )
               print(err1, err2)
               print(popt[2]/err1, popt[3]/err2)
               measurement_info[i, 0:7] = popt
               measurement_info[i, 7] = err1
               measurement_info[i , 8] = err2
               print('save data...')



               #index_emission=(wave >= 2900.0*(1+zqso[i])) | (wave <= 1530.0*(1+zqso[i])) MgII
               index_emission=(wave >= 4100.0*(1+zqso[i])) | (wave <= 2700.0*(1+zqso[i]))


               new_flux[index_emission]=0.0
               new_fit[index_emission]=0.0
               error_final[index_emission]=0.0
               print('line214, index_emission',index_emission)
               print('line217, new_flux',new_flux)
               print('line218, new_fit',new_fit)
               print('line219, error_final',error_final)


               '''
               #print line_info
               #print FWHM_final
               #print SNR_array
               plt.plot(wave, new_flux)
               plt.plot(wave, new_fit)
               plt.plot(wave, error_final)
               plt.axvline((zabs+1)*2796.35)
               plt.xlim(wave.min(), wave.max())
               plt.show()
               '''
               #store the information for injected lines (zabs, EW1, EW2, SL1, SL2)
               abs_info[i,0] = zabs
               abs_info[i,1:5] = line_info
               abs_info[i,5:7] = SNR_array
               abs_info[i,7] = FWHM_final
               spectra_matrix[i,:] = new_flux

           if label_matrix[i] == 0:


               new_flux = noise_generation(wave, fit_final, error_final)

               #index_emission=(wave >= 2900.0*(1+zqso[i])) | (wave <= 1530.0*(1+zqso[i]))
               index_emission=(wave >= 4100.0*(1+zqso[i])) | (wave <= 2700.0*(1+zqso[i]))


               new_flux[index_emission]=0.0
               fit_final[index_emission]=0.0
               error_final[index_emission]=0.0


               spectra_matrix[i,:] = new_flux
               print('line249, index_emission',index_emission)
               print('line252, new_flux',new_flux)
               print('line253, fit_final',fit_final)
               print('line256, error_final',error_final)
               print('line257, spectra_matrix', spectra_matrix)

               '''
               plt.plot(wave, new_flux)
               plt.plot(wave, fit_final)
               plt.plot(wave, error_final)
               plt.show()
               '''

        info_matrix = np.concatenate((info, abs_info), axis=1)
        info_matrix = np.concatenate((info_matrix, bad_spectra), axis=1)
        outfile_name = './Qgen_training/training_gen%s.hdf5' % bink
        with h5py.File(outfile_name,'w') as hf:
            hf.create_dataset("spectra",  data=spectra_matrix)
            hf.create_dataset("info",  data=info_matrix)
            hf.create_dataset("label",  data=label_matrix)
            hf.create_dataset("abs_info",  data=measurement_info)

