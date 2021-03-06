#this script will divide the spectra in DR7 Schneider catalog into different
#subsamples by their emission redshifts (bin size = 0.2)
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
from scipy import signal

	
#get the parameter from the command line (python *.py plot)
args = sys.argv[1:]
plot=args[0]


#import the dr7 catalog here.
hdu=fits.open('QSOs_DR7_by_Schneider.fits')
ra = hdu[1].data['RA']
dec = hdu[1].data['DEC']
mjd = hdu[1].data['MJD']
plate = hdu[1].data['PLATE']
fiber = hdu[1].data['FIBERID']
zqso = hdu[1].data['Z']   #(105783
print('ra\n', ra)
print('dec', dec)
print('mjd====', mjd)
print('plate', plate)
print('fiber', fiber)
print('zqso', zqso)

print('there are %s spectra in this subsample' % zqso.size)


#make z bin here:
#0.2 
z_stat = np.arange(24)*0.2
z_end = np.arange(25)*0.2
z_end = z_end[1:]

wave_window = [3800.00, 9200.00]

for bink in np.arange(24):
#for bink in np.arange(1, 4):
    print('line49, the z range is %s to %s' % (z_stat[bink], z_end[bink]))
    print('this is bin %s' % bink)
    #make the wave grid here:
    wave_window = [3800.00, 9200.00]
    wave_pca=[wave_window[0]/(1+z_end[bink]),wave_window[1]/(1+z_stat[bink])]
    print('line54, wave_pca', wave_pca)
	
    index=(zqso >= z_stat[bink]) & (zqso < z_end[bink])
    print('line57, index = ', index.shape)
    print('line57, index = ', index)
    zqso_sub=zqso[index]
    print('there are %s spectra in this subsample' % zqso_sub.size)
    print('line59, zqso_sub = ', zqso_sub)
    print('line66, zqso_sub = ', zqso_sub.shape)

    mjd_sub=mjd[index]
    plate_sub=plate[index]
    fiber_sub=fiber[index]
    zqso_sub=zqso[index]
    info_matrix=np.concatenate((ra[index].reshape(-1,1), dec[index].reshape(-1,1), mjd[index].reshape(-1,1), \
                    plate[index].reshape(-1,1), fiber[index].reshape(-1,1), zqso[index].reshape(-1,1)), axis = 1)
    #print('line69,info_matrix', info_matrix)
    print(info_matrix.shape)

    dlogw=1.E-4
    no_grids=np.log10(wave_pca[1]/wave_pca[0])/dlogw
    print('line75, no_grids = ', no_grids)
    wave_rest=np.arange(no_grids)*dlogw+np.log10(wave_pca[0])
    print('line77, np.arange = ', np.arange(no_grids))
    print('line77, wave_rest = ', wave_rest)
    wave_rest=10**wave_rest
    print('line79, wave_rest----', wave_rest)
    print(wave_rest)

    spno=zqso_sub.shape[0]
    print('line84, spno ======', spno)
    sp_matrix=np.zeros((spno, wave_rest.size))
    print('line86, sp_matrix ======', sp_matrix)
    error_matrix=np.zeros((spno, wave_rest.size))
    print('line88, error_matrix ======', error_matrix)
    norm_factor=np.zeros((spno,1))
    print('line90, norm_factor ======', norm_factor)
    #spno is number
    print('spno', spno)
    #for i in np.arange(1, 100):
    for i in np.arange(spno):
        print("this is %s" % i)
        sp_data=spSpec_sdss(int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i]), ('wave','flux', 'noise'))
        #print('shape', sp_data)  has wave/flux/noise
        mask_index=(sp_data['flux'] >0.0) & (sp_data['noise'] > 0.0)
        print('mask_index=====', mask_index.shape)
        print('mask_index=====', mask_index)
 

        wave=sp_data['wave'][mask_index]
        print('line101, wave =', wave)
        flux=sp_data['flux'][mask_index]
        print('line103, flux =', flux)
        error=sp_data['noise'][mask_index]
        print('line105, error =', error)
        print('wave', wave.size)

        print('***', wave.size, sp_data['flux'].size*0.8)
        if  wave.size >= sp_data['flux'].size*0.8:
            flux, norm_value = spec_norm_v1(wave, flux, error, zqso_sub[i])
            error = error / norm_value
            print('line113, error = ', error)

#remove the blue part of the Lyman_alpha and red part of the MgII
            wave = wave/(1+zqso_sub[i])
            print('line117, wave = ', wave)

            new_flux = spec_rebin(wave, flux, wave_rest)
            print('line120, new_flux = ', new_flux)
            new_error = spec_rebin(wave, error, wave_rest)
            print('line122, new_error = ', new_error)

            ind_ly_mg = (wave_rest <= 1215.0)

            new_flux[ind_ly_mg] = 0.0
            print('line127, new_flux = ', new_flux)
            new_error[ind_ly_mg] = 0.0
            print('line129, new_error = ', new_error)

            norm_factor[i,0] = norm_value
            print('line132, norm_factor = ', norm_factor)
			
            print('line77 wave_rest', wave_rest)
            print('line125 ind_ly_mg', ind_ly_mg)

            if plot=='yes':

                #'''
                plt.figure()
                plt.plot(wave_rest, new_flux)
                plt.plot(wave_rest, new_error)
                plt.xlim(np.min(wave_rest), np.max(wave_rest))
                plt.show()
                #'''

            sp_matrix[i,:]=new_flux
            error_matrix[i,:]=new_error
			
            print('*************', sp_matrix.shape)

    info_matrix = np.concatenate((info_matrix,norm_factor), axis=1)
    #print info_matrix.shape
    matrix_name='./Qgen_bins/Qgen_zbin%s.hdf5' % bink
    with h5py.File(matrix_name,'w') as hf:
        hf.create_dataset("pca_sp",  data=sp_matrix)
        hf.create_dataset("pca_error",  data=error_matrix)
        hf.create_dataset("pca_info",  data=info_matrix)
        hf.create_dataset("norm_value", data=norm_factor)




#check the number

