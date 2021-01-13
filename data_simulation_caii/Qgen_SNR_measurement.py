#this script will measure the spectra snr info
from sklearn.decomposition import IncrementalPCA
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import linalg as LA
#from loadspec import *
from random import uniform, gauss

wave_window = [3800.00, 9200.0]
dlogw=1.E-4
no_grids=np.log10(wave_window[1]/wave_window[0])/dlogw
wave=np.arange(no_grids)*dlogw+np.log10(wave_window[0])
wave=10**wave


for bink in np.arange(24):
    matrix_name='./Qgen_fit/Qgen_fit_zbin%s.hdf5' % bink
    with h5py.File(matrix_name,'r') as hf:
        spectra = hf["spectra"][:]
        error = hf["error"][:]
        fit = hf["fit_final"][:]
        info = hf["sp_info"][:]
        norm_value = hf["norm_value"][:]
    
    print('line22, spectra = ', spectra)
    print('line23, error = ', error)
    print('line24, info = ', info)
    print('line25, fit = ', fit)
	
    spno = spectra.shape[0]
    print('line29, spno', spno)
    snr_info = np.zeros((spno,1))
    print(spectra.shape[0], spno)
    print('line31, snr_info', snr_info)
    for i in np.arange(spno):
        flux_final = spectra[i,:]
        error_final = error[i,:]
        print('line35, flux_final', flux_final)
        print('line36, error_final', error_final)
        index_snr = (wave > 5000.0) & (wave < 7500.0)
        #CaII index_snr = (wave > 6000.0) & (wave < 9000.0)
        print('ppp: i', i, index_snr)
        print(flux_final[index_snr])
        print(error_final[index_snr])
        #b=CheckIfValid(error_final[index_snr])
        #if b==1:
        snr_info[i,0] = np.mean(flux_final[index_snr]/error_final[index_snr])
            #print('ccc')
        #print np.mean(flux_final[index_snr]/error_final[index_snr])
        print('line46, snr_info', snr_info)
        #plt.plot(wave,flux_final)
        #plt.plot(wave,error_final)
        #plt.show()

    outfile_name = './Qgen_SNR/Qgen_snr_zbin%s.hdf5' % bink
    with h5py.File(outfile_name,'w') as hf:
        hf.create_dataset("SNR_info",  data=snr_info)
