#this script will the eigenvectors to do the continuum fitting for each subsample
from sklearn.decomposition import IncrementalPCA
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import linalg as LA
from loadspec import *
import sys


#get the parameter from the command line (python *.py plot)
args = sys.argv[1:]
plot=args[0]


z_stat = np.arange(24)*0.2
z_end = np.arange(25)*0.2
z_end = z_end[1:]

for kkk in np.arange(24):
    print('this is bin %s' % kkk)
    filename = './Qgen_bins/Qgen_zbin%s.hdf5' % kkk
    with h5py.File(filename, 'r') as hf:
         spectra = hf["pca_sp"][:]
         error = hf["pca_error"][:]
         info = hf["pca_info"][:]
         norm_factor = hf["norm_value"][:]
    		 


    #remove nan value in the spectra before perfoming PCA
    index_list=np.unique(np.argwhere(np.isnan(spectra))[:,0])
    spectra=np.delete(spectra, index_list, axis=0)
    info=np.delete(info, index_list, axis=0)
    error=np.delete(error, index_list, axis=0)
    norm_factor=np.delete(norm_factor, index_list, axis=0)
    print('line34, index_list', index_list)
    print('line35, spectra', spectra)
    print('line36, info', info)
    print('line37, error', error)
    print('line38, norm_factor', norm_factor)


    #import the info here:
    ra=info[:,0]
    dec=info[:,1]
    mjd=info[:,2]
    plate=info[:,3]
    fiber=info[:,4]
    zqso=info[:,5]
    print('line52, zqso', zqso)
	
    spno = zqso.size
    print('line55,there are %s spectra in this sample' % spno)

    mean_spectra=np.mean(spectra,axis=0)
    spectra=spectra-mean_spectra
    print('line58, mean_spectra', mean_spectra)
    print('line59, spectra', spectra)

    #load the eigenvectors
    eigenvector_name='./Qgen_eigen/eigenvector_%s.hdf5' %kkk
    with h5py.File(eigenvector_name,'r') as hf:
         ipca_comp = hf["eigenvector"][:]

    coeff=np.dot(spectra,ipca_comp)
    spectra_recon=np.dot(coeff,ipca_comp.T)
    print('line66, ipca_comp.T', ipca_comp.T)
    print('line66, ipca_comp', ipca_comp)
    print('line68, coeff', coeff)
    print('line69, spectra_recon', spectra_recon)

    spectra_old=spectra+mean_spectra
    spectra_fit=spectra_recon+mean_spectra
    print('line73, spectra_old', spectra_old)
    print('line74, spectra_fit', spectra_fit)

    #plt.plot(spectra_old[1,:])
    #plt.show()
    #print(mean_spectra.shape)
    #exit()

    zmin=z_stat[kkk]
    zmax=z_end[kkk]
    print('line83, zmin', zmin)
    print('line84, zmax', zmax)

    wave_window = [3800.00, 9200.0]
    wave_pca=[wave_window[0]/(1+zmax),wave_window[1]/(1+zmin)]
    print('line88, wave_pca', wave_pca)
    dlogw=1.E-4
    no_grids=np.log10(wave_pca[1]/wave_pca[0])/dlogw
    print('line92, no_grids', no_grids)
    wave_rest=np.arange(no_grids)*dlogw+np.log10(wave_pca[0])
    print('line94, wave_rest', wave_rest)
    wave_rest=10**wave_rest
    print('line96, wave_rest', wave_rest)

    wave_window = [3800.00, 9200.0]
    dlogw=1.E-4
    no_grids=np.log10(wave_window[1]/wave_window[0])/dlogw
    print('line101, no_grids', no_grids)
    wave=np.arange(no_grids)*dlogw+np.log10(wave_window[0])
    print('line103, wave', wave)
    wave=10**wave
    print('line105, wave', wave)


    spectra_data = np.zeros((spno, wave.size))
    print('line109, spectra_data', spectra_data)
    error_data = np.zeros((spno, wave.size))
    print('line111, error_data', error_data)
    fit_data = np.zeros((spno, wave.size))
    print('line112, fit_data', fit_data)


    for i in np.arange(spno):
    #for i in np.arange(1, 100):
        #print(spectra_fit[i,:])
        print('line117, i', i)
        ddd=signal.medfilt(spectra_old[i,:]/spectra_fit[i,:],kernel_size=91)
        print('line121, ddd', ddd)
        new_fit = spectra_fit[i,:]*ddd
        print('line123, new_fit', new_fit)
        spectra_sdss = spec_rebin(wave_rest*(1+zqso[i]), spectra_old[i,:], wave)
        print('line125, spectra_sdss', spectra_sdss)
        fit_final = spec_rebin(wave_rest*(1+zqso[i]), new_fit, wave)
        print('line127, fit_final', fit_final)
        error_final = spec_rebin(wave_rest*(1+zqso[i]), error[i,:], wave)
        print('line129, error_final', error_final)

        if plot=='yes':

            #'''
            plt.plot(wave, spectra_sdss,color='black')
            plt.plot(wave, fit_final,color='blue')
            plt.ylim(0,5)
            plt.plot(wave, error_final)
            plt.show()
            #'''

      #save the data here:

        spectra_data[i,:] = spectra_sdss
        print('line144, spectra_data', spectra_data)
        fit_data[i,:] = fit_final
        print('line146, fit_data', fit_data)
        error_data[i,:] = error_final
        print('line148, error_data', error_data)


    matrix_name='./Qgen_fit/Qgen_fit_zbin%s.hdf5' % kkk
    with h5py.File(matrix_name,'w') as hf:
        hf.create_dataset("spectra",  data=spectra_data)
        hf.create_dataset("error",  data=error_data)
        hf.create_dataset("fit_final",  data=fit_data)
        hf.create_dataset("sp_info",  data=info)
        hf.create_dataset("norm_value", data=norm_factor)

