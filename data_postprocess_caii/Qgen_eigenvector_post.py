#this script will use the IPCA decomposition to generate the eigenvectors for each subsample
#subsamples by their emission redshifts (bin size = 0.2)
from sklearn.decomposition import IncrementalPCA
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import linalg as LA
from loadspec import *

for kkk in np.arange(24):
    print('this is bin %s' % kkk)

    filename = './Qgen_bins/Qgen_zbin%s.hdf5' % kkk
    with h5py.File(filename, 'r') as hf:
        spectra = hf["pca_sp"][:]
        error = hf["pca_error"][:]
        info = hf["pca_info"][:]
        norm_factor = hf["norm_value"][:]
    #remove nan value in the spectra before perfoming PCA
    print('line17, spectra = ', spectra)
    print('this is the matrix shape from loading:', spectra.shape)
    if spectra.size != 0:
        print("Continue...")
        print('zqso = ', info[:,5])
        index_list=np.unique(np.argwhere(np.isnan(spectra))[:,0])
        print('line23, index_list', index_list)
        spectra=np.delete(spectra, index_list, axis=0)
        print('this is the matrix shape:', spectra.shape)
        print('line24, spectra = ', spectra)
        print('lne18, error = ', error)
        print('line19, info = ', info)
        #print('line20, norm_factor = ', norm_factor)

        # do the iteration here:
        #subtrac the mean spectra before perfoming PCA
        mean_spectra=np.mean(spectra,axis=0)
        print('line33, mean_spectra = ', mean_spectra)
        spectra=spectra-mean_spectra
        print('line35, spectra = ', spectra)


        # perform the pca here in iterratively way
        for i in np.arange(3):
            n_components = 20
            ipca=IncrementalPCA(n_components=n_components)
            print('line41, i = ', i)
            print('line42, ipca = ', ipca)
            ipca.fit(spectra)
            ipca_comp=ipca.components_
            print('line46, ipca_comp = \n', ipca_comp)
            print('line46 the ipca_comp shape:\n', ipca_comp.shape) 
            ipca_comp=ipca_comp.T
            print('line47, ipca_comp = ', ipca_comp)
            print('line47 the ipca_comp shape:\n', ipca_comp.shape)


            spectra_old=spectra
            print('line52, spectra_old = ', spectra_old)
            #project the spectra in the eigenvector space
            coeff=np.dot(spectra[:,:],ipca_comp)
            print('line55, coeff = \n', coeff)
            print('line55, coeff.shape = \n', coeff.shape)
            # calculate the norm for each spectra then reject the outliers
            norm_sp=LA.norm(coeff,axis=1)
            print('line58, norm_sp = ', norm_sp)
            fmean, fsig=Iterstat(norm_sp, 4, 20)
            print('line60, fmean = ', fmean)
            print('line60, fsig = ', fsig)

            index_remove=np.argwhere((norm_sp >= fmean+4*fsig) | (norm_sp <= fmean-4*fsig))
            print('line64, index_remove = ', index_remove)
            #remove the outliers
            spectra=np.delete(spectra, index_remove, axis=0)
            print('line69, spectra = ', spectra)
            print('line69 the spectra shape:\n', spectra.shape)


            print('remove %s quasar spectra' % (spectra_old.shape[0]-spectra.shape[0]))


        eigenvector_name='./Qgen_eigen/eigenvector_%s.hdf5' %kkk
        with h5py.File(eigenvector_name,'w') as hf:
            hf.create_dataset("eigenvector",  data=ipca_comp)


    #spectra_recon=np.dot(coeff,ipca_comp.T)
    #print(spectra_recon.shape)

