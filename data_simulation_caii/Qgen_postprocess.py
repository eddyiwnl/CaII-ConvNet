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
#hdu=fits.open('predictionalex_0.hdf5')
filename = './predictionalex_0.hdf5'
with h5py.File(filename, 'r') as hf:
     info = hf["info"][:]
     true_label = hf["true_label"][:]
     pred_label = hf["pred_label"][:]
     print('info\n', info)
     print('true_label', true_label)
     print('pred_label', pred_label)

#import the info here:
ra=info[:,0]
dec=info[:,1]
mjd=info[:,2]
plate=info[:,3]
fiber=info[:,4]
zqso=info[:,5]
print('line34_ra=\n',ra)
print('line35_dec=\n',dec)
print('line36_mjd=\n',mjd)
print('line37_plate=\n',plate)
print('line38_fiber=\n',fiber)

#retrive the index of the labeled data:
index_true=np.argwhere(true_label > 0.0)[:,0]
index_false=np.argwhere(true_label == 0.0)[:,0]
index_pred=np.argwhere(pred_label > 0.0)[:,0]
print('line47_index_true=\n',index_true)
print('line47_index_true_shape=\n',index_true.shape)
print('line49_index_pred=\n',index_pred)
print('line49_index_pred_shape=\n',index_pred.shape)

mjd_sub=mjd[index_pred]
plate_sub=plate[index_pred]
fiber_sub=fiber[index_pred]
zqso_sub=zqso[index_pred]
print('line55_mjd_sub=\n',mjd_sub)
print('line56_plate_sub=\n',plate_sub)
print('line57_fiber_sub=\n',fiber_sub)
print('line58_zqso_sub=\n',zqso_sub)

spno=zqso_sub.shape[0]
print('line64_spno =\n', spno)
#for i in np.arange(spno):
for i in np.arange(10):
    print("this is %s" % i)
    sp_data=spSpec_sdss_dr12(int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i]), ('wave','flux', 'noise'))
    #print('shape', sp_data)  has wave/flux/noise
    mask_index=(sp_data['flux'] >0.0) & (sp_data['noise'] > 0.0)
    print('mask_index=====', mask_index.shape)
    print('mask_index=====', mask_index)
 
    wave=sp_data['wave'][mask_index]
    print('line71, wave =', wave)
    sp_matrix=sp_data['flux'][mask_index]
    print('line73, sp_matrix =', sp_matrix)
    error_matrix=sp_data['noise'][mask_index]
    print('line75, error_matrix =', error_matrix)
    print('wave', wave.size)
    flux=sp_data['flux'][mask_index]
    print('line103, flux =', flux)
    error=sp_data['noise'][mask_index]
    print('line105, error =', error)
    #info_matrix=info[mask_index]

    print('***', wave.size, sp_data['flux'].size*0.8)
    if  wave.size >= sp_data['flux'].size*0.8:
        flux, norm_value = spec_norm_v1(wave, flux, error, zqso_sub[i])
        error = error / norm_value
        print('line113, error = ', error)

        #norm_factor[i,0] = norm_value
        #print('line132, norm_factor = ', norm_factor)

        if plot=='yes':

            #'''
            plt.figure(zqso_sub[i])
            plt.plot(wave, flux)
            #plt.plot(wave, error)
            plt.xlim(np.min(wave), np.max(wave))
            plt.show()
            #'''
				

matrix_name='./labeled_CaII_for_test'
with h5py.File(matrix_name,'w') as hf:
     hf.create_dataset("post_wave",  data=wave)
     hf.create_dataset("post_sp",  data=sp_matrix)
     hf.create_dataset("post_error",  data=error_matrix)
     #hf.create_dataset("podt_info",  data=info_matrix)

#check the number

