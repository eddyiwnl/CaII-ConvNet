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
#hdu=fits.open('predictionalex_1.hdf5')
filename = './predictionalex_3.hdf5'
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
index_pred=np.argwhere(pred_label > 0.0)[:,0]
print('line47_index_true=\n',index_true)
print('line47_index_true_shape=\n',index_true.shape)
print('line49_index_pred=\n',index_pred)
print('line49_index_pred_shape=\n',index_pred.shape)


mjd_sub=mjd[index_pred]
plate_sub=plate[index_pred]
fiber_sub=fiber[index_pred]
zqso_sub=zqso[index_pred]
#mjd_sub=mjd[index_pred]
#plate_sub=plate[index_pred]
#fiber_sub=fiber[index_pred]
#zqso_sub=zqso[index_pred]
print('line55_mjd_sub=\n',mjd_sub)
print('line56_plate_sub=\n',plate_sub)
print('line57_fiber_sub=\n',fiber_sub)
print('line58_zqso_sub=\n',zqso_sub)
print('line55_mjd_sub.shape=\n',mjd_sub.shape)
print('line56_plate_sub.shape=\n',plate_sub.shape)
print('line57_fiber_sub.shape=\n',fiber_sub.shape)
print('line58_zqso_sub.shape=\n',zqso_sub.shape)

index_reject = np.argwhere((zqso_sub > 1.4))
print('line68_index_list=',index_reject)
mjd_sub=np.delete(mjd_sub, index_reject, axis=0)
plate_sub=np.delete(plate_sub, index_reject, axis=0)
fiber_sub=np.delete(fiber_sub, index_reject, axis=0)
zqso_sub=np.delete(zqso_sub, index_reject, axis=0)
print('line70_mjd_sub=\n',mjd_sub)
print('line71_plate_sub=\n',plate_sub)
print('line72_fiber_sub=\n',fiber_sub)
print('line73_zqso_sub=\n',zqso_sub)
print('line70_mjd_sub.shape=\n',mjd_sub.shape)
print('line71_plate_sub.shape=\n',plate_sub.shape)
print('line72_fiber_sub.shape=\n',fiber_sub.shape)
print('line73_zqso_sub.shape=\n',zqso_sub.shape)

spno=zqso_sub.shape[0]
print('line64_spno =\n', spno)


#output file here:
writeFile = open("some_spectrum.txt", 'w')
file_lis = []
#make z bin here:
#0.2 
z_stat = np.arange(24)*0.2
z_end = np.arange(25)*0.2
z_end = z_end[1:]

wave_window = [3800.00, 9200.00]

for bink in np.arange(24):
#for bink in np.arange(1, 12):
    print('line49, the z range is %s to %s' % (z_stat[bink], z_end[bink]))
    print('this is bin %s' % bink)
    #make the wave grid here:
    wave_window = [3800.00, 9200.00]
    wave_pca=[wave_window[0]/(1+z_end[bink]),wave_window[1]/(1+z_stat[bink])]
    print('line54, wave_pca', wave_pca)
	
    index=(zqso_sub >= z_stat[bink]) & (zqso_sub < z_end[bink])
    print('line57, index = ', index.shape)
    print('line57, index = ', index)
    zqso_sub_hit=zqso_sub[index]
    print('there are %s spectra in this subsample' % zqso_sub_hit.size)
    print('line59, zqso_sub = ', zqso_sub_hit)
    print('line66, zqso_sub = ', zqso_sub_hit.shape)

    mjd_sub_hit=mjd_sub[index]
    plate_sub_hit=plate_sub[index]
    fiber_sub_hit=fiber_sub[index]
    #zqso_sub_hit=zqso_sub[index]
    #info_matrix=np.concatenate((ra[index].reshape(-1,1), dec[index].reshape(-1,1), mjd[index].reshape(-1,1), \
     #               plate[index].reshape(-1,1), fiber[index].reshape(-1,1), zqso[index].reshape(-1,1)), axis = 1)
    #print('line69,info_matrix', info_matrix)
    #print(info_matrix.shape)

    dlogw=1.E-4
    no_grids=np.log10(wave_pca[1]/wave_pca[0])/dlogw
    print('line75, no_grids = ', no_grids)
    wave_rest=np.arange(no_grids)*dlogw+np.log10(wave_pca[0])
    print('line77, np.arange = ', np.arange(no_grids))
    print('line77, wave_rest = ', wave_rest)
    wave_rest=10**wave_rest
    print('line79, wave_rest----', wave_rest)
    print(wave_rest)

    spno=zqso_sub_hit.shape[0]
    print('line84, spno ======', spno)
    sp_matrix=np.zeros((spno, wave_rest.size))
    print('line86, sp_matrix ======', sp_matrix)
    error_matrix=np.zeros((spno, wave_rest.size))
    print('line88, error_matrix ======', error_matrix)
    norm_factor=np.zeros((spno,1))
    print('line90, norm_factor size ======', norm_factor.size)
    #spno is number
    print('spno', spno)
    #for i in np.arange(200):
    for i in np.arange(spno):
        print("this is %s" % i)
        sp_data=spSpec_sdss_dr12(int(mjd_sub_hit[i]), int(plate_sub_hit[i]), int(fiber_sub_hit[i]), ('wave','flux', 'noise'))
        #sp_data=spSpec_sdss(int(mjd_sub_hit[i]), int(plate_sub_hit[i]), int(fiber_sub_hit[i]), ('wave','flux', 'noise'))
        fileName = getFileName(int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i]))
        print (fileName)
        #print('shape', sp_data)  has wave/flux/noise
		#writeFile.write('spSpec-%05d-%04d-%03d.fit', % (int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i])))
		#writeFile.write('spSpec-%05d-%04d-%03d.fit', % (int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i])))
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
        #writeFile.write('spSpec-%05d-%04d-%03d.fit', % (int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i])))
        #fileName = ('spSpec-%05d-%04d-%03d.fit', % (int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i])))
        file_lis.append(fileName)
        print('***', wave.size, sp_data['flux'].size*0.8)
        if  wave.size >= sp_data['flux'].size*0.8:

            #flux, norm_value = spec_norm_v1(wave, flux, error, zqso_sub[i])
            #error = error / norm_value
            #print('line170, flux = ', flux)
            #print('line170, norm_value', norm_value)
            #print('line171, error = ', error)

#remove the blue part of the Lyman_alpha and red part of the MgII
            #wave_caii = wave/(1+zqso_sub[i])
            #print('line117, wave = ', wave)

            #new_flux = spec_rebin(wave_caii, flux, wave_rest)
            #print('line120, new_flux = ', new_flux)
            #new_error = spec_rebin(wave_caii, error, wave_rest)
            #print('line122, new_error = ', new_error)

            #ind_ly_mg = (wave_rest <= 1215.0)

            #new_flux[ind_ly_mg] = 0.0
            #print('line127, new_flux = ', new_flux)
            #new_error[ind_ly_mg] = 0.0
            #print('line129, new_error = ', new_error)

            #norm_factor[i,0] = norm_value
            #print('line132, norm_factor size = ', norm_factor.size)
			
            #print('line77 wave_rest', wave_rest)
            #print('line125 ind_ly_mg', ind_ly_mg)

            #if plot=='yes':
            if (0.2 <= zqso_sub[i]) & (zqso_sub[i] < 0.4) :

                #'''
                #plt.figure()
                #plt.plot(wave_rest, new_flux)
                #plt.plot(wave_rest, new_error)
                #plt.xlim(np.min(wave_rest), np.max(wave_rest))
                #plt.show()
                plt.figure()
                plt.plot(wave, flux)
                plt.xlabel('Observed Wavelength [A]')
                plt.ylabel('Normalized Flux')
                plt.title(zqso_sub[i])
				#plt.annotate(Double absorber line' xy=(7000, 1.0), arrowprops=dict(facecolor='black',shrink=0.05) )
                #plt.plot(wave, error)
                plt.xlim(np.min(wave), np.max(wave))
                plt.show()
                #'''
            #sp_matrix[i,:]=new_flux
            #error_matrix[i,:]=new_error
            #sp_matrix[i,:]=flux
            #error_matrix[i,:]=error
            #print('*************', sp_matrix.shape)

    #info_matrix = np.concatenate((info_matrix,norm_factor), axis=1)
    #print info_matrix.shape
    #matrix_name='./Qgen_bins/Qgen_zbin_dr12_prediction_%s.hdf5' % bink
    #with h5py.File(matrix_name,'w') as hf:
        #hf.create_dataset("wave",  data=wave)
        #hf.create_dataset("error",  data=error)
        #hf.create_dataset("pca_info",  data=info_matrix)
       # hf.create_dataset("norm_value", data=norm_factor)


np.savetxt('some_spectrum.lis', np.asarray(file_lis).reshape(-1,1), fmt = '%s')

#check the number

