#this script will combine the subsamples together
import h5py
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import os


#filelist=os.listdir('./Qgen_training/training_gen*.hdf5')
filelist=os.listdir('./Qgen_training/')
filenum=len(filelist)

	
for bink in np.arange(filenum):
    outfile_name = './Qgen_training/'+filelist[bink]
    print(filelist[bink])
    with h5py.File(outfile_name,'r') as hf:
        spec_matrix = hf["spectra"][:]
        info_matrix = hf["info"][:]
        label = hf["label"][:]
        measurement = hf["abs_info"][:]

    bad_spectra = info_matrix[:,-1]
    #bad_snr = info_matrix[:,-2]
    index_list = np.argwhere((bad_spectra < 0.0))
    print('line22, spec_matrix',spec_matrix)
    print('line24, info_matrix',info_matrix)
    print('line25, label',label)
    print('line27, measurement',measurement)
    #print index_bad
    #index_list=np.unique(np.argwhere(np.isnan(spec_matrix))[:,0])
#    print index_list
#    exit()
    spec_matrix=np.delete(spec_matrix, index_list, axis=0)
    info_matrix=np.delete(info_matrix, index_list, axis=0)
    label=np.delete(label, index_list, axis=0)
    measurement = np.delete(measurement, index_list, axis=0)
    print('line36, spec_matrix',spec_matrix)
    print('line37, info_matrix',info_matrix)
    print('line38, label',label)
    print('line39, measurement',measurement)

    print('spec_matrix.shape', spec_matrix.shape)
    print('info_matrix.shape', info_matrix.shape)
    print('label.shape', label.shape)
    if spec_matrix.shape[0] > 0:
        #concatenate them together
        if bink == 0:
            spec = spec_matrix
            info = info_matrix
            label_matrix = label
            measurement_matrix = measurement
           #'''
            print('this is the all: ', label.shape[0])
            index = np.argwhere(label > 0.0)
            cc = label[index]
            print('this is the positive: ', cc.shape[0])
           #'''

        if bink > 0:
            # print spec.shape, info.shape
            #print label_matrix.shape
            spec = np.concatenate((spec, spec_matrix), axis=0)
            info = np.concatenate((info, info_matrix), axis=0)
            label_matrix = np.concatenate((label_matrix, label), axis=0)
            measurement_matrix = np.concatenate((measurement_matrix, measurement), axis=0)
            print('line62, spec',spec)
            print('line63, info',info)
            print('line64, label_matrix',label_matrix)
            print('line65, measurement_matrix',measurement_matrix)

            #'''
            print ('this is the all: ', label.shape[0])
            index = np.argwhere(label > 0.0)
            cc = label[index]
            print ('this is the positive: ', cc.shape[0])
            #'''


with h5py.File('training_artificial.hdf5','w') as hf:
    hf.create_dataset("training_sp",  data=spec)
    hf.create_dataset("training_label",  data=label_matrix)
    hf.create_dataset("training_info",  data=info)
    hf.create_dataset("training_measure",  data=measurement_matrix)
