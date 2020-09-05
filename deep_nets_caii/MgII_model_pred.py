import numpy as np
np.random.seed(123)
from keras.models import Model
from keras.layers import Dense, Input, Activation, Dropout, Flatten, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras import initializers
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
import sys
import pickle
import h5py
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import model_from_json
import json
import time
from loadspec import *



print('available models: alex, revised, resnet, inception, VGG')
model_name = input('Select a model from the above list: ')
indice_no = input('Select a model index number:')

#import the  dr12 data here:


final_spec_name='./training_data/dr12_final_data_for_test0.hdf5'
with h5py.File(final_spec_name,'r') as hf:
    X_final=hf["training_sp"][:]
    y_final=hf["training_label"][:]
    info_final=hf["training_info"][:]
    #flux=hf["flux"][:]
    #spectra=hf["spectra"][:]


print('hf\n', hf)
#print(X_final.shape)

y_final = y_final.reshape((-1, 1))
X_final = X_final.reshape((X_final.shape[0], X_final.shape[1], 1))

# load json and create model
json_file = open('./model_parameters/'+model_name+'_model_'+str(indice_no)+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('./model_parameters/'+model_name+'_weights_'+str(indice_no)+'.hdf5')
print("Loaded model from disk")

# evaluate loaded model on test data

optimization = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])

start_time = time.time()
y_pred=loaded_model.predict(X_final)
y_pred=np.around(y_pred)


print ("-----------------------------------", y_pred)
print (y_pred.shape)
print("--- %s seconds ---" % (time.time() - start_time))
print(accuracy_score(y_final, y_pred.round()) )

index=np.abs(y_pred-y_final) >= 0.5
index=index.reshape(index.size)

#index_hit=np.abs(y_pred-y_final) < 0.5
#index_hit=index.reshape(index.size)


#misclassifcation
info_matrix_mis=info_final[index,:]
label_mis=y_final[index,:]
#info_matrix_hit = info_final[index_hit,:]
#label_hit = y_final[index_hit,:]


print(info_final.shape)
print(info_matrix_mis.shape)
print(label_mis.shape)
print('line68 info_final', info_final)
print('line69 info_matrix_mis', info_matrix_mis)
#print('line70 label_hit', label_hit.shape)



#print('y_pred', y_pred.shape)
#print('info_final', info_final.shape)
totalTrue=0
#for kkk in np.arange(51162):
for kkk in np.arange(10000):
    if y_pred[kkk] == 1:
        totalTrue+=1
        #print('kkk=', kkk)
        info_matrix_hit = info_final[kkk,:]
        #print('info_matrix_hit', info_matrix_hit)
        #flux_hit = flux[kkk,:]
        #print('flux_hit=\n', flux_hit)
        #spectra_hit = spectra[kkk,:]
        #print('spectra_hit=\n', spectra_hit)
		
#        #spSpec_sdss(int(info_final[kkk][2]), int(info_final[kkk][3]), int(info_final[kkk][4]), ('wave','flux', 'noise'))
#print(totalTrue)
#sp_data=spSpec_sdss(int(mjd_sub[i]), int(plate_sub[i]), int(fiber_sub[i]), ('wave','flux', 'noise'))


#save the prediction here:
with h5py.File('prediction'+model_name+'_'+str(indice_no)+'.hdf5','w') as hf:
    hf.create_dataset("info",  data=info_final)
    hf.create_dataset("true_label", data=y_final)
    hf.create_dataset("pred_label", data=y_pred)

#save misclassifcation here:
with h5py.File('misclassifcation'+model_name+'_'+str(indice_no)+'.hdf5','w') as hf:
    hf.create_dataset("mis_info",  data=info_matrix_mis)
    hf.create_dataset("mis_label", data=label_mis)

