import numpy as np
np.random.seed(1337)
from keras.models import Model
from keras.layers import Dense, Input, Activation, Dropout, Flatten, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D, LSTM, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras import initializers
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras import backend as K
import tensorflow as tf
import sys
from deep_model_lib import *
import pickle
import h5py
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,TensorBoard
from keras.models import model_from_json
import json


batch_size=32
epochs=15

model_option = input('List the all the available models to train. Press(y/n): ')

if model_option == 'y':
    print('available models: alex, revised, resnet, inception, VGG')
    model_name = input('Select a model from the above list: ')

if model_option == 'n':
    print('Please modify the models in the deep_model_lib.py ')

#model_name = args[0]

results_arr = np.zeros((10,6))
data_dir = './training_data/'
#testspec_name='C:/Python_Study/for_edward/DL_codes/DL_codes/dr7/DR7_set/training_set/no%s_test_data.hdf5' % k

for i in np.arange(5):
    print('this is training set %s' % i)
    #trainspec_name= data_dir + 'no%s_training_data.hdf5' % i
    trainspec_name= './training_data/training_artificial.hdf5'
    with h5py.File(trainspec_name,'r') as hf:
        X_train=hf["training_sp"][:]
        y_train=hf["training_label"][:]
        info_train=hf["training_info"][:]

    testspec_name= data_dir + 'no%s_test_data.hdf5' % i
    with h5py.File(testspec_name,'r') as hf:
        X_test=hf["test_sp"][:]
        y_test=hf["test_label"][:]
        info_test=hf["test_info"][:]


    final_spec_name= data_dir + 'dr12_final_data_for_test0.hdf5'
    with h5py.File(final_spec_name,'r') as hf:
        X_final=hf["training_sp"][:]
        y_final=hf["training_label"][:]
        info_final=hf["training_info"][:]


    y_train = y_train.reshape((-1, 1))
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    y_test = y_test.reshape((-1, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_final = y_final.reshape((-1, 1))
    X_final = X_final.reshape((X_final.shape[0], X_final.shape[1], 1))



    print("Training...")

    if model_name == 'alex':
        model = Alex1d_model((X_train.shape[1], 1))

    if model_name == 'revised':
        model = revised_model((X_train.shape[1], 1))

    if model_name == 'resnet':
        model = resnet_model((X_train.shape[1], 1))

    if model_name == 'inception':
        model = inception_model((X_train.shape[1], 1))

    if model_name == 'VGG':
        model = VGG1d_model((X_train.shape[1], 1))

    optimization = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    logname='./model_logs/'+model_name+'%slogs' % i
    tbCallBack = TensorBoard(log_dir=logname, histogram_freq=0, write_graph=True, write_images=True)

    """start training"""
    model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    tmp_name = './model_results/'+model_name+'_'+str(i)+'.hdf5'
    model_checkpoint = ModelCheckpoint(tmp_name, monitor='val_loss', save_best_only=True)
    callbacks = [model_checkpoint,tbCallBack]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True,
                    callbacks=callbacks)
    model.summary()

    print("Validating...")
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    print("Predicting...")
    #print(X_final.shape)
    y_pred = model.predict(X_final)
    y_pred=np.around(y_pred)

    print(accuracy_score(y_final, y_pred.round()) )
    print(precision_score(y_final, y_pred.round()) )
    print(recall_score(y_final, y_pred.round()) )
    print(f1_score(y_final, y_pred.round()) )
    print(roc_auc_score(y_final, y_pred.round()) )


    results_arr[i,0]=scores[1]
    results_arr[i,1]=accuracy_score(y_final, y_pred.round())
    results_arr[i,2]=precision_score(y_final, y_pred.round())
    results_arr[i,3]=recall_score(y_final, y_pred.round())
    results_arr[i,4]=f1_score(y_final, y_pred.round())
    results_arr[i,5]=roc_auc_score(y_final, y_pred.round())

    model_json = model.to_json()
    save_model_name='./model_parameters/'+model_name+'_model_'+str(i)+'.json'
    with open(save_model_name, "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    save_weights_name='./model_parameters/'+model_name+'_weights_'+str(i)+'.hdf5'
    model.save_weights(save_weights_name)
    print("Saved model to disk")


result_name=model_name+'_results.hdf5'
with h5py.File(result_name,'w') as hf:
    hf.create_dataset("results",  data=results_arr)

