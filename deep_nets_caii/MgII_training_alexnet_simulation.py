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
#from matplotlib import pyplot as plt
import pickle
import h5py
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,TensorBoard
from keras.models import model_from_json
import json

        #define the training parameters here:
batch_size=32
epochs=30

        #import the dataset here:
        #python MgII_training training number model_name
args = sys.argv[1:]
        #set_num = args[0]
model_name = args[0]

results_arr = np.zeros((5,6))

for i in np.arange(5):
#            print 'this is set %s' % i 
            trainspec_name='training_artificial_sample.hdf5'
            with h5py.File(trainspec_name,'r') as hf:
                  X_train=hf["training_sp"][:]
                  y_train=hf["training_label"][:]
                  info_train=hf["training_info"][:]



        #preprocessing
            y_train = y_train.reshape((-1, 1))
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


        #def training model here:
            def Alex1d_model():
                inputs = Input(shape=(X_train.shape[1], 1))

                layer = Convolution1D(filters=96, kernel_size=15, strides=5,kernel_initializer='glorot_uniform')(inputs)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)

                layer = MaxPooling1D(pool_size=3)(layer)

                layer = Convolution1D(filters=256, kernel_size=5, strides=1,kernel_initializer='glorot_uniform')(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)

                layer = MaxPooling1D(pool_size=3)(layer)

        #        layer = MaxPooling1D(pool_size=7, strides=None)(layer)

                layer = Convolution1D(filters=384, kernel_size=3,strides=1,kernel_initializer='glorot_uniform')(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)

                layer = Convolution1D(filters=384, kernel_size=3,strides=1, kernel_initializer='glorot_uniform')(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)

                layer = Convolution1D(filters=256, kernel_size=3,strides=1,kernel_initializer='glorot_uniform')(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)


                layer = MaxPooling1D(pool_size=3)(layer)


                layer = Flatten()(layer)
                layer = Dense(1024,kernel_initializer='glorot_uniform')(layer) #2048
                layer = Activation('relu')(layer)
                layer = Dropout(0.5)(layer)
                layer = Dense(1024,kernel_initializer='glorot_uniform')(layer) #2048
                layer = Activation('relu')(layer)
                layer = Dropout(0.5)(layer)                
                layer = Dense(1)(layer)
                layer = Activation('sigmoid')(layer)
                return Model(inputs, layer)


            print("Cross Validation Round: %d" % i)
            print("Training...")
            model=Alex1d_model()
            model.summary()
#            optimization=Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            optimization = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            logname='./'+model_name+'%slogs' % i
            tbCallBack = TensorBoard(log_dir=logname, histogram_freq=0, write_graph=True, write_images=True)
            #optimization=Adadelta()
            """start training"""
            model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])
            tmp_name = './'+model_name+'_'+str(i)+'.hdf5'
            model_checkpoint = ModelCheckpoint(tmp_name, monitor='val_loss', save_best_only=True)
            callbacks = [model_checkpoint,tbCallBack]
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=callbacks)
            #model.summary()

