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
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,TensorBoard
from keras.models import model_from_json
import json


#
def Alex1d_model(data_shape):
    inputs = Input(shape=data_shape)#(X_train.shape[1], 1)

    layer = Convolution1D(filters=96, kernel_size=15, strides=5,kernel_initializer='glorot_uniform')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(pool_size=3)(layer)

    layer = Convolution1D(filters=256, kernel_size=5, strides=1,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(pool_size=3)(layer)

    layer = Convolution1D(filters=384, kernel_size=3,strides=1,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=384, kernel_size=3,strides=1, kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=256, kernel_size=3,strides=1)(layer)
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


#
def revised_model(data_shape):
    inputs = Input(shape=data_shape)

    layer = Convolution1D(filters=32, kernel_size=23, strides=5,kernel_initializer='glorot_uniform')(inputs) #128
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=64, kernel_size=15, strides=3,kernel_initializer='glorot_uniform')(layer)  #256  /5 1
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=128, kernel_size=5,strides=1,kernel_initializer='glorot_uniform')(layer)  #128 /3 1
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=3)(layer)

    layer = Flatten()(layer)
    layer = Dense(256,kernel_initializer='glorot_uniform')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(256,kernel_initializer='glorot_uniform')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)

    return Model(inputs, layer)


def resnet_model(data_shape):

    inputs = Input(shape=data_shape)

    layer = Convolution1D(filters=64, kernel_size=1, strides=3,kernel_initializer='glorot_uniform')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=64, kernel_size=7, strides=1, padding='same', kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=256, kernel_size=1,strides=1,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)


    shortcut = Convolution1D(filters=256, kernel_size=1,strides=3,kernel_initializer='glorot_uniform')(inputs)
    shortcut = BatchNormalization()(shortcut)
    layer = add([layer, shortcut])
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=3)(layer)
    layer = Flatten()(layer)

    layer = Dense(64,kernel_initializer='glorot_uniform')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)
    return Model(inputs, layer)


def inception_model(data_shape):

    inputs = Input(shape=data_shape)

    layer = Convolution1D(filters=32, kernel_size=3, strides=3,kernel_initializer='glorot_uniform')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=32, kernel_size=3, strides=3,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=3)(layer)
    branch1X1_cov = Convolution1D(filters=64, kernel_size=1, strides=1,kernel_initializer='glorot_uniform')(layer)
    branch1X1_ba = BatchNormalization()(branch1X1_cov)
    branch1X1_final = Activation('relu')(branch1X1_ba)

    branch3X3_cov = Convolution1D(filters=64, kernel_size=3, strides=3,kernel_initializer='glorot_uniform')(layer)
    branch3X3_ba = BatchNormalization()(branch3X3_cov)
    branch3X3_final = Activation('relu')(branch3X3_ba)


    branch5X5_cov = Convolution1D(filters=64, kernel_size=5, strides=5,kernel_initializer='glorot_uniform')(layer)
    branch5X5_ba = BatchNormalization()(branch5X5_cov)
    branch5X5_final = Activation('relu')(branch5X5_ba)

    layer = concatenate([branch1X1_final, branch3X3_final, branch5X5_final],axis=1)

    layer = Flatten()(layer)
    layer = Dense(64,kernel_initializer='glorot_uniform')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)

    return Model(inputs, layer)


def VGG1d_model(data_shape):

    inputs = Input(shape=data_shape)

    layer = Convolution1D(filters=64, kernel_size=3, strides=3,kernel_initializer='glorot_uniform')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)


    layer = Convolution1D(filters=64, kernel_size=3, strides=3,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)


    layer = MaxPooling1D(pool_size=2)(layer)

    layer = Convolution1D(filters=128, kernel_size=3,strides=3,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=256, kernel_size=3,strides=3,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=256, kernel_size=3,strides=3,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=256, kernel_size=3,strides=3,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=2)(layer)

    layer = Flatten()(layer)
    layer = Dense(4096,kernel_initializer='glorot_uniform')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(4096,kernel_initializer='glorot_uniform')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)

    return Model(inputs, layer)

