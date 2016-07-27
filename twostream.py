import cv2
import numpy as np
import os

import math
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import keras

import imagereaders
from sklearn import preprocessing
import nn
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, ZeroPadding3D, Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adamax, rmsprop, clip_norm, Adagrad
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, splittensor, Softmax4D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization




def combinetrain(model1, model2, weights=None):
    merged = Merge([model1, model2], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(1, activation='linear'))


    if weights:
        model.load_weights(weights)

    return model




def temporalNet(weights=None):
    model = Sequential()

    model.add(Convolution3D(30, 20, 17, 17, activation='relu', trainable=False, subsample=(4,2,2), input_shape=(1, 120,32,32)))


    model.add(MaxPooling3D(pool_size=(13, 2, 2), strides=(13,2, 2), trainable=False))

    model.add(Reshape((60, 4, 4)))


    model.add(Convolution2D(100, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))
    model.add(Flatten())


    model.add(Dense(400, activation='relu', trainable=False))

    model.add(Dense(50, activation='relu'))
    #model.add(Dense(14, activation='relu'))
    model.add(Dense(1, activation='relu'))

    if weights:
        model.load_weights(weights)

    return model

def spatialNet(weights=None):
    model = Sequential()
    leakyrelu = ELU()

    model.add(Convolution3D(30, 10, 17, 17, subsample=(2,2,2), trainable=False, input_shape=(1, 60,32,32)))
    model.add(Activation(leakyrelu))
    model.add(BatchNormalization(mode=2))
    model.add(MaxPooling3D(pool_size=(13, 2, 2),trainable=False, strides=(13,2, 2)))

    model.add(Reshape((60, 4, 4)))


    model.add(Convolution2D(100, 3, 3, trainable=False))
    model.add(Activation(leakyrelu))
    model.add(BatchNormalization(mode=2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))
    model.add(Flatten())


    model.add(Dense(400, trainable=False))
    model.add(Activation(leakyrelu))
    model.add(BatchNormalization(mode=2))
    model.add(Dense(50))
    model.add(Activation(leakyrelu))
    model.add(BatchNormalization(mode=2))

    model.add(Dense(1, activation='sigmoid'))
    model.add(BatchNormalization(mode=2))

    if weights:
        model.load_weights(weights)

    return model







if __name__ == "__main__":


    #collect test videos and resize segments as network input for the spatial network
    testvideospatial = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/FeatureTraining/videos/test/', 30, 15, 4, 2, flow=1)
    testsamplespatial = []
    for vid in testvideospatial:

        bluelist = []
        for x in range(len(vid)):
            blue = cv2.resize(vid[x], (32, 32))

            bluelist.append(blue)
        vid = np.asarray(bluelist)
        vid = imagereaders.getChannelsinVid(vid)
        testsamplespatial.append(vid)



    #collect spatial training video
        trainvideospatial = imagereaders.collectVidSegments(
            '/home/jessiefin/PycharmProjects/FeatureTraining/videos/train/', 30, 15,
            4, 2, flow=1)
    trainsamplespatial = []
    for vid in trainvideospatial:

        bluelist = []
        for x in range(len(vid)):
            blue = cv2.resize(vid[x], (32, 32))

            bluelist.append(blue)
        vid = np.asarray(bluelist)
        vid = imagereaders.getChannelsinVid(vid)
        trainsamplespatial.append(vid)


    #collect temporal training video
    trainvideostemporal = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/FeatureTraining/videos/train/', 30, 15, 4, 2)
    trainsampletemporal = []
    for vid in trainvideostemporal:

        vid = imagereaders.getFlowVid(vid, 32)
        vid = imagereaders.getChannelsinVid(vid)
        trainsampletemporal.append(vid)

    #collect temporal testing video
    testvideostemporal = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/FeatureTraining/videos/test/', 30, 15, 4, 2)
    testsampletemporal = []

    for vid in testvideostemporal:
        vid = imagereaders.getFlowVid(vid, 32)
        vid = imagereaders.getChannelsinVid(vid)
        testsampletemporal.append(vid)

    #read in annotations
    y_train = imagereaders.read_answers(
        '/home/jessiefin/PycharmProjects/FeatureTraining/annotationtrain.txt')
    y_train = np.float32(y_train)
    y_test = imagereaders.read_answers(
        '/home/jessiefin/PycharmProjects/FeatureTraining/annotationtest.txt')
    y_test = np.float32(y_test)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    y_train_norm = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_norm = scaler.transform(y_test.reshape(-1, 1))




    #compile, train, and test two stream networks
    model1 = spatialNet('spatialwtsmerge.h5')
    model2 = temporalNet('temporalwtsmerge.h5')
    model = combinetrain(model1, model2, 'mergedwts.h5')

    opt = SGD(clipvalue=5)
    model.compile(optimizer=opt, loss='mean_squared_error')

    model.fit([np.asarray(trainsamplespatial), np.asarray(trainsampletemporal)], y_train_norm, nb_epoch=10)
    out = model.predict([np.asarray(testsamplespatial), np.asarray(testsampletemporal)])
    out_norm = scaler.inverse_transform(out)
    score = model.evaluate([np.asarray(testsamplespatial), np.asarray(testsampletemporal)], np.asarray(y_test_norm))

    #Pring output.  One estimate for each 4 second clip in a video
    print out
    print out_norm
    print score

