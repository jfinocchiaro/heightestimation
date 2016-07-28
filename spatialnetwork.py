import cv2
import numpy as np
import os

import math
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import keras
from matplotlib import pyplot as plt

import imagereaders
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, ZeroPadding3D, Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adamax, rmsprop, clip_norm, Adagrad
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization


#Combine two models
def combinetrain(model1, model2):
    merged = Merge([model1, model2], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(1, activation='linear'))
    return model


#Temporal Network input is 60 x and y Optical Flow Frames of Size 32x32
def temporalNet(weights=None):
    model = Sequential()

    model.add(Convolution3D(30, 20, 17, 17, activation='relu', subsample=(4,2,2), input_shape=(1, 120,32,32)))
    model.add(MaxPooling3D(pool_size=(13, 2, 2), strides=(13,2, 2)))
    model.add(Reshape((60, 4, 4)))

    model.add(Convolution2D(100, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())


    model.add(Dense(400, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='relu'))

    if weights:
        model.load_weights(weights)

    return model


#Spatial network 60 grayscale frames of size 32x32
def spatialNet(weights=None):
    model = Sequential()
    elu = ELU()

    model.add(Convolution3D(30, 10, 17, 17, subsample=(2,2,2), input_shape=(1, 60,32,32)))
    model.add(Activation(elu))
    model.add(BatchNormalization(mode=2))
    model.add(MaxPooling3D(pool_size=(13, 2, 2), strides=(13,2, 2)))

    model.add(Reshape((60, 4, 4)))


    model.add(Convolution2D(100, 3, 3))
    model.add(Activation(elu))
    model.add(BatchNormalization(mode=2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())


    model.add(Dense(400))
    model.add(Activation(elu))
    model.add(BatchNormalization(mode=2))
    model.add(Dense(50))
    model.add(Activation(elu))
    model.add(BatchNormalization(mode=2))

    model.add(Dense(1, activation='sigmoid'))
    model.add(BatchNormalization(mode=2))

    if weights:
        model.load_weights(weights)

    return model



if __name__ == "__main__":

    trainsamples = []

    trainvideos = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/FeatureTraining/videos/train/', 30, 15, 4, 2, flow=1)


    for vid in trainvideos:
        bluelist = []
        for x in range (len(vid)):
            img = cv2.resize(vid[x], (32,32))
            bluelist.append(img)
        vid = np.asarray(bluelist)
        vid = imagereaders.getChannelsinVid(vid)
        trainsamples.append(vid)

    trainsamples = np.asarray(trainsamples)

    #Fit heights between 0 and 1
    y_train = imagereaders.read_answers('/home/jessiefin/PycharmProjects/FeatureTraining/annotationtrain.txt')
    y_train=np.float32(y_train)
    mean = np.mean(y_train)
    std = np.std(y_train)
    y_test = imagereaders.read_answers('/home/jessiefin/PycharmProjects/FeatureTraining/annotationtest.txt')
    y_test = np.float32(y_test)
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    y_train_norm = scaler.fit_transform(y_train.reshape(-1,1))
    y_test_norm = scaler.transform(y_test.reshape(-1,1))



    testvideos = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/FeatureTraining/videos/test/', 30, 15, 4, 2, flow=1)
    testsamples = []

    for vid in testvideos:
        bluelist = []
        for x in range(len(vid)):
            img = cv2.resize(vid[x], (32, 32))
            bluelist.append(img)
        vid = np.asarray(bluelist)
        vid = imagereaders.getChannelsinVid(vid)
        testsamples.append(vid)

    testsamples=np.asarray(testsamples)



    # Test and train model
    model = spatialNet('spatialwtssmerge.h5')
    opt = SGD(clipvalue=5)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.fit(trainsamples, y_train_norm, nb_epoch=10, verbose=1)
    out = model.predict(testsamples)
    score = model.evaluate(testsamples, y_test_norm)
    x_norm2 = scaler.inverse_transform(out)



    #Print output
    print(out)
    print(x_norm2)
    print(score)
