import cv2
import numpy as np
import os

import math
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import keras

import imagereaders
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, ZeroPadding3D, Convolution3D, MaxPooling3D
from keras.optimizers import SGD
from sklearn import preprocessing




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


if __name__ == "__main__":

    trainsamples = []
    trainvideos = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/FeatureTraining/videos/train/', 30, 15, 4, 2)
    testvideos = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/FeatureTraining/videos/test/', 30, 15, 4, 2)

    for vid in trainvideos:
        vid = imagereaders.getFlowVid(vid, 32)
        vid = imagereaders.getChannelsinVid(vid)
        trainsamples.append(vid)

    y_train = imagereaders.read_answers('/home/jessiefin/PycharmProjects/FeatureTraining/annotationtrain.txt')
    y_train=np.float32(y_train)

    y_test = imagereaders.read_answers('/home/jessiefin/PycharmProjects/FeatureTraining/annotationtest.txt')
    y_test = np.float32(y_test)
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    y_train_norm = scaler.fit_transform(y_train.reshape(-1,1))
    y_test_norm = scaler.transform(y_test.reshape(-1,1))

    trainsamples = np.asarray(trainsamples)
    print trainsamples.shape


    testsamples = []


    for vid in testvideos:
        vid = imagereaders.getFlowVid(vid, 32)
        vid = imagereaders.getChannelsinVid(vid)
        testsamples.append(vid)

    testsamples=np.asarray(testsamples)



    # Test pretrained model
    model = temporalNet('temporalwtsmerge.h5')
    sgd = SGD(lr=0.0001, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    model.fit(trainsamples, y_train_norm, nb_epoch=10, verbose=1)
    out = model.predict(testsamples)
    score = model.evaluate(testsamples, y_test_norm)
    x_norm2 = scaler.inverse_transform(out)

    #Print output
    print(out)
    print x_norm2
    print score
