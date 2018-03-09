#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:49:52 2018

@author: frangarcia
"""

import os

## change this path
os.chdir("/Volumes/GoogleDrive/Mi unidad/Proyectos/PistolsVSsmartphones/")

import numpy as np
import glob as g
import scipy
import cv2


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense,Flatten,Activation, Dropout, GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

#keras CNN layers
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(123)  # for reproducibility


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy'
              ,  metrics=['accuracy'])



model.fit(Xtrain, yTrain, epochs=10, verbose=1)

score = model.evaluate(Xtest, yTest)

prediction = model.predict()
print(prediction)




pathsTrainPistol=g.glob("Train/Pistol/*[!*.ini]")
pathsTrainSmartphone=g.glob("Train/Smartphone/*[!*.ini]")
pathsTest=g.glob("Test/*[!*.ini]")

pistols=[]
smartphones=[]
test=[]
w=128
h=128
totalRows=len(pathsTrainPistol) + len(pathsTrainSmartphone)
labels = np.zeros(shape=(totalRows), dtype=int)

for i,path in enumerate(pathsTrainPistol):
    img=cv2.imread(path)
    img=cv2.resize(img,(w,h))
    pistols.append(img)
    labels[i]=1
   
for path in pathsTrainSmartphone:
    img=cv2.imread(path)
    img=cv2.resize(img,(w,h))
    smartphones.append(img)
    
for path in pathsTest:
    img=cv2.imread(path)
    img=cv2.resize(img,(w,h))
    test.append(img)
    
data=np.append(pistols, smartphones, axis=0)
Xtrain, Xtest, yTrain, yTest = train_test_split(data, labels, 
                                                    test_size = 0.2, 
                                                    stratify=labels)


yTrain = np_utils.to_categorical(yTrain, 2)
yTest = np_utils.to_categorical(yTest, 2)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_shape=(w,h,3)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(Xtrain, yTrain, 
          batch_size=w, nb_epoch=10, verbose=1)
score = model.evaluate(Xtest, yTest, batch_size=128)





#Arquitectura
model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(w,h,3)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#compilamos modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#entrenamos el modelo
model.fit(Xtrain, yTrain, 
          batch_size=w, nb_epoch=10, verbose=1)

#evaluar el modelo
score = model.evaluate(X_test, Y_test, verbose=0)




import random
def generator(features, labels, batch_size, w, h):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, w, h, 3))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= np.random.choice(len(features),1)
     batch_features[i] = features[index]
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels
   
   
   
def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = # Enter number of classes
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])