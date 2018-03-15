#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:50:35 2018

@author: frangarcia
"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.layers import Dense,Flatten,Activation, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Nadam
from keras.applications.inception_v3 import InceptionV3


#-------------------------------------
# Preparamos los datos de MNIST
#-------------------------------------
import os

## Cambiar path 3
#os.chdir("/Volumes/GoogleDrive/Mi unidad/Proyectos/PistolsVSsmartphones/")
os.chdir("C:\\Users\\Garmo\\Dropbox\Master\\MD_avanzada\\kaggle")
import loaddata as ld

WEIGHT = 128
HEIGHT = 128
CHANNELS = 3

EPOCHS = 100
BATCH_SIZE = 8
NUM_CLASSES = 2


X_train, y_train, test = ld.prepare_all_data(WEIGHT, HEIGHT)

#Xtrain, Xtest, yTrain, yTest = train_test_split(data, labels, 
#                                                        test_size = 0.2, 
#                                                        stratify=labels)





#-------------------------------------
# Model
#-------------------------------------

    
model = Sequential()

model.add(Conv2D(8, (3, 3), input_shape=(WEIGHT, HEIGHT, CHANNELS)))
model.add(Activation('relu'))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16,(3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512)) # capa completamente conectada
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.7))


model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
      optimizer=keras.optimizers.Adam(lr=27*1e-04,clipnorm=1., clipvalue=0.5),
      metrics=['accuracy'])


#-------------------------------------
# Entrenamiento
#-------------------------------------

# con data-augmentation
train_gen = ImageDataGenerator(rotation_range=90,width_shift_range=0.03, 
                             height_shift_range=0.03, horizontal_flip=True, 
                             vertical_flip=True, zoom_range=0.08)
#sin
train_gen_no_aug = ImageDataGenerator()

train_generator = train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
train_no_aug_generator = train_gen_no_aug.flow(X_train, y_train, batch_size=BATCH_SIZE)


#entrenamos el modelo
model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0]//BATCH_SIZE, epochs=EPOCHS)
                    #,validation_data=test_generator, validation_steps=X_test.shape[0]//BATCH_SIZE)






#-------------------------------------
# Prediccion
#-------------------------------------
#predict_generator =  train_gen.flow(test, batch_size=BATCH_SIZE)
#prediction = model.predict_generator(predict_generator)

prediction = model.predict(test)
ld.submission(prediction, 4)

