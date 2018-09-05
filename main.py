#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:50:35 2018

@author: frangarcia
"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense,Flatten,Activation, Dropout,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Nadam
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os



#-------------------------------------
# Preparamos los datos de MNIST
#-------------------------------------


## Cambiar path 3
#os.chdir("/Volumes/GoogleDrive/Mi unidad/Proyectos/PistolsVSsmartphones/")
os.chdir("C:\\Users\\Garmo\\Dropbox\Master\\MD_avanzada\\kaggle")
import loaddata as ld


WEIGHT = 128
HEIGHT = 128
CHANNELS = 3

EPOCHS = 30
BATCH_SIZE = 64
VAL_SIZE = 64
NUM_CLASSES = 2


X_train, y_train, test = ld.prepare_all_data(WEIGHT, HEIGHT)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                        test_size = 0.2, 
                                                        stratify=y_train)





#-------------------------------------
# Model
#-------------------------------------

from keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False, 
                   input_shape=(WEIGHT, HEIGHT, CHANNELS))

# freeze layers
for i,layer in enumerate(base_model.layers):
    layer.trainable = False
    print(i,layer.name)
 

# FINE-TUNE
model = Sequential()
 
# Add the vgg convolutional base model
model.add(base_model)
 
# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.7))
model.add(Dense(NUM_CLASSES, activation='softmax'))



# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])



#-------------------------------------
# Entrenamiento
#-------------------------------------
    

## con data-augmentation
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=20,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')
test_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
test_generator = test_gen.flow(X_test, y_test, batch_size=VAL_SIZE)

#entrenamos el modelo
#multiplicamos por 2 steps_per_epoch ya que usamos data augmentation
trained_model = model.fit_generator(train_generator, 
                    steps_per_epoch=2*X_train.shape[0]//BATCH_SIZE, 
                    epochs=EPOCHS,
                    validation_data=test_generator, 
                    validation_steps=X_test.shape[0]//VAL_SIZE)
                
model.save("pretrained_model_all_freeze.h5")


#-------------------------------------
# Resultados evaluación
#-------------------------------------
train_acc = trained_model.history['acc']
val_acc = trained_model.history['val_acc']
train_loss = trained_model.history['loss']
val_loss = trained_model.history['val_loss']
ep = range(len(train_acc))

plt.plot(ep, train_acc, 'b', label='Training Accuracy')
plt.plot(ep, val_acc, 'r', label='Validation Accuracy')
#plt.title('Evaluación del modelo fine-tuned VGG-16 (Accuracy)')
plt.legend()

plt.figure()

plt.plot(ep, train_loss, 'b', label='Training loss')
plt.plot(ep, val_loss, 'r', label='Validation loss')
#plt.title('Evaluación del modelo fine-tuned VGG-16 (loss)')
plt.legend()

#-------------------------------------
# Prediccion
#-------------------------------------
#predict_generator =  train_gen.flow(test, batch_size=BATCH_SIZE)
#prediction = model.predict_generator(predict_generator)

prediction = model.predict(test)
ld.submission(prediction, 14)

