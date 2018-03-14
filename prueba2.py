#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:57:13 2018

@author: frangarcia
"""

import os

## change this path
os.chdir("/Volumes/GoogleDrive/Mi unidad/Proyectos/PistolsVSsmartphones/")

import numpy as np

import scipy



from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense,Flatten,Activation, Dropout, GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

#keras CNN layers
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#guardar y cargar modelos
from keras.models import model_from_json

np.random.seed(123)  # for reproducibility


####
# Leemos los datos
####
import loaddata as ld

data, labels, test = ld.prepare_all_data()

Xtrain, Xtest, yTrain, yTest = train_test_split(data, labels, 
                                                        test_size = 0.2, 
                                                        stratify=labels)

####
# Creamos la arquitectura de la red con InceptionV3
####

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


# entrenar el modelo
model.fit(Xtrain, yTrain, epochs=5, verbose=1)

score = model.evaluate(Xtest, yTest)
score


prediction = model.predict(test_n)

submission(prediction, 2)



# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy',  metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(data, labels, epochs=20, verbose=1)

prediction = model.predict(test_n)



#generamos la submission
submission="ID,Ground_Truth\n"
for i,path in enumerate(pathsTest):
    pred0=int(prediction[i,0])
    pred='0' if pred0==1 else '1'
    submission += path[5:] + "," + pred +"\n"
with open("submission.csv", "w") as f:
    f.write(submission)




# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")







####
# Arquitectura con VGG19
####
# create the base pre-trained model
base_model = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)









def submission(prediction, num=1):
    #generamos la submission
    submission="ID,Ground_Truth\n"
    for i,path in enumerate(pathsTest):
        pred0=int(prediction[i,0])
        pred='0' if pred0==1 else '1'
        submission += path[5:] + "," + pred +"\n"
    with open("submission"+str(num)+".csv", "w") as f:
        f.write(submission)
