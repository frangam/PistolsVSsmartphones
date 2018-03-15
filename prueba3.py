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


#-------------------------------------
# Preparamos los datos de MNIST
#-------------------------------------
import os

## Cambiar path 
os.chdir("/Volumes/GoogleDrive/Mi unidad/Proyectos/PistolsVSsmartphones/")
import loaddata as ld

w=128
h=128

epochs = 10 # 10 for augmented training data, 20 for training data
TRAIN_BATCH_SIZE = 50



data, labels, test = ld.prepare_all_data(128, 128)

Xtrain, Xtest, yTrain, yTest = train_test_split(data, labels, 
                                                        test_size = 0.2, 
                                                        stratify=labels)

steps_per_epoch = len(data) #// 32


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


#-------------------------------------
# Preparamos los datos de MNIST
#-------------------------------------
import os

## Cambiar path 
os.chdir("/Volumes/GoogleDrive/Mi unidad/Proyectos/PistolsVSsmartphones/")
os.chdir("")
import loaddata as ld

WEIGHT=128
HEIGHT=128

EPOCHS = 10
BATCH_SIZE = 50



X_train, y_rain, test = ld.prepare_all_data(WEIGHT, HEIGHT)

Xtrain, Xtest, yTrain, yTest = train_test_split(data, labels, 
                                                        test_size = 0.2, 
                                                        stratify=labels)

steps_per_epoch = len(data) #// 32



#-------------------------------------
# Model
#-------------------------------------
model = Sequential()
    
model.add(Conv2D(32, (3, 3), input_shape=(w,h,1)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512)) # capa completamente conectada
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.7))

model.add(Dense(10))
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
test_gen = ImageDataGenerator()

train_generator = train_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
test_generator = test_gen.flow(X_test, Y_test, batch_size=BATCH_SIZE)
test_aug_generator = train_gen.flow(X_test, Y_test, batch_size=BATCH_SIZE)
train_no_aug_generator = test_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE)

#entrenamos el modelo
model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0]//BATCH_SIZE, epochs=EPOCHS)
                    #,validation_data=test_generator, validation_steps=X_test.shape[0]//BATCH_SIZE)




# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(horizontal_flip=True,  vertical_flip=True,
                                   rotation_range=90,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2)
train_generator = train_datagen.flow_from_directory(
		'Train',
		target_size=(WEIGHT, HEIGHT),
		batch_size=32)








prediction = model.predict(test)

ld.submission(prediction, 3)






# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(horizontal_flip=True,  vertical_flip=True,
                                   rotation_range=90,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2)
train_generator = train_datagen.flow_from_directory(
		'Train',
		target_size=(128, 128),
		batch_size=32)
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
		'test_man',
		target_size=(128, 128),
		batch_size=32)

train_model(model,train_generator,steps_per_epoch)

test_loss = model.evaluate_generator(test_generator, steps=36 // 32)
print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
print(model.metrics_names)

prediction = model.predict(test)

ld.submission(prediction, 3)







def create_model(w=128, h=128):

    #model = Sequential()
    
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
    
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True
    
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
                  loss='categorical_crossentropy',  metrics=['accuracy'])


    model.add(Conv2D(8, (3, 3), input_shape=(w, h,3)))
    model.add(Activation("relu"))

    model.add(Conv2D(8, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=1))

    model.add(Dropout(0.7))

    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(2))

    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
      optimizer=keras.optimizers.Adam(lr=27*1e-04,clipnorm=1., clipvalue=0.5),
      metrics=['accuracy'])
    return model



def train_model(model,train_generator,steps_per_epoch):
    #Using the early stopping technique to prevent overfitting
    earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    
    	#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
    history = model.fit_generator(
    			train_generator,
    			steps_per_epoch= steps_per_epoch,
    			#callbacks=[earlyStopping],
    			epochs=50)
    
    print("Saving the weights")
    model.save_weights('weights.h5')