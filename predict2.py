import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import matplotlib.pyplot as plt
import csv
import re

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

from os import listdir
from os.path import isfile, join
mypath="Test"
image_size=56
num_channels=3
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images=[]
f = open('submission_3.csv','w')
f.write('ID,Ground_Truth')
f.flush();
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./deep_learning_model_3.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()
onlyfiles = sorted_aphanumeric(onlyfiles)
for im in onlyfiles:
    image = cv2.imread(mypath + "/" + im)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ## Let us restore the saved model


    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    if result[0][0] > result[0][1]:
        f.write('\n'+im+','+'0')
        f.flush();
    else:
        f.write('\n'+im + ',' + '1')
        f.flush();

    images=[]


f.close()