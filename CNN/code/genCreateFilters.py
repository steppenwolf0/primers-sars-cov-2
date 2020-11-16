#Declarations******************************************************************************
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
import math
import random
import sys
from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from funcCNN import *
from crossValB import *
import matplotlib.pyplot as plt

dataFolder='../data/'


#Parameters*******************************************************************************
#Maximum number of iterations
iterMax=int(sys.argv[1])
#maximum number of iterations
limit=1.00
#regularization on the weights
beta=0.001
#version of the code
version='gen1'
#size of batch
batchSize=50
#Parameters*******************************************************************************
w1=int(sys.argv[2]) #12
w4=int(sys.argv[3]) #196
h1=int(sys.argv[4]) #148
wd1=int(sys.argv[5]) #21
index=int(sys.argv[6]) #0
kfoldIndex=int(sys.argv[7]) #0
generation=int(sys.argv[8]) #0
#Input Data***********************************************************************************
(test_dataset_Flat,oneHot_test_labels,valid_dataset_Flat,oneHot_valid_labels,
	train_dataset_Flat,oneHot_train_labels,labelSize,vectorSize)=get_Info(
	kfoldIndex,dataFolder)
runs=int(len(oneHot_train_labels)/batchSize)
print(runs)
#Model declaration************************************************************************
import tensorflow as tf
#declare interactive session
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess=tf.InteractiveSession()
#INPUT->CONV LAYER->CONV LAYER->CONV LAYER->RECT FLAT->RECT DROPOUT

#function to declare easily the weights only by shape
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
#function to declare easily the bias only by shape
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#input variable
x = tf.placeholder(tf.float32, [None, vectorSize])
#keep probability to change from dropout 0.50 to 1.0 in validation and test
keep_prob = tf.placeholder(tf.float32)
#expected outputs variable
y_ = tf.placeholder(tf.float32, [None, labelSize])

#arrange the tensor as an image (1*31029) 1 channel
x_image0 = tf.reshape(x, [-1,1,vectorSize,1])
x_image = tf.transpose(x_image0, perm=[0,3,2,1])
#arrange the tensor into 1 channels (1*31029)

#1 LAYER*************************************************************************************
#1 Convolutional Layer Explicit for regularization of the weights
#weigth first layer 1 input channels, 12 output channels, 1x21 filter window size
W_conv1 = weight_variable([1, wd1, 1, w1])
#bias declaration the size has to be the same as the output channels 12
b_conv1 = bias_variable([w1])
#convolution (input weights) moving 1 step each time with a relu
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, 
	strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
#max pooling with a 148 width window size, moving 148 in width by step
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, h1, 1],
	strides=[1, 1, h1, 1], padding='SAME')
#output=545/4
#1 LAYER*************************************************************************************

#Rectifier LAYER*****************************************************************************
#calculated coefficient for the flattening from the size of the 3 convolutional layer
coef=int (h_pool1.get_shape()[1]*h_pool1.get_shape()[2]*h_pool1.get_shape()[3])
h_pool2_flat = tf.reshape(h_pool1, [-1, coef])
#declare the weights considering the constants and 256 output 
W_fc1 = weight_variable([coef, w4])
b_fc1 = bias_variable([w4])

#rectifier (matmul)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#Rectifier LAYER*****************************************************************************

#Rectifier-Dropout LAYER**********************************************************************
#dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#declare weights with the ouput layer in this case 2 (labelSize)
W_fc2 = weight_variable([w4, labelSize])
b_fc2 = bias_variable([labelSize])
#output
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#Rectifier-Dropout LAYER**********************************************************************

#Loss Function********************************************************************************
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[0]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_)+
	beta*tf.nn.l2_loss(W_conv1))
#Optimizer Adam at 1e-5 (literature)**********************************************************
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
#softmax prediction remember we are using one hot labels
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

trueResult=tf.argmax(y_conv,1)
trueTest=tf.argmax(y_,1)
#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Loss Function********************************************************************************
valid_accuracy_global=0.0
test_accuracy_global=0.0


#start
sess.run(tf.initialize_all_variables())
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
saver.restore(sess, "../data/model/model.ckpt")
#Extra to verify sizes************************************************************************
print(x_image.get_shape())
#print(h_conv1.get_shape())
print(h_pool1.get_shape())

print(y_conv.get_shape())
print(y_.get_shape())
#Extra to verify sizes************************************************************************


#out_height = ceil(float(in_height) / float(strides[1]))

#strides2=strides[2]
strides2=1
in_width=vectorSize
filter_width=21
out_width  = np.ceil(float(in_width) / float(strides2))
#pad_along_height = max((out_height - 1) * strides[1] +filter_height - in_height, 0)
pad_along_width = max((out_width - 1) * strides2 +filter_width - in_width, 0)
#pad_top = pad_along_height // 2
#pad_bottom = pad_along_height - pad_top
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left

print("pad_left Conv "+str(pad_left))
print("pad_right Conv "+str(pad_right))

strides2=148
in_width=vectorSize
filter_width=148
out_width  = np.ceil(float(in_width) / float(strides2))
#pad_along_height = max((out_height - 1) * strides[1] +filter_height - in_height, 0)
pad_along_width = max((out_width - 1) * strides2 +filter_width - in_width, 0)
#pad_top = pad_along_height // 2
#pad_bottom = pad_along_height - pad_top
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left

print("out_width HPool "+str(out_width))
print("pad_left HPool "+str(pad_left))
print("pad_right HPool "+str(pad_right))
				  
#Input Data***********************************************************************************
(data,oneHotLabels)=get_InfoTotal('../data/')

#Output:ay amo
units = sess.run(W_conv1,feed_dict={x:data, 
			y_: oneHotLabels, keep_prob: 1.0})
print(units.shape)
units = sess.run(h_conv1,feed_dict={x:data, 
			y_: oneHotLabels, keep_prob: 1.0})
print(units.shape)

dataSize=data.shape
sampleSize=int(dataSize[0])

Mat=np.zeros((sampleSize, vectorSize))
import pandas as pd
for filterIndex in range(0,units.shape[3]):
	for testSize in range(0,sampleSize):
		for inputSize in range (0,vectorSize):
			Mat[testSize][inputSize]=units[testSize][0][inputSize][filterIndex]
	pd.DataFrame(Mat).to_csv("../data/filters/filter_"+str(filterIndex)+".csv", header=None, index =None)

units = sess.run(h_pool1,feed_dict={x:data, 
			y_: oneHotLabels, keep_prob: 1.0})
print(units.shape)
Mat=np.zeros((sampleSize, units.shape[2]))
import pandas as pd
for filterIndex in range(0,units.shape[3]):
	for testSize in range(0,sampleSize):
		for inputSize in range (0,units.shape[2]):
			Mat[testSize][inputSize]=units[testSize][0][inputSize][filterIndex]
	pd.DataFrame(Mat).to_csv("../data/filters/maxPool_"+str(filterIndex)+".csv", header=None, index =None)


sess.close()


