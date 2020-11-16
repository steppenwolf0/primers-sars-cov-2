# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
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

from pandas import read_csv
import pandas as pd 
from sklearn.model_selection import StratifiedKFold

def get_InfoA(indexVar,labelSize,vectorSize):
	
	trainLabels=openVector('trainLabels.txt')
	validLabels=openVector('validateLabels.txt')
	testLabels=openVector('testLabels.txt')
	
	oneHot_train_labels=oneHot(trainLabels,labelSize)
	print(oneHot_train_labels.shape)

	oneHot_valid_labels=oneHot(validLabels,labelSize)
	print(oneHot_valid_labels.shape)

	oneHot_test_labels=oneHot(testLabels,labelSize)
	print(oneHot_test_labels.shape)
	
	trainArray = np.genfromtxt('./'+'Train.matrix', delimiter=' ')
	train=np.array(trainArray)
	print('train set', train.shape)
	
	testArray = np.genfromtxt('./'+'Test.matrix', delimiter=' ')
	test=np.array(testArray)
	print('test set', test.shape)
	
	validateArray = np.genfromtxt('./'+'Validate.matrix', delimiter=' ')
	valid=np.array(validateArray)
	print('valid set', valid.shape)

	return(test,oneHot_test_labels,valid,oneHot_valid_labels,train,oneHot_train_labels)

from sklearn.preprocessing import StandardScaler
def get_InfoOriginal(indexVar,dataFolder):
	#examples=8129

	data=[]
	data = np.genfromtxt(dataFolder+'data.csv', delimiter=',')
	data=np.array(data)
	print('data set', data.shape)
	#print(data[0])
	
	from sklearn import preprocessing
	StandardScaler = preprocessing.StandardScaler()
	#data = StandardScaler.fit_transform(data)

	
	labels=openVector(dataFolder+'labels.csv')
	
	labelSize=int(np.max(labels)+1)
	vectorSize=data.shape[1]
	
	#print(labels)
	testIndex=openVector(dataFolder+'index/'+str(indexVar)+'test_index.txt')
	valIndex=openVector(dataFolder+'index/'+str(indexVar)+'val_index.txt')
	trainIndex=openVector(dataFolder+'index/'+str(indexVar)+'train_index.txt')
	
	testIndex=testIndex.astype(int)
	valIndex=valIndex.astype(int)
	trainIndex=trainIndex.astype(int)
		
	train=[]
	test=[]
	valid=[]
	
	
	trainLabels=[]
	testLabels=[]
	validLabels=[]
	#test***************************************************************************
	for i in range (0,len(testIndex)):
		testLabels.append(labels[testIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[testIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[testIndex[i]][j])
		test.append(temp)
	#valid***************************************************************************
	for i in range (0,len(valIndex)):
		validLabels.append(labels[valIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[valIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[valIndex[i]][j])
		valid.append(temp)
	#train***************************************************************************
	for i in range (0,len(trainIndex)):
		trainLabels.append(labels[trainIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[trainIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[trainIndex[i]][j])
		train.append(temp)
	
	test=np.array(test)
	testLabels=np.array(testLabels)

	valid=np.array(valid)
	validLabels=np.array(validLabels)
	
	train=np.array(train)
	trainLabels=np.array(trainLabels)
	
	
	
	print(train.shape)
	print(trainLabels.shape)
	print(valid.shape)
	print(validLabels.shape)
	print(test.shape)
	print(testLabels.shape)
	
	print(labelSize)
	oneHot_train_labels=oneHot(trainLabels,labelSize)
	print(oneHot_train_labels.shape)

	oneHot_valid_labels=oneHot(validLabels,labelSize)
	print(oneHot_valid_labels.shape)

	oneHot_test_labels=oneHot(testLabels,labelSize)
	print(oneHot_test_labels.shape)


	return(test,oneHot_test_labels,valid,oneHot_valid_labels,train,oneHot_train_labels,labelSize,vectorSize)

# generate random integer values
from random import seed
from random import randint
# seed random number generator
import time
def get_InfoMutation(indexVar,labelSize,vectorSize):
	#examples=8129

	data=[]
	data = np.genfromtxt('./data/'+'data.csv', delimiter=',')
	data=np.array(data)
	print('data set', data.shape)
	#print(data[0])
	
	from sklearn import preprocessing
	StandardScaler = preprocessing.StandardScaler()
	#data = StandardScaler.fit_transform(data)

	
	labels=openVector('./data/labels.csv')
	print(labels)
	#print(labels)
	testIndex=openVector('./data/index/'+str(indexVar)+'test_index.txt')
	valIndex=openVector('./data/index/'+str(indexVar)+'val_index.txt')
	trainIndex=openVector('./data/index/'+str(indexVar)+'train_index.txt')
	
	testIndex=testIndex.astype(int)
	valIndex=valIndex.astype(int)
	trainIndex=trainIndex.astype(int)
		
	train=[]
	test=[]
	valid=[]
	
	
	trainLabels=[]
	testLabels=[]
	validLabels=[]
	#test***************************************************************************
	for i in range (0,len(testIndex)):
		testLabels.append(labels[testIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[testIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[testIndex[i]][j])
		test.append(temp)
	#valid***************************************************************************
	for i in range (0,len(valIndex)):
		validLabels.append(labels[valIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[valIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[valIndex[i]][j])
		valid.append(temp)
	#train***************************************************************************
	for i in range (0,len(trainIndex)):
		trainLabels.append(labels[trainIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[trainIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[trainIndex[i]][j])
		train.append(temp)
	
	test=np.array(test)
	testLabels=np.array(testLabels)

	valid=np.array(valid)
	validLabels=np.array(validLabels)
	
	train=np.array(train)
	trainLabels=np.array(trainLabels)
	
	seed(time.time())
	for i in range (0,len(test)):
		for j in range (0,1551):
			valuePosition = randint(0, len(test[0])-1)
			valueDNA=float(randint(0, 5-1))
			print(str(valuePosition)+"\t"+str(valueDNA)) 
			test[i,valuePosition]=valueDNA*0.25
	print(train.shape)
	print(trainLabels.shape)
	print(valid.shape)
	print(validLabels.shape)
	print(test.shape)
	print(testLabels.shape)
	
	print(labelSize)
	oneHot_train_labels=oneHot(trainLabels,labelSize)
	print(oneHot_train_labels.shape)

	oneHot_valid_labels=oneHot(validLabels,labelSize)
	print(oneHot_valid_labels.shape)

	oneHot_test_labels=oneHot(testLabels,labelSize)
	print(oneHot_test_labels.shape)


	return(test,oneHot_test_labels,valid,oneHot_valid_labels,train,oneHot_train_labels)
import pandas as pd
def get_InfoTotal_Original(labelSize,vectorSize):
	#examples=8129

	data=[]
	data = np.genfromtxt('../data/'+'data.csv', delimiter=',')
	data=np.array(data)
	print('data set', data.shape)
	#print(data[0])
	
	labels=openVector('../data/labels.csv')
	
	
	data=np.array(data)
	labels=np.array(labels)
	
	sort_index = np.argsort(labels)

	size=data.shape
	
	dataSorted=np.zeros((size[0],size[1]))
	labelsSorted=np.zeros(size[0])

	for i in range(0,len(labels)):
		labelsSorted[i]=labels[sort_index[i]]
		dataSorted[i]=data[sort_index[i]]
	
	pd.DataFrame(dataSorted).to_csv("../data/filters/sortedData.csv", header=None, index =None)
	pd.DataFrame(labelsSorted).to_csv("../data/filters/labelsSorted.csv", header=None, index =None)
	

	print(labelSize)
	oneHot_labels=oneHot(labelsSorted,labelSize)
	print(oneHot_labels.shape)

	return(dataSorted,oneHot_labels)

def get_InfoTotal(dataFolder):
	#examples=8129

	dataSeq = read_csv(dataFolder+'sequences.csv', header=None).values.ravel()
	data=dataSeq
	print('data set', data.shape)

	
	labels=read_csv(dataFolder+'labels.csv', header=None).values.ravel()
	
	labelSize=int(np.max(labels)+1)
	
	max=0
	for i in range (0,len(data)):
		if (len(data[i])>max):
			max=len(data[i])
	
	vectorSize=max
	print('vectorSize', vectorSize)
	
	outData=[]
	outLabels=[]
	for i in range (len(data)):
		sample=np.zeros(vectorSize)
		for j in range (0, len(data[i])):
			if(data[i][j]=='C'):
				sample[j]=0.25
			elif(data[i][j]=='T'):
				sample[j]=0.50
			elif(data[i][j]=='G'):
				sample[j]=0.75
			elif(data[i][j]=='A'):
				sample[j]=1.0
			else:
				sample[j]=0.0
		outData.append(sample)
		outLabels.append(labels[i])
	
	data=np.array(outData)
	labels=np.array(outLabels)
	
	sort_index = np.argsort(labels)

	size=data.shape
	
	dataSorted=np.zeros((size[0],size[1]))
	labelsSorted=np.zeros(size[0])
	sequencesSorted=[]

	for i in range(0,len(labels)):
		labelsSorted[i]=labels[sort_index[i]]
		sequencesSorted.append(dataSeq[sort_index[i]])
		dataSorted[i]=data[sort_index[i]]
	
	pd.DataFrame(sequencesSorted).to_csv("../data/filters/sequencesSorted.csv", header=None, index =None)
	#pd.DataFrame(dataSorted).to_csv("../data/filters/sortedData.csv", header=None, index =None)
	pd.DataFrame(labelsSorted).to_csv("../data/filters/labelsSorted.csv", header=None, index =None)
	

	print(labelSize)
	oneHot_labels=oneHot(labelsSorted,labelSize)
	print(oneHot_labels.shape)

	return(dataSorted,oneHot_labels)

def getBatch(data, labels, size, sampleSize):
	index=[]
	for i in range (len(data)):
		index.append(i)
	batch=random.sample(index,  size)
	outData=[]
	outLabels=[]
	for i in range (len(batch)):
		sample=np.zeros(sampleSize)
		for j in range (0, len(data[batch[i]])):
			if(data[batch[i]][j]=='C'):
				sample[j]=0.25
			elif(data[batch[i]][j]=='T'):
				sample[j]=0.50
			elif(data[batch[i]][j]=='G'):
				sample[j]=0.75
			elif(data[batch[i]][j]=='A'):
				sample[j]=1.0
			else:
				sample[j]=0.0
		outData.append(sample)
		outLabels.append(labels[batch[i]])
	return np.array(outData), np.array(outLabels)

def getBatch_run(data, labels, size,run,vector, sampleSize):
	infLimit=run*size
	supLimit=infLimit+size
	if supLimit > len(data):
		supLimit=len(data)
	batch=[]
	for i in range (infLimit,supLimit):
		batch.append(vector[i])
	outData=[]
	outLabels=[]
	for i in range (len(batch)):
		sample=np.zeros(sampleSize)
		for j in range (0, len(data[batch[i]])):
			if(data[batch[i]][j]=='C'):
				sample[j]=0.25
			elif(data[batch[i]][j]=='T'):
				sample[j]=0.50
			elif(data[batch[i]][j]=='G'):
				sample[j]=0.75
			elif(data[batch[i]][j]=='A'):
				sample[j]=1.0
			else:
				sample[j]=0.0
		outData.append(sample)
		outLabels.append(labels[batch[i]])
	return np.array(outData), np.array(outLabels)

def get_Info(indexVar,dataFolder):
	#examples=8129

	data = read_csv(dataFolder+'sequences.csv', header=None).values.ravel()
	print('data set', data.shape)

	
	labels=read_csv(dataFolder+'labels.csv', header=None).values.ravel()
	
	labelSize=int(np.max(labels)+1)
	
	max=0
	for i in range (0,len(data)):
		if (len(data[i])>max):
			max=len(data[i])
	
	vectorSize=max
	print('vectorSize', vectorSize)
	
	#print(labels)
	testIndex=read_csv(dataFolder+'index/'+str(indexVar)+'test_index.txt', header=None).values.ravel()
	valIndex=read_csv(dataFolder+'index/'+str(indexVar)+'val_index.txt', header=None).values.ravel()
	trainIndex=read_csv(dataFolder+'index/'+str(indexVar)+'train_index.txt', header=None).values.ravel()
	
	testIndex=testIndex.astype(int)
	valIndex=valIndex.astype(int)
	trainIndex=trainIndex.astype(int)
		
	train=[]
	test=[]
	valid=[]
	
	
	trainLabels=[]
	testLabels=[]
	validLabels=[]
	#test***************************************************************************
	for i in range (0,len(testIndex)):
		testLabels.append(labels[testIndex[i]])
		test.append(data[testIndex[i]])
	#valid***************************************************************************
	for i in range (0,len(valIndex)):
		validLabels.append(labels[valIndex[i]])
		valid.append(data[valIndex[i]])
	#train***************************************************************************
	for i in range (0,len(trainIndex)):
		trainLabels.append(labels[trainIndex[i]])
		train.append(data[trainIndex[i]])
	

	testLabels=np.array(testLabels)
	validLabels=np.array(validLabels)
	trainLabels=np.array(trainLabels)

	print(len(train))
	print(len(train[0]))
	print(trainLabels.shape)
	print(len(valid))
	print(validLabels.shape)
	print(len(test))
	print(testLabels.shape)
	
	print(labelSize)
	oneHot_train_labels=oneHot(trainLabels,labelSize)
	print(oneHot_train_labels.shape)

	oneHot_valid_labels=oneHot(validLabels,labelSize)
	print(oneHot_valid_labels.shape)

	oneHot_test_labels=oneHot(testLabels,labelSize)
	print(oneHot_test_labels.shape)


	return(test,oneHot_test_labels,valid,oneHot_valid_labels,train,oneHot_train_labels,labelSize,vectorSize)