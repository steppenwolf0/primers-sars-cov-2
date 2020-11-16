import numpy as np
import math
import random
import pandas as pd 

#Ouput*******************************************************************************
def openMatrix (name) :
	dataArray = np.genfromtxt(name, delimiter=' ')
	data=dataArray
	print(dataArray.shape)
	maxData=np.nanmax(dataArray)
	minData=np.nanmin(dataArray)
	meanData=np.nanmean(dataArray)
	distance=maxData-minData
	print('Max ',maxData)
	print('Min ',minData)
	print('meanData ',meanData)
	for i in range (len(dataArray)):
		for j in range (len(dataArray[0])):
			data[i][j]=(data[i][j]-minData)/distance
			#data[i][j]=data[i][j]
			if math.isnan(data[i][j]):
				data[i][j]=-1
	return data

def openVector (name) :
	dat= np.genfromtxt(name, delimiter=' ')
	#data = pd.read_csv(name, header=None)
	print(dat.shape)
	return dat

def saveMatrix(name,var):
	np.savetxt(name, var, fmt='%1.3f', delimiter=' ')

def saveMatrixInt(name,var):
	np.savetxt(name,var, fmt='%i', delimiter=' ')

def saveVectorInt(name,var):
	np.savetxt(name, var, fmt='%i', delimiter=' ')

#Pre Processing*********************************************************************
def oneHot (array, size):
	output=[]
	#print(len(array))
	#print(size)
	#print(array)
	for i in range (len(array)):
		temp=np.zeros(size)
		temp[int(array[i])]=1
		output.append(temp)
	print(len(output))
	print(len(output[0]))
	return np.array(output)

def compressArray(array,size):
	outputArray=[]
	for i in range(len(array)):
		temp=[]
		for j in range(size):
			if (j<len(array[0])):
				temp.append(array[i][j])
			else:
				temp.append(-1)
		outputArray.append(temp)
	return np.array(outputArray)

def getBatch(data, labels, size):
	index=[]
	for i in range (len(data)):
		index.append(i)
	batch=random.sample(index,  size)
	outData=[]
	outLabels=[]
	for i in range (len(batch)):
		outData.append(data[batch[i]])
		outLabels.append(labels[batch[i]])
	return np.array(outData), np.array(outLabels)

def getBatch_c(data, labels, size, constants):
	index=[]
	for i in range (len(data)):
		index.append(i)
	batch=random.sample(index,  size)
	#print(batch)
	outData=[]
	outLabels=[]
	outConstants=[]
	for i in range (len(batch)):
		outData.append(data[batch[i]])
		outLabels.append(labels[batch[i]])
		outConstants.append(constants[batch[i]])
	return np.array(outData), np.array(outLabels), np.array(outConstants)

def getBatch_run_c(data, labels, size, constants,run,vector):
	infLimit=run*size
	supLimit=infLimit+size
	if supLimit > len(data):
		supLimit=len(data)
	batch=[]
	for i in range (infLimit,supLimit):
		batch.append(vector[i])
	outData=[]
	outLabels=[]
	outConstants=[]
	for i in range (len(batch)):
		outData.append(data[batch[i]])
		outLabels.append(labels[batch[i]])
		outConstants.append(constants[batch[i]])
	return np.array(outData), np.array(outLabels), np.array(outConstants)

def print_no_newline(string):
	import sys
	sys.stdout.write(string)
	sys.stdout.flush()

def getBatch_run(data, labels, size,run,vector):
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
		outData.append(data[batch[i]])
		outLabels.append(labels[batch[i]])
	return np.array(outData), np.array(outLabels)
#Pre Processing*********************************************************************