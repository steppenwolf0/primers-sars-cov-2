# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 
import collections

from classifiersMulti import *

# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# this is an incredibly useful function
from pandas import read_csv
import matplotlib.pyplot as plt

directory="data"

def fakeBootStrapper():
	# create folder
	folderName ="./best/"
	if not os.path.exists(folderName) : os.makedirs(folderName)

	orig_stdout = sys.stdout
	f = open('./best/out.txt', 'w')
	sys.stdout = f
	
	runs=10
	directory="run"+str(0)
	f=open("./"+directory+"/results.txt", "r")
	fl =f.readlines()
	blocks=int(len(fl)/8)
	print(blocks)
	results=np.zeros((blocks,runs+1))
	for j in range(0,runs):
		directory="run"+str(j)
		f=open("./"+directory+"/results.txt", "r")
		fl =f.readlines()
		blocks=int(len(fl)/8)
		count=0
		value=0
		variables=np.zeros(blocks)
		accuracy=np.zeros(blocks)
		indexResults=0
		for x in fl:
			a=x.split("\t")
			value=value+float(a[1])/8.0
			count=count+1
			if (count==8):
				accuracy[indexResults]=value
				value=0
				count=0
				indexResults=indexResults+1
		indexResults=0
		
		for i in range (0,blocks):
			dfFeats = (read_csv("./"+directory+"/features_"+str(i)+".csv", header=None)).as_matrix().ravel() 
			variables[i]=len(dfFeats)
			results[i,j+1]=accuracy[i]
			results[i,0]=variables[i]
		
		
	
	pd.DataFrame(results).to_csv("./best/sum.csv", header=None, index =None)
	
	bestVal=np.zeros(runs)
	bestSize=np.zeros(runs)
	bestPos=np.zeros(runs)
	for j in range(0,runs):
		bestVal[j]=np.max(results[:,j+1])
	
	for j in range(0,runs):	
		for i in range (0,blocks):
			if ( bestVal[j]==results[i,j+1]):
				bestSize[j]=int(results[i,0])
				bestPos[j]=i
	print(bestVal)
	print(bestSize)
	print(bestPos)
	
	bestFeatures=[]
	signatures=[]
	for j in range(0,runs):
		dfFeats = (read_csv("./run"+str(j)+"/features_"+str(int(bestPos[j]))+".csv", header=None))
		bestFeatures.extend(dfFeats.values.ravel())
		signatures.append(dfFeats.values.ravel())
	#print(bestFeatures)
	signatures.append(bestVal)
	signatures.append(bestSize)
	pd.DataFrame(signatures).to_csv("./best/signatures.csv", header=None, index =None)
	
	unique, counts = np.unique(bestFeatures, return_counts=True)
	
	resultsFeatures=np.zeros((len(unique),2), 'U16')
	for j in range(0,len(unique)):
		resultsFeatures[j,0]=unique[j]
		resultsFeatures[j,1]=counts[j]
	
	
	pd.DataFrame(resultsFeatures).to_csv("./best/resultsFeatures.csv", header=None, index =None)
	
	print(np.max(bestVal))
	print(np.argmax(bestVal))
	print(int(bestPos[np.argmax(bestVal)]))
	
	runBest=int(np.argmax(bestVal))
	indexBest=int(bestPos[np.argmax(bestVal)])
	
	
	
	# data used for the predictions
	dfData = read_csv("./run"+str(runBest)+"/data_"+str(indexBest)+".csv", header=None, sep=',')
	dfLabels = read_csv("./run"+str(runBest)+"/labels.csv", header=None)
	biomarkers = read_csv("./run"+str(runBest)+"/features_"+str(indexBest)+".csv", header=None)
	
	pd.DataFrame(dfData.values).to_csv("./best/data_0.csv", header=None, index =None)
	pd.DataFrame(biomarkers.values.ravel()).to_csv("./best/features_0.csv", header=None, index =None)
	pd.DataFrame(dfLabels.values.ravel()).to_csv("./best/labels.csv", header=None, index =None)
	
	runFeatureReduce()
	sys.stdout = orig_stdout
	f.close()
	return

if __name__ == "__main__" :
	sys.exit( fakeBootStrapper() )