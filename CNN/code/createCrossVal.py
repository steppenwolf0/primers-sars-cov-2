#argv[1]=total size
#argv[2]=size of validation examples
#argv[3]=number of folds 20 for 5% test, 10 for 10% test
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
from funcCrossVal import *

dataFolder='../NCBI_1503/'

size=int(sys.argv[1])

from sklearn.model_selection import StratifiedKFold
from pandas import read_csv

dfLabels = read_csv(dataFolder+"labels.csv", header=None)
labels = dfLabels.as_matrix().ravel()  

kf = StratifiedKFold(n_splits=int(sys.argv[3]),shuffle=True)
print(kf)

tempA=[]
for i in range (0,size):
	tempA.append(i)

indexCrossVal=0
for train_index, test_index in kf.split(tempA, labels):
	saveVectorInt(str(indexCrossVal)+'test_index.txt',test_index)
	saveVectorInt(str(indexCrossVal)+'trainVal_index.txt',train_index)
	random.shuffle(train_index)
	tempVector=[]
	for j in range (0,int(sys.argv[2])):
		tempVector.append(train_index[j])
	saveVectorInt(str(indexCrossVal)+'val_index.txt',tempVector)
	tempVector=[]
	for j in range (int(sys.argv[2]),len(train_index)):
		tempVector.append(train_index[j])
	saveVectorInt(str(indexCrossVal)+'train_index.txt',tempVector)
	indexCrossVal=indexCrossVal+1;

