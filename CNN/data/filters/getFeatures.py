import numpy as np
import math
import random
import pandas as pd 

filterIndex=0
vector = pd.read_csv("featsVector_"+str(filterIndex)+".csv", header=None).values.ravel()
sequences = pd.read_csv("../sequences.csv", header=None).values.ravel()

print(len(sequences))
print(len(vector))

sizeSeq=int(len(sequences))
sizeVec=int(len(vector))

freqMatrix = np.zeros((sizeSeq,sizeVec))

for i in range (0,sizeSeq):	
	for j in range ( 0, sizeVec):
		if (vector[j] in sequences[i]):
			freqMatrix[i][j]=sequences[i].count(vector[j])
		else:
			freqMatrix[i][j]=0

#pd.DataFrame(freqMatrix).to_csv("data_"+str(filterIndex)+".csv", header=None, index =None)

repeatedList=[]
for i in range (0,sizeSeq):	
	for j in range ( 0, sizeVec):
		if (freqMatrix[i][j]>1):
			if(vector[j] not in repeatedList):
				repeatedList.append(vector[j])

print(repeatedList)

nonRepeatedList=[]
for i in range ( 0, sizeVec):
	if (vector[i] not in repeatedList):
		nonRepeatedList.append(vector[i])

print("Repeated in the same sample: "+str(len(repeatedList)))
print("Non repeated in the same sample: "+str(len(nonRepeatedList)))


sizeSeq=int(len(sequences))
sizeVec=int(len(nonRepeatedList))

freqMatrix = np.zeros((sizeSeq,sizeVec))

for i in range (0,sizeSeq):	
	for j in range ( 0, sizeVec):
		if (nonRepeatedList[j] in sequences[i]):
			freqMatrix[i][j]=sequences[i].count(nonRepeatedList[j])
		else:
			freqMatrix[i][j]=0
			
pd.DataFrame(nonRepeatedList).to_csv("features_"+str(filterIndex)+".csv", header=None, index =None)
pd.DataFrame(freqMatrix).to_csv("data_"+str(filterIndex)+".csv", header=None, index =None)