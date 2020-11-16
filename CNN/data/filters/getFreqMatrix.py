import numpy as np
import math
import random
import pandas as pd 

filterIndex=0
vector = pd.read_csv("features_"+str(filterIndex)+".csv", header=None).values.ravel()
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

pd.DataFrame(freqMatrix).to_csv("data_"+str(filterIndex)+".csv", header=None, index =None)


