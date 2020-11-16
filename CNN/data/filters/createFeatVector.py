import numpy as np
import math
import random
import pandas as pd 

filterIndex=0
posMatrix = pd.read_csv("posPool_"+str(filterIndex)+".csv", header=None).values
matrix = pd.read_csv("sequencesSorted.csv", header=None).values.ravel()

max=0
for i in range (0,len(matrix)):
	if (len(matrix[i])>max):
		max=len(matrix[i])

vectorSize=max
print('vectorSize', vectorSize)

outData=[]
for i in range (len(matrix)):
	sample=np.zeros(vectorSize)
	for j in range (0, len(matrix[i])):
		if(matrix[i][j]=='C'):
			sample[j]=0.25
		elif(matrix[i][j]=='T'):
			sample[j]=0.50
		elif(matrix[i][j]=='G'):
			sample[j]=0.75
		elif(matrix[i][j]=='A'):
			sample[j]=1.0
		else:
			sample[j]=0.0
	outData.append(sample)
	
matrix=np.array(outData)

sizePosMatrix=np.shape(posMatrix)
numberFilters = 21
padding = 10
dataDNA = [[0 for i in range(210 * numberFilters)] for j in range(sizePosMatrix[0])] 
#dataDNA = np.array(shape=(sizePosMatrix[0],210 * numberFilters))

sizeDNAMatrix=np.shape(matrix)
temp=((matrix[0]))
#temp=(str(matrix[0]))
print(temp)
for i in range (0,sizePosMatrix[0]):
	temp=((matrix[i]))
	for j in range ( 0, sizePosMatrix[1]):
		coef=int(posMatrix[i][j])
		for k in range(0, padding+1):
			if ((coef + k) < len(temp)):
				dataDNA[i][ j * numberFilters + padding + k] = temp[ coef + k]
			if ((coef - k) >= 0 and (coef - k)< len(temp)):
				dataDNA[i][ j * numberFilters + padding - k] = temp[ coef - k]


dataDNAString = [[0 for i in range(210 * numberFilters)] for j in range(sizePosMatrix[0])] 

for i in range (0,sizePosMatrix[0]):	
	for j in range ( 0, 210 * numberFilters):
		if (dataDNA[i][ j] == 0.25):
		
			dataDNAString[i][ j] = "C"
		
		elif (dataDNA[i][ j] == 0.50):
		
			dataDNAString[i][ j] = "T"
		
		elif (dataDNA[i][ j] == 0.75):
		
			dataDNAString[i][ j] = "G"
		
		elif (dataDNA[i][ j] == 1.00):
		
			dataDNAString[i][ j] = "A"
		
		else:
		
			dataDNAString[i][ j] = "N"
		
dataDNAFeatures = [[0 for i in range(210)] for j in range(sizePosMatrix[0])] 
for i in range (0,sizePosMatrix[0]):
	for j in range(0,210):
		dataDNAFeatures[i][ j] =str("")

for i in range (0,sizePosMatrix[0]):
	indexFeature = 0
	feature = 0
	for j in range(0,210 * numberFilters):
		dataDNAFeatures[i][ feature] =str(dataDNAFeatures[i][ feature])+str(dataDNAString[i][ j])
		indexFeature=indexFeature+1
		if (indexFeature == numberFilters):
			feature=feature+1
			indexFeature = 0

featsVector=[]            
for i in range (0,sizePosMatrix[0]):
	for j in range(0,210):
		count=featsVector.count(dataDNAFeatures[i][j])
		if (count==0):
			if ("N" not in dataDNAFeatures[i][j]):
				featsVector.append(dataDNAFeatures[i][j])


		
		
#pd.DataFrame(dataDNA).to_csv("dataDNA_"+str(filterIndex)+".csv", header=None, index =None)
#pd.DataFrame(dataDNAString).to_csv("dataDNAString_"+str(filterIndex)+".csv", header=None, index =None)
pd.DataFrame(featsVector).to_csv("featsVector_"+str(filterIndex)+".csv", header=None, index =None)
pd.DataFrame(dataDNAFeatures).to_csv("dataDNAFeatures_"+str(filterIndex)+".csv", header=None, index =None)
		
