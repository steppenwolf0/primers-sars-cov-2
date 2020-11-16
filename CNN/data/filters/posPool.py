import numpy as np
import math
import random
import pandas as pd 

for filterIndex in range(0,12):
	
	data = pd.read_csv("filter_"+str(filterIndex)+".csv", header=None).values
	numberWindows = 210

	sizeData=np.shape(data)

	print(sizeData)

	maxPool = np.zeros(shape=(sizeData[0],numberWindows))
	posPool = np.zeros(shape=(sizeData[0],numberWindows))

	for  i in range(0, sizeData[0]):
		maxPool_windowSize = 148
		pad_left_HPool = 25
		max = -1e6
		index = pad_left_HPool
		position = -1
		indexMax = 0
		for j in range (0, sizeData[1]):
			if data[i][j]>max:
				max=data[i][j]
				position=j
			index=index+1
			if (index == maxPool_windowSize) or (j == sizeData[1] - 1):
				maxPool[i][indexMax] = max
				posPool[i][indexMax] = position
				max = -1e6
				position = -1
				index = 0
				indexMax=indexMax+1
				
	pd.DataFrame(maxPool).to_csv("maxPool_"+str(filterIndex)+".csv", header=None, index =None)
	pd.DataFrame(posPool).to_csv("posPool_"+str(filterIndex)+".csv", header=None, index =None)				
