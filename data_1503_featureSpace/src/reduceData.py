# this is an incredibly useful function
from pandas import read_csv
import numpy as np

import pandas as pd 

def reduceDataset(globalIndex, run) :
	
	# data used for the predictions
	if(globalIndex==0):
		dfData = read_csv("../data/data_"+str(globalIndex)+".csv", header=None, sep=',')
		ids=read_csv("../data/features_"+str(globalIndex)+".csv", header=None, sep=',')
	else:
		dfData = read_csv("./run"+str(run)+"/data_"+str(globalIndex)+".csv", header=None, sep=',')
		ids=read_csv("./run"+str(run)+"/features_"+str(globalIndex)+".csv", header=None, sep=',')
	
	idsRed=read_csv("./run"+str(run)+"/global_"+str(globalIndex)+".csv", sep=',')
	
	data=dfData.values
	idsRed=idsRed.values
	
	ids=ids.values

	print("data Y %d" %(len(data)))
	print("data X %d" %(len(data[0])))
	print(len(ids))
	print(len(idsRed))
	
	tempIds=[]
	for i in range(0,len(idsRed)):
		if (idsRed[i,1]>=1):
			tempIds.append(idsRed[i,0])
	print(len(tempIds))
	count=0
	
	dataRed=np.zeros((len(data),len(tempIds)))

	for i in range(0,len(tempIds)):
		for j in range (0,len(ids)):
			if (ids[j]== tempIds[i]):
				count=count+1
				for k in range(0,len(data)):
					dataRed[k,i]=data[k,j]
	
	
	pd.DataFrame(tempIds).to_csv("./run"+str(run)+"/features_"+str(globalIndex+1)+".csv", header=None, index =None)
	pd.DataFrame(dataRed).to_csv("./run"+str(run)+"/data_"+str(globalIndex+1)+".csv", header=None, index =None)			
	print(count)
	
	return len(ids),len(tempIds)
