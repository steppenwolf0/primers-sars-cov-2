from features import *
from reduceData import *

from joblib import Parallel, delayed
import multiprocessing
#from runClassifiers import *
import time

def mainRun(indexRun) :
	run=indexRun
	start_time = time.time()
	globalAnt=0.0
	globalIndex=0
	globalAccuracy=0.0

	X, y, biomarkerNames = loadDatasetOriginal(run)
	
	if (int(len(y))>1000):
		numberOfTopFeatures = 1000
	else :
		numberOfTopFeatures = int(len(X[0])*0.80)

	variableSize=numberOfTopFeatures;
	while True:
		globalAnt=globalAccuracy
		globalAccuracy=featureSelection(globalIndex,variableSize, run)
		print(globalAccuracy)
		print(globalIndex)
		print(variableSize)
		size,sizereduced=reduceDataset(globalIndex, run)
		
		if((globalAccuracy<(0.50)) and (globalIndex!=0)):
			break
		if(variableSize==0):
			break
		variableSize=int(variableSize*0.80)
		
		globalIndex=globalIndex+1
	elapsed_time = time.time() - start_time
	print("time")
	print(elapsed_time)
	return

def main():
	threads=5
	totalRuns=10
	Parallel(n_jobs=threads, verbose=5, backend="multiprocessing")(delayed(mainRun)(i) for i in range(0,totalRuns))
	return

if __name__ == "__main__" :
	sys.exit( main() )
