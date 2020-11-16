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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
# this is an incredibly useful function
from pandas import read_csv

from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from scipy import stats
def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./best/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./best/labels.csv", header=None)
		
	return dfData.as_matrix(), dfLabels.as_matrix().ravel() # to have it in the format that the classifiers like


def runFeatureReduce() :
	
	# a few hard-coded values
	numberOfFolds = 10
	
	# list of classifiers, selected on the basis of our previous paper "
	classifierList = [
		
			[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			[LogisticRegression(solver='lbfgs',), "LogisticRegression"],
			[PassiveAggressiveClassifier(),"PassiveAggressiveClassifier"],
			[SGDClassifier(), "SGDClassifier"],
			[SVC(kernel='linear'), "SVC(linear)"],
			[RidgeClassifier(), "RidgeClassifier"],
			[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],
			# ensemble
			#[AdaBoostClassifier(), "AdaBoostClassifier"],
			#[AdaBoostClassifier(n_estimators=300), "AdaBoostClassifier(n_estimators=300)"],
			#[AdaBoostClassifier(n_estimators=1500), "AdaBoostClassifier(n_estimators=1500)"],
			#[BaggingClassifier(), "BaggingClassifier"],
			
			#[ExtraTreesClassifier(), "ExtraTreesClassifier"],
			#[ExtraTreesClassifier(n_estimators=300), "ExtraTreesClassifier(n_estimators=300)"],
			 # features_importances_
			#[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			#[GradientBoostingClassifier(n_estimators=1000), "GradientBoostingClassifier(n_estimators=1000)"],
			
			#[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			#[RandomForestClassifier(n_estimators=1000), "RandomForestClassifier(n_estimators=1000)"], # features_importances_

			# linear
			#[ElasticNet(), "ElasticNet"],
			#[ElasticNetCV(), "ElasticNetCV"],
			#[Lasso(), "Lasso"],
			#[LassoCV(), "LassoCV"],
			 # coef_
			#[LogisticRegressionCV(), "LogisticRegressionCV"],
			  # coef_
			 # coef_
			#[RidgeClassifierCV(), "RidgeClassifierCV"],
			 # coef_
			 # coef_, but only if the kernel is linear...the default is 'rbf', which is NOT linear
			
			# naive Bayes
			#[BernoulliNB(), "BernoulliNB"],
			#[GaussianNB(), "GaussianNB"],
			#[MultinomialNB(), "MultinomialNB"],
			
			# neighbors
			#[KNeighborsClassifier(), "KNeighborsClassifier"], # no way to return feature importance
			# TODO this one creates issues
			#[NearestCentroid(), "NearestCentroid"], # it does not have some necessary methods, apparently
			#[RadiusNeighborsClassifier(), "RadiusNeighborsClassifier"],
			
			# tree
			#[DecisionTreeClassifier(), "DecisionTreeClassifier"],
			#[ExtraTreeClassifier(), "ExtraTreeClassifier"],

			]
	
	# this is just a hack to check a few things
	#classifierList = [
	#		[RandomForestClassifier(), "RandomForestClassifier"]
	#		]

	print("Loading dataset...")
	X, y = loadDataset()
	
	print(len(X))
	print(len(X[0]))
	print(len(y))

	
	labels=np.max(y)+1
	# prepare folds
	skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True)
	indexes = [ (training, test) for training, test in skf.split(X, y) ]
	
	# this will be used for the top features
	topFeatures = dict()
	
	# iterate over all classifiers
	classifierIndex = 0
	
	
	
	
	for originalClassifier, classifierName in classifierList :
		
		print("\nClassifier " + classifierName)
		classifierPerformance = []

		cMatrix=np.zeros((labels, labels))
		# iterate over all folds
		
		indexFold = 0

		yTest=[]
		yNew=[]

		for train_index, test_index in indexes :
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize, anyway
			# MinMaxScaler StandardScaler Normalizer
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

		
			
			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			scoreTraining = classifier.score(X_train, y_train)
			scoreTest = classifier.score(X_test, y_test)
			
			y_new = classifier.predict(X_test)
			
			
			
			yNew.append(y_new)
			yTest.append(y_test)
			
			for i in range(0,len(y_new)):
				cMatrix[y_test[i]][y_new[i]]+=1


			print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )

		pd.DataFrame(cMatrix).to_csv("./best/cMatrix"+str(classifierIndex)+".csv", header=None, index =None)
		classifierIndex+=1
		line ="%s \t %.4f \t %.4f \n" % (classifierName, np.mean(classifierPerformance), np.std(classifierPerformance))
		
		
		print(line)
		fo = open("./best/results.txt", 'a')
		fo.write( line )
		fo.close()
	
	return

if __name__ == "__main__" :
	sys.exit( runFeatureReduce() )