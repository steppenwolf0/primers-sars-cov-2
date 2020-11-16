# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys

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
from sklearn.preprocessing import StandardScaler

# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# this is an incredibly useful function
from pandas import read_csv
import pandas as pd 
def loadDatasetOriginal(run) :
	
	# data used for the predictions
	dfData = read_csv("../data/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("../data/labels.csv", header=None)
	biomarkers = read_csv("../data/features_0.csv", header=None)

	# create folder
	folderName ="./run"+str(run)+"/"
	if not os.path.exists(folderName) : os.makedirs(folderName)

	pd.DataFrame(dfData.values).to_csv("./run"+str(run)+"/data_0.csv", header=None, index =None)
	pd.DataFrame(biomarkers.values.ravel()).to_csv("./run"+str(run)+"/features_0.csv", header=None, index =None)
	pd.DataFrame(dfLabels.values.ravel()).to_csv("./run"+str(run)+"/labels.csv", header=None, index =None)
	return dfData.values, dfLabels.values.ravel(), biomarkers.values.ravel() # to have it in the format that the classifiers like


def loadDataset(globalIndex, run) :
	
	# data used for the predictions
	dfData = read_csv("./run"+str(run)+"/data_"+str(globalIndex)+".csv", header=None, sep=',')
	dfLabels = read_csv("./run"+str(run)+"/labels.csv", header=None)
	biomarkers = read_csv("./run"+str(run)+"/features_"+str(globalIndex)+".csv", header=None)

	return dfData.values, dfLabels.values.ravel(), biomarkers.values.ravel() # to have it in the format that the classifiers like

# this function returns a list of features, in relative order of importance
def relativeFeatureImportance(classifier) :
	
	# this is the output; it will be a sorted list of tuples (importance, index)
	# the index is going to be used to find the "true name" of the feature
	orderedFeatures = []

	# the simplest case: the classifier already has a method that returns relative importance of features
	if hasattr(classifier, "feature_importances_") :

		orderedFeatures = zip(classifier.feature_importances_ , range(0, len(classifier.feature_importances_)))
		orderedFeatures = sorted(orderedFeatures, key = lambda x : x[0], reverse=True)
	
	# some classifiers are ensembles, and if each element in the ensemble is able to return a list of feature importances
	# (that are going to be all scored following the same logic, so they could be easily aggregated, in theory)
	elif hasattr(classifier, "estimators_") and hasattr(classifier.estimators_[0], "feature_importances_") :

		# add up the scores given by each estimator to all features
		global_score = np.zeros(classifier.estimators_[0].feature_importances_.shape[0])

		for estimator in classifier.estimators_ :
			for i in range(0, estimator.feature_importances_.shape[0]) :
				global_score[i] += estimator.feature_importances_[i]

		# "normalize", dividing by the number of estimators
		for i in range(0, global_score.shape[0]) : global_score[i] /= len(classifier.estimators_)

		# proceed as above to obtain the ranked list of features
		orderedFeatures = zip(global_score, range(0, len(global_score)))
		orderedFeatures = sorted(orderedFeatures, key = lambda x : x[0], reverse=True)
	
	# the classifier does not have "feature_importances_" but can return a list
	# of all features used by a lot of estimators (typical of ensembles)
	elif hasattr(classifier, "estimators_features_") :

		numberOfFeaturesUsed = 0
		featureFrequency = dict()
		for listOfFeatures in classifier.estimators_features_ :
			for feature in listOfFeatures :
				if feature in featureFrequency :
					featureFrequency[feature] += 1
				else :
					featureFrequency[feature] = 1
			numberOfFeaturesUsed += len(listOfFeatures)
		
		for feature in featureFrequency : 
			featureFrequency[feature] /= numberOfFeaturesUsed

		# prepare a list of tuples (name, value), to be sorted
		orderedFeatures = [ (featureFrequency[feature], feature) for feature in featureFrequency ]
		orderedFeatures = sorted(orderedFeatures, key=lambda x : x[0], reverse=True)

	# the classifier does not even have the "estimators_features_", but it's
	# some sort of linear/hyperplane classifier, so it does have a list of
	# coefficients; for the coefficients, the absolute value might be relevant
	elif hasattr(classifier, "coef_") :
	
		# now, "coef_" is usually multi-dimensional, so we iterate on
		# all dimensions, and take a look at the features whose coefficients
		# more often appear close to the top; but it could be mono-dimensional,
		# so we need two special cases
		dimensions = len(classifier.coef_.shape)
		#print("dimensions=", len(dimensions))
		featureFrequency = None # to be initialized later
		
		# check on the dimensions
		if dimensions == 1 :
			featureFrequency = np.zeros(len(classifier.coef_))
			
			relativeFeatures = zip(classifier.coef_, range(0, len(classifier.coef_)))
			relativeFeatures = sorted(relativeFeatures, key=lambda x : abs(x[0]), reverse=True)
			
			for index, values in enumerate(relativeFeatures) :
				value, feature = values
				featureFrequency[feature] += 1/(1+index)

		elif dimensions > 1 :
			featureFrequency = np.zeros(len(classifier.coef_[0]))
			
			# so, for each dimension (corresponding to a class, I guess)
			for i in range(0, len(classifier.coef_)) :
				# we give a bonus to the feature proportional to
				# its relative order, good ol' 1/(1+index)
				relativeFeatures = zip(classifier.coef_[i], range(0, len(classifier.coef_[i])))
				relativeFeatures = sorted(relativeFeatures, key=lambda x : abs(x[0]), reverse=True)
				
				for index, values in enumerate(relativeFeatures) :
					value, feature = values
					featureFrequency[feature] += 1/(1+index)
			
		# finally, let's sort
		orderedFeatures = [ (featureFrequency[feature], feature) for feature in range(0, len(featureFrequency)) ]
		orderedFeatures = sorted(orderedFeatures, key=lambda x : x[0], reverse=True)

	else :
		print("The classifier does not have any way to return a list with the relative importance of the features")

	return np.array(orderedFeatures)

def featureSelection(globalIndex, variableSize,run) :
	
	# a few hard-coded values
	numberOfFolds = 10
	
	
	# list of classifiers, selected on the basis of our previous paper "
	classifierList = [
			# ensemble
			#[AdaBoostClassifier(), "AdaBoostClassifier"],
			#[AdaBoostClassifier(n_estimators=300), "AdaBoostClassifier(n_estimators=300)"],
			#[AdaBoostClassifier(n_estimators=1500), "AdaBoostClassifier(n_estimators=1500)"],
			#[BaggingClassifier(), "BaggingClassifier"],
			
			[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			[LogisticRegression(), "LogisticRegression"],
			[PassiveAggressiveClassifier(),"PassiveAggressiveClassifier"],
			[SGDClassifier(), "SGDClassifier"],
			[SVC(kernel='linear'), "SVC(linear)"],
			[RidgeClassifier(), "RidgeClassifier"],
			[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],
			#[ExtraTreesClassifier(), "ExtraTreesClassifier"],
			#[ExtraTreesClassifier(n_estimators=300), "ExtraTreesClassifier(n_estimators=300)"],
			#[GradientBoostingClassifier(), "GradientBoostingClassifier"], # features_importances_
			#[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			#[GradientBoostingClassifier(n_estimators=1000), "GradientBoostingClassifier(n_estimators=1000)"],
			#[RandomForestClassifier(), "RandomForestClassifier"],
			#[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			#[RandomForestClassifier(n_estimators=1000), "RandomForestClassifier(n_estimators=1000)"], # features_importances_

			# linear
			#[ElasticNet(), "ElasticNet"],
			#[ElasticNetCV(), "ElasticNetCV"],
			#[Lasso(), "Lasso"],
			#[LassoCV(), "LassoCV"],
			#[LogisticRegression(), "LogisticRegression"], # coef_
			#[LogisticRegressionCV(), "LogisticRegressionCV"],
			#[PassiveAggressiveClassifier(), "PassiveAggressiveClassifier"], # coef_
			#[RidgeClassifier(), "RidgeClassifier"], # coef_
			#[RidgeClassifierCV(), "RidgeClassifierCV"],
			#[SGDClassifier(), "SGDClassifier"], # coef_
			#[SVC(kernel='linear'), "SVC(linear)"], # coef_, but only if the kernel is linear...the default is 'rbf', which is NOT linear
			
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
	#	   [RandomForestClassifier(), "RandomForestClassifier"]
	#	   ]

	print("Loading dataset...")
	if (globalIndex==0):
		X, y, biomarkerNames = loadDatasetOriginal(run)
	else:
		X, y, biomarkerNames = loadDataset(globalIndex, run)
	
	
	
	numberOfTopFeatures=int(variableSize)
	# create folder
	folderName ="./run"+str(run)+"/"
	if not os.path.exists(folderName) : os.makedirs(folderName)
	
	# prepare folds
	skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True)
	indexes = [ (training, test) for training, test in skf.split(X, y) ]
	
	# this will be used for the top features
	topFeatures = dict()
	
	# iterate over all classifiers
	classifierIndex = 0
	
	globalAccuracy=0;
	
	for originalClassifier, classifierName in classifierList :
		
		print("\nClassifier " + classifierName)
		classifierPerformance = []
		classifierTopFeatures = dict()

		# iterate over all folds
		for train_index, test_index in indexes :
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize, anyway
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			scoreTraining = classifier.score(X_train, y_train)
			scoreTest = classifier.score(X_test, y_test)
			
			print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )
			
			# now, let's get a list of the most important features, then mark the ones in the top X
			orderedFeatures = relativeFeatureImportance(classifier) 
			for i in range(0, numberOfTopFeatures) :
				
				feature = int(orderedFeatures[i][1])

				if feature in topFeatures :
					topFeatures[ feature ] += 1
				else :
					topFeatures[ feature ] = 1
				
				if feature in classifierTopFeatures :
					classifierTopFeatures[ feature ] += 1
				else :
					classifierTopFeatures[ feature ] = 1
			
		line ="%s\t%.4f\t%.4f\n" % (classifierName, np.mean(classifierPerformance), np.std(classifierPerformance))
		
		globalAccuracy=globalAccuracy+np.mean(classifierPerformance)
		
		print(line)
		fileName=folderName+"results.txt"
		fo = open(fileName, 'a')
		fo.write( line )
		fo.close()
		# save most important features for the classifier
		with open( os.path.join(folderName, classifierName + ".csv"), "w" ) as fp :
	
			fp.write("feature,frequencyInTop" + str(numberOfTopFeatures) + "\n")
			
			# transform dictionary into list
			listOfClassifierTopFeatures = [ (key, classifierTopFeatures[key]) for key in classifierTopFeatures ]
			listOfClassifierTopFeatures = sorted( listOfClassifierTopFeatures, key = lambda x : x[1], reverse=True )
			
			for feature, frequency in listOfClassifierTopFeatures :
				fp.write( str(biomarkerNames[feature]) + "," + str(float(frequency/numberOfFolds)) + "\n")
	
	# save most important features overall
	with open( os.path.join(folderName, "global_"+str(int(globalIndex))+".csv"), "w" ) as fp :
		
		fp.write("feature,frequencyInTop" + str(numberOfTopFeatures) + "\n")
		
		# transform dictionary into list
		listOfTopFeatures = [ (key, topFeatures[key]) for key in topFeatures ]
		listOfTopFeatures = sorted( listOfTopFeatures, key = lambda x : x[1], reverse=True )
		
		tempIndex=0
		for feature, frequency in listOfTopFeatures :
			if tempIndex<numberOfTopFeatures:
				fp.write( str(biomarkerNames[feature]) + "," + str(float(frequency/numberOfFolds)) + "\n")
			tempIndex=tempIndex+1
	globalAccuracy=globalAccuracy/8
	return globalAccuracy

