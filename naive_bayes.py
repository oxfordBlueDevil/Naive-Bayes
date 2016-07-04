from sys import maxint
from collections import Counter
import math
import numpy as np
import pandas as pd

class GaussianNaiveBayes(object):

	def __init__(self, filename):
	    '''
	    INPUT:
	    - filename: csv filename for input data

	    OUTPUT: None

	    Initializes gnb classifier by loading the data
	    from the csv file.
	    '''
	    self._loadCsv(filename)
	    self.class_counts = None
	    self.class_priors = None
	    self.featuresByClass = None
	    self.featureSummariesByClass = None

	def _computeClassLogProbabilities(self, testVector):
		'''
	    INPUT:
	    - testVector: test data instance

	    OUTPUT:
	    - classLogProbabilities: dict containing log
	    probabilities (value) by class (key)

		Calculate the log probability of the entire 
		data instance testVector belonging to the class.
		'''
		classLogProbabilities = {}
		# classProbabilities = {}

		for classValue, classSummaries in self.featureSummariesByClass.iteritems():
			#compute prior
			# classProbabilities[classValue] = 1
			likelihoods = []
			# loop over each feature
			for i in range(len(classSummaries)):
				mean, stdev = classSummaries[i]
				x = testVector[i]
				likelihood = self._computeGaussianLikelihood(x, mean, stdev)
				likelihoods.append(likelihood)
				#multiply prior with the gaussian likelihood
				# classProbabilities[classValue] *= self.class_priors[classValue] * likelihood
			likelihoods = np.array(likelihoods)
			log_posterior = np.log(self.class_priors[classValue]) + np.dot(testVector, np.log(likelihoods))
			classLogProbabilities[classValue] = log_posterior

			return classLogProbabilities


	def _computeGaussianLikelihood(self, x, mean, stdev):
		'''
		INPUT:
		- x: datapoint
		- mean: mean of feature_j
		- stdev: std. dev. of feature_j

		OUTPUT: Gaussian Likelihood

		Calculate the gaussian likelihood density function of a feature.
		'''
		exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

	def _loadCsv(self, filename):
		'''
		INPUT:
	    - filename: csv filename for input data

	    OTUPUT: None
		Loads a CSV file using pandas.
		'''
		dataset = pd.read_csv(filename)
		self.y = dataset.pop('category').values.astype(int)
		self.X = dataset.values.astype(float)


	def _separateByClass(self):
		'''
		Separate the training dataset instances by 
		class value so that we can calculate statistics
		for each class.
		'''
		self.featuresByClass = {}

		for i in range(len(self.X)):
			vector = self.X[i]
			if self.y[i] not in self.featuresByClass:
				self.featuresByClass[self.y[i]] = []
			self.featuresByClass[self.y[i]].append(vector)

	def _summarize(self, vectors):
		'''
		Calculate the mean and the standard deviation 
		for each attribute.
		'''
		summaries = [(feature.mean(), feature.std()) for feature in vectors.T]
		return summaries
	 
	def _summarizeByClass(self):
		'''
		Calculate the statistic summaries for each feauture.
		'''
		self._separateByClass()
		summaries = {}

		for classValue, instances in self.featuresByClass.iteritems():
			summaries[classValue] = self._summarize(np.array(instances))

		return summaries

	def _predictTestVector(self, testVector):
		'''
		Calculate the probability of a data instance 
		belonging to each class value, we can look for 
		the largest probability and return the associated class.
		'''
		probabilities = self._computeClassLogProbabilities(testVector)
		bestLabel, bestProb = None, -maxint - 1
		for classValue, probability in probabilities.iteritems():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = classValue
		return bestLabel

	def fit(self):
		'''
		INPUT:
		- X: 2d numpy array, feature matrix
		- y: numpy array, labels

		OUTPUT: None

		Compute the class priors and then mean and std. dev.
        for each feature by class.
		'''
        # compute priors
		self.class_counts = Counter(self.y)
		self.class_priors = {}
		for classValue in self.class_counts:
			self.class_priors[classValue] = self.class_counts[classValue]/sum(self.class_counts.values())

		#compute mean and std. dev. for each feature by class
		self.featureSummariesByClass = self._summarizeByClass()
	    
	def predict(self, X_test):
		'''
		INPUT:
		- X_test: test datapoints

		OUTPUT:
		- predictions: classification for each data instance 
		in our test dataset

		Classifies the input data instances
		'''
		predictions = []
		for i in range(len(X_test)):
			result = self._predictTestVector(X_test[i])
			predictions.append(result)
		return np.array(predictions)

	def observe(self, newData, classValue):
		'''
		INPUT:
		- newData: new data instances
		- classValue: integer categories

		Adds the data to the classifier and revise its future predictions
		'''
		if newData.ndim == 1 and len(classValue) == 1:
			self.featuresByClass[classValue].append(newData)
		else:
			for i, dataVector in enumerate(newData):
				self.featuresByClass[classValue[i]].append(dataVector)
		#update feature summaries by class
		self.featureSummariesByClass = self._summarizeByClass()

	def score(self, y_test, y_predict):
		'''
		INPUT:
		- y_test: true labels
		- y_predict: predicted labels

		OUTPUT:
		- accuracy: accuracy ratio

		Calculate the accuracy of the gnb classifier.
		'''
		return np.sum(y_predict == y_test)/float(len(y_test))*100.0

if __name__ == '__main__':
	gnb = GaussianNaiveBayes('trainset.csv')
	X_test = np.loadtxt(open("X_test.csv", "rb"), delimiter=",")
	y_test = np.loadtxt(open("y_test.csv", "rb"), delimiter=",")

	print 'Fitting Gaussian Naive Bayes Classifier for Dataset'
	gnb.fit()
	y_predict = gnb.predict(X_test)
	print 'Accuracy: %s' %gnb.score(y_test, y_predict)


