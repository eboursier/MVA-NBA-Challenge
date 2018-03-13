import numpy as np
from preprocessing import *

class CombineClassifiers():

	def __init__(self, classifiers, nsamples=24, global_stats=False):
		"""
		We initialize with a list of tuples where the first element is a classifier and the second is the name of the classifier
		Whereas nsamples is a constant or a list. If a list, each element corresponds to the number of samples for the corresponding classifier

		For a 
		"""

		self.classifiers = {}
		self.nsamples = {}
		self.nclassif = 0
		self.global_stats = global_stats

		for classifier, name in classifiers:
			self.classifiers[name] = classifier
			if type(nsamples)==int:
				self.nsamples[name] = nsamples
			elif len(nsamples)==len(classifiers):
				self.nsamples[name] = nsamples[self.nclassif]
			else:
				raise ValueError('nsamples has to be either an int or a list of the same length than classifiers')

			self.nclassif += 1
		
		
	def fit(self, X, y):
		"""
		provide X and y in the form of X, y = load_train() here
		fit all the classifiers individually here
		"""
		for k in self.classifiers.keys():
			X_train, y_train = preprocessing(X, y, sampling_rate=1440//self.nsamples[k])

			if 'RNN' in k:
				X_train = X_train.reshape(X_train.shape[0], -1, self.nsamples[k])

			self.classifiers[k].fit(X_train, y_train)


	def predict(self, X):
		"""
		predict the classes for X
		"""
        
        preds = np.zeros((self.nclassif, X.shape[0]))
        
        i = 0
		for k in self.classifiers.keys():
			X_test, _ = preprocessing(X, _, sampling_rate=1440//self.nsamples[k])

			if 'RNN' in k:
				X_test = X_test.reshape(X_test.shape[0], -1, self.nsamples[k])

			preds[i] = self.classifiers[k].predict_proba(X_test)
            i += 1
            
        preds = (np.mean(preds, axis=0)>0.5)
        return preds