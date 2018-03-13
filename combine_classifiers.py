import numpy as np
from preprocessing import *
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection

class CombineClassifiers():

	def __init__(self, classifiers, nsamples=24, stats=False, combine='mean'):
		"""
		We initialize with a list of tuples where the first element is a classifier and the second is the name of the classifier
		Whereas nsamples is a constant or a list. If a list, each element corresponds to the number of samples for the corresponding classifier

		combine_method: 'mean', 'median', 'regression'
						the way to compute the different predictions between the classifiers
						mean: compute the average probability
						median: the predicted class is the one in majority between the classifiers
						regression: train a LinearRegression to compute the weights to combine between the different classifiers
		"""

		self.classifiers = {}
		self.nsamples = {}
		self.nclassif = 0
		self.stats = stats
		self.combine = combine
		if combine == 'regression':
			self.regressor = LogisticRegression()

		for classifier, name in classifiers:
			self.classifiers[name] = classifier
			if type(nsamples)==int:
				self.nsamples[name] = nsamples
			elif len(nsamples)==len(classifiers):
				self.nsamples[name] = nsamples[self.nclassif]
			else:
				raise ValueError('nsamples has to be either an int or a list of the same length than classifiers')

			self.nclassif += 1
		
		
	def fit(self, X, y, valid_ratio=0, batch_size=32, epochs=10):
		"""
		provide X and y in the form of X, y = load_train() here
		fit all the classifiers individually here
		A validation set is used to train the linear regressor in case the combine method is regression
		In that case, you need to provide the size (in ratio) of the validation set
		"""

		self.dim = X.shape[1]//1440

		if valid_ratio>0:
			X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=valid_ratio)
		else:
			X_train, y_train = X, y

		#self.test_label_ = y

		if valid_ratio>0:
			preds = np.zeros((X_valid.shape[0], self.nclassif))
			i=0

		for k in self.classifiers.keys():
			X_train0 = subsampling(X_train, self.nsamples[k])
			if valid_ratio>0:
				X_valid0 = subsampling(X_valid, self.nsamples[k])

			if 'RNN' in k:
				self.classifiers[k].fit(X_train0, y_train, batch_size=batch_size, epochs=epochs, verbose=False)

			else:
				self.classifiers[k].fit(X_train0.reshape(X_train0.shape[0], -1), y_train)
				if valid_ratio>0:
					X_valid0 = X_valid0.reshape(X_valid0.shape[0], -1)

			if valid_ratio>0:
				preds[:, i] = self.classifiers[k].predict_proba(X_valid0)[:, -1].reshape(-1)
				i += 1

		if valid_ratio>0:
			self.regressor.fit(preds, y_valid)


	def predict(self, X):
		"""
		predict the classes for X
		"""

		preds = np.zeros((X.shape[0], self.nclassif))

		i = 0
		for k in self.classifiers.keys():
			X_test = subsampling(X, self.nsamples[k])
			if 'RNN' not in k:
				X_test = X_test.reshape(X_test.shape[0],  -1)

			preds[:, i] = self.classifiers[k].predict_proba(X_test)[:, -1].reshape(-1)
			i += 1

		if self.combine=='mean':
			preds = (np.mean(preds, axis=1)>0.5)
		if self.combine=='median':
			preds = (np.median(preds, axis=1)>0.5)
		if self.combine=='regression':
			preds = (self.regressor.predict(preds)>0.5)
		return preds.astype(int).reshape(-1, 1)
