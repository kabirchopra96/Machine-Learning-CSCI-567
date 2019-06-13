import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		features = np.array(features)
		N, D = features.shape
		p = np.zeros((N, 1))
		a = []
		for i in range(len(self.clfs_picked)):
			clfs_picked_predict = np.array(self.clfs_picked[i].predict(features))
			p += self.betas[i] * np.reshape(clfs_picked_predict,(len(clfs_picked_predict), 1))
		for i in p:
			if i>=0:
				a.append(1)
			else:
				a.append(-1)
		return a


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
	def comp(self, n1: int, n2: int) -> int:
		if n2 != n1:
			return 1
		else:
			return 0

	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		features = np.array(features)
		N, D = features.shape
		w = np.full((N, 1), 1.0 / N, dtype=float)
		for iter in range(0, self.T - 1):
			e = 10000.0
			for j in self.clfs:
				predict = j.predict(features)
				e1 = 0.0
				for i in range(N):
					e1 += w[i] * self.comp(labels[i], predict[i])
				if e1 < e:
					e = e1
					clfs_picked = j
			self.clfs_picked.append(clfs_picked)
			beta = 0.5 * np.log((1.0 - e) / e)
			self.betas.append(beta)
			pred = clfs_picked.predict(features)
			for i in range(N):
				if labels[i] == pred[i]:
					w[i] = w[i] * np.exp(-beta)
				else:
					w[i] = w[i] * np.exp(beta)
			w = np.divide(w, np.sum(w))

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	