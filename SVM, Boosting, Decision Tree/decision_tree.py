import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			c_e = 0
			for i in range(len(branches[0])):
				e = 0
				for j in range(len(branches)):
					t = branches[j][i]/np.array(branches)[:,i].sum()
					if t != 0:
						e += -t*np.log2(t)			
				c_e += e*np.array(branches)[:,i].sum()/np.array(branches).sum()
			return c_e

		min_e = 99999
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			B = np.unique(np.array(self.features)[:,idx_dim]).tolist()
			C = np.unique(self.labels).tolist()
			branches = np.zeros((len(C),len(B)))
			for n in range(len(self.features)):
				branches[C.index(self.labels[n])][B.index(self.features[n][idx_dim])] += 1
			e = conditional_entropy(branches)
			if e < min_e:
				min_e = e
				self.dim_split = idx_dim
				self.feature_uniq_split = B
		if self.dim_split == None:
			self.splittable = False
			self.feature_uniq_split = None
			return



		############################################################
		# TODO: split the node, add child nodes
		############################################################
		self.children = []
		for i in self.feature_uniq_split:
			features = []
			labels = []
			for n in range(len(self.features)):
				if self.features[n][self.dim_split] == i:
					features.append(self.features[n][:self.dim_split]+self.features[n][self.dim_split+1:])
					labels.append(self.labels[n])
			self.children.append(TreeNode(features, labels, np.max(labels)+1))



		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



