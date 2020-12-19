#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrapping", default=False, action="store_true", help="Perform data bootstrapping")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=2, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")

# If you add more arguments, ReCodEx will keep them with your default values.


feature_subsampling = 1
generator = None
max_depth = None


class Node:
	def __init__(self, data, target, parent=None, depth=0):
		self.parent = parent
		self.left = None
		self.right = None
		self.data = data
		self.target = target
		self.depth = depth
		self.class_distribution = None
		self.criterion = None
		self.feature = None
		self.split_point = None

	def compute_classes(self):
		self.class_distribution = {}
		for t in self.target:
			if t not in self.class_distribution:
				self.class_distribution[t] = 1
			else:
				self.class_distribution[t] += 1

	def get_criterion(self):
		if self.class_distribution is None:
			self.compute_classes()
		if self.criterion is None:
			sum = 0
			for c in self.class_distribution:
				probability_c = self.class_distribution[c] / len(self.target)
				sum += probability_c * np.log2(probability_c)
			self.criterion = - sum * len(self.target)
		return self.criterion

	def split(self):
		if max_depth is not None and self.depth >= max_depth:
			return
		if self.get_criterion() == 0:
			return
		minimum = 999999
		number_of_features = self.data.shape[1]
		subsampling = generator.uniform(size=number_of_features) <= feature_subsampling
		for i in range(number_of_features):
			if not subsampling[i]:
				continue
			data = np.c_[self.data, self.target]
			data = data[data[:, i].argsort()]
			values = np.unique(data[:, i])
			split_index = 0
			for split_point in (values[:len(values) - 1] + values[1:]) / 2:
				while data[split_index, i] <= split_point:
					split_index += 1
				left = Node(data[:split_index, :number_of_features], data[:split_index, number_of_features], self, self.depth + 1)
				right = Node(data[split_index:, :number_of_features], data[split_index:, number_of_features], self, self.depth + 1)
				l_c = left.get_criterion()
				p_c = right.get_criterion()
				if minimum > l_c + p_c:
					minimum = l_c + p_c
					self.left = left
					self.right = right
					self.feature = i
					self.split_point = split_point
		return


def recursive_split(node):
	node.split()
	if node.left is not None:
		recursive_split(node.left)
		recursive_split(node.right)


def compute_accuracy(data, target, trees):
	accuracy = 0
	for i in range(len(target)):
		x = data[i]
		t = target[i]
		predictions = {}
		for root in trees:
			node = root
			while node.left is not None:
				if x[node.feature] <= node.split_point:
					node = node.left
				else:
					node = node.right
			maximum = -1
			chosen_class = 0
			for c in node.class_distribution:
				if node.class_distribution[c] > maximum:
					maximum = node.class_distribution[c]
					chosen_class = c
			if chosen_class not in predictions:
				predictions[chosen_class] = 1
			else:
				predictions[chosen_class] += 1
		maximum = -1
		prediction = 0
		for c in predictions:
			if predictions[c] > maximum:
				maximum = predictions[c]
				prediction = c
			elif predictions[c] == maximum and c < prediction:
				prediction = c
		if prediction == t:
			accuracy += 1
	return accuracy / len(target)


def main(args):
	# Use the wine dataset
	data, target = sklearn.datasets.load_wine(return_X_y=True)

	# Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
	# with `test_size=args.test_size` and `random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	global generator
	generator = np.random.RandomState(args.seed)
	global feature_subsampling
	feature_subsampling = args.feature_subsampling
	global max_depth
	max_depth = args.max_depth

	# TODO: Create a random forest on the trainining data.
	#
	# For determinism, create a generator
	#   generator = np.random.RandomState(args.seed)
	# at the beginning and then use this instance for all random number generation.
	#
	# Use a simplified decision tree from the `decision_tree` assignment:
	# - use `entropy` as the criterion
	# - use `max_depth` constraint, so split a node only if:
	#   - its depth is less than `args.max_depth`
	#   - the criterion is not 0 (the corresponding instance targetsare not the same)
	# When splitting nodes, proceed in the depth-first order, splitting all nodes
	# in left subtrees before nodes in right subtrees.
	#
	# Additionally, implement:
	# - feature subsampling: when searching for the best split, try only
	#   a subset of features. When splitting a node, start by generating
	#   a feature mask using
	#     generator.uniform(size=number_of_features) <= feature_subsampling
	#   which gives a boolean value for every feature, with `True` meaning the
	#   feature is used during best split search, and `False` it is not.
	#   (When feature_subsampling == 1, all features are used, but the mask
	#   should still be generated.)
	#
	# - train a random forest consisting of `args.trees` decision trees
	#
	# - if `args.bootstrapping` is set, right before training a decision tree,
	#   create a bootstrap sample of the training data using the following indices
	#     indices = generator.choice(len(train_data), size=len(train_data))
	#   and if `args.bootstrapping` is not set, use the original training data.
	#
	# During prediction, use voting to find the most frequent class for a given
	# input, choosing the one with smallest class index in case of a tie.

	trees = np.empty(args.trees, dtype=Node)
	i = 0
	while i < len(trees):
		if args.bootstrapping:
			indices = generator.choice(len(train_data), size=len(train_data))
			data = train_data[indices]
			target = train_target[indices]
		else:
			data = train_data
			target = train_target
		root = Node(data, target)
		recursive_split(root)
		trees[i] = root
		i += 1

	# TODO: Finally, measure the training and testing accuracy.
	train_accuracy = compute_accuracy(train_data, train_target, trees)
	test_accuracy = compute_accuracy(test_data, test_target, trees)

	return train_accuracy, test_accuracy


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	train_accuracy, test_accuracy = main(args)

	print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
	print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
