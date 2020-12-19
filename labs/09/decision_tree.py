#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")

# If you add more arguments, ReCodEx will keep them with your default values.


gini_index = True


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
		self.leaf = True

	def compute_classes(self):
		self.class_distribution = {}
		for t in self.target:
			if t not in self.class_distribution:
				self.class_distribution[t] = 1
			else:
				self.class_distribution[t] += 1

	def get_most_frequent_class(self):
		maximum = 0
		class_ = 0
		for c in self.class_distribution:
			count_c = self.class_distribution[c]
			if count_c > maximum:
				maximum = count_c
				class_ = c
			elif count_c == maximum and c < class_:
				class_ = c
		return class_

	def get_avg(self):
		return np.sum(self.target) / len(self.target)

	def get_criterion(self):
		if self.class_distribution is None:
			self.compute_classes()
		if self.criterion is None:
			self.criterion = compute_gini_index(self) if gini_index else compute_entropy_criterion(self)
		return self.criterion

	def split(self, max_depth, min_to_split):
		if (min_to_split is not None and len(self.target) < min_to_split) or (max_depth is not None and self.depth >= max_depth):
			return
		if self.get_criterion() == 0:
			return
		minimum = 999999
		features = self.data.shape[1]
		for i in range(features):
			data = np.c_[self.data, self.target]
			data = data[data[:, i].argsort()]
			values = np.unique(data[:, i])
			split_index = 0
			for split_point in (values[:len(values) - 1] + values[1:]) / 2:
				while data[split_index, i] <= split_point:
					split_index += 1
				left = Node(data[:split_index, :features], data[:split_index, features], self, self.depth + 1)
				right = Node(data[split_index:, :features], data[split_index:, features], self, self.depth + 1)
				l_c = left.get_criterion()
				p_c = right.get_criterion()
				if minimum > l_c + p_c:
					minimum = l_c + p_c
					self.left = left
					self.right = right
					self.feature = i
					self.split_point = split_point
		return


def split_array(data, index, split_value):
	l = np.zeros(data.shape())
	r = np.zeros(data.shape())
	li = 0
	ri = 0
	for d in data:
		if d[index] <= split_value:
			l[li] = d
			li += 1
		else:
			r[ri] = d
			ri += 1
	return l, r


def recursive_split(node, max_depth, min_to_split):
	node.split(max_depth, min_to_split)
	if node.left is not None:
		node.leaf = False
		recursive_split(node.left, max_depth, min_to_split)
		recursive_split(node.right, max_depth, min_to_split)


def weird_split(root, max_depth, min_to_split, max_leaves):
	leaves = []
	leaves_n = 0
	queue = [root]
	while len(queue) > 0:
		node = queue[0]
		if node.leaf:
			node.split(max_depth, min_to_split)
			if node.left is not None:
				leaves.append(node)
			leaves_n += 1
		else:
			queue.append(node.left)
			queue.append(node.right)
		queue.remove(node)
	node_ = None
	value = 999999
	for node in leaves:
		diff = node.left.get_criterion() + node.right.get_criterion() - node.get_criterion()
		if diff < value:
			node_ = node
			value = diff
	node_.leaf = False
	if max_leaves > leaves_n:
		weird_split(root, max_depth, min_to_split, max_leaves - 1)


def compute_gini_index(node):
	sum = 0
	for c in node.class_distribution:
		probability_c = node.class_distribution[c] / len(node.target)
		sum += probability_c * (1 - probability_c)
	return sum * len(node.target)


def compute_entropy_criterion(node):
	sum = 0
	for c in node.class_distribution:
		probability_c = node.class_distribution[c] / len(node.target)
		sum += probability_c * np.log2(probability_c)
	return - sum * len(node.target)


def compute_accuracy(data, target, root):
	accuracy = 0
	for i in range(len(target)):
		x = data[i]
		t = target[i]
		node = root
		while not node.leaf:
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
		if chosen_class == t:
			accuracy += 1
	return accuracy / len(target)


def main(args):
	# Use the wine dataset
	data, target = sklearn.datasets.load_wine(return_X_y=True)

	# Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
	# with `test_size=args.test_size` and `random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	tree = Node(train_data, train_target)

	if args.criterion != 'gini':
		global gini_index
		gini_index = False

	# TODO: Create a decision tree on the trainining data.
	#
	# - For each node, predict the most frequent class (and the one with
	#   smallest index if there are several such classes).
	#
	# - When splitting a node, consider the features in sequential order, then
	#   for each feature consider all possible split points ordered in ascending
	#   value, and perform the first encountered split descreasing the criterion
	#   the most. Each split point is an average of two nearest unique feature values
	#   of the instances corresponding to the given node (i.e., for four instances
	#   with values 1, 7, 3, 3 the split points are 2 and 5).
	#
	# - Allow splitting a node only if:
	#   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
	#     depth of the root node is zero;
	#   - there are at least `args.min_to_split` corresponding instances;
	#   - the criterion value is not zero.
	#
	# - When `args.max_leaves` is None, use recursive (left descendants first, then
	#   right descendants) approach, splitting every node if the constraints are valid.
	#   Otherwise (when `args.max_leaves` is not None), always split a node where the
	#   constraints are valid and the overall criterion value (c_left + c_right - c_node)
	#   decreases the most. If there are several such nodes, choose the one
	#   which was created sooner (a left child is considered to be created
	#   before a right child).

	if args.max_leaves is None:
		recursive_split(tree, args.max_depth, args.min_to_split)
	else:
		weird_split(tree, args.max_depth, args.min_to_split, args.max_leaves)

	# TODO: Finally, measure the training and testing accuracy.
	train_accuracy = compute_accuracy(train_data, train_target, tree)
	test_accuracy = compute_accuracy(test_data, test_target, tree)

	return train_accuracy, test_accuracy


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	train_accuracy, test_accuracy = main(args)

	print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
	print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
