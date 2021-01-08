#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=57, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")

# If you add more arguments, ReCodEx will keep them with your default values.


learning_rate = 1
max_depth = None
l2 = 1.


class Node:
	def __init__(self, data, target, parent=None, depth=0, g=None, h=None):
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
		self.weight = None
		self.g = g
		self.h = h

	def get_weight(self):
		if self.weight is None:
			self.weight = - np.sum(self.g) / (l2 + np.sum(self.h))
		return self.weight

	def get_criterion(self):
		if self.criterion is None:
			self.criterion = get_criterion(self.g, self.h)
		return self.criterion

	def split(self):
		g_sum, h_sum = np.sum(self.g), np.sum(self.h)
		self.get_weight()
		self.get_criterion()
		if max_depth is not None and self.depth >= max_depth:
			return
		if len(self.data) <= 1:
			return
		score = 99999999999
		number_of_features = self.data.shape[1]
		# loss_value = g_sum ** 2 / (h_sum + l2)
		for i_feature in range(number_of_features):
			g_l, h_l, g_r, h_r = 0, 0, g_sum, h_sum
			# data = np.c_[self.data, self.target]
			argsort = self.data[:, i_feature].argsort()
			data = self.data[argsort]
			g, h = self.g[argsort], self.h[argsort]
			# values = np.unique(data[:, i_feature])
			for split_index in range(len(data) - 1):
				g_l += g[split_index]
				g_r -= g[split_index]
				h_l += h[split_index]
				h_r -= h[split_index]
				split_index_next = split_index + 1
				data_feature, data_feature_next = data[split_index, i_feature], data[split_index_next, i_feature]
				if data_feature == data_feature_next:
					continue
				# while data[split_index, i_feature] <= split_index:
				# 	split_index += 1
				# new_score = g_l**2 / (h_l + l2) + g_r ** 2 / (h_r + l2) - loss_value
				g_index_to, h_index_to = g[:split_index_next], h[:split_index_next]
				g_index_from, h_index_from = g[split_index_next:], h[split_index_next:]
				l_c = get_criterion(g_index_to, h_index_to)
				p_c = get_criterion(g_index_from, h_index_from)
				if score > l_c + p_c + 0.0001:
					score = l_c + p_c
					self.left = Node(data[:split_index_next], None, self, self.depth + 1, g_index_to, h_index_to)
					self.right = Node(data[split_index_next:], None, self, self.depth + 1, g_index_from, h_index_from)
					self.feature = i_feature
					self.split_point = (data_feature + data_feature_next) / 2

		self.left.split()
		self.right.split()
		return


def get_criterion(g, h):
	return - 1 / 2 * (np.sum(g) ** 2) / (np.sum(h) + l2)


def predict(x, trees):
	prediction = 0
	for root in trees:
		node = root
		while node.left is not None:
			if x[node.feature] <= node.split_point:
				node = node.left
			else:
				node = node.right
		prediction += node.weight
	return prediction * learning_rate


def compute_accuracy(data, target, trees):
	accuracy = np.zeros(trees.shape[1])

	for i_data in range(len(data)):
		for t in range(trees.shape[1]):
			predictions = np.zeros(len(trees))
			for c in range(len(trees)):
				predictions[c] = predict(data[i_data], trees[c, :t + 1])
			prediction = np.argmax(predictions)
			if target[i_data] == prediction:
				accuracy[t] += 1

	return accuracy / len(target)


def get_gi_hi(predictions_x, target, i_class):
	sum_c = np.sum(np.exp(predictions_x))
	c_pred = np.exp(predictions_x[i_class])
	softmax_c = c_pred / sum_c
	gi = softmax_c
	if target == i_class:
		gi = softmax_c - 1
	hi = softmax_c * (1 - softmax_c)
	return gi, hi


def main(args):
	# Use the given dataset
	data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

	global learning_rate
	learning_rate = args.learning_rate
	global max_depth
	max_depth = args.max_depth
	global l2
	l2 = args.l2

	# Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
	# with `test_size=args.test_size` and `random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	classes = np.max(target) + 1

	# TODO: Create a gradient boosted trees on the classification training data.
	#
	# Notably, train for `args.trees` iteration. During iteration `t`:
	# - the goal is to train `classes` regression trees, each predicting
	#   raw weight for the corresponding class.
	# - compute the current predictions `y_t(x_i)` for every training example `i` as
	#     y_t(x_i)_c = \sum_{j=1}^t args.learning_rate * tree_{iter=j,class=c}.predict(x_i)
	#     (note that y_0 is zero)
	# - loss in iteration `t` is
	#     L = (\sum_i NLL(target_i, softmax(y_{t-1}(x_i) + trees_to_train_in_iter_t.predict(x_i)))) +
	#         1/2 * args.l2 * (sum of all node values in trees_to_train_in_iter_t)
	# - for every class `c`:
	#   - start by computing `g_i` and `h_i` for every training example `i`;
	#     the `g_i` is the first derivative of NLL(target_i_c, softmax(y_{t-1}(x_i))_c)
	#     with respect to y_{t-1}(x_i)_c, and the `h_i` is the second derivative of the same.
	#   - then, create a decision tree minimizing the above loss L. According to the slides,
	#     the optimum prediction for a given node T with training examples I_T is
	#       w_T = - (\sum_{i \in I_T} g_i) / (args.l2 + sum_{i \in I_T} h_i)
	#     and the value of the loss with the above prediction is
	#       c_GB = - 1/2 (\sum_{i \in I_T} g_i)^2 / (args.l2 + sum_{i \in I_T} h_i)
	#     which you should use as a splitting criterion.
	#
	# During tree construction, we split a node if:
	# - its depth is less than `args.max_depth`
	# - there is more than 1 example corresponding to it (this was covered by
	#     a non-zero criterion value in the previous assignments)

	forest = np.empty(shape=[classes, args.trees], dtype=Node)
	# for i_class in range(classes):
	# 	forest[i_class][0] = Node(train_data, train_target)

	for t in range(args.trees):
		predictions = np.empty(shape=(classes, len(train_data)))
		for i_class in range(classes):
			for i_data in range(len(train_data)):
				x = train_data[i_data]
				predictions[i_class, i_data] = predict(x, forest[i_class, :t])
		a = 0
		for i_class in range(classes):
			g, h = np.empty(len(train_data)), np.empty(len(train_data))
			for i_data in range(len(train_data)):
				g[i_data], h[i_data] = get_gi_hi(predictions[:, i_data], train_target[i_data], i_class)

			new_tree_root = Node(train_data, train_target, g=g, h=h)
			new_tree_root.split()
			forest[i_class, t] = new_tree_root

	# TODO: Finally, measure your training and testing accuracies when
	# using 1, 2, ..., `args.trees` of the created trees.
	#
	# To perform a prediction using t trees, compute the y_t(x_i) and return the
	# class with the highest value (and the smallest class if there is a tie).
	train_accuracies = compute_accuracy(train_data, train_target, forest)
	test_accuracies = compute_accuracy(test_data, test_target, forest)

	return train_accuracies, test_accuracies


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	train_accuracies, test_accuracies = main(args)

	for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
		print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
			i + 1, 100 * train_accuracy, 100 * test_accuracy))
