#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.


def gaussian(frequency_table):
	mean_var = {};
	for t in frequency_table:
		mean_var[t] = [];
		for i in range(len(frequency_table[t][1])):
			mean_var[t].append([0, 0]);
			for x in frequency_table[t][1][i]:
				mean_var[t][i][0] += frequency_table[t][1][i][x] * x;
			mean_var[t][i][0] /= frequency_table[t][0];
			mean = mean_var[t][i][0];
			for x in frequency_table[t][1][i]:
				diff = x - mean;
				mean_var[t][i][1] += frequency_table[t][1][i][x] * diff * diff / frequency_table[t][0];
			mean_var[t][i][1] = np.sqrt(mean_var[t][i][1] / frequency_table[t][0] + args.alpha);
	return mean_var;


def normal_distribution(x, mean, variance):
	# return scipy.stats.norm.logpdf(x, loc=mean, scale=variance);
	diff = x - mean
	variance_ = np.exp(-diff * diff / (2 * variance * variance)) / (np.sqrt(2 * np.pi) * variance)
	return np.log2(variance_)


def main(args):
	# Use the digits dataset.
	data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

	# Split the dataset into a train set and a test set.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	# TODO: Fit the naive Bayes classifier on the train data.
	#
	# The `args.naive_bayes_type` can be one of:
	# - "gaussian": Fit Gaussian NB, by estimating mean and variance of the input
	#   features. For variance estimation use
	#     1/N * \sum_x (x - mean)^2
	#   and additionally increase all estimated variances by `args.alpha`.
	# - "multinomial": Multinomial NB with smoothing factor `args.alpha`
	# - "bernoulli": Bernoulli NB with smoothing factor `args.alpha`

	frequency_table = {}
	targets = {}

	for i in range(len(train_data)):
		x = train_data[i];
		t = train_target[i];
		if t not in frequency_table:
			frequency_table[t] = [0, []]
		t_t = frequency_table[t]
		t_t[0] += 1
		for j in range(len(x)):
			if len(t_t[1]) <= j:
				frequency_table[t][1].append({});
			if x[j] not in t_t[1][j]:
				t_t[1][j][x[j]] = 0;
			t_t[1][j][x[j]] += 1;

	# TODO: Predict the test data classes and compute test accuracy.
	#
	# You can use `scipy.stats.norm` to compute probability density function
	# of a Gaussian distribution -- it offers `pdf` and `logpdf` methods, among
	# others.

	test_accuracy = 0;

	if args.naive_bayes_type == "gaussian":
		mean_var = gaussian(frequency_table);
		for i in range(len(test_data)):
			x = test_data[i];
			t = test_target[i];
			k = 0;
			maxim = -9999999999;
			for c_k in mean_var:
				probability = np.log(frequency_table[c_k][0] / len(train_target));
				for i in range(len(mean_var[c_k])):
					probability += normal_distribution(x[i], mean_var[c_k][i][0], mean_var[c_k][i][1]);
				if probability > maxim:
					maxim = probability;
					k = c_k;
			if t == k:
				test_accuracy += 1;

	return test_accuracy / len(test_target);


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	test_accuracy = main(args)

	print("Test accuracy {:.2f}%".format(100 * test_accuracy))
