#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x),
										help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
	# Load Boston housing dataset
	dataset = sklearn.datasets.load_boston()

	# The input data are in dataset.data, targets are in dataset.target.

	# If you want to learn about the dataset, uncomment the following line.
	# print(dataset.DESCR)

	# TODO: Append a new feature to all input data, with value "1"
	X = np.c_[dataset.data, np.ones((len(dataset.data), 1))]

	# TODO: Split the dataset into a train set and a test set.
	# Use `sklearn.model_selection.train_test_split` method call, passing
	# arguments `test_size=args.test_size, random_state=args.seed`.
	X_train, X_test, t_train, t_test = \
		sklearn.model_selection.train_test_split(X, dataset.target, test_size=args.test_size, random_state=args.seed)
	# TODO: Solve the linear regression using the algorithm from the lecture,
	# explicitly computing the matrix inverse (using `np.linalg.inv`).

	w = np.linalg.inv(np.transpose(X_train) @ X_train) @ np.transpose(X_train) @ t_train

	# TODO: Compute root mean square error on the test set predictions
	rmse = np.sqrt(np.sum(np.power(X_test @ w - t_test, 2)) / len(t_test))

	return rmse


# TODO: Predict target values on the test set

if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	rmse = main(args)
	print("{:.2f}".format(rmse))
