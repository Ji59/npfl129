#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=40, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--range", default=3, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
										help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
	# Create the data
	xs = np.linspace(0, 7, num=args.data_size)
	ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

	rmses = []
	for order in range(1, args.range + 1):
		# TODO: Create features of x^1, ..., x^order.
		x = xs.reshape(len(xs), 1)
		for i in range(2, order + 1):
			x = np.c_[x, np.power(x[:, 0], i)]

		# TODO: Split the data into a train set and a test set.
		# Use `sklearn.model_selection.train_test_split` method call, passing
		# arguments `test_size=args.test_size, random_state=args.seed`.
		x_train, x_test, y_train, y_test = \
			sklearn.model_selection.train_test_split(x, ys, test_size=args.test_size, random_state=args.seed)

		# TODO: Fit a linear regression model using `sklearn.linear_model.LinearRegression`.
		model = sklearn.linear_model.LinearRegression().fit(x_train, y_train)

		# TODO: Predict targets on the test set using the trained model.
		y_predicted = model.predict(x_test)

		# TODO: Compute root mean square error on the test set predictions
		rmse = np.sqrt(np.sum(np.power(y_test - y_predicted, 2)) / len(y_test))

		rmses.append(rmse)

	return rmses


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	rmses = main(args)
	for order, rmse in enumerate(rmses):
		print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))
