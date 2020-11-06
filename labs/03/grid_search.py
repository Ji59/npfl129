#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.


def main(args):
	# Load digit dataset
	dataset = sklearn.datasets.load_digits()
	dataset.target = dataset.target % 2

	# If you want to learn about the dataset, uncomment the following line.
	# print(dataset.DESCR)

	# TODO: Split the dataset into a train set and a test set.
	# Use `sklearn.model_selection.train_test_split` method call, passing
	# arguments `test_size=args.test_size, random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

	# TODO: Create a pipeline, which
	# 1. performs sklearn.preprocessing.MinMaxScaler()
	# 2. performs sklearn.preprocessing.PolynomialFeatures()
	# 3. performs sklearn.linear_model.LogisticRegression(random_state=args.seed)
	#
	# Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
	# train performance of all combinations of the the following parameters:
	# - polynomial degree: 1, 2
	# - LogisticRegression regularization C: 0.01, 1, 100
	# - LogisticRegression solver: lbfgs, sag
	#
	# For the best combination of parameters, compute the test set accuracy.
	#
	# The easiest way is to use `sklearn.model_selection.GridSearchCV`.

	pipe = sklearn.pipeline.Pipeline([
		('scaler', sklearn.preprocessing.MinMaxScaler()),
		('poly', sklearn.preprocessing.PolynomialFeatures()),
		('logistic', sklearn.linear_model.LogisticRegression(random_state=args.seed))
	])

	parameters = {'poly__degree': [1, 2], 'logistic__C': [0.01, 1, 100], 'logistic__solver': ['lbfgs', 'sag']}

	model = sklearn.model_selection.GridSearchCV(pipe, parameters, n_jobs=-1, cv=sklearn.model_selection.StratifiedKFold(5), return_train_score=True)
	model.fit(train_data, train_target)

	pipe.set_params(**model.best_params_)
	pipe.fit(train_data, train_target)

	for (param, avg_score) in zip(model.cv_results_['params'], model.cv_results_['mean_train_score']):
		print(f'Cross-val: {avg_score * 100:>5.1f}% lr_C: {str(param["logistic__C"]):<4s} lr__solver: {param["logistic__solver"]:<5s} polynomial__degree: {str(param["poly__degree"])}')

	test_accuracy = pipe.score(test_data, test_target)

	return test_accuracy


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	test_accuracy = main(args)
	print("Test accuracy: {:.2f}".format(100 * test_accuracy))
