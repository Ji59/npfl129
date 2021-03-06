#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")


# If you add more arguments, ReCodEx will keep them with your default values.


def get_poly_kernel(deg, gamma, x, y):
	out = 1;
	temp = gamma * x @ y + 1;
	for i in range(deg):
		out *= temp;
	return out;


def get_rbf_kernel(gamma, x, y):
	z = x - y;
	return np.exp(-gamma * z @ z);


def get_kernel(args, x, y):
	# TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
	# - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
	# - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}

	rbf = args.kernel == "rbf";
	deg = args.kernel_degree;
	gamma = args.kernel_gamma;
	out = np.zeros((len(x), len(y)));
	for i in range(len(x)):
		x_i = x[i]
		for j in range(len(y)):
			y_j = y[j]
			out[i, j] = get_rbf_kernel(gamma, x_i, y_j) if (rbf) else get_poly_kernel(deg, gamma, x_i, y_j);
	return out;


def get_prediction(weights, b, kernel):
	return kernel @ weights + b;


def get_prediction_v2(args, x, b, support_vectors, weights):
	kernel = get_kernel(args, [x], support_vectors);
	return np.sum(kernel @ weights) + b;


# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(args, train_data, train_target, test_data, test_target):
	# Create initial weights
	a, b, c = np.zeros(len(train_data)), 0, args.C;
	c_tol, tolerance = c - args.tolerance, args.tolerance;
	train_weights = np.zeros(len(a));

	generator = np.random.RandomState(args.seed)

	train_kernel = get_kernel(args, train_data, train_data);
	test_kernel = get_kernel(args, test_data, train_data);

	passes_without_as_changing = 0
	train_accs, test_accs = [], []
	for _ in range(args.max_iterations):
		as_changed = 0
		# Iterate through the data
		for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
			# We want j != i, so we "skip" over the value of i
			j = j + (j >= i)

			# TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.

			t_i = train_target[i];
			prediction_i = get_prediction(train_weights, b, train_kernel[i])
			a_i = a[i]
			if ((a_i <= tolerance or t_i * prediction_i <= 1 + tolerance) and (a_i >= c_tol or t_i * prediction_i >= 1 - tolerance)):
				continue;

			# If the conditions do not hold, then
			# - compute the updated unclipped a_j^new.
			#
			#   If the second derivative of the loss with respect to a[j]
			#   is > -`args.tolerance`, do not update a[j] and continue
			#   with next i.

			t_j = train_target[j];
			e_i, e_j = prediction_i - t_i, get_prediction(train_weights, b, train_kernel[j]) - t_j;

			train_kernel_ii, train_kernel_ij, train_kernel_jj = train_kernel[i, i], train_kernel[i, j], train_kernel[j, j]
			dL_da_j_2 = 2 * train_kernel_ij - train_kernel_ii - train_kernel_jj;
			if (dL_da_j_2 > -tolerance):
				continue;
			a_j = a[j]
			a_j_new = a_j - t_j * (e_i - e_j) / dL_da_j_2;

			# - clip the a_j^new to suitable [L, H].
			#
			#   If the clipped updated a_j^new differs from the original a[j]
			#   by less than `args.tolerance`, do not update a[j] and continue
			#   with next i.

			l, h = max(0, a_j - (c - a_i if (t_i == t_j) else a_i)), min(c, a_j + (a_i if (t_i == t_j) else c - a_i));
			a_j_clipped = max(l, min(h, a_j_new));
			a_j_diff = a_j_clipped - a_j;
			if (abs(a_j_diff) < tolerance):
				continue;

			# - update a[j] to a_j^new, and compute the updated a[i] and b.
			#
			#   During the update of b, compare the a[i] and a[j] to zero by
			#   `> args.tolerance` and to C using `< args.C - args.tolerance`.

			a[i] = a_i_new = a_i - t_i * t_j * a_j_diff;
			a[j] = a_j_clipped;

			a_i_diff = a_i_new - a_i;
			if (a_i_new > tolerance and a_i_new < c_tol):
				b -= e_i + t_i * a_i_diff * train_kernel_ii + t_j * a_j_diff * train_kernel_ij;
			elif (a_j_clipped > tolerance and a_j_clipped < c_tol):
				b -= e_j + t_i * a_i_diff * train_kernel_ij + t_j * a_j_diff * train_kernel_jj;
			else:
				sum_ = e_i + e_j + t_i * a_i_diff * (train_kernel_ii + train_kernel_ij) + t_j * a_j_diff * (train_kernel_ij + train_kernel_jj);
				b -= sum_ / 2;

			train_weights[i] = a_i_new * t_i;
			train_weights[j] = a_j_clipped * t_j;

			# - increase `as_changed`

			as_changed += 1;

		# TODO: After each iteration, measure the accuracy for both the
		# train set and the test set and append it to `train_accs` and `test_accs`.
		train_accs.append(np.sum(abs(train_target + [1 if p > 0 else -1 for p in get_prediction(train_weights, b, train_kernel)])) / (2 * len(train_target)));
		if (len(test_data) > 0):
			test_accs.append(np.sum(abs(test_target + [1 if p > 0 else -1 for p in get_prediction(train_weights, b, test_kernel)])) / (2 * len(test_target)));
		elif (len(test_accs) == 0):
			test_accs.append(0);

		# Stop training if max_passes_without_as_changing passes were reached
		passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
		if passes_without_as_changing >= args.max_passes_without_as_changing:
			break

		if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
			print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
				len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

	print("Training finished after iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
		len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

	# TODO: Create an array of support vectors (in the same order in which they appeared
	# in the training data; to avoid rounding errors, consider a training example
	# a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
	# Note that until now the full `a` should have been for prediction_i.
	support_vector_indexes = [i for i in range(len(a)) if a[i] > tolerance]
	support_vectors, support_vector_weights = np.array([train_data[i] for i in support_vector_indexes]), np.array([a[i] * train_target[i] for i in support_vector_indexes])

	return support_vectors, support_vector_weights, b, train_accs, test_accs


def main(args):
	# Generate an artifical regression dataset, with +-1 as targets
	data, target = sklearn.datasets.make_classification(
		n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
	target = 2 * target - 1

	# Split the dataset into a train set and a test set.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	# Run the SMO algorithm
	support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
		args, train_data, train_target, test_data, test_target)

	if args.plot:
		import matplotlib.pyplot as plt
		def plot(predict, support_vectors):
			xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
			ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
			predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
			test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
			plt.figure()
			plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
			plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
			plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
			plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
			plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
			plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
			plt.legend(loc="upper center", ncol=4)

		# If you want plotting to work (not required for ReCodEx), you need to
		# define `predict_function` computing SVM prediction `y(x)` for the given x.
		predict_function = lambda x: get_prediction_v2(args, x, bias, support_vectors, support_vector_weights)

		plot(predict_function, support_vectors)
		if args.plot is True:
			plt.show()
		else:
			plt.savefig(args.plot, transparent=True, bbox_inches="tight")

	return support_vectors, support_vector_weights, bias, train_accs, test_accs


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	main(args)
