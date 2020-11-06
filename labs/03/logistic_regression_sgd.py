#!/usr/bin/env python3
import argparse
import math
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
										help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
	# Create a random generator with a given seed
	generator = np.random.RandomState(args.seed)

	# Generate an artifical regression dataset
	data, target = sklearn.datasets.make_classification(
		n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

	# TODO: Append a constant feature with value 1 to the end of every input data

	data = np.c_[data, np.ones((len(data), 1))]

	# TODO: Split the dataset into a train set and a test set.
	# Use `sklearn.model_selection.train_test_split` method call, passing
	# arguments `test_size=args.test_size, random_state=args.seed`.

	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

	# Generate initial linear regression weights
	weights = generator.uniform(size=train_data.shape[1])

	for iteration in range(args.iterations):
		permutation = generator.permutation(train_data.shape[0])

		# TODO: Process the data in the order of `permutation`.
		# For every `args.batch_size`, average their gradient, and update the weights.
		# You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

		for j in range(int(train_data.shape[0] / args.batch_size)):

			gradient = np.zeros(weights.shape[0])

			for k in range(args.batch_size):
				i = permutation[j * args.batch_size + k]
				if train_target[i] == 0:
					gradient -= train_data[i] * math.pow(math.e, train_data[i] @ weights) / (1 + math.pow(math.e, train_data[i] @ weights))
				else:
					gradient += train_data[i] / (1 + math.pow(math.e, train_data[i] @ weights))
			gradient = - gradient / args.batch_size
			weights -= args.learning_rate * gradient

		# TODO: After the SGD iteration, measure the average loss and accuracy for both the
		# train test and the test set. The loss is the average MLE loss (i.e., the
		# negative log likelihood, or crossentropy loss, or KL loss) per example.

		train_loss = 0
		for i in range(train_data.shape[0]):
			sigmoid = 1 / (1 + np.exp(- np.transpose(train_data[i]) @ weights))
			if train_target[i] == 0:
				train_loss += np.log(1 - sigmoid)
			else:
				train_loss += np.log(sigmoid)
		train_loss /= -train_data.shape[0]
		
		test_loss = 0
		for i in range(test_data.shape[0]):
			sigmoid = 1 / (1 + np.exp(- np.transpose(test_data[i]) @ weights))
			if test_target[i] == 0:
				test_loss += np.log(1 - sigmoid)
			else:
				test_loss += np.log(sigmoid)
		test_loss /= -test_data.shape[0]

		train_accuracy = 1 - np.sum(np.square(train_target - [0 if i <= 0 else 1 for i in train_data @ weights])) / train_data.shape[0]
		test_accuracy = 1 - (np.sum(np.square(test_target - [0 if i <= 0 else 1 for i in test_data @ weights])) / test_data.shape[0])


		print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(iteration + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

		if args.plot:
			import matplotlib.pyplot as plt

			if args.plot is not True:
				if not iteration: plt.figure(figsize=(6.4 * 3, 4.8 * (args.iterations + 2) // 3))
				plt.subplot(3, (args.iterations + 2) // 3, 1 + iteration)
			xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
			ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
			predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
			plt.contourf(xs, ys, predictions, levels=21, cmap=plt.cm.RdBu, alpha=0.7)
			plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
			plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
			plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
			plt.legend(loc="upper right")
			if args.plot is True:
				plt.show()
			else:
				plt.savefig(args.plot, transparent=True, bbox_inches="tight")

	return weights


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	weights = main(args)
	print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
