#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
import random
from sklearn.naive_bayes import GaussianNB;


KONSTANTA = 100000


class Dataset:
	def __init__(self,
							 name="isnt_it_ironic.train.zip",
							 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
		if not os.path.exists(name):
			print("Downloading dataset {}...".format(name), file=sys.stderr)
			urllib.request.urlretrieve(url + name, filename=name)

		# Load the dataset and split it into `data` and `target`.
		self.data = []
		self.target = []

		with zipfile.ZipFile(name, "r") as dataset_file:
			with dataset_file.open(name.replace(".zip", ".txt"), "r") as train_file:
				for line in train_file:
					label, text = line.decode("utf-8").rstrip("\n").split("\t")
					self.data.append(text)
					self.target.append(int(label))
		self.target = np.array(self.target, np.int32)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


def main(args):
	if args.predict is None:
		# We are training a model.
		np.random.seed(args.seed)
		train = Dataset()

		words_deceptikon = {}
		data = np.zeros(KONSTANTA);
		n = 0;
		k = 1;
		ones = 0;
		all = 0;
		target = np.zeros(KONSTANTA);
		for i in range(len(train.target)):
			words = train.data[i].lower().split();
			if (train.target[i] == 1):
				ones += len(words);
			all += len(words);
			for word in words:
				if (word not in words_deceptikon):
					words_deceptikon[word] = k;
					k += 1;
				data[n] = words_deceptikon[word];
				target[n] = train.target[i]
				n += 1;
				if (n > len(data)):
					data = np.append(data, np.zeros(KONSTANTA));
					target = np.append(target, np.zeros(KONSTANTA));
		data = data[:n];
		target = target[:n];

		# TODO: Train a model on the given dataset and store it in `model`.
		model = GaussianNB();
		model.fit(data.reshape(-1, 1), target);


		# Serialize the model.
		with lzma.open(args.model_path, "wb") as model_file:
			pickle.dump([words_deceptikon, model, ones, all], model_file)

	else:
		# Use the model and return test set predictions.
		test = Dataset(args.predict)

		with lzma.open(args.model_path, "rb") as model_file:
			file = pickle.load(model_file)
			words = file[0]
			model = file[1]
			ones = file[2]
			all = file[3]

		# TODO: Generate `predictions` with the test set predictions, either
		# as a Python list of a NumPy array.

		predictions = np.zeros(len(test.target));

		for i in range(len(test.data)):
			sum = 0;
			n = 0;
			for word in test.data[i].lower().split():
				if word in words:
					sum += model.predict(np.array([words[word]]).reshape(-1, 1));
				else:
					sum += 1 if random.uniform(0, 1) >= ones / all else 0;
				n += 1;
			if sum > n / 2:
				predictions[i] = 1;

		return predictions


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	main(args)
