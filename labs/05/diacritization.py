#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier


before = 4
after = 4
size = 100000


class Dataset:
	LETTERS_NODIA = "acdeeinorstuuyz"
	LETTERS_DIA = "áčďéěíňóřšťúůýž"

	# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
	DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

	def __init__(self,
								name="fiction-train.txt",
								url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
		if not os.path.exists(name):
			print("Downloading dataset {}...".format(name), file=sys.stderr)
			urllib.request.urlretrieve(url + name, filename=name)
			urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

		# Load the dataset and split it into `data` and `target`.
		with open(name, "r", encoding="utf-8") as dataset_file:
			self.target = dataset_file.read()
		self.data = self.target.translate(self.DIA_TO_NODIA)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


def main(args):
	if args.predict is None:
		# We are training a model.
		np.random.seed(args.seed)
		train = Dataset()

		data_words = train.data.split(" ")
		target_words = train.target.split(" ")

		one_hot = get_one_hot()

		# TODO: Train a model on the given dataset and store it in `model`.z
		model = MLPClassifier(max_iter=420, learning_rate_init=0.003, learning_rate='adaptive', hidden_layer_sizes=(420, 107, 69	), verbose=True)

		data_size = features * (before + after + 1)
		data = np.zeros((size, data_size))
		targets = np.zeros(size)
		n = 0
		for i in range(len(data_words)):
			word_i = data_words[i]
			len_word_i = len(word_i)
			for j in range(len_word_i):
				if word_i[j].lower() in Dataset.LETTERS_NODIA:
					get_one_hot_vector(data, features, j, len_word_i, n, one_hot, word_i)

					targets[n] = char_to_byte(target_words[i][j])
					n += 1
					if n >= data.shape[0]:
						data = np.concatenate((data, np.zeros((size, data_size))), axis=0)
						targets = np.append(targets, np.zeros(size))
			if i % 3000 == 0:
				print("{} / {}".format(i, len(data_words)))

		model.fit(data[:n], targets[:n])

		# Serialize the model.
		with lzma.open(args.model_path, "wb") as model_file:
			pickle.dump(model, model_file)

	else:
		# Use the model and return test set predictions, as either a Python list or a NumPy array.
		test = Dataset(args.predict)

		with lzma.open(args.model_path, "rb") as model_file:
			model = pickle.load(model_file)

		# TODO: Generate `predictions` with the test set predictions. Specifically,
		# produce a diacritized `str` with exactly the same number of words as `test.data`.

		predictions = ""
		one_hot = get_one_hot()
		data_size = features * (before + after + 1)

		words = test.data.split()
		data = np.zeros((size, data_size))

		n = 0
		for i in range(len(words)):
			word_i = words[i]
			len_word_i = len(word_i)
			for j in range(len_word_i):
				if word_i[j].lower() in Dataset.LETTERS_NODIA:
					get_one_hot_vector(data, features, j, len_word_i, n, one_hot, word_i)

					n += 1
					if n >= data.shape[0]:
						data = np.concatenate((data, np.zeros((size, data_size))), axis=0)

		predicted_all = model.predict(data)

		n = 0
		for i in range(len(words)):
			word = words[i]
			out = ""
			for j in range(len(word)):
				l = word[j]
				if l.lower() in Dataset.LETTERS_NODIA:
					lower = l.lower() == l

					word_vect = np.zeros((1, data_size))
					get_one_hot_vector(word_vect, features, j, len(word), 0, one_hot, word)
					predicted = byte_to_char(predicted_all[n])
					l = predicted if lower else predicted.upper()
					n += 1
				out += l
			words[i] = out
		predictions = " ".join(words)

		print(predictions)
		return predictions


def get_one_hot_vector(data, features, j, len_word_i, n, one_hot, word_i):
	l = before - j
	for k in range(max(j - before, 0), min(j + after + 1, len_word_i)):
		i = char_to_byte(word_i[k])
		if i > 0:
			data[n, (k + l) * features + i - 1] = 1


def get_one_hot():
	one_hot = OneHotEncoder(handle_unknown='ignore')
	one_hot.fit(str_to_byte_array(letters).reshape(-1, 1))
	return one_hot


letters = "abcdefghijklmnopqrstuvwxyz"
letters_dia = Dataset.LETTERS_DIA


def create_converter():
	i = 0
	out = {'.': i}

	for c in letters + letters_dia:
		if out.get(c) is None:
			i += 1
			out[c] = i
	return out


converter = create_converter()
features = len(letters)


def create_inv_converter():
	out = {0: '.'}
	for l in letters + letters_dia:
		i = converter[l]
		out[i] = l
	return out


inv_converter = create_inv_converter()


def char_to_byte(c):
	value = converter.get(c.lower())
	return 0 if value is None else value


def str_to_byte_array(s):
	return np.array([char_to_byte(c) for c in s]).reshape(-1, 1)


def byte_to_char(b):
	c = inv_converter.get(int(b))
	return None if c is None else c


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	main(args)
