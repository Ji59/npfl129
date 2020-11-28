#!/usr/bin/env python3
import os
import urllib.request
import sys

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


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants


before = 3
after = 3


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

		# TODO: Train a model on the given dataset and store it in `model`.
		model = MLPClassifier(max_iter=100, learning_rate_init=0.01, learning_rate='adaptive', hidden_layer_sizes=(100), verbose=True)

		data_size = len(one_hot.get_feature_names()) * (before + after + 1)
		data = np.array([]).reshape(0, data_size)
		targets = np.array([])
		for i in range(len(data_words)):
			word_i = data_words[i]
			for j in range(len(word_i)):
				if word_i[j].lower() in Dataset.LETTERS_NODIA:
					word = ""
					target = np.array([char_to_byte(target_words[i][j])])
					for k in range(j - before, j + after + 1):
						if k < 0 or k >= len(word_i):
							word += '.'
						else:
							word += word_i[k]
					data = np.concatenate((data, one_hot.transform(str_to_byte_array(word)).toarray().reshape(1, -1)))
					targets = np.concatenate((targets, target))

		model.fit(data, targets)

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

		words = test.data.split()
		for word in words:
			for i in range(len(word)):
				l = word[i]
				if l.lower() in Dataset.LETTERS_NODIA:
					lower = l.lower() == l
					word_temp = ""
					for k in range(i - before, i + after + 1):
						if k < 0 or k >= len(word):
							word_temp += '.'
						else:
							word_temp += word[k]
					word_vect = one_hot.transform(str_to_byte_array(word_temp)).toarray().reshape(1, -1)
					predicted = byte_to_char(model.predict(word_vect)[0])
					predictions += l if predicted is None else predicted if lower else predicted.upper()
				else:
					predictions += l
			predictions += ' '

		return predictions


def get_one_hot():
	one_hot = OneHotEncoder(handle_unknown='ignore')
	one_hot.fit(str_to_byte_array("abcdefghijklmnopqrstuvwxyz" + Dataset.LETTERS_DIA).reshape(-1, 1))
	return one_hot


letters_dia = "abcdefghijklmnopqrstuvwxyz" + Dataset.LETTERS_DIA


def create_converter():
	i = 0
	out = {'.': i}

	for c in letters_dia:
		if out.get(c) is None:
			i += 1
			out[c] = i
	return out


converter = create_converter()


def create_inv_converter():
	out = {0: '.'}
	for l in letters_dia:
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
