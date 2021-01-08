#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os

import numpy as np
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix


class Dataset:
	CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

	def __init__(self, name):
		if not os.path.exists(name):
			raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

		# Load the dataset and split it into `data` and `target`.
		self.data, self.prompts, self.levels, self.target = [], [], [], []
		with open(name, "r", encoding="utf-8") as dataset_file:
			for line in dataset_file:
				target, prompt, level, text = line.rstrip("\n").split("\t")
				self.data.append(text)
				self.prompts.append(prompt)
				self.levels.append(level)
				self.target.append(-1 if not target else self.CLASSES.index(target))
		self.target = np.array(self.target, np.int32)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")

words_before, words_after = 2, 1


def split(text):
	return text.split()


def main(args):
	if args.predict is None:
		# We are training a model.
		np.random.seed(args.seed)
		train = Dataset("nli_dataset/nli_dataset.train.txt")
		dev = Dataset("nli_dataset/nli_dataset.dev.txt")

		words = {}
		n = 1
		sentences = [[]] * len(train.data)
		for i in range(len(train.data)):
			data_words = split(train.data[i])
			sentences[i] = [0] * len(data_words)
			sentence = sentences[i]
			for j in range(len(data_words)):
				word = data_words[j]
				word_id = words.get(word)
				if word_id is None:
					words[word] = [n, 1]
					sentence[j] = n
					n += 1
				else:
					sentence[j] = word_id[0]
					words[word][1] += 1

		words_final = {}
		indexes = {}
		n = 1
		for word in words:
			if words.get(word)[1] >= 424:
				words_final[word] = n
				indexes[words.get(word)[0]] = n
				n += 1

		n_words = 0
		for sentence in sentences:
			n_words += len(sentence)
			for i in range(len(sentence)):
				sentence_i_ = sentence[i]
				if sentence_i_ in indexes:
					sentence[i] = indexes[sentence_i_]
				else:
					sentence[i] = 0

		data, target = get_data_target(len(words_final), n_words, sentences, train.target)

		# TODO: Train a model on the given dataset and store it in `model`.
		model = MLPClassifier(solver="adam", hidden_layer_sizes=173, verbose=True, learning_rate="invscaling", learning_rate_init=0.01)
		model.fit(data, target)

		# Serialize the model.
		with lzma.open(args.model_path, "wb") as model_file:
			pickle.dump([model, words_final, len(train.CLASSES)], model_file)

	else:
		# Use the model and return test set predictions.
		test = Dataset(args.predict)

		with lzma.open(args.model_path, "rb") as model_file:
			model, words, classes = pickle.load(model_file)

		predictions = np.zeros(len(test.data))
		for i in range(len(test.data)):
			data_words = split(test.data[i])
			essay_words = len(data_words)
			sentence = [0] * essay_words
			for j in range(essay_words):
				word = data_words[j]
				if word in words:
					sentence[j] = words.get(word)
			data, _ = get_data_target(len(words), essay_words, [sentence])
			distribution = model.predict(data)
			frequency = np.zeros(classes)
			for j in distribution:
				frequency[int(j)] += 1
			predictions[i] = frequency.argmax()

		# TODO: Generate `predictions` with the test set predictions, either
		# as a Python list of a NumPy array.

		return predictions


def get_data_target(n, n_words, sentences, targets=None):
	n_sentences = len(sentences)
	data = lil_matrix((n_words, (words_before + words_after + 1) * n))
	if targets is not None:
		target = np.empty(n_words)
	else:
		target = None
	m = 0
	for i in range(n_sentences):
		if targets is not None:
			target_i = targets[i]
		sentence = sentences[i]
		n_sentence = len(sentence)
		for j in range(n_sentence):
			for k in range(j - words_before, j + words_after + 1):
				if k < 0 or k >= n_sentence or sentence[k] == 0:
					continue
				data[m + j, (k - j + words_before) * n + sentence[k] - 1] = 1
				if targets is not None:
					target[m + j] = target_i
		m += n_sentence
	return data, target


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	main(args)
