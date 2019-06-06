#!/usr/bin/env python
"""
Example code for part of speech tagging using LSTM.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2019, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging
from pathlib import Path
import collections
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
#from keras.layers import LSTM # Slow, do not use
from keras.layers import CuDNNLSTM as LSTM

import nltk

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

import config

def load_dataset(corpus='brown', test_ratio=0.1):
    """Load corpus

    Parameters
    ----------
    test_ratio: float
        Proportion of dataset to be assigned to test set

    Returns
    -------
    (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag): list of tuples
        x_train: ndarray
            Training dataset
        y_train: ndarray
            Ground-truth values for training dataset
        x_test: ndarray
            Test dataset
        x_test: ndarray
            Ground-truth values for test dataset
        word2id: dict
            Mapping from word to ID
        id2word: dict
            Mappint from id to word
        tag2id: dict
            Mapping from tag to ID
        id2tag: dict
            Mapping from ID to tag
    """
    nltk.download(corpus)

    word2id = collections.defaultdict(lambda: len(word2id))  # 0-based index
    tag2id = collections.defaultdict(lambda: len(tag2id))

    word2id["<UNK>"] = 0
    word2id["<PAD>"] = 1

    tag2id["<PAD>"] = 0

    word_id_only_sentences = list()
    tag_id_only_sentences = list()

    if corpus == 'brown:':
        tagged_sentences = nltk.corpus.brown.tagged_sents()  # [[(w11, t11), (w12, t12), ... ], [(w21, t21), (w22, t22)], ... ]
    elif corpus == 'conll2002':
        tagged_sentences = nltk.corpus.conll2002.tagged_sents()  # [[(w11, t11), (w12, t12), ... ], [(w21, t21), (w22, t22)], ... ]
    else:
        raise ValueError("Invalid corpus")

    for tagged_sentence in tagged_sentences:
        word_id_only_sentence = [word2id[word] for word, tag in tagged_sentence]
        tag_id_only_sentence = [tag2id[tag] for word, tag in tagged_sentence]

        # Create placeholder ndarrays filled with <PAD>
        word_id_only_sentence_np = np.full(config.MAX_SEQUENCE_SIZE, word2id["<PAD>"], dtype=np.int32)
        tag_id_only_sentence_np = np.full(config.MAX_SEQUENCE_SIZE, tag2id["<PAD>"], dtype=np.int32)

        # Copy sentence to numpy array
        word_id_only_sentence_np[:len(word_id_only_sentence)] = word_id_only_sentence
        tag_id_only_sentence_np[:len(tag_id_only_sentence)] = tag_id_only_sentence

        word_id_only_sentences.append(word_id_only_sentence_np)
        tag_id_only_sentences.append(tag_id_only_sentence_np)

    # Convert both list to ndarray
    word_id_only_sentences_np = np.array(word_id_only_sentences)
    tag_id_only_sentences_np = np.array(tag_id_only_sentences)

    # Split to training set and test set
    num_samples = word_id_only_sentences_np.shape[0]
    num_training_samples = int(num_samples * (1 - test_ratio))
    num_test_samples = (num_samples - num_training_samples)

    x_train = word_id_only_sentences_np[:num_training_samples]
    x_test = word_id_only_sentences_np[num_training_samples:]
    y_train = tag_id_only_sentences_np[:num_training_samples]
    y_test = tag_id_only_sentences_np[num_training_samples:]

    # Reverse dict
    id2word = {id: word for word, id in word2id.items()}
    id2tag = {id: tag for tag, id in tag2id.items()}

    print(x_train.shape)
    x_train = x_train.reshape((num_training_samples, config.MAX_SEQUENCE_SIZE, 1))
    y_train = y_train.reshape((num_training_samples, config.MAX_SEQUENCE_SIZE, 1))
    x_test = x_test.reshape((num_test_samples, config.MAX_SEQUENCE_SIZE, 1))
    y_test = y_test.reshape((num_test_samples, config.MAX_SEQUENCE_SIZE, 1))

    return (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag)

