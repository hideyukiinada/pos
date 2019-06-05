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

BASE_DIR = "/tmp/pos"
WEIGHTS_PATH = Path(BASE_DIR) / Path("weights.h5")
LOG_DIR_PATH = Path(BASE_DIR) / Path("log")

EPOCHS = 200
BATCH_SIZE = 64
MAX_SEQUENCE_SIZE = 256


def load_data(test_ratio=0.1):
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
    nltk.download('brown')

    word2id = collections.defaultdict(lambda: len(word2id))  # 0-based index
    tag2id = collections.defaultdict(lambda: len(tag2id))

    word2id["<UNK>"] = 0
    word2id["<PAD>"] = 1

    tag2id["<PAD>"] = 0

    word_id_only_sentences = list()
    tag_id_only_sentences = list()

    tagged_sentences = nltk.corpus.brown.tagged_sents()  # [[(w11, t11), (w12, t12), ... ], [(w21, t21), (w22, t22)], ... ]
    for tagged_sentence in tagged_sentences:
        word_id_only_sentence = [word2id[word] for word, tag in tagged_sentence]
        tag_id_only_sentence = [tag2id[tag] for word, tag in tagged_sentence]

        # Create placeholder ndarrays filled with <PAD>
        word_id_only_sentence_np = np.full(MAX_SEQUENCE_SIZE, word2id["<PAD>"], dtype=np.int32)
        tag_id_only_sentence_np = np.full(MAX_SEQUENCE_SIZE, tag2id["<PAD>"], dtype=np.int32)

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
    x_train = x_train.reshape((num_training_samples, MAX_SEQUENCE_SIZE, 1))
    y_train = y_train.reshape((num_training_samples, MAX_SEQUENCE_SIZE, 1))
    x_test = x_test.reshape((num_test_samples, MAX_SEQUENCE_SIZE, 1))
    y_test = y_test.reshape((num_test_samples, MAX_SEQUENCE_SIZE, 1))

    return (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag)

def convert_input_sentence(sentence, word2id):
    """Convert a sentence to tokens.

    Parameters
    ----------
    sentence: str
        A sentence in a string format.

    Returns
    -------
    """
#    tokenizer = nltk.WhitespaceTokenizer() # Note that punctuation is kept: ['Julie', 'is', 'very', 'pretty.']
    tokenizer = nltk.TreebankWordTokenizer() # Note that punctuation is kept: ['Julie', 'is', 'very', 'pretty.']

    tokens = tokenizer.tokenize(sentence)
    log.info("Tokens:%s" % (repr(tokens)))

    # this time you need to set unknown word to UNK
    word_ids = list()
    for word in tokens:
        if word not in word2id:
            word = "<UNK>"
        word_id = word2id[word]
        word_ids.append(word_id)

    # Create placeholder ndarrays filled with <PAD>
    word_id_only_sentence_np = np.full(MAX_SEQUENCE_SIZE, word2id["<PAD>"], dtype=np.int32)

    # Copy sentence to numpy array
    word_id_only_sentence_np[:len(word_ids)] = word_ids
    word_count = len(word_ids)

    x = word_id_only_sentence_np
    x = x.reshape((1, MAX_SEQUENCE_SIZE, 1))

    return x, word_count

# insfin

def main():
    """Defines an application's main functionality"""

    log.info("Started.")

    base_path = Path(BASE_DIR)
    if base_path.exists() is False:
        base_path.mkdir(exist_ok=True)
        
    (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag) = load_data(test_ratio=0.1)
    num_tags = len(id2tag)

    y_train_oh = keras.utils.np_utils.to_categorical(y_train, num_tags)
    y_test_oh = keras.utils.np_utils.to_categorical(y_test, num_tags)

    log.info("Number of unique tags: %d" % (num_tags))
    # Set up a model
    model = Sequential()
    model.add(LSTM(100, input_shape=(MAX_SEQUENCE_SIZE, 1),
                   return_sequences=True))  # input = [batch_size, ts, 1], output = [batch_size, ts, 1]
    model.add(TimeDistributed(Dense(num_tags, activation='softmax')))

    # model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])

    # Do not use sparese for a possible accuracy shape issue.
    #
    # model.fit(x=x_train, y=y_train_oh,
    #           validation_data = (x_test, y_test_oh),
    #           batch_size=128, epochs=1,
    #           verbose=1) # progress bar

    model.load_weights(WEIGHTS_PATH)

    while True:
        sentence = input("Enter a sentence (press 'q' to quit): ")
        if sentence == 'q':
            break

        word_ids, word_count = convert_input_sentence(sentence, word2id)
        y_hat = model.predict(word_ids)

        word_ids = word_ids.reshape((MAX_SEQUENCE_SIZE))
        print(word_ids.shape)
        #print(y_hat)

        y_hat = y_hat[0]
        print(y_hat.shape)

        for i in range(word_count):
            tag_id = np.argmax(y_hat[i])
            tag = id2tag[tag_id]

            word_id = word_ids[i]
            print("[%d] word: %s tag: %s" % (i, id2word[word_id], tag))


if __name__ == "__main__":
    main()
