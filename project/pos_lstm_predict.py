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

from load_data import load_data
import config
import model_architecture

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

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
    word_id_only_sentence_np = np.full(config.MAX_SEQUENCE_SIZE, word2id["<PAD>"], dtype=np.int32)

    # Copy sentence to numpy array
    word_id_only_sentence_np[:len(word_ids)] = word_ids
    word_count = len(word_ids)

    x = word_id_only_sentence_np
    x = x.reshape((1, config.MAX_SEQUENCE_SIZE, 1))

    return x, word_count

# insfin

def main():
    """Defines an application's main functionality"""

    log.info("Started.")

    # base_path = Path(config.BASE_DIR)
    # if base_path.exists() is False:
    #     base_path.mkdir(exist_ok=True)
        
    (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag) = load_data(test_ratio=0.1)
    num_tags = len(id2tag)

    log.info("Number of unique tags: %d" % (num_tags))
    model = model_architecture.build_model(num_tags)
    model.load_weights(config.WEIGHTS_PATH)

    while True:
        sentence = input("Enter a sentence (press 'q' to quit): ")
        if sentence == 'q':
            break

        word_ids, word_count = convert_input_sentence(sentence, word2id)
        y_hat = model.predict(word_ids)

        word_ids = word_ids.reshape((config.MAX_SEQUENCE_SIZE))
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
