#!/usr/bin/env python
"""
Predict POS from a trained model.
After loading the saved weights, it will prompt you to enter a sentence followed by a carriage return.
It displays the corresponding POS for each word.
Press q to exit.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2019, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging
import numpy as np

import nltk

from load_data import load_dataset
import config
import model_architecture

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def convert_input_sentence(sentence, word2id, use_embedding=False):
    """Convert a sentence to tokens.

    Parameters
    ----------
    sentence: str
        A sentence in a string format.
    word2id: dict
        Mapping from words to IDs.
    use_embedding: bool
        If true, use embedding layer.

    Returns
    -------
    x: ndarray
        dataset
    word_count: int
        Number of words
    """

    #    tokenizer = nltk.WhitespaceTokenizer()
    tokenizer = nltk.TreebankWordTokenizer()

    tokens = tokenizer.tokenize(sentence)
    log.info("Tokens:%s" % (repr(tokens)))

    word_ids = list()
    for word in tokens:
        if word not in word2id:
            word = config.unk_string
        word_id = word2id[word]
        word_ids.append(word_id)

    # Create placeholder ndarrays filled with <PAD>
    word_id_only_sentence_np = np.full(config.max_sequence_size, word2id[config.pad_string], dtype=np.int32)

    # Copy sentence to numpy array
    word_id_only_sentence_np[:len(word_ids)] = word_ids
    word_count = len(word_ids)

    x = word_id_only_sentence_np

    if use_embedding:
        x = x.reshape((1, config.max_sequence_size))
    else:
        x = x.reshape((1, config.max_sequence_size, 1))

    return x, word_count


def main():
    """Defines an application's main functionality"""

    log.info("Started.")

    (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag) = load_dataset(config.corpus,
                                                                                              test_ratio=0.1)
    voc_size = len(word2id)
    num_tags = len(id2tag)

    log.info("Number of unique tags: %d" % (num_tags))

    if config.use_embedding is False:
        model = model_architecture.build_model(num_tags)
    else:
        model = model_architecture.build_model_with_embedding(num_tags, voc_size, config.sample_dimension)

    model.load_weights(config.weights_path)

    while True:
        sentence = input("Enter a sentence (press 'q' to quit): ")
        if sentence == 'q':
            break

        word_ids, word_count = convert_input_sentence(sentence, word2id, config.use_embedding)
        y_hat = model.predict(word_ids)

        word_ids = word_ids.reshape((config.max_sequence_size))
        print(word_ids.shape)
        # print(y_hat)

        y_hat = y_hat[0]
        print(y_hat.shape)

        for i in range(word_count):
            tag_id = np.argmax(y_hat[i])
            tag = id2tag[tag_id]

            word_id = word_ids[i]
            print("[%d] word: %s tag: %s" % (i, id2word[word_id], tag))


if __name__ == "__main__":
    main()
