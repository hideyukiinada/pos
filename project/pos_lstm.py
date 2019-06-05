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

def main():
    """Defines an application's main functionality"""

    log.info("Started.")

    base_path = Path(config.BASE_DIR)
    if base_path.exists() is False:
        base_path.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag) = load_data(test_ratio=0.1)
    num_tags = len(id2tag)

    y_train_oh = keras.utils.np_utils.to_categorical(y_train, num_tags)
    y_test_oh = keras.utils.np_utils.to_categorical(y_test, num_tags)

    log.info("Number of unique tags: %d" % (num_tags))
    # Set up a model
    model = model_architecture.build_model(num_tags)
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])

    model.fit(x=x_train, y=y_train_oh,
              validation_data = (x_test, y_test_oh),
              batch_size=128, epochs=config.EPOCHS,
              verbose=1) # progress bar

    model.save_weights(config.WEIGHTS_PATH)

if __name__ == "__main__":
    main()
