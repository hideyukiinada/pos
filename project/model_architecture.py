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

import keras
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
#from keras.layers import LSTM # Slow, do not use
from keras.layers import CuDNNLSTM as LSTM

import config

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

def build_model(num_tags):
    """Build a model

    Parameters
    ----------
    num_tags: int
        Number of units in the final dense layer.
    """
    # Set up a model
    model = Sequential()
    model.add(LSTM(100, input_shape=(config.MAX_SEQUENCE_SIZE, 1),
                   return_sequences=True))  # input = [batch_size, ts, 1], output = [batch_size, ts, 1]
    model.add(TimeDistributed(Dense(num_tags, activation='softmax')))

    # Do not use sparese for a possible accuracy shape issue.

    return model
