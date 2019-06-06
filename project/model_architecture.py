#!/usr/bin/env python
"""
Model definition.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2019, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging

from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed
# from keras.layers import LSTM # Slow.  Use CuDNNLSTM instead if your hardware supports it.
from keras.layers import CuDNNLSTM as LSTM

import config

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def build_model_with_embedding(num_tags, voc_size, sample_dimension):
    """Build a model

    Parameters
    ----------
    num_tags: int
        Number of units in the final dense layer.
    voc_size: int
        Size of the vocabulary
    sample_dimension:
        Dimension of a sample

    Returns
    -------
    model: Sequential
        Keras Sequential model
    """
    model = Sequential()

    model.add(Embedding(voc_size, sample_dimension))
    model.add(LSTM(sample_dimension, return_sequences=True))
    model.add(TimeDistributed(Dense(num_tags, activation='softmax')))

    return model


def build_model(num_tags):
    """Build a model

    Parameters
    ----------
    num_tags: int
        Number of units in the final dense layer.

    Returns
    -------
    model: Sequential
        Keras Sequential model
    """
    model = Sequential()

    model.add(LSTM(100, input_shape=(config.max_sequence_size, 1),
                   return_sequences=True))  # input = [batch_size, sequence len, 1], output = [batch_size, sequence len, 1]
    model.add(TimeDistributed(Dense(num_tags, activation='softmax')))

    return model
