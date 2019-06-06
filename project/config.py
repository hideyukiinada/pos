#!/usr/bin/env python
"""
Example code for part of speech tagging using LSTM.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2019, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

from pathlib import Path

BASE_DIR = "./result"
WEIGHTS_PATH = Path(BASE_DIR) / Path("weights.h5")
LOG_DIR_PATH = Path(BASE_DIR) / Path("log")

EPOCHS = 20
#EPOCHS = 1

BATCH_SIZE = 64
MAX_SEQUENCE_SIZE = 256

