#!/usr/bin/env python
"""
Configurable parameters of the scripts.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2019, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

from pathlib import Path

#corpus = "conll2002"
corpus = 'brown'

# Weight
base_dir = "./result"
weights_path = Path(base_dir) / Path(corpus + "_weights.h5")
log_dir_path = Path(base_dir) / Path("log")

epochs = 20
#epochs = 1 # for testing the flow

# Data shape
batch_size = 64
max_sequence_size = 256 # For Brown
#max_sequence_size = 256 # for CONLL2002

# For embedding
sample_dimension = 100
use_embedding = True

# Special word definitions
unk_string = "<UNK>"
pad_string = "<PAD>"
