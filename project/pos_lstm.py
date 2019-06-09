#!/usr/bin/env python
"""
Example code for part of speech tagging using LSTM.

Credit
------
1. I used [1] below as a reference for the model.
2. I used [2] as a reference for loading Brown corpus including conversion between word and tag to IDs.

References
----------
[1] Brownlee, Jason, "How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras,"
 https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/, 2017.
[2] Chainer team, postagging.py, https://github.com/chainer/chainer/blob/master/examples/pos/postagging.py.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2019, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging
from pathlib import Path
import keras

from load_data import load_dataset
import config
import model_architecture

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def main():
    """Defines an application's main functionality"""

    log.info("Started.")

    base_path = Path(config.base_dir)
    if base_path.exists() is False:
        base_path.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test), (word2id, id2word), (tag2id, id2tag) = load_dataset(config.corpus,
                                                                                              test_ratio=0.1,
                                                                                              data_dir=config.base_dir)
    voc_size = len(word2id)
    num_tags = len(id2tag)

    y_train_oh = keras.utils.np_utils.to_categorical(y_train, num_tags)
    y_test_oh = keras.utils.np_utils.to_categorical(y_test, num_tags)

    log.info("Data information")
    log.info("Size of training set: %d" % (x_train.shape[0]))
    log.info("Shape of training set: %s" % (repr(x_train.shape)))
    log.info("Size of test set: %d" % (x_test.shape[0]))
    log.info("Number of unique wordss: %d" % (len(word2id)))
    log.info("Number of unique tags: %d" % (num_tags))
    log.info("Weights path: %s" % (config.weights_path))

    if config.use_embedding is False:
        model = model_architecture.build_model(num_tags)
    else:
        model = model_architecture.build_model_with_embedding(num_tags, voc_size, config.sample_dimension)

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])

    model.fit(x=x_train, y=y_train_oh,
              validation_data=(x_test, y_test_oh),
              batch_size=128, epochs=config.epochs,
              verbose=1)  # Use progress bar

    model.save_weights(config.weights_path)


if __name__ == "__main__":
    main()
