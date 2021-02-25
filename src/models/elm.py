"""Extreme Learning Machine Classifier."""
from time import time
from typing import Tuple
from hpelm import ELM as ELMachine
from numpy import ndarray
import numpy as np
from .model_base import Classifier
import logging


DEFAULT_NEURONS = ((1500, "sigm"), (1500, "rbf_l2"))
logger = logging.getLogger(__name__)


class ELM(Classifier):
    def __init__(self, neurons: Tuple[Tuple] = None) -> None:
        clf = None
        self.neurons = neurons if neurons else DEFAULT_NEURONS
        super().__init__(clf)

    def fit(self, x_train: ndarray, y_train: ndarray, *args, **kwargs)\
            -> None:
        self.classifier = ELMachine(x_train.shape[1], y_train.shape[1])
        for neuron in self.neurons:
            logger.info("Adding {} neurons with '{}' function.".format(
                neuron[0], neuron[1]
            ))
            self.classifier.add_neurons(neuron[0], neuron[1])
        logger.debug("Training the Extreme Learning Machine Classifier...")
        start = time()
        self.classifier.train(x_train, y_train, **kwargs)
        logger.debug("Done training in {} seconds.".format(time() - start))

    def predict(self, x_test: ndarray) -> ndarray:
        logger.debug("Predicting {} samples...".format(x_test.shape[0]))
        start = time()
        predictions = np.argmax(self.classifier.predict(x_test), axis=-1)
        logger.debug("Done all predictions in {} seconds.".
                     format(time() - start))
        return predictions

    def predict_proba(self, x_test: ndarray) -> float:
        logger.debug("Predicting {} samples...".format(x_test.shape[0]))
        start = time()
        predictions = self.classifier.predict(x_test)
        logger.debug("Done all predictions in {} seconds.".
                     format(time() - start))
        return predictions
