"""Temporal Convolutional Network Classifier."""
from time import time
from numpy import ndarray
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tcn import TCN as KerasTcn
from .model_base import Classifier
import logging


logger = logging.getLogger(__name__)


class TCN(Classifier):
    def __init__(self) -> None:
        clf = None
        super().__init__(clf)

    def fit(self, x_train: ndarray, y_train: ndarray, epochs: int = 100,
              **kwargs) -> None:
        logger.debug("Temporal Convolutional Network Classifier for {} "
                     "epochs...".format(epochs))
        start = time()
        if not self.classifier:
            self.classifier = default_arch((x_train.shape[1],
                                            x_train.shape[2]), y_train.shape[1])
        self.classifier.fit(x_train, y_train, epochs=epochs, **kwargs)
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
        predictions = self.classifier.predict_proba(x_test)
        logger.debug("Done all predictions in {} seconds.".
                     format(time() - start))
        return predictions


def default_arch(shape, n_classes):
    print(n_classes)
    input = Input(shape=shape)
    layer1 = KerasTcn()(input)
    layer2 = Dense(units=n_classes, activation='linear')(layer1)
    model = Model(inputs=[input], outputs=[layer2])
    model.summary(print_fn=logger.info)
    model.compile('adam', 'mae')
    return model
