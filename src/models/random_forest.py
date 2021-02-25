"""Random Forest Classifier."""
from time import time
from sklearn.ensemble import RandomForestClassifier
from numpy import ndarray
import numpy as np
from .model_base import Classifier
import logging


logger = logging.getLogger(__name__)


class RandomForest(Classifier):
    def __init__(self, n_estimators: int = 100) -> None:
        rf = RandomForestClassifier(n_estimators=n_estimators)
        super().__init__(rf)

    def fit(self, x_train: ndarray, y_train: ndarray) -> None:
        logger.debug("Training the Random Forest Classifier...")
        start = time()
        self.classifier.fit(x_train, np.argmax(y_train, axis=-1))
        logger.debug("Done training in {} seconds.".format(time() - start))

    def predict(self, x_test: ndarray) -> ndarray:
        logger.debug("Predicting {} samples...".format(x_test.shape[0]))
        start = time()
        predictions = self.classifier.predict(x_test)
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
