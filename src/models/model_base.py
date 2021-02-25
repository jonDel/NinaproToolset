"""Base class for Feature extractors"""
from abc import ABC, abstractmethod
from numpy import ndarray
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sn


class Classifier(ABC):
    def __init__(self, classifier_obj=None):
        self.classifier = classifier_obj

    @abstractmethod
    def fit(self, train_x: ndarray, train_y: ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, input: ndarray) -> ndarray:
        pass

    @abstractmethod
    def predict_proba(self, input: ndarray) -> ndarray:
        pass

    @staticmethod
    def accuracy(y_pred: ndarray, y_true: ndarray) -> float:
        return accuracy_score(y_pred, y_true)

    @staticmethod
    def balanced_accuracy(y_pred: ndarray, y_true: ndarray) -> float:
        return balanced_accuracy_score(y_pred, y_true)

    @staticmethod
    def f1_score(y_pred: ndarray, y_true: ndarray,
                 average: str = "weighted") -> float:
        return f1_score(y_pred, y_true, average=average)

    @staticmethod
    def confusion_matrix(y_pred: ndarray, y_true: ndarray, **kwargs)\
        -> ndarray:
        return confusion_matrix(y_pred, y_true, **kwargs)

    @staticmethod
    def plot_confusion_matrix(conf_matrix: ndarray,
                              figsize=[40/2.54, 30/2.54],
                              title="Confusion matrix of classifier",
                              annot=False,
                              annot_kws=None,
                              cmap=None,
                              axis_fontsize=(16, 16),
                              context="notebook",
                              style="darkgrid",
                              palette="deep",
                              font="sans-serif",
                              font_scale=0.3,
                              color_codes=True,
                              rc=None) -> None:
        plt.clf()
        sn.heatmap(conf_matrix, annot=annot, annot_kws=annot_kws, cmap=cmap)
        sn.set_theme(context=context, style=style, palette=palette,
                     font=font, font_scale=font_scale, 
                     color_codes=color_codes, rc=rc)
        plt.rcParams['figure.figsize'] = figsize
        plt.xlabel("Predicted label",  fontsize=axis_fontsize[0])
        plt.ylabel("True label",  fontsize=axis_fontsize[1])
        plt.title(title, fontsize=18)
        plt.show()
