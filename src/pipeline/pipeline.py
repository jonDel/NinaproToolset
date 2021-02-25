"""Base class for Pipeline execution"""
from typing import Type
import logging
from time import time
import re
from pathlib import Path
import numpy as np


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, preprocessor: Type,
                 feature_extractors: [],
                 classifiers: [],
                 database: str,
                 datasets: list,
                 database_path: str,
                 datasets_paths: dict = {},
                 window_duration: int = 200,
                 window_overlap: int = 10,
                 which_reps: list = [],
                 which_moves: list = [],
                 downloader: Type = None):
        self.downloader = downloader
        self.preprocessor = preprocessor
        self.feature_extractors = feature_extractors
        self.classifiers = classifiers
        self.database = database
        self.datasets = datasets
        self.database_path = database_path
        self.datasets_path = self._infer_datasets_paths(datasets_paths)
        self.window_duration = window_duration
        self.window_overlap = window_overlap
        self.which_reps = which_reps
        self.which_moves = which_moves
        self.results = {}

    def _infer_datasets_paths(self, datasets_path):
        if datasets_path:
            return datasets_path
        for folder in Path(self.database_path).glob("*"):
            patt = re.search(r"^s(\d+)$", folder.name, re.I)
            if not patt:
                continue
            datasets_path[int(patt.groups()[0])] = str(folder)
        return datasets_path

    def run(self):
        self.results = {}
        overall = {}
        logger.info("Start running the pipeline...")
        start_run = time()
        if self.downloader:
            self.download()
        for dataset in self.datasets:
            self.results[dataset] = {}
            x_train, y_train, x_test, y_test = self.preprocess(dataset)
            extracted_data = self.extract_features(x_train, x_test)
            start_train = time()
            logger.info("Start training the models {} for dataset {}...".
                        format(", ".join(
                            [type(clf).__name__ for clf in self.classifiers]),
                            dataset
                        ))
            for clf in self.classifiers:
                clf_name = type(clf).__name__
                self.results[dataset][clf_name] = {}
                for fname, fdata in extracted_data.items():
                    self.results[dataset][clf_name][fname] = {}
                    clf.fit(fdata["train"], y_train)
                    predictions = clf.predict(fdata["test"])
                    self.results[dataset][clf_name][fname]["f1-score"] = \
                        clf.f1_score(predictions, y_test)
                    self.results[dataset][clf_name][fname]["accuracy"] = \
                        clf.accuracy(predictions, y_test)
                    self.results[dataset][clf_name][fname]["balanced_accuracy"] = \
                        clf.balanced_accuracy(predictions, y_test)
                    self.results[dataset][clf_name][fname]["confusion_matrix"] = \
                        clf.confusion_matrix(predictions, y_test,
                                             normalize="true")
            logger.info("All models  for subject {} were trained in {}"
                        " seconds.".format(dataset, time() - start_train))
        for clf_name in [type(clf).__name__ for clf in self.classifiers]:
            overall[clf_name] = {} 
            for exct in [type(fext).__name__
                         for fext in self.feature_extractors]:
                overall[clf_name][exct] = {}
                for metric in ["balanced_accuracy", "accuracy", "f1-score"]:
                    mets = [sub[clf_name][exct][metric]
                            for sub in self.results.values()]
                    mean = np.mean(mets)
                    std = np.std(mets)
                    overall[clf_name][exct][metric] = {
                        "mean": mean,
                        "std": std
                    }
        self.results["overall"] = overall
        logger.info("The pipeline execution finished in {} seconds.".format(
            time() - start_run))

    def download(self):
        logger.info("Start downloading data from subjects {} "
                    "from database {}...".format(", ".join(
                        [str(dts) for dts in self.datasets]),
                                                 self.database)
                    )
        start = time()
        for dataset in self.datasets:
            self.downloader.download_dataset(self.database, dataset)
        logger.info("All datasets were downloaded in {} seconds.".format(
                    time() - start))
        self.datasets_path = self._infer_datasets_paths({})
    
    def preprocess(self, dataset):
        sub_data = self.preprocessor(dataset, self.datasets_path[dataset])
        # Normalize
        sub_data.normalize()
        # Split into overlapping time windows
        x_win, y_win, r_win = sub_data.get_windows(self.window_duration,
                                                   self.window_overlap,
                                                   self.which_reps,
                                                   self.which_moves)
        # Split data into train an test sets
        x_train, y_train, x_test, y_test = sub_data.split_data(x_win,
                                                               y_win,
                                                               r_win)
        return x_train, y_train, x_test, y_test

    def extract_features(self, x_train, x_test):
        features = {}
        logger.info("Start extracting features using extractors {}.".format(
            ", ".join([type(ftr).__name__ for ftr in self.feature_extractors]))
        )
        start = time()
        for extractor in self.feature_extractors:
            features[type(extractor).__name__] = {}
            features[type(extractor).__name__]["train"] = \
                extractor.extract(x_train)
            features[type(extractor).__name__]["test"] = \
                extractor.extract(x_test)
        logger.info("All features were extracted in {} seconds.".format(
                    time() - start))
        return features