"""Helper classes for preprocessing NinaPro databases."""
import logging
from typing import Tuple
from time import time
import numpy as np
from ..database.db import SubjectDB1, SubjectDB2, SubjectDB3
from ..database.dbinfo import DB1_INFO, DB2_INFO, DB3_INFO
from .utils import normalize, time_windows, split_sets


logger = logging.getLogger(__name__)


class PreprocessorDB1(SubjectDB1):
    def __init__(self, subject_number: int, dataset_path: str,
                 subsamp_rate: int = 10) -> None:
        self.subsamp_rate = subsamp_rate
        super().__init__(subject_number, dataset_path)
        self.load_dataset()
        self.norm_emg_data = None
        self.emg_data = self.emg_data[::subsamp_rate]
        self.rerepetition_data = self.rerepetition_data[::subsamp_rate]
        self.restimulus_data = self.restimulus_data[::subsamp_rate]

    def normalize(self, train_reps: list = DB1_INFO["default_trainreps"],
                  movements: list = [], which_moves: list = []):
        """Preprocess train+test data to mean 0, std 1 based on training data only.

        Args:
            train_reps (array): Which repetitions are in the training set
            movements (array, optional): Movement labels, required if using which_moves
            which_moves (array, optional): Which movements to return - if None use all

        Returns:
            array: Rescaled EMG data
        """
        logger.debug("Start normalizing EMG data...")
        start = time()
        self.norm_emg_data = normalize(self.emg_data, self.rerepetition_data,
                                       train_reps, movements, which_moves)
        logger.debug("Finished normalizing EMG data in {} seconds.".format(
            time() - start
        ))

    def get_windows(self, duration: int = 200, increment: int = 10,
                    which_reps: list = [], which_moves: list = [],
                    dtype: type = np.float32) -> Tuple[np.ndarray]:
        """Get set of windows based on repetition and movement criteria and\
            associated label + repetition data.

        Args:
            which_reps (array): Which repetitions to return
            window_len (int): Desired window length in miliseconds
            window_inc (int): Desired window increment in miliseconds
            which_moves (array, optional): Which movements to return - if None use all
            dtype (TYPE, optional): What precision to use for EMG data

        Returns:
            X_data (array): Windowed EMG data
            Y_data (array): Movement label for each window
            R_data (array): Repetition label for each window
        """
        which_reps = self.rep_labels if not which_reps else which_reps
        logger.debug("Start spliting EMG data into time windows...")
        start = time()
        x_data, y_data, r_data = time_windows(self.norm_emg_data,
                                              self.rerepetition_data,
                                              self.frequency,
                                              self.restimulus_data,
                                              duration,
                                              increment,
                                              which_reps,
                                              which_moves,
                                              dtype)
        logger.debug("Finished spliting EMG data in {} seconds.".format(
            time() - start
        ))
        return x_data, y_data, r_data

    def split_data(self, x_data: np.ndarray, y_data: np.ndarray,
                   r_data: np.ndarray,
                   train_reps: list = DB1_INFO["default_trainreps"],
                   test_reps: list = DB1_INFO["default_testreps"]) \
            -> Tuple[np.ndarray]:
        return split_sets(x_data, y_data, r_data, train_reps, test_reps)


class PreprocessorDB2(SubjectDB2):
    def __init__(self, subject_number: int, dataset_path: str,
                 subsamp_rate: int = 10) -> None:
        self.subsamp_rate = subsamp_rate
        super().__init__(subject_number, dataset_path)
        self.load_dataset()
        self.norm_emg_data = None
        self.emg_data = self.emg_data[::subsamp_rate]
        self.rerepetition_data = self.rerepetition_data[::subsamp_rate]
        self.restimulus_data = self.restimulus_data[::subsamp_rate]

    def normalize(self, train_reps: list = DB2_INFO["default_trainreps"],
                  movements: list = [], which_moves: list = []):
        """Preprocess train+test data to mean 0, std 1 based on training data only.

        Args:
            train_reps (array): Which repetitions are in the training set
            movements (array, optional): Movement labels, required if using which_moves
            which_moves (array, optional): Which movements to return - if None use all

        Returns:
            array: Rescaled EMG data
        """
        logger.debug("Start normalizing EMG data...")
        start = time()
        self.norm_emg_data = normalize(self.emg_data, self.rerepetition_data,
                                       train_reps, movements, which_moves)
        logger.debug("Finished normalizing EMG data in {} seconds.".format(
            time() - start
        ))

    def get_windows(self, duration: int = 200, increment: int = 10,
                    which_reps: list = [], which_moves: list = [],
                    dtype: type = np.float32) -> Tuple[np.ndarray]:
        """Get set of windows based on repetition and movement criteria and\
            associated label + repetition data.

        Args:
            which_reps (array): Which repetitions to return
            window_len (int): Desired window length in miliseconds
            window_inc (int): Desired window increment in miliseconds
            which_moves (array, optional): Which movements to return - if None use all
            dtype (TYPE, optional): What precision to use for EMG data

        Returns:
            X_data (array): Windowed EMG data
            Y_data (array): Movement label for each window
            R_data (array): Repetition label for each window
        """
        which_reps = self.rep_labels if not which_reps else which_reps
        logger.debug("Start spliting EMG data into time windows...")
        start = time()
        x_data, y_data, r_data = time_windows(self.norm_emg_data,
                                              self.rerepetition_data,
                                              self.frequency,
                                              self.restimulus_data,
                                              duration,
                                              increment,
                                              which_reps,
                                              which_moves,
                                              dtype)
        logger.debug("Finished spliting EMG data in {} seconds.".format(
            time() - start
        ))
        return x_data, y_data, r_data

    def split_data(self, x_data: np.ndarray, y_data: np.ndarray,
                   r_data: np.ndarray,
                   train_reps: list = DB2_INFO["default_trainreps"],
                   test_reps: list = DB2_INFO["default_testreps"]) \
            -> Tuple[np.ndarray]:
        return split_sets(x_data, y_data, r_data, train_reps, test_reps)


class PreprocessorDB3(SubjectDB3):
    def __init__(self, subject_number: int, dataset_path: str,
                 subsamp_rate: int = 10) -> None:
        self.subsamp_rate = subsamp_rate
        super().__init__(subject_number, dataset_path)
        self.load_dataset()
        self.norm_emg_data = None
        self.emg_data = self.emg_data[::subsamp_rate]
        self.rerepetition_data = self.rerepetition_data[::subsamp_rate]
        self.restimulus_data = self.restimulus_data[::subsamp_rate]

    def normalize(self, train_reps: list = DB3_INFO["default_trainreps"],
                  movements: list = [], which_moves: list = []):
        """Preprocess train+test data to mean 0, std 1 based on training data only.

        Args:
            train_reps (array): Which repetitions are in the training set
            movements (array, optional): Movement labels, required if using which_moves
            which_moves (array, optional): Which movements to return - if None use all

        Returns:
            array: Rescaled EMG data
        """
        logger.debug("Start normalizing EMG data...")
        start = time()
        self.norm_emg_data = normalize(self.emg_data, self.rerepetition_data,
                                       train_reps, movements, which_moves)
        logger.debug("Finished normalizing EMG data in {} seconds.".format(
            time() - start
        ))

    def get_windows(self, duration: int = 200, increment: int = 10,
                    which_reps: list = [], which_moves: list = [],
                    dtype: type = np.float32) -> Tuple[np.ndarray]:
        """Get set of windows based on repetition and movement criteria and\
            associated label + repetition data.

        Args:
            which_reps (array): Which repetitions to return
            window_len (int): Desired window length in miliseconds
            window_inc (int): Desired window increment in miliseconds
            which_moves (array, optional): Which movements to return - if None use all
            dtype (TYPE, optional): What precision to use for EMG data

        Returns:
            X_data (array): Windowed EMG data
            Y_data (array): Movement label for each window
            R_data (array): Repetition label for each window
        """
        which_reps = self.rep_labels if not which_reps else which_reps
        logger.debug("Start spliting EMG data into time windows...")
        start = time()
        x_data, y_data, r_data = time_windows(self.norm_emg_data,
                                              self.rerepetition_data,
                                              self.frequency,
                                              self.restimulus_data,
                                              duration,
                                              increment,
                                              which_reps,
                                              which_moves,
                                              dtype)
        logger.debug("Finished spliting EMG data in {} seconds.".format(
            time() - start
        ))
        return x_data, y_data, r_data

    def split_data(self, x_data: np.ndarray, y_data: np.ndarray,
                   r_data: np.ndarray,
                   train_reps: list = DB3_INFO["default_trainreps"],
                   test_reps: list = DB3_INFO["default_testreps"]) \
            -> Tuple[np.ndarray]:
        return split_sets(x_data, y_data, r_data, train_reps, test_reps)
