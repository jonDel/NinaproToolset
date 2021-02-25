"""Utility functions for preprocessing data."""
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler


def to_categorical(y, nb_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.

    Taken from:
    https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
    v2.0.2 of Keras to remove unnecessary Keras dependency
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def get_idxs(in_array, to_find):
    """Find the positions of observations of one array in another an array.

    Args:
        in_array (array): Array in which to locate elements of to_find
        to_find (array): Array of elements to locate in in_array

    Returns:
        TYPE: Indices of all elements of to_find in in_array
    """
    targets = ([np.where(in_array == x) for x in to_find])
    return np.squeeze(np.concatenate(targets, axis=1))


def normalize(emg_data: np.ndarray, rerepetition_data: np.ndarray,
              train_reps: list, movements: list = [],
              which_moves: list = []) -> np.ndarray:
    """Preprocess train+test data to mean 0, std 1 based on training data only.

    Args:
        train_reps (array): Which repetitions are in the training set
        movements (array, optional): Movement labels, required if using which_moves
        which_moves (array, optional): Which movements to return - if None use all

    Returns:
        array: Rescaled EMG data
    """
    train_targets = get_idxs(rerepetition_data, train_reps)
    # Keep only selected movement(s)
    if which_moves and movements:
        move_targets = get_idxs(movements[train_targets], which_moves)
        train_targets = train_targets[move_targets]
    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).\
        fit(emg_data[train_targets, :])
    return scaler.transform(emg_data)


def time_windows(emg_data: np.ndarray, rerepetition_data: np.ndarray,
                 frequency: int, restimulus_data, duration: int = 200,
                 increment: int = 10, which_reps: list = [],
                 which_moves: list = [],
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
    window_len = int(duration*frequency*0.001)
    window_inc = int(increment*frequency*0.001)
    nb_obs = emg_data.shape[0]
    nb_channels = emg_data.shape[1]
    # All possible window end locations given an increment size
    possible_targets = np.array(range(window_len - 1, nb_obs, window_inc))
    targets = get_idxs(rerepetition_data[possible_targets],
                       which_reps)
    # Re-adjust back to original range (for indexinging into rep/move)
    targets = (window_len - 1) + targets * window_inc
    # Keep only selected movement(s)
    if which_moves:
        move_targets = get_idxs(restimulus_data[targets], which_moves)
        targets = targets[move_targets]
    x_data = np.zeros([targets.shape[0], window_len, nb_channels, 1],
                      dtype=dtype)
    y_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    r_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    for i, win_end in enumerate(targets):
        win_start = win_end - (window_len - 1)
        if restimulus_data[win_start] == restimulus_data[win_end]:
            x_data[i, :, :, 0] = emg_data[win_start:win_end + 1, :]  # Include end
            y_data[i] = restimulus_data[win_end]
            r_data[i] = rerepetition_data[win_end]
    return x_data, y_data, r_data


def split_sets(x_data: np.ndarray, y_data: np.ndarray, r_data: np.ndarray,
               train_reps: list, test_reps) -> Tuple[np.ndarray]:
    train_idx = get_idxs(r_data, train_reps)
    x_train = x_data[train_idx]
    test_idx = get_idxs(r_data, test_reps)
    x_test = x_data[test_idx]
    y_train = y_data[train_idx]
    y_test = y_data[test_idx]
    y_train = to_categorical(y_train)
    return x_train, y_train, x_test, y_test
