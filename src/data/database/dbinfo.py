import numpy as np


def get_db3_nb_moves(subject):
    if subject not in [1, 3, 10]:
        return DB3_INFO["moves"]
    elif subject == 1:
        return DB3_INFO["s1_moves"]
    elif subject == 3:
        return DB3_INFO["s3_moves"]
    return DB3_INFO["s10_moves"]


def get_db3_nb_channels(subject):
    if subject not in [7, 8]:
        return DB3_INFO["channels"]
    return DB3_INFO["s7_and_8_channels"]


def get_db3_mov_labels(subject):
    if subject not in [1, 3, 10]:
        return DB3_INFO["move_labels"]
    elif subject == 1:
        return DB3_INFO["s1_move_labels"]
    elif subject == 3:
        return DB3_INFO["s3_move_labels"]
    return DB3_INFO["s10_move_labels"]


DB1_INFO = {
    "database_number": 1,
    "number_of_reps": 10,
    "number_of_subjects": 27,
    "acquisition_freq": 100,
    "channels": lambda subject: 10,
    "moves": lambda subject: 53,
    "move_labels": lambda subject: np.array(range(1, 54)),
    "default_testreps": [2, 5, 7],
    "default_trainreps":  [1, 3, 4, 6, 8, 9, 10] 
}
DB2_INFO = {
    "database_number": 2,
    "number_of_reps": 6,
    "number_of_subjects": 40,
    "acquisition_freq": 2000,
    "channels": lambda subject: 12,
    "moves": lambda subject: 50,
    "move_labels": lambda subject: np.array(range(1, 51)),
    "default_testreps": [2, 5],
    "default_trainreps": [1, 3, 4, 6]
}
DB3_INFO = {
    "database_number": 3,
    "number_of_reps": 6,
    "number_of_subjects": 11,
    "acquisition_freq": 2000,
    "channels": get_db3_nb_channels,
    "s7_and_8_channels": 10,
    "moves": get_db3_nb_moves,
    "s1_moves": 39,
    "s3_moves": 49,
    "s10_moves": 43,
    "move_labels": get_db3_mov_labels,
    "s1_move_labels": np.array(range(1, 40)),
    "s3_move_labels": np.array(range(1, 50)),
    "s10_move_labels": np.array(range(1, 44)),
    "default_testreps": [2, 5], 
    "default_trainreps": [1, 3, 4, 6]
}