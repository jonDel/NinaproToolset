"""NinaPro database 3 helper class."""
from pathlib import Path
import logging
from time import time
from numpy import ndarray
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from .base import Dataset
from .dbinfo import DB1_INFO, DB2_INFO, DB3_INFO


logger = logging.getLogger(__name__)


class Subject(Dataset):
    def __init__(self, db_number: int, subject_number: int,
                 db_info: dict, dataset_path: str):
        self.nb_subjects = db_info["number_of_subjects"]
        self.nb_reps = db_info["number_of_reps"]
        self.frequency = db_info["acquisition_freq"]
        self.nb_channels = db_info["channels"](subject_number)
        self.nb_moves = db_info["moves"](subject_number)
        self.move_labels = np.array(range(1, self.nb_moves+1))
        self.rep_labels = np.array(range(1, self.nb_reps+1))
        self.emg_data: ndarray = None
        self.rerepetition_data = None
        self.restimulus_data = None
        super().__init__(db_number, subject_number, dataset_path)


class SubjectDB1(Subject):
    def __init__(self, subject_number: int, dataset_path: str):
        super().__init__(1, subject_number, DB1_INFO, dataset_path)

    def load_dataset(self, rest_length_cap: int = 1000) -> None:
        mat_files = self._get_mat_files()
        logger.debug("Loading file {} ...".format(mat_files[0]))
        start = time()
        data = sio.loadmat(mat_files[0])
        logger.debug("Done loading file in {} seconds.".format(time() - start))
        emg = np.squeeze(np.array(data['emg']))
        rep = np.squeeze(np.array(data['rerepetition']))
        move = np.squeeze(np.array(data['restimulus']))
        for mfile in mat_files[1:]:
            logger.debug("Loading file {} ...".format(mfile))
            start = time()
            data = sio.loadmat(mfile)
            logger.debug("Done loading file in {} seconds.".
                         format(time() - start))
            emg = np.vstack((emg, np.array(data['emg'])))
            rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
            # Fix for numbering
            move_tmp = np.squeeze(np.array(data['restimulus']))
            move_tmp[move_tmp != 0] += max(move)
            move = np.append(move, move_tmp)
        move = move.astype('int8')  # To minimise overhead
        # Label repetitions using new block style: rest-move-rest regions
        move_regions = np.where(np.diff(move))[0]
        rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
        nb_reps = int(round(move_regions.shape[0] / 2))
        last_end_idx = int(round(move_regions[0] / 2))
        nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
        cur_rep = 1
        rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
        for i in range(nb_reps - 1):
            rep_regions[2 * i] = last_end_idx
            midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                      move_regions[2 * (i + 1)]) / 2)) + 1
            trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
            if trailing_rest_samps <= rest_length_cap * self.frequency:
                rep[last_end_idx:midpoint_idx] = cur_rep
                last_end_idx = midpoint_idx
                rep_regions[2 * i + 1] = midpoint_idx - 1
            else:
                rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                               int(round(rest_length_cap * self.frequency)))
                rep[last_end_idx:rep_end_idx] = cur_rep
                last_end_idx = ((move_regions[2 * (i + 1)] -
                                 int(round(rest_length_cap * self.frequency))))
                rep_regions[2 * i + 1] = rep_end_idx - 1
            cur_rep += 1
            if cur_rep > nb_unique_reps:
                cur_rep = 1
        end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
        rep[last_end_idx:end_idx] = cur_rep
        rep_regions[-2] = last_end_idx
        rep_regions[-1] = end_idx - 1
        self.emg_data = emg
        self.rerepetition_data = rep
        self.restimulus_data = move


class SubjectDB2(Subject):
    def __init__(self, subject_number: int, dataset_path: str):
        super().__init__(2, subject_number, DB2_INFO, dataset_path)

    def load_dataset(self, rest_length_cap: int = 1000) -> None:
        mfile_e1, mfile_e2, mfile_e3 = self._get_mat_files()
        logger.debug("Loading file {} ...".format(mfile_e1))
        start = time()
        data = sio.loadmat(mfile_e1)
        logger.debug("Done loading file in {} seconds.".format(time() - start))
        emg = np.squeeze(np.array(data['emg']))
        rep = np.squeeze(np.array(data['rerepetition']))
        move = np.squeeze(np.array(data['restimulus']))
        logger.debug("Loading file {} ...".format(mfile_e2))
        start = time()
        data = sio.loadmat(mfile_e2)
        logger.debug("Done loading file in {} seconds.".format(time() - start))
        emg = np.vstack((emg, np.array(data['emg'])))
        rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
        move_tmp = np.squeeze(np.array(data['restimulus']))
        move = np.append(move, move_tmp)  # Note no fix needed for this exercise
        logger.debug("Loading file {} ...".format(mfile_e3))
        start = time()
        data = sio.loadmat(mfile_e3)
        logger.debug("Done loading file in {} seconds.".format(time() - start))
        emg = np.vstack((emg, np.array(data['emg'])))
        data['repetition'][-1] = 0  # Fix for diffing
        rep = np.append(rep, np.squeeze(np.array(data['repetition'])))
        # Movements number in non-logical pattern [0  1  2  4  6  8  9 16 32 40]
        # Also note that for last file there is no 'rerepetition or 'restimulus'
        data['stimulus'][-1] = 0  # Fix for diffing
        data['stimulus'][np.where(data['stimulus'] == 1)] = 41
        data['stimulus'][np.where(data['stimulus'] == 2)] = 42
        data['stimulus'][np.where(data['stimulus'] == 4)] = 43
        data['stimulus'][np.where(data['stimulus'] == 6)] = 44
        data['stimulus'][np.where(data['stimulus'] == 8)] = 45
        data['stimulus'][np.where(data['stimulus'] == 9)] = 46
        data['stimulus'][np.where(data['stimulus'] == 16)] = 47
        data['stimulus'][np.where(data['stimulus'] == 32)] = 48
        data['stimulus'][np.where(data['stimulus'] == 40)] = 49
        move_tmp = np.squeeze(np.array(data['stimulus']))
        move = np.append(move, move_tmp)
        move = move.astype('int8')  # To minimise overhead
        # Label repetitions using new block style: rest-move-rest regions
        move_regions = np.where(np.diff(move))[0]
        rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
        nb_reps = int(round(move_regions.shape[0] / 2))
        last_end_idx = int(round(move_regions[0] / 2))
        # To account for 0 regions
        nb_unique_reps = np.unique(rep).shape[0] - 1
        cur_rep = 1
        rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
        for i in range(nb_reps - 1):
            rep_regions[2 * i] = last_end_idx
            midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                      move_regions[2 * (i + 1)]) / 2)) + 1

            trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
            if trailing_rest_samps <= rest_length_cap * self.frequency:
                rep[last_end_idx:midpoint_idx] = cur_rep
                last_end_idx = midpoint_idx
                rep_regions[2 * i + 1] = midpoint_idx - 1
            else:
                rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                               int(round(rest_length_cap * self.frequency)))
                rep[last_end_idx:rep_end_idx] = cur_rep
                last_end_idx = (move_regions[2 * (i + 1)] -
                                int(round(rest_length_cap * self.frequency)))
                rep_regions[2 * i + 1] = rep_end_idx - 1
            cur_rep += 1
            if cur_rep > nb_unique_reps:
                cur_rep = 1
        end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
        rep[last_end_idx:end_idx] = cur_rep
        rep_regions[-2] = last_end_idx
        rep_regions[-1] = end_idx - 1
        self.emg_data = emg
        self.rerepetition_data = rep
        self.restimulus_data = move


class SubjectDB3(Subject):
    def __init__(self, subject_number: int, dataset_path: str):
        super().__init__(3, subject_number, DB2_INFO, dataset_path)

    def load_dataset(self, rest_length_cap: int = 1000) -> None:
        mfile_e1, mfile_e2, mfile_e3 = self._get_mat_files()
        logger.debug("Loading file {} ...".format(mfile_e1))
        start = time()
        data = sio.loadmat(mfile_e1)
        logger.debug("Done loading file in {} seconds.".format(time() - start))
        emg = np.squeeze(np.array(data['emg']))
        rep = np.squeeze(np.array(data['rerepetition']))
        move = np.squeeze(np.array(data['restimulus']))
        logger.debug("Loading file {} ...".format(mfile_e2))
        start = time()
        data = sio.loadmat(mfile_e2)
        logger.debug("Done loading file in {} seconds.".format(time() - start))
        emg = np.vstack((emg, np.array(data['emg'])))
        rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
        move_tmp = np.squeeze(np.array(data['restimulus']))
        move = np.append(move, move_tmp)  # Note no fix needed for this exercise
        last_mov = max(move)
        logger.debug("Loading file {} ...".format(mfile_e3))
        start = time()
        data = sio.loadmat(mfile_e3)
        logger.debug("Done loading file in {} seconds.".format(time() - start))
        emg = np.vstack((emg, np.array(data['emg'])))
        rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
        data['restimulus'][np.where(data['restimulus'] == 1)] = last_mov + 1
        data['restimulus'][np.where(data['restimulus'] == 2)] = last_mov + 2
        data['restimulus'][np.where(data['restimulus'] == 4)] = last_mov + 3
        data['restimulus'][np.where(data['restimulus'] == 6)] = last_mov + 4
        data['restimulus'][np.where(data['restimulus'] == 8)] = last_mov + 5
        data['restimulus'][np.where(data['restimulus'] == 9)] = last_mov + 6
        data['restimulus'][np.where(data['restimulus'] == 16)] = last_mov + 7
        data['restimulus'][np.where(data['restimulus'] == 32)] = last_mov + 8
        data['restimulus'][np.where(data['restimulus'] == 40)] = last_mov + 9
        move_tmp = np.squeeze(np.array(data['restimulus']))
        # Note no fix needed for this exercise
        move = np.append(move, move_tmp)
        move = move.astype('int8')  # To minimise overhead
        # Label repetitions using new block style: rest-move-rest regions
        move_regions = np.where(np.diff(move))[0]
        rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
        nb_reps = int(round(move_regions.shape[0] / 2))
        last_end_idx = int(round(move_regions[0] / 2))
        # To account for 0 regions
        nb_unique_reps = np.unique(rep).shape[0] - 1
        cur_rep = 1
        rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
        for i in range(nb_reps - 1):
            rep_regions[2 * i] = last_end_idx
            midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                      move_regions[2 * (i + 1)]) / 2)) + 1
            trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
            if trailing_rest_samps <= rest_length_cap * self.frequency:
                rep[last_end_idx:midpoint_idx] = cur_rep
                last_end_idx = midpoint_idx
                rep_regions[2 * i + 1] = midpoint_idx - 1
            else:
                rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                               int(round(rest_length_cap * self.frequency)))
                rep[last_end_idx:rep_end_idx] = cur_rep
                last_end_idx = (move_regions[2 * (i + 1)] -
                                int(round(rest_length_cap * self.frequency)))
                rep_regions[2 * i + 1] = rep_end_idx - 1
            cur_rep += 1
            if cur_rep > nb_unique_reps:
                cur_rep = 1
        end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
        rep[last_end_idx:end_idx] = cur_rep
        rep_regions[-2] = last_end_idx
        rep_regions[-1] = end_idx - 1      
        self.emg_data = emg
        self.rerepetition_data = rep
        self.restimulus_data = move
