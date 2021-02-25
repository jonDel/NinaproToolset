"""Base class for working with NinaPro datasets."""
from abc import ABC, abstractmethod
from pathlib import Path


class Dataset(ABC):
    def __init__(self, db_number: int, subject_number: int, dataset_path: str):
        self.db_number = db_number
        self.subject_number = subject_number
        self.dataset_path = dataset_path

    @property    
    def rep_labels(self):
        return self._rep_labels

    @rep_labels.setter
    def rep_labels(self, value):
        self._rep_labels = value

    @property    
    def move_labels(self):
        return self._move_labels

    @move_labels.setter
    def move_labels(self, value):
        self._move_labels = value

    @property    
    def db_number(self):
        return self._db_number

    @db_number.setter
    def db_number(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "db_number must be an integer, a type '{}' was given!".
                format(type(value).__name__)
            )
        self._db_number = value

    @property    
    def nb_subjects(self):
        return self._nb_subjects

    @nb_subjects.setter
    def nb_subjects(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "nb_subjects must be an integer, a type '{}' was given!".
                format(type(value).__name__)
            )
        self._nb_subjects = value

    @property    
    def subject_number(self):
        return self._subject_number

    @subject_number.setter
    def subject_number(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "subject_number must be an integer, a type '{}' was given!".
                format(type(value).__name__)
            )
        if value > self.nb_subjects:
            raise TypeError(
                "subject_number must no be greater than the total number of "
                "subjects in database, number '{}' was given!".
                format(type(value).__name__)
            )
        self._subject_number = value

    @property    
    def nb_channels(self):
        return self._nb_channels

    @nb_channels.setter
    def nb_channels(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "nb_channels must be an integer, a type '{}' was given!".
                format(type(value).__name__)
            )
        self._nb_channels = value

    @property    
    def nb_reps(self):
        return self._nb_reps

    @nb_reps.setter
    def nb_reps(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "nb_reps must be an integer, a type '{}' was given!".
                format(type(value).__name__)
            )
        self._nb_reps = value

    @property    
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "frequency must be an integer, a type '{}' was given!".
                format(type(value).__name__)
            )
        self._frequency = value

    @property    
    def nb_moves(self):
        return self._nb_moves

    @nb_moves.setter
    def nb_moves(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "nb_moves must be an integer, a type '{}' was given!".
                format(type(value).__name__)
            )
        self._nb_moves = value

    @abstractmethod
    def load_dataset(self, rest_length_cap: int = 1000) -> None:
        """Function for extracting data from raw NinaPro files.

        Args:
            folder_path (string): Path to folder containing raw mat files
            subject (int): 1-27 which subject's data to import
            rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

        Returns:
            Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
                demarked and the number of repetitions with capped off rest data
        """
        pass

    def _get_mat_files(self) -> list:
        dataset_path = Path(self.dataset_path)
        mat_files = sorted([sfile for sfile in dataset_path.rglob('*.*')
                           if str(sfile).lower().endswith('.mat')])
        if not mat_files:
            raise FileNotFoundError("No *.mat files found in folder {}!".
                                    format(dataset_path))
        return mat_files
