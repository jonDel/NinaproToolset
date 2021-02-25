"""Time Domain Feature extractors."""
from numpy import ndarray
from tqdm import tqdm
import numpy as np
from time import time
from .fext_base import FeatureExtractor
import logging


logger = logging.getLogger(__name__)


class IdentityExtractor(FeatureExtractor):
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Returning the same data reshaped.")
        return self.transform(np.array(windowed_data))


class RmsExtractor(FeatureExtractor):
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting RMS feature from {} time windows...".
                     format(windowed_data.shape[0]))
        start = time()
        feat_array = [np.sqrt(np.mean(np.square(sample), axis=0))
                      for sample in tqdm(windowed_data)]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))


class MavExtractor(FeatureExtractor):
    """Calculate the mean absolute value."""
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting Mean Absolute Value feature from {}"
                     " time windows...".format(windowed_data.shape[0]))
        start = time()
        feat_array = [np.sqrt(np.mean(np.square(sample), axis=0))
                      for sample in tqdm(windowed_data)]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))


class ZeroCrossExtractor(FeatureExtractor):
    """Count the number of zero crosses."""
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting Zero Cross Count feature from {}"
                     " time windows...".format(windowed_data.shape[0]))
        start = time()
        feat_array = []
        for window_sample in tqdm(windowed_data):
            temp = np.swapaxes(window_sample, 0, 0)
            feat_array.append(np.sum(np.abs(
                np.sign(temp[:-1])-np.sign(temp[1:])), axis=0)/2)
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))


class SlopeChangesExtractor(FeatureExtractor):
    """Count the number of slope changes."""
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting Slope Changes Count feature from {}"
                     " time windows...".format(windowed_data.shape[0]))
        start = time()
        feat_array = [self._num_slope_changes(sample) for sample in
                      tqdm(windowed_data)]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))

    def _num_slope_changes(self, sample):
        window_size = sample.shape[0]
        num_channels = sample.shape[1]
        num_slope_changes = np.zeros(num_channels, dtype=np.uint16)
        for i in range(num_channels):
            for j in range(window_size):
                # Check for slope changes
                if (j > 0) and (j < window_size - 1):
                    left = sample[j-1][i]
                    mid = sample[j][i]
                    right = sample[j+1][i]
                    condition_1 = (mid > left) and (mid > right)
                    condition_2 = (mid < left) and (mid < right)
                    if condition_1 or condition_2:
                        num_slope_changes[i] += 1
        return num_slope_changes


class WaveformLengthExtractor(FeatureExtractor):
    """Extract the waveform length."""
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting Waveform Length feature from {}"
                     " time windows...".format(windowed_data.shape[0]))
        start = time()
        feat_array = [self._waveform_length(sample) for sample in
                      tqdm(windowed_data)]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))

    def _waveform_length(self, sample: ndarray):
        window_size = sample.shape[0]
        num_channels = sample.shape[1]
        waveform_length = np.zeros(num_channels, dtype=np.uint16)
        for i in range(num_channels):
            for j in range(window_size):
                # Compute waveform length
                if j > 0:
                    left = sample[j - 1][i]
                    mid = sample[j][i]
                    waveform_length[i] += np.abs(mid - left)
        return waveform_length


class VarianceExtractor(FeatureExtractor):
    """Calculate the variance."""
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting Variance feature from {}"
                     " time windows...".format(windowed_data.shape[0]))
        start = time()
        feat_array = [np.var(sample, axis=0)
                      for sample in tqdm(windowed_data)]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))


class HistogramExtractor(FeatureExtractor):
    """Extract the histogram."""
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting Histogram feature from {}"
                     " time windows...".format(windowed_data.shape[0]))
        start = time()
        feat_array = [self._hist(sample) for sample in
                      tqdm(windowed_data)]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))

    def _hist(self, sample, bins=10, ranges=None, axis=1):
        if not ranges:
            ranges = [sample.min(axis=axis).min(),
                      sample.max(axis=axis).max()]
        return np.apply_along_axis(lambda a: np.histogram(a, bins=bins,
                                   range=ranges)[0], axis, sample)
