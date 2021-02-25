"""Frequency Domain Feature extractors."""
from time import time
import logging
from numpy import ndarray
from pywt import wavedec
from tqdm import tqdm
import numpy as np
from scipy.signal import spectrogram
from .fext_base import FeatureExtractor


logger = logging.getLogger(__name__)


class MDWTExtractor(FeatureExtractor):
    """Marginal Discrete Wavelet Transform."""
    def extract(self, windowed_data: ndarray) -> ndarray:
        logger.debug("Extracting Marginal Discrete Wavelet Transform feature"
                     " from {} time windows...".format(windowed_data.shape[0]))
        start = time()
        feat_array = [self._mdwt(sample) for sample in
                      tqdm(windowed_data)]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))

    def _mdwt(self, sample, num_levels=3, mother_wavelet="db7"):
        """marginal discrete wavelet transform"""
        num_channels = sample.shape[1]
        all_coeff = []
        for i in range(num_channels):
            coeffs = wavedec(sample[:, i], mother_wavelet, level=num_levels)
            # "Marginal" of each level
            for j in range(num_levels):
                all_coeff.append(np.sum(np.abs(coeffs[j])))
        all_coeff = np.array(all_coeff)
        return all_coeff


class SpectrogramExtractor(FeatureExtractor):
    """Spectrogram"""
    def extract(self, windowed_data: ndarray, frequency: int = 200) -> ndarray:
        logger.debug("Extracting Marginal Discrete Wavelet Transform feature"
                     " from {} time windows...".format(windowed_data.shape[0]))
        start = time()
        #feat_array = [spectrogram(sample, frequency)[2]
        #              for sample in tqdm(windowed_data)]
        feat_array = spectrogram(windowed_data, frequency)[2]
        logger.debug("Done extracting feature in {} seconds.".
                     format(time() - start))
        return self.transform(np.array(feat_array))
