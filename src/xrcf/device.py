import os
import sys
import errno
import logging

import numpy as np
import mcareader as mca

logger = logging.getLogger('xrcf')

# suppress warning messages from mcareader
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

class SDD:

    def __init__(self, file_path):
        self.file_path = file_path

        logger.debug('Opening file: {}'.format(self.file_path))

        # check if SDD data file exists
        if not os.path.isfile(self.file_path):
            raise IOError(errno.ENOENT, 'No such file or directory', self.file_path)

        self.data = mca.Mca(self.file_path)
        bins, self.counts = self.data.get_points(trim_zeros=False)
        steps = np.gradient(bins)
        self.bins = np.append(bins, bins[-1] + steps[-1])

class GPC:

    def __init__(self, file_path, merge_bins=False):
        self.file_path = file_path

        logger.debug('Opening file: {}'.format(self.file_path))

        # check if GPC data file exists
        if not os.path.isfile(self.file_path):
            raise IOError(errno.ENOENT, 'No such file or directory', self.file_path)

        self.number_bins = np.loadtxt(self.file_path, skiprows=11, max_rows=1, dtype=int)[-1] + 1
        self._counts = np.loadtxt(self.file_path, skiprows=12, max_rows=self.number_bins, dtype=int)
        self._bins = np.linspace(0, self.number_bins, self.number_bins+1, dtype=int)
        self.counts = self._counts
        self.bins = self._bins

        if merge_bins > 1:
            n = merge_bins
            self.counts = self._counts.reshape((self._counts.shape[0]//n, n)).sum(axis=1)
            self.bins = self._bins[:-1].reshape((self._bins.shape[0]//n, n))[:, 0]
            steps = np.gradient(self.bins)
            self.bins = np.append(self.bins, self.bins[-1] + steps[-1]).astype(int)

