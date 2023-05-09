import os
import sys
import errno
import logging

import numpy as np
import xrcf.mcareader as mca

logger = logging.getLogger('xrcf')

# suppress warning messages from mcareader
if not sys.warnoptions:
    import warnings
    warnings.filterwarnings('ignore', module='xrcf.mcareader')

class SDD:

    def __init__(self, file_path):
        self.file_path = file_path

        logger.debug('Opening file: {}'.format(self.file_path))

        # check if SDD data file exists
        if not os.path.isfile(self.file_path):
            raise IOError(errno.ENOENT, 'No such file or directory', self.file_path)

        # use mcareader to get SDD data
        self.data = mca.Mca(self.file_path)
        bins, self.counts = self.data.get_points(trim_zeros=False)
        steps = np.gradient(bins)
        self.bins = np.append(bins, bins[-1] + steps[-1])
        self.live_time = float(self.data.get_variable('LIVE_TIME'))
        self.real_time = float(self.data.get_variable('REAL_TIME'))

class GPC:

    def __init__(self, file_path, merge_bins=False):
        self.file_path = file_path

        logger.debug('Opening file: {}'.format(self.file_path))

        # check if GPC data file exists
        if not os.path.isfile(self.file_path):
            raise IOError(errno.ENOENT, 'No such file or directory', self.file_path)

        # get GPC data from file

        self.number_bins = -1
        interval = [ -1, -1 ]
        time = [ -1, -1 ]
        skip_rows = -1

        with open(self.file_path) as f:
            time_idx = -1
            for idx, line in enumerate(f, 1):
                # can't use next() here as doing so will skip the next
                # iteration in the iterator loop
                if line.startswith('$MEAS_TIM:'):
                    time_idx = idx+1
                if idx == time_idx:
                    time = [ eval(_) for _ in line.strip().split() ]
                # next() can be used here since the iterator loop will be
                # terminated
                if line.startswith('$DATA:'):
                    interval = [ eval(_) for _ in next(f, '').strip().split() ]
                    self.number_bins = interval[-1] - interval[0] + 1
                    skip_rows = idx+1
                    break

        self._counts = np.loadtxt(self.file_path, skiprows=skip_rows, max_rows=self.number_bins, dtype=int)
        self._bins = np.linspace(interval[0], interval[-1]+1, self.number_bins+1, dtype=int)
        self.counts = self._counts
        self.bins = self._bins
        self.live_time = time[0]
        self.run_time = time[1]

        # merge bins
        if merge_bins > 1:
            n = merge_bins
            self.counts = self._counts.reshape((self._counts.shape[0]//n, n)).sum(axis=1)
            self.bins = self._bins[:-1].reshape((self._bins.shape[0]//n, n))[:, 0]
            steps = np.gradient(self.bins)
            self.bins = np.append(self.bins, self.bins[-1] + steps[-1]).astype(int)

