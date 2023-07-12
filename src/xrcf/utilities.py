import os
import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger('xrcf')

amptek_dir = '../src/xrcf/data/amptek/'
henke_dir = '../src/xrcf/data/henke/'
sri_dir = '../src/xrcf/data/sri/'
gasses = [ 'p10' ]
pressures = [ 400, 600, 800 ]

#//////////////////////////////////////////////////////////////////////////////
# convert channel to energy [eV]
#//////////////////////////////////////////////////////////////////////////////

def electronvolts(channel, α, β):
    """
    electronvolts(channel, α, β)

    Conversion of channel to energy in electron volts [eV].

    Parameters
    ----------
    channel : array_like of floats
        Channel(s).
    α : float
        Channel offset.
    β : float
        Conversion factor [channel / eV].

    Returns
    -------
    (channel - α) / β : array_like of floats

    """
    return (channel - α) / β

#//////////////////////////////////////////////////////////////////////////////
# efficiency of Amptek SDD with C1/C2 window
#//////////////////////////////////////////////////////////////////////////////
def sdd_c1_c2_efficiency_function(window='c1', efficiency='pe'):

    # check if Amptek data file exists
    file_path = amptek_dir + 'c1_c2_efficiency.txt'
    if not os.path.isfile(file_path):
        raise IOError(errno.ENOENT, 'No such file or directory', file_path)

    data = np.loadtxt(file_path, skiprows=12, dtype=float)

    x = data[:, 0] * 1e3
    y = data[:, 2]

    if window == 'c1' and efficiency == 'total':
        y = data[:, 1]
    elif window == 'c1' and efficiency == 'pe':
        y = data[:, 2]
    elif window == 'c2' and efficiency == 'total':
        y = data[:, 3]
    elif window == 'c2' and efficiency == 'pe':
        y = data[:, 4]
    else:
        raise ValueError('Invalid (window, efficiency) values: ({}, {}); valid window values: c1 or c2; valid efficiency values: total or pe'.format(window, efficiency))

    return interp1d(x, y)

#//////////////////////////////////////////////////////////////////////////////
# efficiency of SRI CMOS imager
#//////////////////////////////////////////////////////////////////////////////
def sri_cmos_efficiency_function():

    # check if SRI data file exists
    file_path = sri_dir + 'EIS_no_ARC_CMOS.dat'
    if not os.path.isfile(file_path):
        raise IOError(errno.ENOENT, 'No such file or directory', file_path)

    data = np.loadtxt(file_path, skiprows=0, dtype=float)

    x = data[:, 0]
    y = data[:, 1]

    return interp1d(x, y)

#//////////////////////////////////////////////////////////////////////////////
# transmission of VYNS window
#//////////////////////////////////////////////////////////////////////////////

def transmission_vyns_function():

    # check if Henke data file exists
    file_path = henke_dir + 'vyns.txt'
    if not os.path.isfile(file_path):
        raise IOError(errno.ENOENT, 'No such file or directory', file_path)

    data = np.loadtxt(file_path, skiprows=2, dtype=float)

    x = data[:, 0]
    y = data[:, 1]

    return interp1d(x, y)

#//////////////////////////////////////////////////////////////////////////////
# transmission of gas
#//////////////////////////////////////////////////////////////////////////////

def transmission_gas_function(gas, pressure):

    # check if gas is valid
    if gas not in gasses:
        raise ValueError('Gas {} is not valid; valid gasses: {}'.format(gas, gasses))

    # check if pressure is valid
    if pressure not in pressures:
        raise ValueError('Pressure {} Torr is not valid; valid pressures [Torr]: {}'.format(pressure, pressures))

    # check if Henke data file exists
    pressure = str(int(pressure))
    file_path = henke_dir + gas + '-' + pressure + '-torr.txt'
    if not os.path.isfile(file_path):
        raise IOError(errno.ENOENT, 'No such file or directory', file_path)

    data = np.loadtxt(file_path, skiprows=2, dtype=float)

    x = data[:, 0]
    y = data[:, 1]

    return interp1d(x, y)

