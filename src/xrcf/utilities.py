import os
import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger('xrcf')

henke_dir = '../src/xrcf/data/henke/'
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

