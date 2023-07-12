#!/usr/bin/env python

import os
import errno
import sys
import importlib
import argparse
import logging
import math

import configparser
import ast
import json
import collections
from collections import OrderedDict

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

spec = importlib.util.spec_from_file_location('xrcf', '../src/xrcf/__init__.py')
xrcf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = xrcf
spec.loader.exec_module(xrcf)

from xrcf.device import SDD, GPC
from xrcf.utilities import electronvolts, transmission_vyns_function, \
    transmission_gas_function, transmission_silicon_function, \
    silicon_escape_probability_function

#//////////////////////////////////////////////////////////////////////////////
# logging
#//////////////////////////////////////////////////////////////////////////////
xrcf.logger.setLevel(logging.DEBUG)
# xrcf.logger.setLevel(logging.INFO)

# test
# xrcf.logger.debug('This is a debug message')
# xrcf.logger.warning('This is a warning message')
# xrcf.logger.error('This is an error message')
# xrcf.logger.info('This is an info message')

#//////////////////////////////////////////////////////////////////////////////
# parse arguments
#//////////////////////////////////////////////////////////////////////////////

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to configuration file')
args = parser.parse_args()

#//////////////////////////////////////////////////////////////////////////////
# config parser
#//////////////////////////////////////////////////////////////////////////////

# check if configuration file exists
if not os.path.isfile(args.file):
    raise IOError(errno.ENOENT, 'No such file or directory', args.file)

config = configparser.ConfigParser()
config.read(args.file)

energy = config.getfloat('configuration', 'energy')

savefig = ast.literal_eval(config.get('configuration', 'savefig'))

sdd_file_path = ast.literal_eval(config.get('configuration', 'sdd_file'))
gpc_file_path = ast.literal_eval(config.get('configuration', 'gpc_file'))

distance_sdd = config.getfloat('SDD', 'distance')
area_sdd     = config.getfloat('SDD', 'area')
start_sdd    = config.getint('SDD', 'start')
stop_sdd     = config.getint('SDD', 'stop')

gas_gpc             = ast.literal_eval(config.get('GPC', 'gas'))
pressure_gpc        = config.getfloat('GPC', 'pressure')
# voltage_gpc         = config.getfloat('GPC', 'voltage')
distance_gpc        = config.getfloat('GPC', 'distance')
diameter_gpc        = config.getfloat('GPC', 'diameter')
transmission_mesh   = config.getfloat('GPC', 'transmission_mesh')

transmission_window = None
try:
    transmission_window = config.getfloat('GPC', 'transmission_window')
except:
    transmission_window = None

transmission_gas    = None
try:
    transmission_gas = config.getfloat('GPC', 'transmission_gas')
except:
    transmission_gas = None

start_gpc           = config.getint('GPC', 'start')
stop_gpc            = config.getint('GPC', 'stop')

title_sdd  = ast.literal_eval(config.get('SDD', 'title'))
xlabel_sdd = ast.literal_eval(config.get('SDD', 'xlabel'))
ylabel_sdd = ast.literal_eval(config.get('SDD', 'ylabel'))
xlim_sdd   = ast.literal_eval(config.get('SDD', 'xlim'))
logy_sdd   = config.getboolean('SDD', 'logy')

title_gpc  = ast.literal_eval(config.get('GPC', 'title'))
xlabel_gpc = ast.literal_eval(config.get('GPC', 'xlabel'))
ylabel_gpc = ast.literal_eval(config.get('GPC', 'ylabel'))
xlim_gpc   = ast.literal_eval(config.get('GPC', 'xlim'))
logy_gpc   = config.getboolean('GPC', 'logy')

#//////////////////////////////////////////////////////////////////////////////
# fetch data from files
#//////////////////////////////////////////////////////////////////////////////

sdd = SDD(sdd_file_path)
gpc = GPC(gpc_file_path)

#//////////////////////////////////////////////////////////////////////////////
# quantum efficiency calculation
#//////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------
# gas proportional counter
#-----------------------------------------------------------------------

if not transmission_gas:
    transmission_gas_ = transmission_gas_function(gas_gpc, pressure_gpc)
    transmission_gas = transmission_gas_(energy)
    xrcf.logger.info(
        'Transmission of gas is not set in the configuration file. '
        'A transmission value of {} will be used for the {} gas assuming a pressure of {} Torr and a temperature of 295 K.'
        .format(transmission_gas, gas_gpc, pressure_gpc))

if not transmission_window:
    transmission_window_ = transmission_vyns_function()
    transmission_window = transmission_window_(energy)
    xrcf.logger.info(
        'Transmission of window is not set in the configuration file. '
        'A transmission value of {} will be used for the window.'
        .format(transmission_window))

absorption_gas = 1 - transmission_gas
qe_gpc = transmission_mesh * transmission_window * absorption_gas

counts_gpc = np.sum(gpc.counts[start_gpc:stop_gpc])
area_gpc = math.pi * diameter_gpc * diameter_gpc / 4

time_gpc = gpc.live_time
rate_gpc = counts_gpc / time_gpc
solid_angle_gpc = area_gpc / distance_gpc / distance_gpc

xrcf.logger.info('Counts from gas proportional counter: {}'.format(counts_gpc))
xrcf.logger.info('Live time of gas proportional counter [s]: {}'.format(time_gpc))
xrcf.logger.info('Estimated count rate of gas proportional counter [counts/s]: {}'.format(rate_gpc))
xrcf.logger.info('Estimated quantum efficiency of gas proportional counter: {}'.format(qe_gpc))

#-----------------------------------------------------------------------
# detector
#-----------------------------------------------------------------------

transmission_silicon_ = transmission_silicon_function()
absorption_silicon = 1 - transmission_silicon_(energy)
silicon_escape_probability_ = silicon_escape_probability_function()
silicon_escape_probability = silicon_escape_probability_(energy)
p = silicon_escape_probability * absorption_silicon

counts_det = np.sum(sdd.counts[start_sdd:stop_sdd]).astype(int)
time_det = sdd.live_time
area_det = area_sdd
distance_det = distance_sdd
counts_silicon = counts_det / (1-p)

rate_det = counts_det / time_det
rate_silicon = counts_silicon / time_det
solid_angle_det = area_det / distance_det / distance_det

# estimated quantum efficiency of detector
qe_det = rate_det / solid_angle_det / rate_gpc * solid_angle_gpc * qe_gpc
qe_silicon = rate_silicon / solid_angle_det / rate_gpc * solid_angle_gpc * qe_gpc

xrcf.logger.info('Absorption of 500-micron-thick silicon: {}'.format(absorption_silicon))
xrcf.logger.info('Silicon escape probability: {}'.format(silicon_escape_probability))
xrcf.logger.info('p: {}'.format(p))
xrcf.logger.info('Counts from detector: {}'.format(counts_det))
xrcf.logger.info('Estimated counts from silicon: {}'.format(counts_silicon))
xrcf.logger.info('Live time of detector [s]: {}'.format(time_det))
xrcf.logger.info('Estimated count rate of detector [counts/s]: {}'.format(rate_det))
xrcf.logger.info('Estimated count rate of silicon [counts/s]: {}'.format(rate_silicon))
xrcf.logger.info('Estimated quantum efficiency of detector: {}'.format(qe_det))
xrcf.logger.info('Estimated quantum efficiency of silicon: {}'.format(qe_silicon))

#//////////////////////////////////////////////////////////////////////////////
# plot data
#//////////////////////////////////////////////////////////////////////////////

fig = plt.figure(figsize=(9, 7))       
gs = fig.add_gridspec(nrows=2, ncols=1)

ax_sdd = fig.add_subplot(gs[0, 0])
ax_gpc = fig.add_subplot(gs[1, 0])

# SDD plot

title_sdd = r'{}'.format(title_sdd)
ax_sdd.set_title(title_sdd, fontsize=14)

ax_sdd.stairs(sdd.counts, sdd.bins, fill=True, color='C0')
ax_sdd.stairs(sdd.counts[start_sdd:stop_sdd], sdd.bins[start_sdd:stop_sdd+1], fill=True, color='C2')
# ax_sdd.stairs(sdd.counts, sdd.bins, fill=True, color='C0', label='1 channel per bin')
ax_sdd.axvline(x=start_sdd, color='k', linestyle='--', linewidth=1)
ax_sdd.axvline(x=stop_sdd, color='k', linestyle='--', linewidth=1)

ax_sdd.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax_sdd.set_xlim(xlim_sdd)

ax_sdd.set_xlabel(xlabel_sdd, horizontalalignment='right', x=1.0, fontsize=14)
ax_sdd.set_ylabel(ylabel_sdd, horizontalalignment='right', y=1.0, fontsize=14)

ax_sdd.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

ax_sdd.xaxis.set_minor_locator(AutoMinorLocator())
ax_sdd.yaxis.set_minor_locator(AutoMinorLocator())

ax_sdd.yaxis.offsetText.set_fontsize(12)
ax_sdd.tick_params(axis='both', which='major', labelsize=12)

# ax_sdd.legend(loc='lower right', fontsize=12)

if logy_sdd:
    ax_sdd.set_yscale('log')
    ax_sdd.set_ylim(bottom=0.5)
else:
    ax_sdd.set_ylim(bottom=0)

# GPC plot

title_gpc = r'{}'.format(title_gpc)
ax_gpc.set_title(title_gpc, fontsize=14)

ax_gpc.stairs(gpc.counts, gpc.bins, fill=True, color='C0')
ax_gpc.stairs(gpc.counts[start_gpc:stop_gpc], gpc.bins[start_gpc:stop_gpc+1], fill=True, color='C2')
# ax_gpc.stairs(gpc.counts, gpc.bins, fill=True, color='C0', label='1 channel per bin')
# ax_gpc.legend(loc='lower right', fontsize=12)
ax_gpc.axvline(x=start_gpc, color='k', linestyle='--', linewidth=1)
ax_gpc.axvline(x=stop_gpc, color='k', linestyle='--', linewidth=1)

ax_gpc.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax_gpc.set_xlim(xlim_gpc)

ax_gpc.set_xlabel(xlabel_gpc, horizontalalignment='right', x=1.0, fontsize=14)
ax_gpc.set_ylabel(ylabel_gpc, horizontalalignment='right', y=1.0, fontsize=14)

ax_gpc.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

ax_gpc.xaxis.set_minor_locator(AutoMinorLocator())
ax_gpc.yaxis.set_minor_locator(AutoMinorLocator())

ax_gpc.yaxis.offsetText.set_fontsize(12)
ax_gpc.tick_params(axis='both', which='major', labelsize=12)

if logy_gpc:
    ax_gpc.set_yscale('log')
    ax_gpc.set_ylim(bottom=0.5)
else:
    ax_gpc.set_ylim(bottom=0)

plt.tight_layout()

if savefig:
    plt.savefig(savefig, bbox_inches='tight')
else:
    plt.show()

plt.cla()
plt.clf()
plt.close()

