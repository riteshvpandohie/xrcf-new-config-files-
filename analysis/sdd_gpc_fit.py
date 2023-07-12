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
from xrcf.utilities import electronvolts

#//////////////////////////////////////////////////////////////////////////////
# logging
#//////////////////////////////////////////////////////////////////////////////
xrcf.logger.setLevel(logging.DEBUG)

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

savefig = ast.literal_eval(config.get('configuration', 'savefig'))

sdd_file_path = ast.literal_eval(config.get('configuration', 'sdd_file'))
gpc_file_path = ast.literal_eval(config.get('configuration', 'gpc_file'))

sdd_title  = ast.literal_eval(config.get('SDD', 'title'))
sdd_xlabel = ast.literal_eval(config.get('SDD', 'xlabel'))
sdd_ylabel = ast.literal_eval(config.get('SDD', 'ylabel'))
sdd_xlim   = ast.literal_eval(config.get('SDD', 'xlim'))
sdd_logy   = config.getboolean('SDD', 'logy')
sdd_a      = config.getfloat('SDD', 'amplitude')
sdd_mu     = config.getfloat('SDD', 'mean')
sdd_sigma  = config.getfloat('SDD', 'sigma')
sdd_start  = config.getint('SDD', 'start')
sdd_stop   = config.getint('SDD', 'stop')

sdd_α      = config.getfloat('SDD', 'α')
sdd_β      = config.getfloat('SDD', 'β')

gpc_title  = ast.literal_eval(config.get('GPC', 'title'))
gpc_xlabel = ast.literal_eval(config.get('GPC', 'xlabel'))
gpc_ylabel = ast.literal_eval(config.get('GPC', 'ylabel'))
gpc_xlim   = ast.literal_eval(config.get('GPC', 'xlim'))
gpc_logy   = config.getboolean('GPC', 'logy')
gpc_merge_bins = config.getint('GPC', 'merge_bins')
gpc_a      = config.getfloat('GPC', 'amplitude')
gpc_mu     = config.getfloat('GPC', 'mean')
gpc_sigma  = config.getfloat('GPC', 'sigma')
gpc_start  = config.getint('GPC', 'start')
gpc_stop   = config.getint('GPC', 'stop')

def power2(x):
    return (math.log(x) / math.log(2)).is_integer()

if gpc_merge_bins > 1:
    if not power2(gpc_merge_bins):
        raise ValueError('Not a power of two:', gpc_merge_bins)

#//////////////////////////////////////////////////////////////////////////////
# fetch data from files
#//////////////////////////////////////////////////////////////////////////////

sdd = SDD(sdd_file_path)
gpc = GPC(gpc_file_path, gpc_merge_bins)
gpc_ = GPC(gpc_file_path)

#//////////////////////////////////////////////////////////////////////////////
# fit data
#//////////////////////////////////////////////////////////////////////////////
sdd_fit = xrcf.mle.minimize(sdd.bins, sdd.counts, sdd_mu, sdd_sigma, sdd_a, start=sdd_start, stop=sdd_stop)
sdd_x = 0.5*(sdd_fit['bins_train'][1:]+sdd_fit['bins_train'][:-1])
sdd_y = sdd_fit['values']['a'] * norm.pdf(sdd_x, sdd_fit['values']['mu'], sdd_fit['values']['sigma']) * sdd_fit['values']['sigma'] * np.sqrt(2*np.pi)

# chi2
sdd_chi2 = ((sdd_y - sdd_fit['counts_train'])**2 / sdd_y).sum()
sdd_dof = len(sdd_y) - len(sdd_fit['values'])

# reduced chi2
print('chi2 / dof (SDD) = {} / {} = {}'.format(sdd_chi2, sdd_dof, sdd_chi2/sdd_dof))

sdd_x = np.linspace(sdd_fit['bins_train'][0], sdd_fit['bins_train'][-1], len(sdd_x)*100)
sdd_y = sdd_fit['values']['a'] * norm.pdf(sdd_x, sdd_fit['values']['mu'], sdd_fit['values']['sigma']) * sdd_fit['values']['sigma'] * np.sqrt(2*np.pi)

if gpc_merge_bins > 1:
    gpc_start = gpc_start // gpc_merge_bins
    gpc_stop = gpc_stop // gpc_merge_bins

gpc_fit = xrcf.mle.minimize(gpc.bins, gpc.counts, gpc_mu, gpc_sigma, gpc_a, start=gpc_start, stop=gpc_stop)
gpc_x = 0.5*(gpc_fit['bins_train'][1:]+gpc_fit['bins_train'][:-1])
gpc_y = gpc_fit['values']['a'] * norm.pdf(gpc_x, gpc_fit['values']['mu'], gpc_fit['values']['sigma']) * gpc_fit['values']['sigma'] * np.sqrt(2*np.pi)

# chi2
gpc_chi2 = ((gpc_y - gpc_fit['counts_train'])**2 / gpc_y).sum()
gpc_dof = len(gpc_y) - len(gpc_fit['values'])

# reduced chi2
print('chi2 / dof (GPC) = {} / {} = {}'.format(gpc_chi2, gpc_dof, gpc_chi2/gpc_dof))

gpc_x = np.linspace(gpc_fit['bins_train'][0], gpc_fit['bins_train'][-1], len(gpc_x)*100)
gpc_y = gpc_fit['values']['a'] * norm.pdf(gpc_x, gpc_fit['values']['mu'], gpc_fit['values']['sigma']) * gpc_fit['values']['sigma'] * np.sqrt(2*np.pi)

#//////////////////////////////////////////////////////////////////////////////
# plot data
#//////////////////////////////////////////////////////////////////////////////

fig = plt.figure(figsize=(9, 7))       
gs = fig.add_gridspec(nrows=2, ncols=1)

ax_sdd = fig.add_subplot(gs[0, 0])
ax_gpc = fig.add_subplot(gs[1, 0])

# SDD plot

sdd_title = r'{}'.format(sdd_title)
ax_sdd.set_title(sdd_title, fontsize=14)

ax_sdd.stairs(sdd.counts, sdd.bins, fill=True, color='C0', label='1 channel per bin')
ax_sdd.plot(sdd_x, sdd_y, c='C1', label='likelihood fit')

ax_sdd.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax_sdd.set_xlim(sdd_xlim)

ax_sdd.set_xlabel(sdd_xlabel, horizontalalignment='right', x=1.0, fontsize=14)
ax_sdd.set_ylabel(sdd_ylabel, horizontalalignment='right', y=1.0, fontsize=14)

ax_sdd.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

ax_sdd.xaxis.set_minor_locator(AutoMinorLocator())
ax_sdd.yaxis.set_minor_locator(AutoMinorLocator())

ax_sdd.yaxis.offsetText.set_fontsize(12)
ax_sdd.tick_params(axis='both', which='major', labelsize=12)

ax_sdd.legend(loc='lower right', fontsize=12)

# text
mu = sdd_fit['values']['mu']
mu_err = sdd_fit['errors']['mu']
sigma = sdd_fit['values']['sigma']
sigma_err = sdd_fit['errors']['sigma']
a = sdd_fit['values']['a']
a_err = sdd_fit['errors']['a']

# these are matplotlib.patch.Patch properties
properties = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '\n'.join((
    r'$\mu=%.3f\pm%.3f$' % (mu, mu_err, ),
    r'$\sigma=%.3f\pm%.3f$' % (sigma, sigma_err, ),
    r'$a=%.0f\pm%.0f$' % (a, a_err, ),
    r'$\chi^2 \; / \; \mathrm{d.\!o.\!f.} = %.2f \; / \; %d=%.2f$' % (sdd_chi2, sdd_dof, sdd_chi2/sdd_dof, )))

# place a text box in upper left in axes coords
ax_sdd.text(0.9825, 0.95, textstr, transform=ax_sdd.transAxes, fontsize=14,
            ha='right', va='top', ma='left', bbox=properties)

textstr = '\n'.join((
    r'$\mu=%.6f\pm%.6f$ keV' % (electronvolts(mu, sdd_α, sdd_β)/1e3, mu_err/sdd_β/1e3, ),
    r'$\sigma=%.6f\pm%.6f$ keV' % (sigma/sdd_β/1e3, sigma_err/sdd_β/1e3, )))

# place a text box in upper left in axes coords
ax_sdd.text(0.9825, 0.525, textstr, transform=ax_sdd.transAxes, fontsize=14,
            ha='right', va='top', ma='left', bbox=properties)

if sdd_logy:
    ax_sdd.set_yscale('log')
    ax_sdd.set_ylim(bottom=0.5)
else:
    ax_sdd.set_ylim(bottom=0)

# GPC plot

gpc_title = r'{}'.format(gpc_title)
ax_gpc.set_title(gpc_title, fontsize=14)

ax_gpc.stairs(gpc.counts, gpc.bins, fill=True, color='C0')
ax_gpc.plot(gpc_x, gpc_y, c='C1')

if gpc_merge_bins:
    ax_gpc.stairs(gpc.counts, gpc.bins, fill=True, color='C0', label='{} channels per bin'.format(gpc_merge_bins))
    ax_gpc.stairs(gpc_.counts, gpc_.bins, fill=True, color='C2', label='1 channel per bin')
    ax_gpc.plot(gpc_x, gpc_y, c='C1', label='likelihood fit')
    ax_gpc.legend(loc='lower right', fontsize=12)
else:
    ax_gpc.stairs(gpc.counts, gpc.bins, fill=True, color='C0', label='1 channel per bin')
    ax_gpc.plot(gpc_x, gpc_y, c='C1', label='likelihood fit')
    ax_gpc.legend(loc='lower right', fontsize=12)

ax_gpc.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax_gpc.set_xlim(gpc_xlim)

ax_gpc.set_xlabel(gpc_xlabel, horizontalalignment='right', x=1.0, fontsize=14)
ax_gpc.set_ylabel(gpc_ylabel, horizontalalignment='right', y=1.0, fontsize=14)

ax_gpc.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

ax_gpc.xaxis.set_minor_locator(AutoMinorLocator())
ax_gpc.yaxis.set_minor_locator(AutoMinorLocator())

ax_gpc.yaxis.offsetText.set_fontsize(12)
ax_gpc.tick_params(axis='both', which='major', labelsize=12)

# text
mu = gpc_fit['values']['mu']
mu_err = gpc_fit['errors']['mu']
sigma = gpc_fit['values']['sigma']
sigma_err = gpc_fit['errors']['sigma']
a = gpc_fit['values']['a']
a_err = gpc_fit['errors']['a']

textstr = '\n'.join((
    r'$\mu=%.3f\pm%.3f$' % (mu, mu_err, ),
    r'$\sigma=%.3f\pm%.3f$' % (sigma, sigma_err, ),
    r'$a=%.0f\pm%.0f$' % (a, a_err, ),
    r'$\chi^2 \; / \; \mathrm{d.\!o.\!f.} = %.2f \; / \; %d=%.2f$' % (gpc_chi2, gpc_dof, gpc_chi2/gpc_dof, )))

# place a text box in upper left in axes coords
ax_gpc.text(0.9825, 0.95, textstr, transform=ax_gpc.transAxes, fontsize=14,
            ha='right', va='top', ma='left', bbox=properties)

if gpc_logy:
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

