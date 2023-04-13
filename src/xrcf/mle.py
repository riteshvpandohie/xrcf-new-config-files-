import logging
import numpy as np
from scipy.stats import norm
from scipy.special import loggamma
from iminuit import Minuit

logger = logging.getLogger('xrcf')

#//////////////////////////////////////////////////////////////////////////////
# likelihood function
#//////////////////////////////////////////////////////////////////////////////

def make_nll(bin_edges, counts):
    """
    make_nll(bin_edges, counts)

    Closure of a negative log-likelihood function for a normal (Gaussian)
    distribution under the assumption that the probability of measuring a
    single value is given by the Poisson probability mass function.

    Parameters
    ----------
    bin_edges : array_like of floats
        The bin edges along the first dimension.
    counts : array_like of floats
        Single-dimensional histogram.

    Returns
    -------
    nll : function

    """
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    def nll(mu, sigma, a):
        _nll = 0.
        for idx in range(len(counts)):
            x = bin_centers[idx]
            z = counts[idx]
            if z < 1e-9:
                continue
            f = a * norm.pdf(x, mu, sigma) * sigma * np.sqrt(2*np.pi)
            if f < 1e-9:
                continue
            _nll += f - z*np.log(f) + loggamma(z+1)
        return _nll
    return nll

#//////////////////////////////////////////////////////////////////////////////
# minimization routine
#//////////////////////////////////////////////////////////////////////////////

def minimize(bins, counts, mu, sigma, a, start=0, stop=-1):

    #--------------------------------------------------------------------------
    # select training sample
    #--------------------------------------------------------------------------
    flag = np.zeros(bins.shape, dtype=bool)
    flag[start:stop+1] = True

    bins_train = bins[flag]
    counts_train = counts[flag[:-1]][:-1]

    #//////////////////////////////////////////////////////////////////////////
    # run minuit
    #//////////////////////////////////////////////////////////////////////////

    # construct minuit object
    minuit = Minuit(
        fcn=make_nll(bins_train, counts_train),
        mu=mu,
        sigma=sigma,
        a=a,
        )

    # set step sizes for minuit's numerical gradient estimation
    minuit.errors = (1e-5, 1e-5, 1e-1)

    # set limits for each parameter
    minuit.limits = [ None, (0, None), (0, None) ]

    # set errordef for a negative log-likelihood (NLL) function
    minuit.errordef = Minuit.LIKELIHOOD
    # minuit.errordef = Minuit.LEAST_SQUARES  # for a least-squares cost function

    # run migrad minimizer
    minuit.migrad(ncall=1000000)

    # print estimated parameters
    logger.debug('minuit.values:\n{}'.format(minuit.values))

    # run hesse algorithm to compute asymptotic errors
    minuit.hesse()

    # print estimated errors on estimated parameters
    logger.debug('minuit.errors:\n{}'.format(minuit.errors))

    # run minos algorithm to compute confidence intervals
    minuit.minos()

    # print estimated parameters
    logger.debug('minuit.params:\n{}'.format(minuit.params))

    # print estimated errors on estimated parameters
    logger.debug('minuit.merrors:\n{}'.format(minuit.merrors))

    output = {
            'counts'       : counts,
            'bins'         : bins,
            'counts_train' : counts_train,
            'bins_train'   : bins_train,
            'values'       : minuit.values,
            'errors'       : minuit.errors,
            'params'       : minuit.params,
            'merrors'      : minuit.merrors,
        }

    return output


