# -*- coding: utf-8 -*-

import numpy as np
from emcee.moves.red_blue import RedBlueMove
from emcee.moves.de import DEMove


class ADEMove(RedBlueMove):
    r"""A proposal using adaptive differential evolution.

    This is the `Adaptive Differential evolution proposal` as suggested by 
    <https://gregorboehl.com/live/ademc_boehl.pdf>`_.
    Args:
        sigma (float): The standard deviation of the Gaussian used to stretch
            the proposal vector.
        gamma (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}`
            as recommended by the two references.
        threshold: (Optional[float]): the threshold maximum proability below which an alternative chain will be proposed.
        verbose: (Optional[bool]): print message whenever chains are exchanged.
    """

    def __init__(self, sigma=1.0e-5, gamma=None, threshold=None, verbose=False, **kwargs):
        self.sigma = sigma
        self.gamma = gamma
        self.delta = threshold
        self.verbose = verbose
        kwargs["nsplits"] = 1
        super(ADEMove, self).__init__(**kwargs)

    def setup(self, coords):
        self.g0 = self.gamma
        self.d0 = self.delta
        npar, ndim = coords.shape

        if self.g0 is None:
            # pure MAGIC:
            self.g0 = 2.38 / np.sqrt(2 * ndim)

        if self.d0 is None:
            # more MAGIC: 
            # a rather conservative default
            self.d0 = npar*ndim/(3*npar - 1)/(ndim - 1)

    def get_proposal(self, x, dummy, random):

        # calculate distribution stats
        ndim, npar = x.shape
        mean = np.mean(x, axis=0)
        cov = np.cov(x.T)

        # calculate squared Mahalanobis distances
        # einsum is probably the most efficient way to do this
        d2s = np.einsum('ij,ji->i', (x - mean) @ np.linalg.inv(cov), (x - mean).T)
        maxprobs = npar/d2s

        # calculate outliers
        outbool = maxprobs < self.d0
        xchange = random.multivariate_normal(mean, cov, size=sum(outbool))
        # substitute outliers for sample from Gaussian distribution
        x[outbool] = xchange

        if self.verbose and sum(outbool):
            print(f"[ADEMove:] resampling {sum(outbool)} draw(s) with (smallest) prob. < {min(maxprobs[outbool]):.2%}.")

        i0 = np.arange(ndim) + random.randint(ndim-1, size=ndim)
        i1 = np.arange(ndim) + random.randint(ndim-2, size=ndim)
        i1[i1 >= i0] += 1
        f = self.sigma * random.randn(ndim)
        q = x + self.g0*(x[i0 % ndim] - x[i1 % ndim]) + f[:,np.newaxis]

        return q, np.zeros(ndim, dtype=np.float64)

