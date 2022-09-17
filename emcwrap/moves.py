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

    def __init__(self, sigma=1.0e-5, gamma=None, target=0.18, threshold=None, sma_decay=.8, verbose=False, **kwargs):

        self.sigma = sigma
        self.gamma = gamma
        # for a gaussian posterior the expected acceptance ratio is 0.23 if the dimensionality is large
        self.target = target
        self.threshold = threshold
        self.sma_decay = sma_decay
        self.verbose = verbose

        self.naccepted = None
        self.state = None
        self.sma = None

        kwargs["nsplits"] = 1
        super(ADEMove, self).__init__(**kwargs)

    def setup(self, coords):

        ndim, npar = coords.shape

        self.g0 = self.gamma
        self.d0 = self.threshold

        if self.g0 is None:
            # pure MAGIC
            self.g0 = 2.38 / np.sqrt(2 * npar)

        if self.sma is None:
            # more MAGIC
            self.sma = self.target*ndim

        if self.d0 is None:
            # even more MAGIC
            self.d0 = npar*ndim/(3*npar - 1)/(ndim - 1)

    def propose(self, model, state):
        # wrap original propose
        state, accepted = super(ADEMove, self).propose(model, state)
        self.naccepted = sum(accepted)
        self.state = state
        return state, accepted

    def get_proposal(self, x, dummy, random):

        # calculate distribution stats
        ndim, npar = x.shape

        # get ndraws
        if self.state is not None:
            # weighted MA
            self.sma = self.sma_decay*self.sma + (1 - self.sma_decay)*self.naccepted
            sortinds = np.argsort(self.state.log_prob)
            ndraws = max(0, int(self.target*ndim - self.sma))
        else:
            sortinds = np.arange(ndim)
            ndraws = 0

        xleft = x[sortinds[ndraws:]]

        # calculate distribution stats for remaining chains
        mean = np.mean(xleft, axis=0)
        cov = np.cov(xleft.T)

        # calculate squared Mahalanobis distances
        # einsum is probably the most efficient way to do this
        d2s = np.einsum('ij,ji->i', (xleft - mean) @ np.linalg.inv(cov), (xleft - mean).T)
        maxprobs = npar/d2s
        # calculate "outliers"
        outbool = maxprobs < self.d0

        if ndraws or sum(outbool):
            mean = np.mean(xleft[~outbool], axis=0)
            cov = np.cov(xleft[~outbool].T)
            xchange = random.multivariate_normal(mean, cov, size=ndraws+sum(outbool))
            x[sortinds[:ndraws]] = xchange[:ndraws]
            x[sortinds[ndraws:][outbool]] = xchange[ndraws:]

        if self.verbose and ndraws:
            print(f"[ADEMove:] resampling {ndraws+sum(outbool)} draw(s) ({sum(outbool)} with high leverage); MA-AF: {self.sma/ndim:0.2%}.")

        i0 = np.arange(ndim) + random.randint(ndim-1, size=ndim)
        i1 = np.arange(ndim) + random.randint(ndim-2, size=ndim)
        i1[i1 >= i0] += 1
        f = self.sigma * random.randn(ndim)
        q = x + self.g0*(x[i0 % ndim] - x[i1 % ndim]) + f[:,np.newaxis]

        return q, np.zeros(ndim, dtype=np.float64)

