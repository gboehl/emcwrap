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

    def __init__(self, sigma=1.0e-5, gamma=None, target=0.15, sma_decay=.7, verbose=False, **kwargs):
        self.sigma = sigma
        self.gamma = gamma
        self.target = target
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

        if self.g0 is None:
            # pure MAGIC
            self.g0 = 2.38 / np.sqrt(2 * npar)

        if self.sma is None:
            # more MAGIC
            self.sma = self.target*ndim


    def propose(self, model, state):
        state, accepted = super(ADEMove, self).propose(model, state)
        # self.accepance_fraction = sum(accepted)/len(accepted)
        self.naccepted = sum(accepted)
        self.state = state
        return state, accepted

    def get_proposal(self, x, dummy, random):

        # calculate distribution stats
        ndim, npar = x.shape
        mean = np.mean(x, axis=0)
        cov = np.cov(x.T)

        # get ndraws
        if self.state is not None:
            # decaying MA
            self.sma = self.sma_decay*self.sma + (1 - self.sma_decay)*self.naccepted
            sortinds = np.argsort(self.state.log_prob)
            ndraws = max(0, int(self.target*ndim - self.sma))
        else:
            ndraws = 0

        if ndraws:
            xchange = random.multivariate_normal(mean, cov, size=ndraws)
            x[sortinds[:ndraws]] = xchange

        if self.verbose and ndraws:
            print(f"[ADEMove:] resampling {ndraws} draw(s)/ MA-AF: {self.sma/ndim:0.2%}.")

        i0 = np.arange(ndim) + random.randint(ndim-1, size=ndim)
        i1 = np.arange(ndim) + random.randint(ndim-2, size=ndim)
        i1[i1 >= i0] += 1
        f = self.sigma * random.randn(ndim)
        q = x + self.g0*(x[i0 % ndim] - x[i1 % ndim]) + f[:,np.newaxis]

        return q, np.zeros(ndim, dtype=np.float64)

