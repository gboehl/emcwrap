# -*- coding: utf-8 -*-

import numpy as np
from emcee.moves.red_blue import RedBlueMove
from emcee.moves.de import DEMove
import scipy.stats as ss


class ODEMove(RedBlueMove):
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
        super(ODEMove, self).__init__(**kwargs)

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
        state, accepted = super(ODEMove, self).propose(model, state)
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
            print(f"[ODEMove:] resampling {ndraws+sum(outbool)} draw(s) ({sum(outbool)} with high leverage); MA-AF: {self.sma/ndim:0.2%}.")

        i0 = np.arange(ndim) + random.randint(ndim-1, size=ndim)
        i1 = np.arange(ndim) + random.randint(ndim-2, size=ndim)
        i1[i1 >= i0] += 1
        f = self.sigma * random.randn(ndim)
        q = x + self.g0*(x[i0 % ndim] - x[i1 % ndim]) + f[:,np.newaxis]

        return q, np.zeros(ndim, dtype=np.float64)


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

    def __init__(self, sigma=1.0e-5, gamma=None, aimh_prob=.05, nsamples_proposal_dist=None, df_proposal_dist=10, **kwargs):

        self.sigma = sigma
        self.gamma = gamma
        self.aimh_prob = aimh_prob
        self.npdist = nsamples_proposal_dist
        self.dft = df_proposal_dist

        self.accepted = None
        self.cov = None

        kwargs["nsplits"] = 1
        super(ADEMove, self).__init__(**kwargs)

    def setup(self, coords):

        nchain, npar = coords.shape
        self.g0 = self.gamma

        if self.g0 is None:
            # pure MAGIC
            self.g0 = 2.38 / np.sqrt(2 * npar)

        if self.npdist is None:
            # more MAGIC
            self.npdist = int(0.5*npar*(npar + 3))

        if self.cov is None:
            # even more MAGIC
            self.cov = np.cov(coords.T, ddof=1)
            self.mean = np.mean(coords, axis=0)

    def propose(self, model, state):
        # wrap original propose to grasp some information
        state, accepted = super(ADEMove, self).propose(model, state)
        self.accepted = accepted
        return state, accepted

    def get_proposal(self, x, dummy, random):

        # calculate distribution stats
        nchain, npar = x.shape

        # differential evolution: draw the indices of the complementary chains
        i0 = np.arange(nchain) + random.randint(nchain-1, size=nchain)
        i1 = np.arange(nchain) + random.randint(nchain-2, size=nchain)
        i1[i1 >= i0] += 1
        # add small noise and calculate proposal
        f = self.sigma * random.randn(nchain)
        q = x + self.g0*(x[i0 % nchain] - x[i1 % nchain]) + f[:,np.newaxis]
        factors = np.zeros(nchain, dtype=np.float64)

        # skip in zeroth' iteration or if chain did not update
        if self.accepted is not None and sum(self.accepted) > 1:

            xaccepted = x[self.accepted]
            naccepted = sum(self.accepted)

            # only use newly accepted to update Gaussian
            ncov = np.cov(xaccepted.T, ddof=1)
            nmean = np.mean(xaccepted, axis=0)

            self.cov = (self.npdist - naccepted)/(self.npdist-1)*self.cov + (naccepted-1)/(self.npdist-1)*ncov
            self.mean = (1 - naccepted/self.npdist)*self.mean + naccepted/self.npdist*nmean

        # also skip in zeroth' iteration
        if self.cov is not None:
            # draw chains for AIMH sampling
            xchnge = random.rand(nchain) <= self.aimh_prob

            # draw alternative candidates and calculate their proposal density
            dist = ss.multivariate_t(self.mean, self.cov*(self.dft-2)/self.dft, df=self.dft)
            xcand = dist.rvs(sum(xchnge), random_state=random)
            lprop_old = dist.logpdf(x[xchnge])
            lprop_new = dist.logpdf(xcand)

            # update proposals and factors
            q[xchnge] = xcand
            factors[xchnge] = lprop_old - lprop_new

        return q, factors

