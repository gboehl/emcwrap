# -*- coding: utf-8 -*-

from emcee.moves.red_blue import RedBlueMove
from emcee.moves.de import DEMove
from scipy.special import logsumexp
import numpy as np
import scipy.stats as ss


class DIMEMove(RedBlueMove):
    r"""A proposal using adaptive differential-independence mixture ensemble MCMC.

    This is the `Differential-Independence Mixture Ensemble proposal` as developed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_ (previousy ADEMC).

    Parameters
    ----------
    sigma : float, optional
        standard deviation of the Gaussian used to stretch the proposal vector.
    gamma : float, optional
        mean stretch factor for the proposal vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}` as recommended by `ter Braak (2006) <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_.
    aimh_prob : float, optional
        probability to draw an adaptive independence Metropolis Hastings (AIMH) proposal. By default this is set to :math:`0.05`.
    df_proposal_dist : float
        degrees of freedom of the multivariate t distribution used for AIMH proposals. Defaults to :math:`10`.
    """

    def __init__(
        self,
        sigma=1.0e-5,
        gamma=None,
        aimh_prob=0.05,
        df_proposal_dist=10,
        **kwargs
    ):

        self.sigma = sigma
        self.g0 = gamma
        self.aimh_prob = aimh_prob
        self.dft = df_proposal_dist

        kwargs["nsplits"] = 1
        super(DIMEMove, self).__init__(**kwargs)

    def setup(self, coords):
        # set some sane defaults

        nchain, npar = coords.shape

        if self.g0 is None:
            # pure MAGIC
            self.g0 = 2.38 / np.sqrt(2 * npar)

        if not hasattr(self, "cov"):
            # even more MAGIC
            self.cov = np.eye(npar)
            self.mean = np.zeros(npar)
            self.accepted = np.ones(nchain, dtype=bool)
            self.cumlweight = -np.inf

    def propose(self, model, state):
        # wrap original propose to get the boolean array of accepted proposals
        self.lprobs = state.log_prob
        state, accepted = super(DIMEMove, self).propose(model, state)
        self.accepted = accepted
        return state, accepted

    def get_proposal(self, x, dummy, random):
        # actual proposal function

        # calculate distribution stats
        nchain, npar = x.shape

        # differential evolution: draw the indices of the complementary chains
        i0 = np.arange(nchain) + random.randint(1, nchain, size=nchain)
        i1 = np.arange(nchain) + random.randint(1, nchain - 1, size=nchain)
        i1[i1 >= i0] += 1
        # add small noise and calculate proposal
        f = self.sigma * random.randn(nchain)
        q = x + self.g0 * (x[i0 % nchain] - x[i1 % nchain]) + f[:, np.newaxis]
        factors = np.zeros(nchain, dtype=np.float64)

        # log weight of current ensemble
        lweight = logsumexp(self.lprobs) + \
            np.log(sum(self.accepted)) - np.log(nchain)

        # calculate stats for current ensemble
        ncov = np.cov(x.T, ddof=1)
        nmean = np.mean(x, axis=0)

        # update AIMH proposal distribution
        newcumlweight = np.logaddexp(self.cumlweight, lweight)
        self.cov = np.exp(self.cumlweight - newcumlweight) * \
            self.cov + np.exp(lweight - newcumlweight) * ncov
        self.mean = np.exp(self.cumlweight - newcumlweight) * \
            self.mean + np.exp(lweight - newcumlweight) * nmean
        self.cumlweight = newcumlweight

        # draw chains for AIMH sampling
        xchnge = random.rand(nchain) <= self.aimh_prob

        # draw alternative candidates and calculate their proposal density
        dist = ss.multivariate_t(
            self.mean, self.cov * (self.dft - 2) / self.dft, df=self.dft, allow_singular=True)
        xcand = dist.rvs(sum(xchnge), random_state=random)
        lprop_old = dist.logpdf(x[xchnge])
        lprop_new = dist.logpdf(xcand)

        # update proposals and factors
        q[xchnge] = xcand
        factors[xchnge] = lprop_old - lprop_new

        return q, factors


ADEMove = DIMEMove  # set alias for compatibility
