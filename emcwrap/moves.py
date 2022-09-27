# -*- coding: utf-8 -*-

from emcee.moves.red_blue import RedBlueMove
from emcee.moves.de import DEMove
import numpy as np
import scipy.stats as ss


class DIMEMove(RedBlueMove):
    r"""A proposal using adaptive differential-independence mixture enseble MCMC.

    This is the `Differential-Independence Mixture Ensemble proposal` as developed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_ (previousy ADEMC).

    Parameters
    ----------
    sigma : float, optional
        standard deviation of the Gaussian used to stretch the proposal vector.
    gamma : float, optional
        mean stretch factor for the proposal vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}` as recommended by `ter Braak (2006) <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_.
    aimh_prob : float, optional
        probability to draw an adaptive independence Metropolis Hastings (AIMH) proposal. By default this is set to :math:`0.05`.
    nsamples_proposal_dist : int
        window size used to calculate the rolling-window covariance estimate. By default this is the number of unique elements in the proposal mean and covariance :math:`0.5 \mathrm{ndim}(\mathrm{ndim}+3)`.
    df_proposal_dist : float
        degrees of freedom of the multivariate t distribution used for AIMH proposals. Defaults to :math:`10`.
    """

    def __init__(
        self,
        sigma=1.0e-5,
        gamma=None,
        aimh_prob=0.05,
        nsamples_proposal_dist=None,
        df_proposal_dist=10,
        **kwargs
    ):

        self.sigma = sigma
        self.gamma = gamma
        self.aimh_prob = aimh_prob
        self.nsamples_proposal_dist = nsamples_proposal_dist
        self.dft = df_proposal_dist

        kwargs["nsplits"] = 1
        super(DIMEMove, self).__init__(**kwargs)

    def setup(self, coords):
        # set some sane defaults

        nchain, npar = coords.shape
        self.g0 = self.gamma
        self.npdist = self.nsamples_proposal_dist

        if self.g0 is None:
            # pure MAGIC
            self.g0 = 2.38 / np.sqrt(2 * npar)

        if self.npdist is None:
            # more MAGIC
            self.npdist = int(0.5 * npar * (npar + 3))

        if not hasattr(self, "cov"):
            # even more MAGIC
            self.cov = np.cov(coords.T, ddof=1)
            self.mean = np.mean(coords, axis=0)

    def propose(self, model, state):
        # wrap original propose to get the boolean array of accepted proposals
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

        # skip if chain did not update
        if hasattr(self, "accepted") and sum(self.accepted) > 1:

            xaccepted = x[self.accepted]
            naccepted = sum(self.accepted)

            # only use newly accepted to update AIMH proposal distribution
            ncov = np.cov(xaccepted.T, ddof=1)
            nmean = np.mean(xaccepted, axis=0)

            self.cov = (self.npdist - naccepted) / (self.npdist - 1) * self.cov + (
                naccepted - 1
            ) / (self.npdist - 1) * ncov
            self.mean = (
                1 - naccepted / self.npdist
            ) * self.mean + naccepted / self.npdist * nmean

        if hasattr(self, "cov"):
            # draw chains for AIMH sampling
            xchnge = random.rand(nchain) <= self.aimh_prob

            # draw alternative candidates and calculate their proposal density
            dist = ss.multivariate_t(
                self.mean, self.cov * (self.dft - 2) / self.dft, df=self.dft
            )
            xcand = dist.rvs(sum(xchnge), random_state=random)
            lprop_old = dist.logpdf(x[xchnge])
            lprop_new = dist.logpdf(xcand)

            # update proposals and factors
            q[xchnge] = xcand
            factors[xchnge] = lprop_old - lprop_new

        return q, factors


ADEMove = DIMEMove  # set alias for compatibility
