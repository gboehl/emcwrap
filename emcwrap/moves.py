# -*- coding: utf-8 -*-

import numpy as np

from emcee.moves.red_blue import RedBlueMove
from emcee.moves.de import DEMove
from emcee.state import State
from emcee.ensemble import walkers_independent

__all__ = ["ADEMove", "DEMove"]


class ADEMove(RedBlueMove):
    r"""A proposal using adaptive differential evolution following
    `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://arxiv.org/abs/1311.5229>`_.
    Args:
        sigma (float): The standard deviation of the Gaussian used to stretch
            the proposal vector.
        gamma0 (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}`
            as recommended by the two references.
        threshold: (Optional[float]): the threshold proability for adapting an alternative walker. Should be _very_ low (e.g. `1e-32`).
        log_threshold: (Optional[float]): the threshold proability for adapting an alternative walker in log space. Should be _very_ low (e.g. `-100`).
    """

    def __init__(self, sigma=1.0e-5, gamma0=None, threshold=0, log_threshold=None, verbose=False, **kwargs):
        self.sigma = sigma
        self.gamma0 = gamma0
        self.verbose = verbose

        if log_threshold and threshold:
            raise RuntimeError(
                "Either provide `threshold` OR `log_threshold`.")

        log_threshold_from_threshold = np.log(
            threshold) if threshold else -np.inf
        self.log_threshold = log_threshold if log_threshold else log_threshold_from_threshold

        kwargs["nsplits"] = 3
        super(ADEMove, self).__init__(**kwargs)

    def setup(self, coords):
        self.g0 = self.gamma0
        if self.g0 is None:
            # Pure MAGIC:
            ndim = coords.shape[1]
            self.g0 = 2.38 / np.sqrt(2 * ndim)

    def get_proposal(self, s, c, random):
        Ns = len(s)
        Nc = list(map(len, c))
        ndim = s.shape[1]
        q = np.empty((Ns, ndim), dtype=np.float64)
        f = self.sigma * random.randn(Ns)
        for i in range(Ns):
            w = np.array([c[j][random.randint(Nc[j])] for j in range(2)])
            random.shuffle(w)
            g = np.diff(w, axis=0) * self.g0 + f[i]
            q[i] = s[i] + g
        return q, np.zeros(Ns, dtype=np.float64)

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance, but adapted for use with ADEMove
        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.
        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        if not walkers_independent(state.coords):
            raise ValueError(
                "Current state has a large condition number. "
                "Make sure that your walkers are linearly independent for the "
                "best performance"
            )

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)

        prop_adapts, accept_adapts = 0, 0

        for split in range(self.nsplits):
            S1 = inds == split

            # Get the splits of the ensemble.
            sets = [state.coords[inds == j] for j in range(self.nsplits)]

            # Get probs
            probs = state.log_prob[S1]

            # compare each walker _within_ split
            shuffled_inds = np.arange(len(probs))
            model.random.shuffle(shuffled_inds)

            # adapt those that are extremely unlikely
            adapt = probs - probs[shuffled_inds] < self.log_threshold

            s = sets[split]
            # get proposal for adapted walker.
            # if proposal is rejected, nothing will happen
            s[adapt] = s[shuffled_inds][adapt]
            c = sets[:split] + sets[split + 1:]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(s, c, model.random)

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)

            # Loop over the walkers and update them accordingly.
            for i, (j, f, nlp) in enumerate(
                zip(all_inds[S1], factors, new_log_probs)
            ):
                lnpdiff = f + nlp - state.log_prob[j]
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True

            new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
            state = self.update(state, new_state, accepted, S1)

            prop_adapts += sum(adapt)
            accept_adapts += sum(accepted[S1][adapt])

        if self.verbose and prop_adapts:
            print(
                f"(ADEMove:) Accepted {accept_adapts} of {prop_adapts} proposed adaptation(s).")

        return state, accepted
