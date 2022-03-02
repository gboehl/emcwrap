#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as ss
from scipy.special import gammaln


class InvGammaDynare(ss.rv_continuous):

    name = "inv_gamma_dynare"

    def _logpdf(self, x, s, nu):

        if x < 0:
            lpdf = -np.inf
        else:
            lpdf = (
                np.log(2)
                - gammaln(nu / 2)
                - nu / 2 * (np.log(2) - np.log(s))
                - (nu + 1) * np.log(x)
                - 0.5 * s / np.square(x)
            )

        return lpdf

    def _pdf(self, x, s, nu):
        return np.exp(self._logpdf(x, s, nu))


def inv_gamma_spec(mu, sigma):

    # directly taken from dynare/matlab

    def ig1fun(nu):
        return (
            np.log(2 * mu ** 2)
            - np.log((sigma ** 2 + mu ** 2) * (nu - 2))
            + 2 * (gammaln(nu / 2) - gammaln((nu - 1) / 2))
        )

    nu = np.sqrt(2 * (2 + mu ** 2 / sigma ** 2))
    nu2 = 2 * nu
    nu1 = 2
    err = ig1fun(nu)
    err2 = ig1fun(nu2)

    if err2 > 0:
        while nu2 < 1e12:  # Shift the interval containing the root.
            nu1 = nu2
            nu2 = nu2 * 2
            err2 = ig1fun(nu2)
            if err2 < 0:
                break
        if err2 > 0:
            raise ValueError(
                "[inv_gamma_spec:] Failed in finding an interval containing a sign change! You should check that the prior variance is not too small compared to the prior mean..."
            )

    # Solve for nu using the secant method.
    while abs(nu2 / nu1 - 1) > 1e-14:
        if err > 0:
            nu1 = nu
            if nu < nu2:
                nu = nu2
            else:
                nu = 2 * nu
                nu2 = nu
        else:
            nu2 = nu
        nu = (nu1 + nu2) / 2
        err = ig1fun(nu)

    s = (sigma ** 2 + mu ** 2) * (nu - 2)

    if (
        abs(
            np.log(mu)
            - np.log(np.sqrt(s / 2))
            - gammaln((nu - 1) / 2)
            + gammaln(nu / 2)
        )
        > 1e-7
    ):
        raise ValueError(
            "[inv_gamma_spec:] Failed in solving for the hyperparameters!")
    if abs(sigma - np.sqrt(s / (nu - 2) - mu * mu)) > 1e-7:
        raise ValueError(
            "[inv_gamma_spec:] Failed in solving for the hyperparameters!")

    return s, nu
