#!/bin/python
# -*- coding: utf-8 -*-

import os
import emcee
import numpy as np
import scipy.stats as ss
import scipy.optimize as so
from grgrlib import parse_yaml
from .dists import inv_gamma_spec, InvGammaDynare


def get_prior(prior, verbose=False):

    prior_lst = []
    initv, lb, ub = [], [], []

    if verbose:
        print("Adding parameters to the prior distribution...")

    for pp in prior:

        dist = prior[str(pp)]

        if len(dist) == 3:
            initv.append(None)
            lb.append(None)
            ub.append(None)
            ptype, pmean, pstdd = dist
        elif len(dist) == 6:
            initv.append(eval(str(dist[0])))
            lb.append(dist[1])
            ub.append(dist[2])
            ptype, pmean, pstdd = dist[3:]
        else:
            raise NotImplementedError(
                "Prior specification must either be 3 or 6 inputs but is %s." % pp)

        # simply make use of frozen distributions
        if str(ptype) == "uniform":
            prior_lst.append(ss.uniform(loc=pmean, scale=pstdd - pmean))

        elif str(ptype) == "normal":
            prior_lst.append(ss.norm(loc=pmean, scale=pstdd))

        elif str(ptype) == "gamma":
            b = pstdd ** 2 / pmean
            a = pmean / b
            prior_lst.append(ss.gamma(a, scale=b))

        elif str(ptype) == "beta":
            a = (1 - pmean) * pmean ** 2 / pstdd ** 2 - pmean
            b = a * (1 / pmean - 1)
            prior_lst.append(ss.beta(a=a, b=b))

        elif str(ptype) == "inv_gamma":

            def targf(x):
                y0 = ss.invgamma(x[0], scale=x[1]).std() - pstdd
                y1 = ss.invgamma(x[0], scale=x[1]).mean() - pmean
                return np.array([y0, y1])

            ig_res = so.root(targf, np.array([4, 4]), method="lm")

            if ig_res["success"] and np.allclose(targf(ig_res["x"]), 0):
                prior_lst.append(ss.invgamma(
                    ig_res["x"][0], scale=ig_res["x"][1]))
            else:
                raise ValueError(
                    f"Can not find inverse gamma distribution with mean {pmean} and std {pstdd}.")

        elif str(ptype) == "inv_gamma_dynare":
            s, nu = inv_gamma_spec(pmean, pstdd)
            ig = InvGammaDynare()(s, nu)
            prior_lst.append(ig)

        else:
            raise NotImplementedError(
                f" Distribution {ptype} not implemented.")
        if verbose:
            if len(dist) == 3:
                print(
                    "   - %s as %s with mean %s and std/df %s"
                    % (pp, ptype, pmean, pstdd)
                )
            if len(dist) == 6:
                print(
                    "   - %s as %s (%s, %s). Init @ %s, with bounds (%s, %s)"
                    % (pp, ptype, pmean, pstdd, dist[0], dist[1], dist[2])
                )

    return prior_lst, lambda x: log_prior(x, prior_lst), initv, (lb, ub)


def log_prior(par, frozen_prior):

    prior = 0
    for i, pl in enumerate(frozen_prior):
        prior += pl.logpdf(par[i])

    return prior


def find_mode_simple(lprob, init, frozen_prior, sd=True, verbose=False, **kwargs):

    def objective(x):

        ll = -lprob(x) - log_prior(x, frozen_prior)

        if verbose:
            print(-ll)

        return ll

    if not 'method' in kwargs:
        kwargs['method'] = 'Nelder-Mead'

    # minimize objective
    result = so.minimize(objective, init, **kwargs)

    # Compute standard deviation if required
    if sd:
        H, nfev_total = hessian(objective, result.x,
                                nfev=result.nfev, f_x0=result.fun)
        Hinv = np.linalg.inv(H)
        x_sd = np.sqrt(np.diagonal(Hinv))
    else:
        nfev_total = result.nfev
        x_sd = np.zeros_like(result.x)

    return result, x_sd, nfev_total


def save_to_backend(backend, content):

    with backend.open("a") as f:
        g = f[backend.name]
        g["keys"] = list(content.keys())
        for key in content:
            g[key] = content[key]

    return


def load_backend(backend):
    """just a shortcut"""
    reader = emcee.backends.HDFBackend(backend, read_only=True)

    storage_dict = {}
    with reader.open() as f:

        g = f[reader.name]
        if 'keys' in g:
            keys = g["keys"][...]

            for key in keys:
                exec('reader.%s = g[key][...]' % key.decode())

    return reader


def remove_backend(backend):
    """just a shortcut"""

    os.remove(backend)

    return


rm_backend = remove_backend
