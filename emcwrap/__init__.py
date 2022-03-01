#!/bin/python
# -*- coding: utf-8 -*-

import yaml
import emcee
import os
import numpy as np
import importlib.machinery
import importlib.util
import scipy.optimize as so
import scipy.stats as ss
from grgrlib import parse_yaml
from grgrlib import load_as_module as load_model

from .emcwrap_dists import inv_gamma_spec, InvGammaDynare

from pydsge.to_emcwrap import get_prior


def find_mode_simple(lprob, init, frozen_prior, sd=True, verbose=False, **kwargs):

    def objective(x):

        ll = -lprob(x) - lprior(x, frozen_prior)

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
        keys = g["keys"][...]

        for key in keys:
            exec('reader.%s = g[key][...]' %key.decode())

    return reader
